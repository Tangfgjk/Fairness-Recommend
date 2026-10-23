"""M1 scoring and M2 training-cohort-only exposure diagnostics."""
import gc
from pathlib import Path
import numpy as np
import torch

from debias.io import TOP_NS, read_json, write_json, sha256
from debias.scoring import fixed_statistics
from debias.prepare import prepared
from debias.context import load_scores
from debias.evaluation import evaluate_file, write_score_matrix
from .io import lock_json, write_csv
from .scoring import EducationScale, fuse_logits, score_rec
from .model import load_frozen, export_logits
from .exposure import rule_targets, exposure_loss, exposure_snapshot, smooth_top_n
from .evaluation import save_comparison


def checked_test_logits(state, source):
    source = Path(source)
    rows = read_json(source/'uid_ex_scores.json')
    if [int(r['uid'].removeprefix('uid')) for r in rows] != state.test.tolist():
        raise ValueError('Stored test logit UID row order mismatch')
    logits = np.load(source/'uid_ex_scores.logits.npy').astype(float)
    probability = np.array([r['scores'] for r in rows])
    if logits.shape != (len(state.test), len(state.q)) or not np.isfinite(logits).all():
        raise ValueError('Bad original true logit matrix')
    if not np.allclose(torch.sigmoid(torch.tensor(logits)).numpy(), probability, atol=2e-7, rtol=2e-6):
        raise ValueError('True logits and original exported probabilities differ')
    marker = read_json(source/'eval/evaluation_complete.json')
    if marker['contract']['scores_sha256'] != sha256(source/'uid_ex_scores.json'):
        raise ValueError('Original score artifact checksum changed')
    if marker['metrics_sha256'] != sha256(source/'eval/metrics.json'):
        raise ValueError('Original metrics checksum changed')
    return logits, probability


def run_m1(state, source, output, protocol):
    output = Path(output)
    ctx, arrays, manifest = prepared(state.dataset)
    z, probabilities = checked_test_logits(state, source)
    scale = EducationScale.fit(state, clip=protocol['distance_clip'])
    write_json(output/'education_scale.json', scale.to_dict())
    results = {'E0': read_json(source/'eval/metrics.json')}
    # Keep E0's float32-probability tie semantics exactly for beta=0.
    base_rank = np.argsort(-probabilities, axis=1, kind='stable')
    logit_rank = np.argsort(-z, axis=1, kind='stable')
    tie_audit = {'E0_true_logit_vs_serialized_probability_Top50_different_users':
                 int(np.any(base_rank[:, :50] != logit_rank[:, :50], axis=1).sum()),
                 'beta0_export_policy': 'reuse original probabilities exactly; positive beta uses actual fused logits',
                 'evaluation_scope': 'exploratory endpoint-state fusion, not a validated future-outcome method'}
    for beta in protocol['betas']:
        method = 'Q_beta'+str(beta).replace('.', 'p')
        folder = output/method
        scores = probabilities if beta == 0 else fuse_logits(z, state.distance[state.test], scale, beta)
        write_score_matrix(folder/'uid_ex_scores.json', ctx, scores)
        results[method] = evaluate_file(ctx, arrays, manifest, folder/'uid_ex_scores.json', folder/'eval',
                                       baseline=source/'uid_ex_scores.json')
        if beta == 0:
            if not np.array_equal(scores, load_scores(source/'uid_ex_scores.json', ctx)):
                raise ValueError('beta=0 failed exact serialized score bypass')
        print(f'{state.dataset} M1 {method} complete', flush=True)
    rules = -state.distance[state.test]
    folder = output/'R'
    write_score_matrix(folder/'uid_ex_scores.json', ctx, rules)
    results['R'] = evaluate_file(ctx, arrays, manifest, folder/'uid_ex_scores.json', folder/'eval',
                                 baseline=source/'uid_ex_scores.json')
    save_comparison(output, results, budgets=protocol['quality_budgets'])
    write_json(output/'tie_and_scope_audit.json', tie_audit)
    lines = [f'# {state.dataset} M1：冻结 ConvE 的教育评分融合', '',
             '所有 beta 均为运行前登记；未根据测试结果选优。R 是规则参考。',
             '本结果使用既有末端状态，属于诊断；没有更新模型参数。', '',
             '|方法|N|Ada|KG4Ex-NOV-Set|Ep@0.5|DED|', '|---|---:|---:|---:|---:|---:|']
    for method, result in results.items():
        for n in TOP_NS:
            m = result['top_n'][str(n)]
            lines.append(f'|{method}|{n}|{m["Ada"]:.6f}|{m["KG4Ex-NOV-Set"]:.6f}|{m["Ep_t0.5"]:.6f}|{m["DED"]:.6f}|')
    (output/'M1教育评分融合结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(output/'complete.json', {'optimizer_updates': 0, 'betas': protocol['betas'],
               'source_checkpoint': sha256(source/'best.pt'), 'test_hyperparameter_selection': False,
               'quality_summary': read_json(output/'quality_summary.json')})


def training_logits(state, source, output, device):
    output = Path(output)
    contract = {'checkpoint': sha256(source/'best.pt'), 'state': state.provenance['state_hash'],
                'uids': state.train.tolist(), 'columns': 'Q exercise-row order', 'score': 'true logit'}
    lock_json(output/'train_logit_contract.json', contract)
    path = output/'train_logits.npy'
    marker = output/'train_logits_complete.json'
    if marker.exists():
        if read_json(marker)['sha256'] != sha256(path):
            raise ValueError('Cached training logits changed')
        values = np.load(path)
        if values.shape != (len(state.train), len(state.q)) or not np.isfinite(values).all():
            raise ValueError('Bad cached training logit matrix')
        return values
    model, bundle = load_frozen(state, source, device)
    # Verify this exact export path against the already stored test logits first.
    expected, _ = checked_test_logits(state, source)
    # Match the original array_split batch shape: cuDNN may select different
    # convolution kernels for a four-user probe and the original ~64-user batch.
    probe_users = np.array_split(state.test, max(1, int(np.ceil(len(state.test)/64))))[0]
    probe = export_logits(model, bundle, state, probe_users, batch_size=len(probe_users))
    error = float(np.max(np.abs(probe-expected[:len(probe_users)])))
    if not np.allclose(probe, expected[:len(probe_users)], atol=3e-5, rtol=3e-5):
        raise ValueError('Original checkpoint/export alignment failed')
    values = export_logits(model, bundle, state, state.train)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    write_json(marker, {'sha256': sha256(path), 'checkpoint': contract['checkpoint'], 'optimizer_updates': 0,
                       'export_alignment_max_error': error, 'probe_batch_users': len(probe_users)})
    del model, bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values


def _changed(before, after, ns):
    a = np.argsort(-before, axis=1, kind='stable')
    b = np.argsort(-after, axis=1, kind='stable')
    return [int(sum(set(x[:n]) != set(y[:n]) for x, y in zip(a, b))) for n in ns]


def _model_direction_probe(state, source, uids, scale, beta, temperature, device, factors):
    """Finite parameter-direction probes; temporary copy, no optimizer, always restore."""
    model, bundle = load_frozen(state, source, device)
    target = rule_targets(state.training_distance(uids), TOP_NS)
    d = torch.tensor(state.distance[uids], dtype=next(model.parameters()).dtype, device=device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    saved = [p.detach().clone() for p in parameters]
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    rows = []
    try:
        with fixed_statistics(model), torch.backends.cudnn.flags(enabled=False):
            z = fuse_logits(score_rec(model, bundle, uids, len(state.q)), d, scale, beta)
            initial = z.detach().cpu().double().numpy()
            loss, _ = exposure_loss(z, target, TOP_NS, temperature)
            grad = torch.autograd.grad(loss, parameters, allow_unused=True)
            norm = float(torch.sqrt(sum(g.double().square().sum() for g in grad if g is not None)))
            pnorm = float(torch.sqrt(sum(p.detach().double().square().sum() for p in parameters)))
            if not np.isfinite(norm) or norm == 0:
                return {'status': 'zero/nonfinite exposure gradient; not passed', 'gradient_norm': norm}
            before = exposure_snapshot(initial, target, TOP_NS, temperature)
            for factor in factors:
                with torch.no_grad():
                    for p, old, g in zip(parameters, saved, grad):
                        p.copy_(old if g is None else old-factor*pnorm/norm*g)
                    perturbed = fuse_logits(score_rec(model, bundle, uids, len(state.q)), d, scale, beta).cpu().double().numpy()
                rows.append({'relative_parameter_step': factor, **exposure_snapshot(perturbed, target, TOP_NS, temperature),
                             'changed_user_sets': _changed(initial, perturbed, TOP_NS)})
        return {'status': 'diagnostic parameter-copy directions; not training', 'training_users': uids.tolist(),
                'gradient_norm': norm, 'initial': before, 'probes': rows,
                'parameter_update_saved': False}
    finally:
        with torch.no_grad():
            for p, old in zip(parameters, saved):
                p.copy_(old)
        if any(not torch.equal(v, buffers[k]) for k, v in model.named_buffers()):
            raise ValueError('Frozen model diagnostic changed BN/feature buffers')
        del model, bundle


def run_m2(state, source, output, protocol, device='cpu'):
    output = Path(output)
    train_z = training_logits(state, source, output, device)
    scale = EducationScale.fit(state, clip=protocol['distance_clip'])
    beta = protocol['primary_beta']
    fused = fuse_logits(train_z, state.distance[state.train], scale, beta)
    centered = fused-fused.mean(1, keepdims=True)
    logit_iqr = float(np.quantile(centered, .75)-np.quantile(centered, .25))
    score_scale = max(logit_iqr, 1e-3)
    rng = np.random.default_rng(protocol['diagnostic_seed'])
    selected = np.sort(rng.choice(len(state.train), min(protocol['matrix_users'], len(state.train)), replace=False))
    uids = state.train[selected]
    initial = fused[selected]
    target = rule_targets(state.training_distance(uids), TOP_NS)
    full_target = rule_targets(state.training_distance(state.train), TOP_NS)
    write_json(output/'training_demand_targets.json', {'uids': state.train.tolist(), 'top_ns': list(TOP_NS),
               'target': full_target.tolist(), 'test_users_in_target': False})
    all_runs = []; invariants = []
    for factor in protocol['temperature_factors']:
        temperature = factor*score_scale
        base = exposure_snapshot(initial, target, TOP_NS, temperature)
        # Fixed perturbations demonstrate soft-score/actual-rank discrepancies.
        rank = np.argsort(-initial, axis=1, kind='stable')
        sorted_z = np.take_along_axis(initial, rank, axis=1)
        gaps = -np.diff(sorted_z, axis=1)
        positive = gaps[gaps>1e-12]
        jitter_size = float(positive.min()/4) if len(positive) else 0.
        # A monotone perturbation preserves exact ties as well as strict ordering.
        jitter = jitter_size*np.tanh(initial/score_scale)
        crossing = initial.copy()
        for i in range(len(initial)):
            crossing[i, rank[i, 10]] = initial[i, rank[i, 9]]+score_scale*.01
        for name, modified in [('common_shift', initial+10*score_scale), ('score_shrink', initial*.5),
                               ('no_crossing_jitter', initial+jitter), ('cross_N10_boundary', crossing)]:
            snap = exposure_snapshot(modified, target, TOP_NS, temperature)
            invariants.append({'temperature_factor': factor, 'perturbation': name,
                               'soft_mean_TV': snap['soft_mean_TV'],
                               'soft_change': snap['soft_mean_TV']-base['soft_mean_TV'],
                               'hard_N10_TV': snap['hard_TV'][1],
                               'changed_N10_user_sets': _changed(initial, modified, (10,))[0]})
        z = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        initial_trace = {**base, 'step': 0, 'changed_user_sets': [0]*len(TOP_NS)}
        trace = [initial_trace]
        for step in range(protocol['matrix_steps']):
            loss, _ = exposure_loss(z, target, TOP_NS, temperature)
            (grad,) = torch.autograd.grad(loss, z)
            rms = float(torch.sqrt(torch.mean(grad**2)))
            if not np.isfinite(rms) or rms < 1e-20:
                break
            z = (z.detach()-protocol['matrix_step_scale']*score_scale/rms*grad).requires_grad_(True)
            current = z.detach().numpy()
            trace.append({'step': step+1, **exposure_snapshot(current, target, TOP_NS, temperature),
                          'changed_user_sets': _changed(initial, current, TOP_NS)})
        end = trace[-1]
        # Fixed before results: soft reduction alone never passes this gate.
        passed = (end['soft_mean_TV'] < base['soft_mean_TV']-1e-6 and
                  end['hard_TV'][1] < base['hard_TV'][1]-1e-6 and
                  np.mean(end['hard_TV']) < np.mean(base['hard_TV'])-1e-6 and
                  end['changed_user_sets'][1] > 0)
        run = {'temperature_factor': factor, 'temperature': temperature, 'trace': trace,
               'matrix_diagnostic_gate_pass': bool(passed)}
        all_runs.append(run)
        print(f'{state.dataset} M2 temp={factor}: gate={passed}, hard@10 {base["hard_TV"][1]:.4f}->{end["hard_TV"][1]:.4f}', flush=True)
    primary = next(r for r in all_runs if r['temperature_factor'] == protocol['primary_temperature_factor'])
    probe_uids = uids[:protocol['model_probe_users']]
    direction = _model_direction_probe(state, source, probe_uids, scale, beta, primary['temperature'], device,
                                       protocol['model_relative_steps'])
    probes = direction.get('probes', [])
    before = direction.get('initial')
    model_gate = bool(before and any(p['soft_mean_TV'] < before['soft_mean_TV']-1e-6 and
                      p['hard_TV'][1] < before['hard_TV'][1]-1e-6 and
                      np.mean(p['hard_TV']) < np.mean(before['hard_TV'])-1e-6 and
                      p['changed_user_sets'][1]>0 for p in probes))
    w = smooth_top_n(torch.tensor(initial, dtype=torch.float64), 10, primary['temperature'])
    numerical = {'max_row_mass_error_N10': float((w.sum(1)-10).abs().max()),
                 'max_shift_difference_N10': float((w-smooth_top_n(torch.tensor(initial+10*score_scale),10,primary['temperature'])).abs().max())}
    result = {'dataset': state.dataset, 'training_users': uids.tolist(), 'beta': beta,
              'temperature_calibration': 'training-row-centered logit IQR', 'logit_scale': score_scale,
              'soft_hard_diagnostics': all_runs, 'numerical_invariants': numerical,
              'model_direction_probe': direction, 'model_direction_gate_pass': model_gate,
              'primary_matrix_gate_pass': primary['matrix_diagnostic_gate_pass'],
              'ready_for_M4': bool(primary['matrix_diagnostic_gate_pass'] and model_gate),
              'readiness_scope': 'M2 mechanism gate only; does not replace M0 protocol review or quality gates',
              'optimizer_updates': 0, 'diagnostic_matrix_steps_only': protocol['matrix_steps'],
              'temporary_model_parameter_probes': True, 'saved_model_weights_modified': False,
              'test_exposure_used_for_optimization': False}
    write_json(output/'diagnostics.json', result)
    write_csv(output/'perturbations.csv', invariants)
    trace_rows = []
    for r in all_runs:
        for t in r['trace']:
            for j,n in enumerate(TOP_NS):
                trace_rows.append({'temperature_factor': r['temperature_factor'], 'step': t['step'], 'N': n,
                    'soft_TV': t['soft_TV'][j], 'hard_TV': t['hard_TV'][j], 'changed_user_sets': t['changed_user_sets'][j]})
    write_csv(output/'soft_hard_exposure_trace.csv', trace_rows)
    (output/'M2曝光代理诊断.md').write_text(
        f'# {state.dataset} M2 曝光代理诊断\n\n'
        f'- 主温度矩阵诊断通过：{primary["matrix_diagnostic_gate_pass"]}\n'
        f'- 原模型参数方向诊断通过：{model_gate}\n'
        f'- M2 机制条件合并通过：{result["ready_for_M4"]}\n'
        '- 未使用测试人群曝光构造目标或选温度；所有温度与扰动完整记录。\n'
        '- 只修改矩阵副本、临时参数副本，没有 optimizer，也未保存修改后的模型。\n'
        '- ready_for_M4 只表示本项机制诊断，不表示 M0 或最终教育质量已经通过。\n', encoding='utf-8')
    return result
