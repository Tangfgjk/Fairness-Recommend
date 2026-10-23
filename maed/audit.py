"""M0 evidence ledger: distinguish observed array checks from unproven lineage."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .io import ROOT, read_json, write_json, sha256, write_csv


def _optional(path):
    return read_json(path) if path.exists() else None


def elapsed_axis(frame, basis, unit):
    """Replay the original time conversion without importing its CLI-only common module."""
    if basis == 'interaction_steps':
        return np.arange(len(frame), dtype=float)
    values = pd.to_numeric(frame['timestamp'], errors='coerce')
    if values.notna().all():
        result = values.to_numpy(dtype=float)
        if unit == 'milliseconds' or (unit == 'auto' and np.max(np.abs(result)) >= 1e12):
            return result/1000
        if unit == 'minutes':
            return result*60
        if unit == 'days':
            return result*86400
        if unit in ('auto', 'seconds'):
            return result
        raise ValueError('Unsupported timestamp unit')
    parsed = pd.to_datetime(frame['timestamp'], errors='raise')
    return parsed.astype('int64').to_numpy(dtype=float)/1e9


def audit_state(state, output):
    output = Path(output)
    front = state.data_root / state.dataset / 'front_features'
    protocol_path = front / 'protocol.json'
    protocol = read_json(protocol_path)
    frame = pd.read_csv(state.raw / 'interactions_all.csv', low_memory=False)
    split = pd.read_csv(state.raw / 'student_split.csv')
    expected = split.set_index('uid')['split']
    if not frame['uid'].map(expected).eq(frame['split']).all():
        raise ValueError('Raw interaction/student split mismatch')
    if not set(frame.response.unique()).issubset({0, 1}):
        raise ValueError('Responses must be binary')
    protocol_split_ok = (sorted(protocol['outer_train_uids']) == state.train.tolist() and
                         sorted(protocol['outer_test_uids']) == state.test.tolist())
    if not protocol_split_ok:
        raise ValueError('Front protocol and formal cohorts differ')
    ordered = frame.sort_values(['uid', 'sequence_order'], kind='stable')
    per_user = []
    intervals = []; interval_uids = []
    forget_manifest_path = front / 'forgetting/forgetting_manifest.json'
    forget_manifest = _optional(forget_manifest_path)
    basis = (forget_manifest or {}).get('time_basis', 'interaction_steps' if state.dataset.lower() == 'eedi' else 'timestamp_seconds')
    unit = (forget_manifest or {}).get('timestamp_unit', 'auto')
    kc_forget = np.asarray(read_json(front / 'stu2know_forget.json'), float)
    pending_deltas = []
    for u, group in ordered.groupby('uid', sort=True):
        times = elapsed_axis(group, basis, unit)
        last_seen = np.full(state.q.shape[1], np.nan)
        for e, t in zip(group.question.astype(int), times):
            last_seen[np.flatnonzero(state.q[e])] = t
        seen = np.isfinite(last_seen)
        delta = np.abs(times[-1]-last_seen[seen])
        pending_deltas.append((int(u), seen, delta))
        intervals.extend(delta.tolist()); interval_uids.extend([int(u)]*len(delta))
        per_user.append({'uid': int(u), 'split': str(expected.loc[u]), 'rows': len(group),
                         'sequence_min': int(group.sequence_order.min()), 'sequence_max': int(group.sequence_order.max()),
                         'duplicate_sequence_orders': int(group.sequence_order.duplicated().sum()),
                         'first_timestamp_raw': str(group.timestamp.iloc[0]), 'last_timestamp_raw': str(group.timestamp.iloc[-1]),
                         'effective_axis_end': float(times[-1]), 'backward_time_steps': int((np.diff(times)<0).sum()),
                         'history_scope': 'all supplied observed rows; no separate future cutoff'})
    delta = np.asarray(intervals); delta_uids = np.asarray(interval_uids)
    calibration = {'status': 'unconfirmed: missing forgetting manifest', 'time_basis': basis,
                   'missing_manifest': forget_manifest is None}
    if forget_manifest is not None:
        theta = float(forget_manifest['theta'])
        reconstructed = np.zeros_like(kc_forget)
        for u, seen, gaps in pending_deltas:
            reconstructed[u, seen] = 1-np.exp(-gaps/theta)
        calibration.update(theta=theta, theta_source=forget_manifest['theta_source'],
                           max_abs_kc_reconstruction_error=float(np.max(np.abs(reconstructed-kc_forget))))
        if forget_manifest['theta_source'] == 'auto_quantile':
            c = forget_manifest['calibration']; quantile = c['quantile']; goal = c['target_forgetting']
            def estimate(mask):
                values = delta[mask & (delta>0)]
                return float(np.quantile(values, quantile)/-math.log1p(-goal)) if len(values) else None
            all_theta = estimate(np.ones(len(delta), dtype=bool))
            train_theta = estimate(np.isin(delta_uids, state.train))
            calibration.update(all_cohort_theta_recomputed=all_theta, train_only_theta_recomputed=train_theta,
                               matches_all_cohort=bool(np.isclose(theta, all_theta, rtol=1e-6)),
                               matches_train_only=bool(train_theta is not None and np.isclose(theta, train_theta, rtol=1e-6)),
                               status='snapshot/transductive shared calibration; not evidence of future-answer use')
        else:
            calibration['status'] = 'manual shared theta recorded; original selection rationale requires provenance'
    frac_q = state.q/state.q.sum(1, keepdims=True)
    aggregate = kc_forget @ frac_q.T
    frontend_comparisons = {}
    for name, graph_values in [('stu2know_mastery.json', state.mastery), ('stu2ex_forget.json', state.forget)]:
        candidate = np.asarray(read_json(front / name), float)
        same_shape = candidate.shape == graph_values.shape
        frontend_comparisons[name] = {'same_shape': same_shape,
            'max_abs_difference': float(np.max(np.abs(candidate-graph_values))) if same_shape else None}
    mirt_path = front/'mirt/mirt_manifest.json'
    ektm_path = front/'ektm_mirt/ektm_manifest.json'
    mirt = read_json(mirt_path); ektm = _optional(ektm_path)
    ck_path = front/'ektm_mirt/best.pt'
    checkpoint = torch.load(ck_path, map_location='cpu', weights_only=False)
    checkpoint_keys = list(checkpoint)
    checkpoint_evidence = {k: checkpoint[k] for k in ('train_uids', 'test_uids', 'max_seq_len') if k in checkpoint}
    config = checkpoint.get('config', {})
    if isinstance(config, dict):
        checkpoint_evidence['config_scope'] = {k: v for k, v in config.items() if any(s in k for s in ('seq_len','train_uid','test_uid'))}
    del checkpoint
    q_mirt_path = front/'mirt/q_mirt.pt'
    qmirt = torch.load(q_mirt_path, map_location='cpu', weights_only=False)
    qmirt_train = list(map(int, qmirt.get('train_uids', [])))
    del qmirt
    files = [state.raw/'interactions_all.csv', state.raw/'student_split.csv', state.raw/'Q.txt', protocol_path,
             front/'stu2know_mastery.json', front/'stu2know_forget.json', front/'stu2ex_forget.json',
             mirt_path, ck_path, q_mirt_path, front/'semantic_kg_features/feature_manifest.json']
    files += [p for p in (forget_manifest_path, ektm_path) if p.exists()]
    # Hash feature files and code used for the lineage claims; absence stays explicit.
    files += [ROOT/'front_pipeline'/name for name in ('prepare_front_protocol.py', 'generate_forgetting.py',
              'train_q_mirt.py', 'train_ektm_mirt.py', 'build_er_graph.py')]
    file_hashes = {str(p): sha256(p) for p in files}
    result = {'dataset': state.dataset, 'state': state.provenance, 'files': file_hashes,
              'history_policy': protocol['test_history_policy'], 'cutoff_status': 'no supplied future evaluation cutoff',
              'snapshot_status': 'diagnostic use only until explicit protocol acceptance; not certified leakage-free',
              'cohorts_match': protocol_split_ok,
              'declared_shared_parameter_policy': protocol.get('shared_parameter_policy'),
              'declared_evaluation_label_policy': protocol.get('evaluation_label_policy'),
              'q_mirt_checkpoint_train_users_match': sorted(qmirt_train) == state.train.tolist(),
              'mirt_manifest': {k: mirt.get(k) for k in ('train_users','test_users_theta_adapted_with_frozen_item_parameters',
                                                        'test_theta_estimation','normalization')},
              'ektm_manifest_present': ektm is not None,
              'ektm_manifest_scope': {k: ektm.get(k) for k in ('training_users','test_users_frozen_inference_only','mastery_source')} if ektm else None,
              'ektm_checkpoint_keys': checkpoint_keys, 'ektm_checkpoint_scope_evidence': checkpoint_evidence,
              'mastery_replay_status': 'not replayed through frozen EKTM; hashes and export equality do not prove training lineage',
              'forgetting_calibration': calibration, 'frontend_graph_comparisons': frontend_comparisons,
              'exercise_forgetting_KC_mean_max_error': float(np.max(np.abs(aggregate-state.forget))),
              'individual_history_rows': len(frame), 'time_basis': basis,
              'original_files_modified': False, 'optimizer_updates': 0}
    write_csv(output/'student_history_scope.csv', per_user)
    write_json(output/'audit.json', result)
    text = [f'# {state.dataset} 教育状态审计', '',
            '- 状态时点：全部已提供历史的末端；没有独立未来评价截点。',
            '- 当前诊断不构成“严格 train-only、无时间泄漏”的认证。',
            f'- 遗忘轴：`{basis}`；Eedi 的 interaction_steps 不是实际时间。',
            f'- 遗忘标定证据：`{calibration}`',
            f'- EKTM 导出清单存在：{ektm is not None}；未重放完整前处理训练轨迹。',
            f'- Q-MIRT checkpoint 训练学生吻合：{result["q_mirt_checkpoint_train_users_match"]}。',
            '- 共享参数范围主要由源码/清单声明；来源缺失项保持待证实。',
            '- 正确历史集合仅供 NOV 评价；不输入 Q 评分或曝光目标。',
            '- 新诊断采用现有快照协议，不覆盖原状态，也不自动重做数据划分。', '']
    output.mkdir(parents=True, exist_ok=True)
    (output/'教育状态来源与时间边界审计.md').write_text('\n'.join(text), encoding='utf-8')
    return result
