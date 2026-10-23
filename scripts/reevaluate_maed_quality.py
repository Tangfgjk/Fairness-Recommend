"""Reassess existing M1 metrics; never infer, train, select beta or overwrite v1 results."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT, DATASETS, TOP_NS, read_json, write_json, sha256
from maed.io import source_hash, lock_json, write_csv
from maed.evaluation import compare_results, quality_budgets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/论文格式文档统计/maed_m0_m2_20261023')
    parser.add_argument('--baseline-run', type=Path, default=ROOT/'runs/merit_frequency_20261022')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/论文格式文档统计/maed_quality_reassessment_20261023')
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        raise ValueError('Reassessment output must be outside the historical results directory')
    budgets = quality_budgets()
    records = {}; originals = {}
    for dataset in DATASETS:
        root = args.source/dataset/'M1'
        marker = read_json(root/'stage_complete.json')
        contract = read_json(args.source/dataset/'source_contract.json')
        baseline = args.baseline_run/dataset/'E0/seed2024/eval/metrics.json'
        if sha256(baseline) != contract['E0']['eval/metrics.json']:
            raise ValueError('Baseline metrics changed')
        paths = sorted(root.glob('Q_beta*/eval/metrics.json'))
        if {p.parents[1].name for p in paths} != {'Q_beta0p0','Q_beta0p25','Q_beta0p5','Q_beta1p0'}:
            raise ValueError('Expected all four preregistered beta settings; do not select a subset')
        records[dataset] = (read_json(baseline), [])
        originals[str(baseline)] = sha256(baseline)
        originals[str(root/'stage_complete.json')] = sha256(root/'stage_complete.json')
        originals[str(args.source/dataset/'source_contract.json')] = sha256(args.source/dataset/'source_contract.json')
        for path in paths:
            if sha256(path) != marker['output_files'][path.relative_to(root).as_posix()]:
                raise ValueError(f'Historical result changed: {path}')
            originals[str(path)] = sha256(path)
            records[dataset][1].append((path.parents[1].name, read_json(path)))
    lock_json(args.output/'protocol.json', {'source_hash': source_hash(), 'source_files': originals,
              'old_budgets': {'Ada': .01, 'KG4Ex-NOV-Set': .01}, 'new_budgets': budgets,
              'interpretation': 'absolute difference, not relative percent',
              'revision_status': 'exploratory threshold change after observing v1; fixed prospectively for next round',
              'student_HarmRate_threshold_unchanged': .01, 'M2_gate_changed': False,
              'training_or_inference': False, 'beta_selection': False})
    rows = []; risks = []; people = []; summaries = []
    for dataset, (baseline, methods) in records.items():
        for method, result in methods:
            stat, individual = compare_results(result, baseline, method, budgets=budgets)
            risks.extend({'dataset': dataset, **r} for r in stat)
            people.extend({'dataset': dataset, **r} for r in individual)
            failed = []; old_pass_count = 0
            for n in TOP_NS:
                values = result['top_n'][str(n)]; base = baseline['top_n'][str(n)]
                da = values['Ada']-base['Ada']; dn = values['KG4Ex-NOV-Set']-base['KG4Ex-NOV-Set']
                old_ok = da >= -.01-1e-12 and dn >= -.01-1e-12
                new_ok = da >= -.01-1e-12 and dn >= -.03-1e-12
                old_pass_count += int(old_ok)
                if not new_ok:
                    failed.append(n)
                # Check student-level and stored population averages use the same grain.
                for metric, delta in [('Ada', da), ('KG4Ex-NOV-Set', dn)]:
                    actual = next(r['mean_delta'] for r in stat if r['N']==n and r['metric']==metric)
                    if abs(actual-delta) > 1e-9:
                        raise ValueError('Stored mean differs from paired student result')
                rows.append({'dataset': dataset, 'method': method, 'N': n,
                             'Ada_E0': base['Ada'], 'Ada_new': values['Ada'], 'delta_Ada': da,
                             'NOV_E0': base['KG4Ex-NOV-Set'], 'NOV_new': values['KG4Ex-NOV-Set'], 'delta_NOV': dn,
                             'old_joint_pass': old_ok, 'new_joint_pass': new_ok,
                             'Ep_t0.5': values['Ep_t0.5'], 'DED': values['DED'], 'delta_DED': values['DED']-base['DED']})
            summaries.append({'dataset': dataset, 'method': method, 'old_joint_passed_N': old_pass_count,
                              'new_joint_passed_N': len(TOP_NS)-len(failed), 'total_N': len(TOP_NS),
                              'failed_N': ','.join(map(str,failed)), 'all_N_pass': not failed})
    write_csv(args.output/'逐N新旧门槛复评.csv', rows)
    write_csv(args.output/'模型通过情况汇总.csv', summaries)
    write_csv(args.output/'学生尾部风险汇总.csv', risks)
    write_csv(args.output/'逐学生损失分布.csv', people)
    write_json(args.output/'summary.json', {'new_budgets': budgets, 'all_methods': summaries,
               'no_new_training_or_inference': True, 'historical_scores_and_metrics_unchanged': True,
               'next_round_rule_fixed': True, 'historical_reassessment_is_exploratory': True})
    lines = ['# MAED M1 新质量门槛复评（2026-10-23）', '',
             'Ada 绝对下降≤0.01；KG4Ex-NOV-Set 绝对下降≤0.03。相对同轮 E0，每个 N 单独验收。',
             '本次门槛调整发生在前期结果已知之后；历史复评属于探索性调整，新门槛从下一轮固定使用。',
             '原评分、指标、旧预算结果及 M2 门槛均未覆盖或修改；不根据本表选择 beta。',
             '学生 HarmRate 继续按下降超过0.01报告，并额外给出超过各指标新预算的人数。', '',
             '|数据集|方法|旧门槛通过N数|新门槛通过N数|仍未通过的N|', '|---|---|---:|---:|---|']
    for r in summaries:
        lines.append(f'|{r["dataset"]}|{r["method"]}|{r["old_joint_passed_N"]}/10|{r["new_joint_passed_N"]}/10|{r["failed_N"] or "无"}|')
    lines += ['', '## 主设置 beta=0.5', '']
    for r in summaries:
        if r['method'] == 'Q_beta0p5':
            lines.append(f'- {r["dataset"]}：{r["new_joint_passed_N"]}/10；未通过N：{r["failed_N"] or "无"}。')
    lines += ['', '保留当前Q结构，优先诊断M2。残余质量问题继续跟踪；质量门槛放宽不改变M2尚未通过的结论。', '']
    (args.output/'新质量门槛复评结论_20261023.md').write_text('\n'.join(lines), encoding='utf-8')
    if any(sha256(Path(p)) != h for p,h in originals.items()):
        raise ValueError('Historical source changed during reassessment')
    write_json(args.output/'complete.json', {'source_files_unchanged': True, 'optimizer_updates': 0,
               'outputs': {p.name:sha256(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='complete.json'}})
    for row in summaries:
        print(row)
    print(args.output)


if __name__ == '__main__':
    main()
