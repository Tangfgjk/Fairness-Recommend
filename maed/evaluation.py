"""Student-aligned quality budgets and tail-risk reports, separate from scoring."""
import math
import numpy as np

from debias.io import TOP_NS
from .io import write_csv, write_json


def loss_statistics(delta, budget=.01):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Expected finite student deltas')
    harm = np.maximum(-delta, 0)
    worst = math.ceil(.1*len(delta))
    return {'mean_delta': float(delta.mean()), 'harm_count': int((delta < -budget).sum()),
            'HarmRate': float((delta < -budget).mean()),
            'Worst10PercentLoss': float(np.sort(harm)[-worst:].mean()),
            'MaxLoss': float(harm.max()), 'tail_students': worst,
            'population_quality_pass': bool(delta.mean() >= -budget-1e-12)}


def compare_results(new, baseline, method, reference='E0', budget=.01):
    summary = []; people = []
    rng = np.random.default_rng(20261023)
    for n in TOP_NS:
        a = {int(r['uid']): r for r in new['per_user'] if r['N'] == n}
        b = {int(r['uid']): r for r in baseline['per_user'] if r['N'] == n}
        if set(a) != set(b):
            raise ValueError('Risk audit student cohorts differ')
        uids = sorted(a)
        indices = rng.integers(0, len(uids), size=(1000, len(uids)))
        for metric in ('Ada', 'KG4Ex-NOV-Set'):
            delta = np.array([a[u][metric]-b[u][metric] for u in uids])
            stats = loss_statistics(delta, budget)
            estimates = delta[indices].mean(1)
            summary.append({'method': method, 'reference': reference, 'N': n, 'metric': metric,
                            'value': new['top_n'][str(n)][metric], **stats,
                            'paired_mean_CI95_low': float(np.quantile(estimates, .025)),
                            'paired_mean_CI95_high': float(np.quantile(estimates, .975))})
            people.extend({'method': method, 'reference': reference, 'uid': u, 'N': n, 'metric': metric,
                           'delta': float(d), 'loss': max(0., -float(d)), 'harm_gt_budget': bool(d < -budget)}
                          for u, d in zip(uids, delta))
    return summary, people


def save_comparison(output, results):
    summary = []; people = []; metrics = []
    for method, result in results.items():
        for n, values in result['top_n'].items():
            metrics.append({'method': method, 'N': n, **values})
        if method != 'R':
            a, b = compare_results(result, results['E0'], method)
            summary.extend(a); people.extend(b)
    write_csv(output/'all_metrics.csv', metrics)
    write_csv(output/'quality_budget_and_tail_risk.csv', summary)
    write_csv(output/'student_quality_deltas.csv', people)
    write_json(output/'quality_summary.json', {'budget': .01, 'reference': 'same-round E0',
               'methods': {m: {'passed_quality_positions': sum(r['population_quality_pass'] for r in summary if r['method']==m),
                                'quality_positions': sum(r['method']==m for r in summary)} for m in results if m != 'R'},
               'personal_hard_constraints': False, 'scope': 'snapshot diagnostic; no beta chosen from test results'})
