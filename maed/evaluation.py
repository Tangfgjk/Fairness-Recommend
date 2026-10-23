"""Student-aligned quality budgets and tail-risk reports, separate from scoring."""
import math
import numpy as np

from debias.io import TOP_NS
from .io import write_csv, write_json

QUALITY_BUDGETS = {'Ada': .01, 'KG4Ex-NOV-Set': .03}


def quality_budgets(budgets=None):
    values = dict(QUALITY_BUDGETS if budgets is None else budgets)
    if set(values) != set(QUALITY_BUDGETS) or any(not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('Explicit finite nonnegative Ada/NOV absolute budgets required')
    return values


def loss_statistics(delta, budget=.01, harm_threshold=.01):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Expected finite student deltas')
    if any(not np.isfinite(v) or v < 0 for v in (budget, harm_threshold)):
        raise ValueError('Budgets and harm thresholds must be nonnegative finite values')
    harm = np.maximum(-delta, 0)
    worst = math.ceil(.1*len(delta))
    return {'mean_delta': float(delta.mean()), 'quality_budget': float(budget),
            'harm_threshold': float(harm_threshold),
            'harm_count': int((delta < -harm_threshold).sum()),
            'HarmRate': float((delta < -harm_threshold).mean()),
            'individual_budget_exceed_count': int((delta < -budget).sum()),
            'individual_budget_exceed_rate': float((delta < -budget).mean()),
            'Worst10PercentLoss': float(np.sort(harm)[-worst:].mean()),
            'MaxLoss': float(harm.max()), 'tail_students': worst,
            'population_quality_pass': bool(delta.mean() >= -budget-1e-12)}


def compare_results(new, baseline, method, reference='E0', budgets=None, harm_threshold=.01):
    budgets = quality_budgets(budgets)
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
            budget = budgets[metric]
            stats = loss_statistics(delta, budget, harm_threshold)
            estimates = delta[indices].mean(1)
            summary.append({'method': method, 'reference': reference, 'N': n, 'metric': metric,
                            'value': new['top_n'][str(n)][metric], **stats,
                            'paired_mean_CI95_low': float(np.quantile(estimates, .025)),
                            'paired_mean_CI95_high': float(np.quantile(estimates, .975))})
            people.extend({'method': method, 'reference': reference, 'uid': u, 'N': n, 'metric': metric,
                           'delta': float(d), 'loss': max(0., -float(d)),
                           'quality_budget': budget, 'harm_threshold': harm_threshold,
                           'harm_gt_threshold': bool(d < -harm_threshold), 'harm_gt_budget': bool(d < -budget)}
                          for u, d in zip(uids, delta))
    return summary, people


def save_comparison(output, results, budgets=None):
    budgets = quality_budgets(budgets)
    summary = []; people = []; metrics = []
    for method, result in results.items():
        for n, values in result['top_n'].items():
            metrics.append({'method': method, 'N': n, **values})
        if method != 'R':
            a, b = compare_results(result, results['E0'], method, budgets=budgets)
            summary.extend(a); people.extend(b)
    write_csv(output/'all_metrics.csv', metrics)
    write_csv(output/'quality_budget_and_tail_risk.csv', summary)
    write_csv(output/'student_quality_deltas.csv', people)
    write_json(output/'quality_summary.json', {'quality_budgets': budgets, 'absolute_not_relative': True,
               'student_harm_threshold': .01, 'reference': 'same-round E0',
               'methods': {m: {'passed_quality_positions': sum(r['population_quality_pass'] for r in summary if r['method']==m),
                                'quality_positions': sum(r['method']==m for r in summary),
                                'joint_passed_N': sum(all(r['population_quality_pass'] for r in summary if r['method']==m and r['N']==n) for n in TOP_NS)} for m in results if m != 'R'},
               'personal_hard_constraints': False, 'scope': 'snapshot diagnostic; no beta chosen from test results'})
