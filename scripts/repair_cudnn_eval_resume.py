"""One-time audited metadata migration for the E0-complete/E1-before-first-step CUDA failure.

No checkpoints or scores are modified. Only the precisely reviewed source transition is allowed.
"""
import argparse
import copy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT, read_json, write_json, sha256, digest, source_digest

BEFORE = "06551af1d46f06f3ccbbcb1ba20c03a716cf8afd1585090c9a6fd9bcbd835ff6"
AFTER = "5e3e17b5e152d775ad58b4d593b3c54d30a6419c01ac84fc46a56d1d97f8054c"


def repair(run_id, apply=False):
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('Invalid run id')
    run = ROOT / 'runs' / run_id
    if source_digest() != AFTER:
        raise ValueError('Source differs from the reviewed CUDA-only repair')
    backup = run / 'repairs' / 'cudnn_aux_backward_20261022'
    record = backup / 'migration.json'
    if record.exists():
        print('Migration already recorded; no further changes.')
        return
    protocol_path = run / 'protocol.json'
    protocol = read_json(protocol_path)
    if protocol['source_hash'] != BEFORE:
        raise ValueError('Unexpected original protocol version')
    state = read_json(run / 'status.json')
    if (state['dataset'],state['seed'],state['method'],state['state']) != ('Eedi',2024,'E1','interrupted'):
        raise ValueError('This migration only supports the verified Eedi/E1 first-step interruption')
    e0 = run / 'Eedi/E0/seed2024'
    e1 = run / 'Eedi/E1/seed2024'
    # No treatment has completed an update/epoch in the known failure scenario.
    if list(e1.glob('*.pt')) or list((e1 / 'epochs').glob('*.json')):
        raise ValueError('E1 already has training state; refusing migration')
    all_checkpoints = list(run.glob('*/*/seed*/*.pt'))
    if any(p.parent != e0 for p in all_checkpoints):
        raise ValueError('Other trained groups exist; require a separate compatibility review')
    paths = [protocol_path,e0/'run_config.json',e0/'training_complete.json',
             e0/'eval/metrics.json',e0/'eval/evaluation_complete.json',e1/'run_config.json']
    values = {p:read_json(p) for p in paths}
    c0,c1 = values[e0/'run_config.json'],values[e1/'run_config.json']
    complete = values[e0/'training_complete.json']
    if c0['source_hash'] != BEFORE or c1['source_hash'] != BEFORE or c0['weight'] != 0:
        raise ValueError('Training configuration mismatch')
    if complete['epochs'] != 25 or complete['contract_hash'] != digest(c0):
        raise ValueError('E0 is not a verified completed baseline')
    if sha256(e0/'best.pt') != complete['best_sha256']:
        raise ValueError('E0 checkpoint changed')
    marker,metrics = values[e0/'eval/evaluation_complete.json'],values[e0/'eval/metrics.json']
    if sha256(e0/'eval/metrics.json') != marker['metrics_sha256'] or marker['contract'] != metrics['provenance']:
        raise ValueError('Evaluation provenance mismatch')
    if marker['contract']['source_hash'] != BEFORE or sha256(e0/'uid_ex_scores.json') != marker['contract']['scores_sha256']:
        raise ValueError('Scores or evaluation version changed')
    protected = {p.relative_to(run).as_posix():sha256(p) for p in
                 (e0/'best.pt',e0/'last.pt',e0/'uid_ex_scores.json',e0/'uid_ex_scores.logits.npy')}
    replacements = copy.deepcopy(values)
    for path in (protocol_path,e0/'run_config.json',e1/'run_config.json'):
        replacements[path]['source_hash'] = AFTER
    replacements[e0/'training_complete.json']['contract_hash'] = digest(replacements[e0/'run_config.json'])
    replacements[e0/'eval/metrics.json']['provenance']['source_hash'] = AFTER
    replacements[e0/'eval/metrics.json']['source_repair'] = {
        'previous_provenance':metrics['provenance'],
        'reason':'Auxiliary CUDA forward repair; E0 has zero fairness weight; evaluation math is unchanged.'}
    replacements[e0/'eval/evaluation_complete.json']['contract']['source_hash'] = AFTER
    print('Verified: E0 25 epochs and evaluation complete; E1 has no saved epoch. Checkpoints/scores will be preserved.')
    if not apply:
        print('Dry run only. Add --apply to migrate the six metadata files.')
        return
    for p in paths:
        target = backup / 'before' / p.relative_to(run)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(p,target)
        elif sha256(target) != sha256(p):
            raise ValueError('Backup differs; refusing overwrite')
    # Keep original metadata and hashes in an explicit repair record.
    for p in paths:
        if p.name != 'evaluation_complete.json':
            write_json(p,replacements[p])
    em = e0/'eval/evaluation_complete.json'
    replacements[em]['metrics_sha256'] = sha256(e0/'eval/metrics.json')
    write_json(em,replacements[em])
    if any(sha256(run/p) != h for p,h in protected.items()):
        raise ValueError('Protected artifact changed')
    write_json(record,{'before_source':BEFORE,'after_source':AFTER,'reason':'cuDNN eval-GRU backward repair',
                       'protected_unchanged':protected,'metadata_backups':[p.relative_to(run).as_posix() for p in paths]})
    print(f'Migration complete. Audit: {record}')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',default='merit_frequency_20261022')
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    repair(a.run_id,a.apply)
