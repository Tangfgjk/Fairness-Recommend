"""Read-only provenance freeze/verification. Never trains or rewrites run artifacts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RELEASE = 'MAED-v1'
DIRECTORY = ROOT / 'docs' / RELEASE
MANIFEST = DIRECTORY / '冻结清单_20261024.json'
ENVIRONMENT = DIRECTORY / '环境记录_20261024.json'
RUN = ROOT / 'runs/maed_controlled_monitored_seed2024_20261023'
OLD = ROOT / 'runs/merit_frequency_20261022'
REPORTS = ROOT / 'runs/论文格式文档统计'
VERIFIED = REPORTS / 'maed_seed2024_mechanism_R_verified_20261024'
SOURCE_HASH = '6e414a4fac60e2a384d652bf96af49054ff8140eaaad0c6c95f29ddd7295179d'
LEGACY_HASH = '5e3e17b5e152d775ad58b4d593b3c54d30a6419c01ac84fc46a56d1d97f8054c'
DATASETS = ('Eedi', 'algebra2005', 'XES3G5M-sub-small')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8') + b'\n'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inside(root, relative):
    """Reject outside/absolute inventory paths, including symlink escapes."""
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Invalid inventory path: {relative}')
    result = (root / relative).resolve()
    result.relative_to(root.resolve())
    return result


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def environment():
    import torch
    versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                if d.metadata.get('Name')}
    try:
        driver = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
            timeout=10, stderr=subprocess.STDOUT).decode(errors='replace').strip()
    except (OSError, subprocess.SubprocessError) as exc:
        driver = {'unavailable': type(exc).__name__}
    return {'scope': 'Environment observed at freeze; not a historical training-time capture',
            'captured_UTC': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': dict(sorted(versions.items(), key=lambda x: x[0].lower())),
            'torch_version': torch.__version__, 'cuda_build': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(),
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'driver_query': driver, 'training_deterministic': False}


def verify_inventory(root, entries):
    failures = []
    for name, info in entries.items():
        path = inside(root, name)
        if not path.is_file():
            failures.append((name, 'missing'))
        elif path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            failures.append((name, 'changed'))
    if failures:
        raise ValueError(f'{len(failures)} frozen files failed verification: {failures[:10]}')


def check_source():
    from maed.io import source_hash
    from debias.io import source_digest
    if source_hash() != SOURCE_HASH or source_digest() != LEGACY_HASH:
        raise ValueError('Production implementation differs from the completed training protocols')


def freeze():
    if MANIFEST.exists():
        # Repeat invocation only verifies; never replaces an established inventory.
        return verify(False)
    if git('tag', '--list', RELEASE).strip():
        raise ValueError('Release tag already exists; do not replace it')
    check_source()
    from maed.state import load_state
    from maed.io import input_hashes
    if read(RUN / 'status.json') != {
        'state': 'complete', 'stages': 9, 'training_jobs': 6,
        'report': str(REPORTS / RUN.name)}:
        raise ValueError('Expected completed nine-stage/six-training run')
    protocol = read(RUN / 'protocol.json')
    if (protocol['source_hash'], protocol['seed'], protocol['beta'], protocol['M4_exposure_weight']) != (
            SOURCE_HASH, 2024, .5, .01):
        raise ValueError('Unexpected formal run protocol')
    # Only this new archive environment file is created. Model/data/run files stay untouched.
    if not ENVIRONMENT.exists():
        ENVIRONMENT.write_bytes(encode(environment()))
    paths = {}
    expected = {}

    def add(path, group, expected_hash=None):
        path = Path(path).resolve()
        rel = path.relative_to(ROOT.resolve()).as_posix()
        inside(ROOT, rel)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.setdefault(rel, set()).add(group)
        if expected_hash is not None:
            if rel in expected and expected[rel] != expected_hash:
                raise ValueError(f'Conflicting recorded source hash: {rel}')
            expected[rel] = expected_hash

    def tree(path, group):
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(path)
        for file in sorted(path.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.tmp':
                add(file, group)

    # Include tracked and non-ignored new code/docs/tests, excluding the self-referential manifest.
    names = git('ls-files', '-c', '-o', '--exclude-standard', '-z').decode('utf-8').split('\0')
    for name in sorted(set(filter(None, names))):
        path = ROOT / name
        if path.resolve() != MANIFEST.resolve():
            add(path, 'repository')
    add(ENVIRONMENT, 'repository')
    tree(RUN, 'formal_run')
    tree(REPORTS / RUN.name, 'original_paper_tables')
    tree(VERIFIED, 'verified_R_mechanisms')
    for name in ('protocol.json', 'status.json'):
        add(OLD / name, 'prior_comparisons')
    datasets, models = {}, {}
    for dataset in DATASETS:
        state = load_state(dataset)
        for rel, value in input_hashes(state).items():
            add(ROOT / rel, 'formal_data', value)
        context = ROOT / 'artifacts/debias_context' / dataset
        tree(context, 'fixed_matching_context')
        matching = read(context / 'manifest.json')
        add(context / 'pairs.npz', 'fixed_matching_context', matching['pairs_sha256'])
        tree(ROOT / 'artifacts/baseline_original' / dataset / 'seed2024', 'archived_original_seed2024')
        assets = read(RUN / 'assets' / dataset / 'assets.json')
        if assets['state_hash'] != state.provenance['state_hash']:
            raise ValueError(f'Educational state changed: {dataset}')
        for source, value in assets['source_files'].items():
            add(source, 'temperature_reference', value)
        datasets[dataset] = {
            'graph': state.graph.relative_to(ROOT).as_posix(),
            'raw': state.raw.relative_to(ROOT).as_posix(),
            'train_students': len(state.train), 'test_students': len(state.test),
            'items': len(state.q), 'KCs': state.q.shape[1],
            'state_hash': assets['state_hash'], 'matching_contract': matching['contract'],
            'matching_context_id': matching['context_id'],
            'scale': assets['scale'], 'temperatures': assets['temperatures']}
        for method in ('E0', 'E1', 'E2', 'R'):
            stage = OLD / dataset / method / 'seed2024'
            tree(stage, 'prior_comparisons')
            add(stage / 'eval/metrics.json', 'prior_comparisons')
            if method in ('E0', 'E1'):
                receipt = read(stage / 'training_complete.json')
                add(stage / 'best.pt', 'prior_comparisons', receipt['best_sha256'])
        for method in ('M3', 'M4'):
            stage = RUN / dataset / method / 'seed2024'
            receipt = read(stage / 'training_complete.json')
            if receipt['epochs'] != 25 or len(list((stage / 'epochs').glob('*.json'))) != 25:
                raise ValueError(f'Incomplete epoch history: {stage}')
            for name, value in receipt['checkpoints'].items():
                add(stage / name, 'formal_run', value)
            # Existing stage receipts cover scores, config and all epoch logs as well.
            for rel, value in read(stage / 'stage_complete.json')['outputs'].items():
                add(stage / rel, 'formal_run', value)
            models[f'{dataset}/{method}/seed2024'] = {
                'directory': stage.relative_to(ROOT).as_posix(),
                'best_epoch': receipt['best_epoch'], 'checkpoints': receipt['checkpoints'],
                'checkpoint_payload': 'model, optimizer and Python/NumPy/Torch/CUDA/auxiliary RNG; see training.py'}
    # Verify and retain the exact source closure already used by the observational audit.
    for source, value in read(VERIFIED / 'audit_manifest.json')['sources'].items():
        add(source, 'audit_source', value)
    entries = {}
    for i, (rel, groups) in enumerate(sorted(paths.items()), 1):
        path = ROOT / rel
        value = sha256(path)
        if rel in expected and value != expected[rel]:
            raise ValueError(f'Historical recorded checksum changed: {rel}')
        entries[rel] = {'sha256': value, 'bytes': path.stat().st_size, 'groups': sorted(groups)}
        if i % 250 == 0:
            print(f'Fingerprinted {i}/{len(paths)} files', flush=True)
    value = {
        'release': RELEASE, 'release_ref': f'refs/tags/{RELEASE}',
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'source_HEAD_before_archive_commit': git('rev-parse', 'HEAD').decode().strip(),
        'branch': git('branch', '--show-current').decode().strip(),
        'commit_identity_policy': 'Resolve annotated tag after archive commit; no self-referential SHA',
        'scope': 'Stages 1-3; no retraining, no historical rewrite, paper chapter deferred',
        'data_backup': False, 'training_seed': 2024,
        'training_source_hash': SOURCE_HASH, 'inherited_source_digest': LEGACY_HASH,
        'protocol': protocol, 'datasets': datasets, 'models': models,
        'checkpoint_payload_scope': 'File hashes preserve existing payloads; no checkpoint serialization changes',
        'quality_reference': {'Ada_max_absolute_decline': .01, 'NOV_max_absolute_decline': .03,
                              'Algebra2005_M4_original_gate_fail_Ns': [20, 40, 45, 50],
                              'accepted_tradeoff_does_not_relabel_failures': True},
        'inventory': entries, 'file_count': len(entries),
        'total_bytes': sum(v['bytes'] for v in entries.values()),
        'group_counts': dict(sorted(Counter(g for v in entries.values() for g in v['groups']).items())),
        'manifest_self_integrity': 'Git annotated release tag protects this manifest; manifest excludes itself'}
    # Exclusive creation prevents overwriting even if another process creates it meanwhile.
    with MANIFEST.open('xb') as stream:
        stream.write(encode(value))
    print(json.dumps({'frozen_files': len(entries), 'bytes': value['total_bytes'],
                      'new_training': False, 'tag_pending': RELEASE}))


def verify(require_tag=False):
    value = read(MANIFEST)
    if value['release'] != RELEASE or value['file_count'] != len(value['inventory']):
        raise ValueError('Invalid release manifest')
    check_source()
    verify_inventory(ROOT, value['inventory'])
    commit = None
    if require_tag:
        if git('cat-file', '-t', f'refs/tags/{RELEASE}').strip() != b'tag':
            raise ValueError('Expected annotated release tag')
        commit = git('rev-parse', f'{RELEASE}^{{commit}}').decode().strip()
        rel = MANIFEST.relative_to(ROOT).as_posix()
        if git('show', f'{RELEASE}:{rel}') != MANIFEST.read_bytes():
            raise ValueError('Manifest differs from the tagged archive')
        # Also bind the recorded repository code/doc paths to the tagged Git tree.
        for name, info in value['inventory'].items():
            if 'repository' in info['groups']:
                blob = git('show', f'{RELEASE}:{name}')
                working = (ROOT / name).read_bytes()
                # Git may normalize CRLF for pre-existing text files. Artifact hashes remain exact.
                if blob != working and blob != working.replace(b'\r\n', b'\n'):
                    raise ValueError(f'Tagged repository file differs: {name}')
    recorded = read(ENVIRONMENT)
    current = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch')}
    differences = {name: {'frozen': recorded['packages'].get(name), 'current': version}
                   for name, version in current.items() if recorded['packages'].get(name) != version}
    print(json.dumps({'verified_files': len(value['inventory']), 'new_training': False,
                      'release': RELEASE, 'tag_commit': commit, 'environment_differences': differences}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('freeze', 'verify'), default='verify')
    parser.add_argument('--require-tag', action='store_true')
    args = parser.parse_args()
    if args.stage == 'freeze' and args.require_tag:
        parser.error('--require-tag is a verification option')
    freeze() if args.stage == 'freeze' else verify(args.require_tag)
