"""Independent M0/M1/M2 runner. No formal model-training command is exposed yet."""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debias.io import ROOT, DATASETS, TOP_NS, sha256, write_json, read_json
from maed.io import VERSION, source_hash, lock_json, input_hashes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan','M0','M1','M2','all-diagnostics'), default='plan')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--seed', type=int, choices=(2024,), default=2024,
                        help='This diagnostic version uses the completed same-round seed2024 E0 only')
    parser.add_argument('--run-id', default='maed_m0_m2_quality_v2_20261023')
    parser.add_argument('--baseline-run', type=Path, default=ROOT/'runs/merit_frequency_20261022')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    parser.add_argument('--torch-threads', type=int, default=4)
    return parser.parse_args()


def protocol():
    # Fixed exploratory settings, NOT test-selected winners or final training hyperparameters.
    return {'version': VERSION, 'stages_implemented': ['M0','M1','M2'],
            'betas': [0., .25, .5, 1.], 'primary_beta': .5, 'distance_clip': 3.,
            'temperature_factors': [.5, 1., 2.], 'primary_temperature_factor': 1.,
            'matrix_users': 48, 'matrix_steps': 16, 'matrix_step_scale': .02,
            'model_probe_users': 16, 'model_relative_steps': [.0001, .001, .01],
            'diagnostic_seed': 20261023, 'top_ns': list(TOP_NS),
            'quality_budgets': {'Ada': .01, 'KG4Ex-NOV-Set': .03},
            'quality_budget_type': 'absolute decrease from same-round E0, checked at every N',
            'quality_revision': '2026-10-23 user decision after v1 outcomes; exploratory historical reassessment, fixed for next round',
            'student_harm_reporting_threshold': .01,
            'matrix_gate': 'primary temperature: soft mean, hard@10 and hard mean TV decrease >1e-6; Top10 sets change',
            'model_gate': 'at least one preregistered parameter direction decreases soft mean, hard@10 and hard mean TV >1e-6 and changes Top10 sets',
            'candidate_policy': 'full original exercise catalog; no added history filter',
            'exposure_targets': 'equal training-student mass, per-N rule TopN, matched batch cohort',
            'test_parameter_selection': False, 'scope': 'exploratory supplied-history snapshot diagnostics',
            'formal_training_enabled': False, 'M0_pending_evidence_blocks_automatic_formal_training': True}


def main():
    args = parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        raise ValueError('run-id must be a single directory name using letters/digits/_/-')
    if args.torch_threads < 1:
        raise ValueError('torch-threads must be positive')
    plan = protocol()
    stages = ['M0','M1','M2'] if args.stage == 'all-diagnostics' else [args.stage]
    out = ROOT/'runs/论文格式文档统计'/args.run_id
    if args.stage == 'plan':
        print(json.dumps({'output': str(out), 'datasets': args.datasets, **plan}, ensure_ascii=False, indent=2))
        return
    import torch
    torch.set_num_threads(args.torch_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    # Lock settings and execution code BEFORE loading new diagnostic outcomes.
    contract = {'protocol': plan, 'source_hash': source_hash(), 'datasets': args.datasets,
                'baseline_run': str(args.baseline_run.resolve()), 'seed': args.seed, 'device': args.device}
    lock_json(out/'protocol.json', contract)
    from maed.state import load_state
    from maed.audit import audit_state
    from maed.diagnostics import run_m1, run_m2
    for dataset in args.datasets:
        state = load_state(dataset)
        source = args.baseline_run/dataset/'E0'/f'seed{args.seed}'
        evidence = ['best.pt','uid_ex_scores.json','uid_ex_scores.logits.npy','eval/metrics.json']
        originals = {rel: sha256(source/rel) for rel in evidence}
        lock_json(out/dataset/'source_contract.json', {'state': state.provenance, 'E0': originals,
                                                      'all_input_files': input_hashes(state)})
        for stage in stages:
            folder = out/dataset/stage
            marker = folder/'stage_complete.json'
            if marker.exists():
                completed = read_json(marker)
                for rel, h in completed['output_files'].items():
                    if sha256(folder/rel) != h:
                        raise ValueError(f'Completed diagnostic output changed: {folder/rel}')
                print(f'Skip completed {dataset}/{stage}', flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            print(f'Start {dataset}/{stage}', flush=True)
            if stage == 'M0':
                audit_state(state, folder)
            elif stage == 'M1':
                run_m1(state, source, folder, plan)
            else:
                run_m2(state, source, folder, plan, args.device)
            if {rel: sha256(source/rel) for rel in evidence} != originals:
                raise ValueError('Read-only E0 artifacts unexpectedly changed')
            write_json(marker, {'optimizer_updates': 0, 'source_hash': contract['source_hash'],
                       'output_files': {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f'Diagnostics saved to {out}; no formal model training was run.', flush=True)


if __name__ == '__main__':
    main()
