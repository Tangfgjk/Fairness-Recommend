"""MAED-v2 engineering entry. Formal training is intentionally not exposed at P0."""
import argparse
from datetime import datetime
import platform
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('p0',),default='p0')
    parser.add_argument('--datasets',default='Eedi,algebra2005,XES3G5M-sub-small')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--chunks',default='4096,16384,65536,131072')
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--measured',type=int,default=5)
    parser.add_argument('--query-limit',type=int,default=2,help='0=full test cohort; default is engineering sample only')
    parser.add_argument('--run-id',default=None)
    args=parser.parse_args()
    import torch
    from debias.io import DATASETS,write_json,sha256
    from maed_v2.p0 import run_dataset
    torch.set_num_threads(2)
    datasets=args.datasets.split(',');chunks=tuple(map(int,args.chunks.split(',')))
    if not datasets or len(set(datasets))!=len(datasets) or any(x not in DATASETS for x in datasets):parser.error('Invalid datasets')
    if not chunks or min(chunks)<1 or args.warmup<1 or args.measured<1 or args.query_limit<1 and args.query_limit!=0:
        parser.error('Positive profile counts/chunks required')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    run_id=args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    if Path(run_id).name!=run_id or any(x in run_id for x in ('/','\\',':')):parser.error('run-id must be a folder name')
    output=ROOT/'artifacts/maed_v2'/f'P0_{run_id}'
    output.mkdir(parents=True,exist_ok=False)
    sources=list((ROOT/'maed_v2').glob('*.py'))+[Path(__file__)]
    manifest={'arguments':vars(args),'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),
              'source_files':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
              'formal_training_started':False,'output':str(output)}
    write_json(output/'run_manifest.json',manifest)
    print(f'P0 output: {output}',flush=True)
    completed=[]
    for name in datasets:
        try:
            print(f'P0 begin {name}',flush=True)
            result=run_dataset(name,output,torch.device(args.device),chunks,args.warmup,args.measured,args.query_limit)
            completed.append({'dataset':name,'status':result['status']})
            write_json(output/'summary.json',{'completed':completed,'formal_training_started':False})
        except Exception as error:
            write_json(output/'failure.json',{'dataset':name,'error':repr(error),'formal_training_started':False})
            raise
    print(f'P0 finished: {output}',flush=True)


if __name__=='__main__':main()
