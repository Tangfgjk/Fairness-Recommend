"""Re-evaluate one full-cohort score file with KG4Ex-NOV-Set and Ep threshold 0.5."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from debias.prepare import prepared
from debias.evaluation import evaluate_file
from debias.io import DATASETS

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=DATASETS)
    p.add_argument('--scores',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    ctx,arrays,manifest=prepared(a.dataset)
    evaluate_file(ctx,arrays,manifest,a.scores,a.output,a.baseline)
