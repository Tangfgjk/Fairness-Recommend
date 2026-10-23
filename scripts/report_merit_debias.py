"""Regenerate per-metric tables from completed experiment stages."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from debias.io import ROOT
from debias.report import report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',default='merit_frequency_20261022')
    a=p.parse_args()
    if Path(a.run_id).name != a.run_id or a.run_id in ('.','..'):
        p.error('run-id must be a folder name')
    print(report(ROOT/'runs'/a.run_id))
