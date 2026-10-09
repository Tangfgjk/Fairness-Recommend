# Final rec=50 drop-zero experiments

Run one dataset per computer. The selected graph is always built with
`(theta_m, theta_f, rec)=(0.8, 0.8, 50)` and removes only training
`exfr0.00` triples. Eedi uses the 20261007 student split; Algebra2005
and XES3G5M-sub-small use their original student splits.

The Git branch contains code only. Copy these directories to the same
repository-relative locations on each computer:

| Computer | Data directories to copy |
| --- | --- |
| Eedi | `Data_Fin_New/_resplit_seed20261007/Eedi/` and `Data_Fin_New/_resplit_seed20261007/_ep_readouts/Eedi/` |
| Algebra2005 | `Data_Fin_New/algebra2005/` and `Data_Fin_New/_ep_readouts/algebra2005/` |
| XES | `Data_Fin_New/XES3G5M-sub-small/` and `Data_Fin_New/_ep_readouts/XES3G5M-sub-small/` |

The dataset folder must include its `front_features`, `raw` (or XES
`raw_compact`) and the selected `forgetting_variants*` graph. To
reuse existing seed 2024 runs, copy the matching `_forgetting_edge_runs*`
and `_forgetting_edge_comparison_runs*` directories as well. Without
those result directories, seed 2024 will run again.

From the repository root, using a Python environment with CUDA PyTorch:

```powershell
$py = 'C:\path\to\your\python.exe'
& $py .\scripts\run_final_dropzero_three_seeds.py --dataset Eedi --plan
& $py .\scripts\run_final_dropzero_three_seeds.py --dataset Eedi
```

Replace `Eedi` with `algebra2005` or `XES3G5M-sub-small` on the
other computers. The default `--phase all` runs all six ablations for
seeds 2024, 2025 and 2026, checks all 18 evaluation outputs, then runs
the comparisons in this order: EB-CF, SB-CF, CBF, TransE,
TransE-adv, RotatE, DistMult, ComplEx. The three CF methods are
deterministic dataset-level methods and have one evaluation result each;
the five KGE models run all three seeds. Pass `--phase ablation` or
`--phase comparison` to run the stages separately. Repeating the
command resumes completed model/seed runs.

Results are written under each dataset's existing
`_forgetting_edge_runs*` and `_forgetting_edge_comparison_runs*`
directories. `--plan` validates inputs and prints commands without
starting training.
