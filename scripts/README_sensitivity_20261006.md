# Three-computer sensitivity and ablation run

Branch: `V-Fin8-20261006`. The branch contains code only; `Data_Fin_New/` is ignored.

Each computer needs Python dependencies compatible with the `rtx5060` environment, a CUDA GPU, and one dataset copied under `<repository>/Data_Fin_New/<dataset>/`. Copy the matching Ep readout directory to `<repository>/Data_Fin_New/_ep_readouts/<dataset>/` as well. Include all dataset files, especially `raw` (or XES `raw_compact`), `front_features`, and `er_graph`; model checkpoints and `.npy` files are required. The runner checks these inputs before starting.

From the repository root in PowerShell, run one command per computer:

```powershell
conda run --no-capture-output -n rtx5060 python .\scripts\run_ablation_sensitivity.py --datasets Eedi --resume
conda run --no-capture-output -n rtx5060 python .\scripts\run_ablation_sensitivity.py --datasets algebra2005 --resume
conda run --no-capture-output -n rtx5060 python .\scripts\run_ablation_sensitivity.py --datasets XES3G5M-sub-small --resume
```

Only run the relevant line on each computer. `--resume` skips completed runs and resumes interrupted model stages. `--plan` performs input checks and prints the ordered commands without training. The default seed is 2024, the six models run in the paper's ablation order, and each metric uses N=5,10,...,50 with Ep thresholds 0.4, 0.5, 0.6, 0.7.

The runner executes two dataset-specific parameter groups first, then the shared `(theta_m, theta_f, rec)=(0.8, 0.8, 10)` group:

| Dataset | First | Second | Third |
| --- | --- | --- | --- |
| Eedi | (0.8, 0.6, 10) | (0.8, 0.6, 50) | (0.8, 0.8, 10) |
| algebra2005 | (0.8, 0.5, 50) | (0.8, 0.4, 50) | (0.8, 0.8, 10) |
| XES3G5M-sub-small | (0.8, 0.4, 50) | (0.8, 0.4, 100) | (0.8, 0.8, 10) |

Generated variant graphs are saved under `Data_Fin_New/<dataset>/variants/<variant>/er_graph`. Training runs are saved under `Data_Fin_New/_sensitivity_runs/<variant>/<dataset>/sensitivity_seed2024`. Neither folder is committed to Git. The graph manifest stores the source paths used on the computer that built it; these are provenance fields, and the runner resolves actual input files from the local repository path.
