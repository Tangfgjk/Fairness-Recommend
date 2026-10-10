# Remaining final comparisons on AutoDL

This command runs only 17 unfinished comparison model/seed evaluations on
the selected `(theta_m, theta_f, rec) = (0.8, 0.8, 50)` graphs with zero
exercise-forgetting training edges removed. It does not rerun the six
ablations or already completed comparison models.

| Dataset | Model | Seeds to run |
| --- | --- | --- |
| algebra2005 | RotatE | 2025, 2026 |
| algebra2005 | DistMult | 2024, 2025, 2026 |
| algebra2005 | ComplEx | 2024, 2025, 2026 |
| Eedi (20261007 student resplit) | RotatE | 2024, 2025, 2026 |
| Eedi (20261007 student resplit) | DistMult | 2024, 2025, 2026 |
| Eedi (20261007 student resplit) | ComplEx | 2024, 2025, 2026 |

## Code and environment

Clone branch `V-Fin8-20261010` into a persistent AutoDL storage location,
for example `/root/autodl-tmp`:

```bash
cd /root/autodl-tmp
git clone --depth 1 --single-branch --branch V-Fin8-20261010 \
  https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'
```

Use an AutoDL image with a CUDA-enabled PyTorch environment. All paths in
the new entry point are resolved relative to the cloned repository.

## Data to copy

The Git branch excludes `Data_Fin_New`. Copy these directories into the
listed repository-relative locations. The selected `er_graph` directory
must be complete, including its text embeddings, dictionaries, triples,
recommendations, and manifest.

```text
Data_Fin_New/
  algebra2005/
    raw/
    front_features/
    forgetting_variants_rec50/drop_exfr0p00/er_graph/
  _ep_readouts/algebra2005/
  _resplit_seed20261007/
    Eedi/
      raw/
      front_features/
      forgetting_variants/drop_exfr0p00_rec50/er_graph/
    _ep_readouts/Eedi/
```

Optional: to continue the interrupted algebra2005 RotatE seed 2025
instead of restarting it, copy `checkpoint` and `config.json` from:

```text
Data_Fin_New/_forgetting_edge_comparison_runs_rec50/drop_exfr0p00/
  algebra2005/forgetting_rec50_seed2024/RotatE/seed2025/
```

to the identical repository-relative path on AutoDL. The checkpoint was
saved at step 16500. The new script automatically passes `--resume` when
that run directory exists. Do not copy the checkpoint into the dataset's
graph directory.

## Run

From the cloned repository root, first validate both data layouts, then
run all 17 jobs sequentially on the selected GPU:

```bash
python scripts/run_remaining_autodl.py --dataset all --plan
python scripts/run_remaining_autodl.py --dataset all
```

To run datasets separately, replace `all` with `algebra2005` or `Eedi`.
Repeat the same run command after an interruption; evaluated tasks are
skipped and available KGE checkpoints are resumed. Each model/seed runs
training, recommendation scoring, Ada/NOV/Ep evaluation, and a partial
comparison summary. The four Ep response thresholds are 0.4, 0.5, 0.6,
and 0.7; recommendation list sizes are 5 to 50 in steps of 5.

Outputs are written to:

```text
Data_Fin_New/_forgetting_edge_comparison_runs_rec50/drop_exfr0p00/
  algebra2005/forgetting_rec50_seed2024/
Data_Fin_New/_resplit_seed20261007/_forgetting_edge_comparison_runs/
  drop_exfr0p00_rec50/Eedi/resplit_seed20261007_modelseed2024/
```

Existing completed results from the other computers are needed only when
assembling the final combined tables. They are not needed to run the 17
unfinished jobs. In particular, algebra2005 RotatE seed 2024 is already
complete and is not included in this AutoDL job list.
