# V-Fin7 Compact Raw 前置文件生成

`assist2009-sub` 与 `XES3G5M-sub-small` 使用经过迭代双侧筛选的
`raw_compact`；`Eedi`、`algebra2005`、`statics2011` 继续使用原始 `raw`。
所有前置特征与 ER 图仍输出到各数据集既有的 `front_features` 和
`er_graph`，因此重新运行会覆盖该数据集的旧前置结果。

| Dataset | Raw input | Learner / exercise threshold |
| --- | --- | --- |
| Eedi | `raw` | unchanged |
| algebra2005 | `raw` | unchanged |
| statics2011 | `raw` | unchanged |
| assist2009-sub | `raw_compact` | 20 / 10 |
| XES3G5M-sub-small | `raw_compact` | 250 / 100 |

Run from the repository root after activating `kg4er_cuda`:

```powershell
python front_pipeline\run_front_pipeline.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small `
  --data-fin-root Data_Fin `
  --mirt-epochs 70 `
  --mirt-batch-size 1024 `
  --ektm-epochs 30 `
  --ektm-batch-size 16 `
  --device cuda `
  --theta auto `
  --force
```

The runner automatically selects `raw_compact` only for the two compact
datasets. Do not pass `--compact-datasets` unless intentionally changing this
mapping.

After it completes, validate every regenerated graph:

```powershell
python front_pipeline\validate_front_pipeline.py `
  --datasets Eedi,algebra2005,statics2011 `
  --data-fin-root Data_Fin `
  --raw-name raw `
  --require-graph

python front_pipeline\validate_front_pipeline.py `
  --datasets assist2009-sub,XES3G5M-sub-small `
  --data-fin-root Data_Fin `
  --raw-name raw_compact `
  --require-graph
```
