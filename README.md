# Fairness-Recommend-New：教育适配条件下的频率去偏

基于 KG4ER-New 的 `V-Fin8-20261015`（`99068a9`），在原 SemanticConvE 上增加默认关闭的训练内频率正则。正式 NOV 使用 **KG4Ex-NOV-Set**。

当前正式数据为 `Data_Fin_New` 下的 Algebra2005、XES3G5M-sub-small，以及 `_resplit_seed20261007` 下的新 Eedi；沿用 `(0.8,0.8,50)` 和训练图 drop-exfr0.00 协议。数据与基线归档独立复制，均不提交 Git。

```powershell
python -u scripts/run_merit_debias_suite.py --plan
python -u scripts/run_merit_debias_suite.py
```

默认先 Eedi，再 Algebra2005，最后 XES；执行三个种子的 E0–E4 和规则参照，支持同命令续跑。先检查单种子可加 `--seeds 2024`。

- [实施方案](docs/教育适配条件下的频率去偏_完整实施方案_20261022.md)
- [实现、验证与一键运行说明](docs/频率去偏实现与一键运行说明_20261022.md)
- [三数据集基线与题对诊断](docs/三数据集基线与题对诊断_20261022.md)

首次部署到新机器需准备 `Data_Fin_New/` 与 `artifacts/`。`scripts/prepare_debias_context.py` 支持从原完整结果目录导入一次，后续实验仅依赖当前项目。

原前端 MIRT/EKTM、图生成、比较模型和 V11 依赖仍保留；旧文档中的 Data_Fin 默认路径是历史流程，新实验以以上运行说明为准。此次只实现和验证代码，新方法的实际效果待受控实验确认。
