# 最终数据集与去偏实验代码同步（2026-10-15）

## 数据来源与位置

本项目现在直接使用 `Code/Data_Fin_New`。它来自本机 `KG4ER-New/Data_Fin_New` 的最终数据；所需文件已复制到本项目，逐文件 SHA-256 比对通过。运行时不再依赖另一个仓库的目录。`Code/Data_Fin_New` 被 `.gitignore` 排除，换机器运行时须另行复制此目录。

| 数据集 | 本项目的数据目录 | 训练/测试学生 | 固定推荐图 |
|---|---|---:|---|
| Eedi | `Code/Data_Fin_New/_resplit_seed20261007/Eedi` | 701 / 234 | `forgetting_variants/drop_exfr0p00_rec50/er_graph` |
| algebra2005 | `Code/Data_Fin_New/algebra2005` | 430 / 144 | `forgetting_variants_rec50/drop_exfr0p00/er_graph` |
| XES3G5M-sub-small | `Code/Data_Fin_New/XES3G5M-sub-small` | 727 / 241 | `forgetting_variants_rec50/drop_exfr0p00/er_graph` |

三个图均来自 `(θm, θf, rec)=(0.8,0.8,50)`，训练图删去零值习题遗忘边。Eedi 使用 2026-10-07 新划分；其余两个数据集使用原划分。原始交互、知识点掌握度、习题遗忘度、推荐图、冻结 KT 模型与 Ep readout 都位于 `Code/Data_Fin_New`。Eedi 的 readout 在 `_resplit_seed20261007/_ep_readouts/Eedi`；其他两个在 `_ep_readouts/<dataset>`。

`Code/codes-New-ConvE/final_dataset.py` 是唯一的默认数据选择入口。它校验文件存在、图 manifest 的参数、划分人数和保留的遗忘边数。旧的 `Code/Data_Fin` 仅供显式指定旧路径的历史实验使用，不能混入新结果。

## 同步后的实验流程

- `run_v3_kppd_pipeline.py`、`run_v3_debias_experiments.py`、`run_fairness_pipeline.py`、`run_fair_model_experiments.py` 与统一入口 `run_v3_ablation_and_comparison.py` 默认定位上述三个图；Pre 图写入各自数据集目录下，不覆盖原图。
- Pre 候选池默认 `M=100`，大于最终推荐图的 `rec=50`；在最终数据下使用 Pre 且 `M≤50` 会被拒绝。训练默认种子为 2024、2025、2026，评价 `N=5,10,…,50`。
- 对比实验直接入口 `comparison_models/run_v9_comparison_experiments.py` 也使用同一默认图。统一入口列出八个对比模型：EB-CF、SB-CF、CBF、TransE、TransE-adv、RotatE、DistMult、ComplEx。
- 原主模型与对比模型评估保留 Ada 与公平性指标。在三个最终数据集及其 Pre 派生图上，正式 `NOV` 等于 `KG4Ex-NOV-Set`：对每位测试学生，将 Top-N 习题的知识点先取并集，再与该学生历史答对题目的知识点集合计算 `1-|K_{TopN}(u)∩C(u)|/|K_{TopN}(u)∪C(u)|`，最后对测试学生平均。逐题计算再平均的版本另存为 `KG4Ex-NOV`，旧的时间加权值另存为 `NOV_time_weighted`。非最终图继续沿用旧 NOV，以便历史实验兼容。另补算论文 Ep 阈值 0.4、0.5、0.6、0.7。对比模型原有 `Ep_sim` 保留为单独字段。`ACC` 与 Ada 相同，用于兼容原论文表名。
- Ep 使用本项目内的 `front_pipeline`、`v11_pipeline`、`ep_kt_evaluation`、`mastery_ep_v2` 代码及本地 readout。评估前须为完整测试学生群体生成推荐分数；仅给一名学生的临时分数文件不足以计算论文 Ep。
- 复制的图 manifest 可能仍记录来源机器上的绝对路径。公平性与对比评估优先从本项目数据目录的 `raw` 或 `raw_compact` 找交互记录。

去偏模型结构仍是 Fairness-Recommend 原实现；数据与评估口径同步并不意味着旧权重能直接视为新设置下的实验结果。用新图训练的 Pre/In/Post 结果应写入新的 batch/run ID，并与这三个图上重算的基线比较。

## 检查与运行

在 `Code` 目录下，可先运行以下只读数据检查：

```powershell
python -c "import sys; sys.path.insert(0, 'codes-New-ConvE'); from final_dataset import select_dataset; [print(d, select_dataset(d).graph) for d in ('Eedi', 'algebra2005', 'XES3G5M-sub-small')]"
```

统一入口使用 `python codes-New-ConvE/run_v3_ablation_and_comparison.py --batch-id <新的批次名>`。应先查看 `--help`，选择所需阶段和算力设置。当前入口先跑八个对比模型，再跑非 KPPD 与 KPPD 消融，用户后续可单独调整顺序。所有数据输入均在本项目的 `Code/Data_Fin_New`；结果默认位于本项目 `Code/runs`。`Code/Data_Fin.7z`、`Code/runs.zip` 未被修改。

已完成验证：六个数据目录逐文件校验；三个图及 Ep 上下文载入；Eedi 234 位测试学生的正式 NOV-Set 与原 KG4ER 报表脚本 Top-5…50 逐项一致（差异仅来自小数舍入）；Eedi 主模型实际评估输出 `NOV=KG4Ex-NOV-Set`，逐题版单独保留；代码编译和单元测试通过。另完成 Eedi 主模型、KPPD 各一批次 GPU 训练及 Pre 图生成的轻量检查。未在本次同步中完成正式多种子训练。历史结果与论文 PDF 位于本项目本地目录，见 `docs/Fairness_V4_最终数据与NOV同步变更_20261015_221859.md`。
