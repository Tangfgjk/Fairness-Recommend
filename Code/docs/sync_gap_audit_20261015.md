# KG4ER-New → Fairness-Recommend 同步核查（2026-10-15）

最终三个数据集、训练图、原始交互、KT 检查点与 Ep readout 已复制到本项目的 `Code/Data_Fin_New`。Eedi 使用新划分；三图均采用 `(θm, θf, rec)=(0.8,0.8,50)`，并从训练图中删除零值习题遗忘边。正式随机种子为 2024、2025、2026，评价 N=5,10,…,50。

独立主模型、公平性和对比实验脚本的默认种子及评价 N 已同步；原始图构建入口的 `top_k_rec` 改为 50。Fairness Pre 候选池 M=100、Post 重排的 Top-100 与候选数是去偏设置，和原图 `rec=50`、论文评价 N≤50 含义不同，因此保留。旧 `Data_Fin` 历史数据与非最终数据集仍需显式指定路径。

正式 `NOV` 使用 `KG4Ex-NOV-Set`，对 Top-N 全部习题知识点的并集与测试学生答对过的知识点计算 Jaccard 新颖性，再对测试学生取平均。`KG4Ex-NOV` 保留逐题版，`NOV_time_weighted` 保留原时间加权版。原主模型与对比模型的评估入口一致；Fairness 的 Pre/In/Post 去偏算法与模型结构未改。

KG4ER-New 2026-10-10 历史结果已复制到本项目 `Code/historical_kg4er_results_20261015`（2,378 个文件、20,769,178,216 字节），论文 PDF 单独复制到 `Code/paper_pdfs_20261015`（36 份），五个报表脚本复制到 `Code/scripts/kg4er_reports_20261015`。历史结果和数据集由 Git 忽略，需单独备份或复制到其他机器；PDF、代码、文档纳入版本控制。报表脚本的部分旧敏感性分析入口仍需对应的历史 `_sensitivity_runs` 数据，不能直接用于新 Fairness 结果。

统一入口暂保留“对比实验 → 非 KPPD 消融 → KPPD 消融”的阶段顺序。完整多种子 Fairness 实验尚待正式运行。详细变更与验证见 `Fairness_V4_最终数据与NOV同步变更_20261015_221859.md`。
