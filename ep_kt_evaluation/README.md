# 独立 Ep 评估

本目录只读取已有 V-Fin8 推荐分数、三个数据集的历史交互和冻结的 `EKTM_mirt/best.pt`；不改动原训练、推荐或旧 `Ep_sim`。旧 `Ep_sim` 的固定 0.1 增益**不是**这里的 Ep。

## 一键运行论文三数据集与所有对比模型

在 PowerShell 执行下列一条命令。`rtx5060` 是 Conda 环境名，`--device cuda` 才表示使用 NVIDIA RTX 5060。每个学生伯努利抽样 30 次、阈值法 1 次，推荐长度为 10。先用单进程占用显卡，避免三个 Python 进程同时加载 KT 模型；脚本会自动检查 63 份推荐分数文件是否齐全。

```powershell
conda run --no-capture-output -n rtx5060 python .\ep_kt_evaluation\run_all_ep.py --output-root .\ep_kt_results --top-n 10 --draws 30 --device cuda --workers 1
```

模型包括主模型 `SemanticConvE_feature_only`、五个知识图谱对比模型 `RotatE / DistMult / ComplEx / TransE / TransE-adv`（每个三个推荐随机种子）以及三个传统对比模型 `EB-CF / SB-CF / CBF`（各一份确定性结果）。每个数据集 21 份运行结果，三个数据集共 63 份；所有模型使用同一个数据集的 KT checkpoint、相同历史预热、相同推荐长度、相同模拟随机种子与同一 Ep 公式。伯努利抽样在相同学生及抽样编号的各模型间使用同一随机数流。传统模型没有三个推荐种子，汇总表不为其虚构跨种子标准差。

运行完毕后查看 `ep_kt_paper_models.csv`（三数据集 × 九模型 × 两作答规则，共 54 行）、`ep_kt_per_run.csv`（63 运行 × 两规则，共 126 行），以及各运行目录的 `ep_kt_summary.json`、`ep_kt_per_draw.json`。再次执行会重新计算，避免本地输入文件变化时误用旧结果。

## 计算方法

对每位测试学生，将其历史作答按原 EKTM 的 `max_seq_len` 截断，从零隐状态逐题回放，得到推荐前隐状态。读取对应模型的 `uid_ex_scores.json`，按分数降序取前 10 道题，分数相同按题目编号升序。对每道题：

1. 用冻结的 EKTM `response_head` 和当前隐状态预测答对概率 `p`。
2. 伯努利模式执行 `Bernoulli(p)`；阈值模式执行 `int(p >= 0.5)`。
3. 把题目表示和模拟 0/1 作答送入冻结的 `GRUCell`，更新该学生隐状态。

每个知识点的状态由其 Q 矩阵关联题目的预测答对概率取平均而来。推荐前后都使用同一组关联题（锚定题），因此 Ep 比较的是相同知识点集合。连续归一化增益与所附参考代码的 `Eval_Global_Gain_Mastery` 公式一致：

`Ep = (Σ_k state_after[k] − Σ_k state_before[k]) / (K − Σ_k state_before[k])`。

分母不大于 `1e-5` 时按参考代码设为 0。伯努利模式先对每位学生的 30 次结果取均值，再对测试学生取均值；最后对推荐模型三个种子取均值与样本标准差。阈值模式每位学生计算一次。

## 与参考代码的差异和解释边界

参考代码使用单独训练的 DKT、按知识点推荐与模拟，并直接读取 DKT 的知识点概率。本项目推荐对象是**具体题目**，因此使用项目中已训练的 EKTM，模拟作答时输入题目及其 0/1 结果。现存三个 `EKTM_mirt/best.pt` 的 `mastery_loss_weight` 均为 0，`mastery_head` 没有接受知识点掌握度监督，不能直接把它或旧 `stu2know_mastery.json` 当作可靠的 Ep 状态。这里通过**已训练的答题预测头**在固定 Q 关联题上求均值，构造可随隐状态变化的知识点预测掌握度。这是参考思路的项目适配，数值定义与参考代码的 DKT 直接输出并不完全相同，论文中需如实说明。

参考代码用历史前半段预热 DKT，并跳过没有目标知识点的学生。这里使用与项目推荐时相同的完整、截断历史作初始状态，对所有拥有历史的测试学生做全局知识点增益评估。Ep 是**模型模拟下的预期学习增益估计**，不等同于真人完成题目后的实测效果，也不能单独作为推荐质量的真实标签。若 Ep 为负，应如实报告；不能以旧固定增益或选择性删结果替代。

模型的 Q、文本 token 和 MIRT 缓冲区由 checkpoint 恢复；运行时会验证当前原始 Q 与 checkpoint 中的 Q 逐项一致。本地历史交互文件仍需保持与论文运行时相同的版本；若它们以后改变，程序能检查学生与题目数量，却无法仅凭 checkpoint 还原原始历史。

## 检查

```powershell
conda run --no-capture-output -n rtx5060 python .\ep_kt_evaluation\test_ep_kt.py
```

测试覆盖参考公式及边界、两种作答规则、63 份论文运行文件的完整发现，以及三个数据集的历史隐状态回放与原模型 `forward` 对齐。
