# 原始 2CKG4ER ConvE 基线与 Algebra2005 的 Post 降分分析（2026-10-20）

## 新增对照

新增 `2CKG4ER-ConvE`：使用当前固定的 `(0.8, 0.8, 50)`、删除零遗忘边后的数据，训练 `baseline`，主干为 ConvE，不使用 Pre、KPPD 或 Post。训练设置为 25 epochs、batch size 1024、学习率 0.001、负采样比 5，默认随机种子 2024。结果单独写入 `Code/runs/<dataset>/<dataset>_V5_2024_v3_original_conve/baseline/seed2024`。

它不同于已有的 `Pre+In+Post (ConvE)`：后者只是去掉 CompGCN，仍保留三个去偏阶段。先用新基线与历史 2CKG4ER 同种子结果对照，确认足够接近后再考虑将它固定到正式表格。历史三种子均值不能直接与当前单种子表格混排。

先检查运行计划：

```powershell
$repo = 'C:\Users\21709\Desktop\Fairness-Recommend\Fairness-Recommend'
$py = 'C:\Users\21709\.conda\envs\rtx5060\python.exe'
Set-Location -LiteralPath $repo
& $py .\Code\codes-New-ConvE\run_original_conve_baseline.py --plan
```

正式运行时去掉 `--plan`；可用 `--datasets algebra2005` 先只跑一个数据集。原有 V5 一键入口也支持显式增加 `--include-original-conve`，已完成阶段会按原有续跑机制跳过。

## 历史分数与当前分数

旧版 Algebra2005 的 `2CKG4ER` Ada@5 三种子均值为约 0.790；其中 2024 种子的原始值为 0.788694。当前 V5 的 `CompGCN+ConvE` 为 0.698889，`Pre+In` 为 0.771389，`Pre+In+Post` 为 0.357930。

旧图和当前图的 `Q.txt`、`test_triples.txt`、`triples.txt` 逐字节一致，均是固定参数、删零遗忘边的 Algebra2005 图。将旧版 2024 推荐分数交给当前评估器计算，仍得到 Ada@5 = 0.788694，因此这次差异不是测试学生、图或 Ada 公式造成的。旧模型为原始 ConvE，当前表格的无去偏基线使用 CompGCN+ConvE；新加入的纯 ConvE 基线将帮助判断差距来自主干变化，还是来自训练实现的其他变化。在该基线重跑前，不能断言 CompGCN 是唯一原因。

## Post 为什么使 Algebra2005 的 Ada 明显下降

同一批 144 位测试学生，`Pre+In` 与 `Pre+In+Post` 只差后处理：Ada@5 从 0.771389 降到 0.357930，Ada@10 从 0.744201 降到 0.458772，Ada@50 从 0.638854 降到 0.597341。Top-5 两个列表的平均重合率仅 20%。Post 后的 Top-5 题在原排序中的中位名次约为 131，而候选池为前 150 道。

后处理代码优先按“新增知识点数量”选择题目，然后才比较混合分数和原推荐相关性。Algebra2005 上，Top-5 每题平均涉及的知识点由 1 个增加到 5.68 个；平均习题掌握度从约 0.647 降到 0.177。Ada 使用 `1 - |0.8 - 习题掌握度|`，所以这些掌握度很低的多知识点题会直接拉低 Ada。前 10 道的有效知识点覆盖目标为 23 个，导致这种前排替换格外显著；Top-50 原列表重合率约 89%，降幅也随之缩小。Eedi 的 Top-5 重合率约 79%，Ada 变化很小，说明这是数据集与当前硬配额重排共同作用的结果。

现阶段建议保留原始分数与 Post 分数两套结果。后续若要兼顾 Ada 与曝光公平，应调整 Post 的知识点覆盖目标、候选选择顺序或添加相关性/掌握度约束，并重新评估两类指标；不能仅凭 Ada 下降认定去偏无效。
