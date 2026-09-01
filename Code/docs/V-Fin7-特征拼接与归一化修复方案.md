# V-Fin7：特征拼接与归一化修复方案

## 1. 背景与已定位问题

当前 `feature_only` 主模型采用原始特征拼接后经 MLP 映射到 ConvE 所需的 200 维空间：

```text
uid:      theta(1) -> RawConcatFusion -> 200
exercise: text(100) + difficulty(1) + discrimination(1) -> RawConcatFusion -> 200
relation: type_embedding(16) + strength(1) -> RawConcatFusion -> 200
kc:       kc_ID_embedding(200) -> RawConcatFusion -> 200
```

但 `RawConcatFusion` 当前在 MLP 前统一使用 `LayerNorm(input_dim)`。这导致两类问题：

1. **学生 theta 被完全抹平。**
   `theta` 只有一维。对单个标量做 `LayerNorm(1)` 时，其均值就是自身、方差为零；不同学生的 theta 会被变成同一个常数，再进入 MLP。因此模型不能利用学生能力差异。
2. **题目与关系的异构特征被混合归一化。**
   题目输入的 100 维文本与 2 维教育学数值被一起做 `LayerNorm(102)`；关系的 16 维类型向量与 1 维强度被一起做 `LayerNorm(17)`。这会改变已经归一化到 `0~1` 的难度、区分度、强度的绝对含义，并使其数值尺度依赖文本或类型 embedding 的当前分布。

本次 Eedi 的局部推荐证据验证了第一个问题：虽然 234 位测试学生的 theta 范围已恢复为 `0.3225~0.6546`，但 Top-20 推荐题、每道题分数在 234 位学生间完全相同。这不是正常的个性化推荐结果。

## 2. 修改目标

在不改变 ConvE 编码器、关系图、训练损失、负采样或评价指标的前提下：

- 保留每个输入特征已经完成的语义处理和归一化；
- 直接拼接原始特征后输入实体/关系专属 MLP；
- 继续对 MLP 输出的 200 维表示做 LayerNorm，以维持 ConvE 输入的训练稳定性；
- 恢复 theta 对学生表示和推荐分数的实际影响。

## 3. 具体模型修改

### 3.1 修改 `RawConcatFusion`

目标文件：`codes-New-ConvE/semantic_conve_model.py`。

将当前结构：

```text
features -> LayerNorm(input_dim) -> Linear -> ReLU -> Dropout -> Linear -> LayerNorm(200)
```

改为：

```text
features -> Linear -> ReLU -> Dropout -> Linear -> LayerNorm(200)
```

即移除 `RawConcatFusion` 的输入层 `LayerNorm(input_dim)`，保留输出层 `LayerNorm(200)`。输出仍是固定 200 维，因此 ConvE 的 `reshape -> concat(h, r) -> Conv2D -> scoring` 主体无需改动。

实现上可采用以下两种等价形式之一：

```python
self.input_norm = nn.Identity()
```

或直接在 `forward()` 中将 `features` 输入 MLP。推荐保留 `input_norm` 属性并改为 `nn.Identity()`，以减少代码改动并便于后续试验切换。

### 3.2 学习者实体

```text
theta_norm(1, already in [0, 1])
  -> MLP_uid(1 -> 400 -> 200)
  -> LayerNorm(200)
```

- 不对 theta 做 `LayerNorm(1)`；
- theta 已在前置文件中归一化，直接保留其大小关系；
- MLP 参数在 SemanticConvE 训练期间学习。

预期：theta 不同的学生应产生不同的 200 维 learner embedding；同一批测试学生的推荐分数不应再逐元素完全相同。

### 3.3 习题实体

```text
concat(topic_text_100, difficulty_norm_1, discrimination_norm_1)
  -> MLP_exercise(102 -> 400 -> 200)
  -> LayerNorm(200)
```

- `topic_text_100`：由共享 EKTM_mirt Bi-GRU 在 SemanticConvE 训练时动态编码的题目文本表示；
- `difficulty_norm`、`discrimination_norm`：MIRT 前置文件的归一化教育学参数；
- 不再对这 102 维混合 LayerNorm；
- 三类特征仍由 MLP 自主学习权重与交互关系。

### 3.4 知识点实体

```text
kc_ID_embedding_200
  -> MLP_kc(200 -> 400 -> 200)
  -> LayerNorm(200)
```

知识点当前只采用 ID embedding；不新增知识点文本，以保持本阶段模型定义不变。

### 3.5 关系文本语义与强度表示

当前关系侧仅使用 `relation_type_embedding(16) + strength(1)`。V-Fin7 将其扩展为“关系文本语义 + 连续强度”的双分支表示，并移除独立的 `relation_type_embedding`：

```text
relation_text_100
strength_1 -> MLP_strength(1 -> 16)

concat(relation_text_100, strength_embedding_16)
  -> MLP_relation(116 -> 400 -> 200)
  -> LayerNorm(200)
```

这里的 16 维 `strength_embedding` 不等于把同一个数复制 16 次。它由可训练的非线性小 MLP 产生，使模型能够学习不同强度区间对关系表示的不同影响。该分支的输入是单维 strength，因此不使用 `LayerNorm(1)`。

不建议一开始将 strength 扩展到 100 维：强度本身仅包含一个连续数值，16 维已足够表达非线性映射，也避免在维度数量上人为压过关系文本。后续可将 `1/16/100` 维 strength embedding 作为单独容量消融。

### 3.6 关系描述模板

关系文本只描述“关系的教育认知含义和三元组方向”，不把具体数值 strength 写入文本。四条固定模板为：

```text
mlkc:
A cognitive mastery relation from a knowledge concept to a learner,
where the relation strength denotes the learner's mastery level of the concept.

pkc:
A cognitive demanding relation from a knowledge concept to a learner,
where the relation strength denotes the probability that the concept will be
encountered by the learner in the next interaction.

exfr:
A cognitive forgetting relation from an exercise to a learner,
where the relation strength denotes the learner's forgetting degree for the exercise.

rec:
A recommendation relation from a learner to an exercise,
where the exercise is recommended according to the learner's cognitive mastery,
sequence, and forgetting states.
```

它们分别对应图中的：

```text
kc --mlkc_strength--> uid
kc --pkc_strength--> uid
ex --exfr_strength--> uid
uid --rec--> ex
```

### 3.7 `relation_text_100` 如何由共享 Bi-GRU 得到

V-Fin7 不再使用独立的 Relation Bi-GRU。关系文本和题目文本使用**同一套词表、词嵌入层和 Bi-GRU 文本编码器**，即与 EKTM_mirt 的 `TopicRNNModel` 保持一致。知识点实体只使用 ID，不再构造或使用知识点文本表示。

首先用以下文本共同构建词表：

```text
全部题目文本
+ 4 条固定关系描述
```

以题目作答序列监督训练 EKTM_mirt 的 Bi-GRU 文本编码器。关系描述不构成作答样本，但其 token 已位于共享词表，不会被映射为 `<unk>`。SemanticConvE 初始化时加载该 Bi-GRU 权重，并在三元组损失下继续端到端微调：

```text
exercise / relation template tokens
  -> shared word embedding
  -> shared bidirectional GRU(hidden_size=50 per direction)
  -> concat(last_forward_hidden, last_backward_hidden)
  -> text embedding (100)
```

具体实现要求：

1. 在前置特征目录新增 `relation_semantics.json`，保存四种 relation type、三元组方向与第 3.6 节文本模板；该文件固定且与数据集无关。
2. EKTM_mirt 的词表构建逻辑接收题目文本和关系模板两类语料；训练完成后保存词表和 `TopicRNNModel` checkpoint。
3. `feature_loader.py` 读取题目和关系文本的 token id、有效长度、共享词表及 EKTM_mirt 文本编码器 checkpoint，并将它们注册为模型输入；删除知识点文本 embedding 的读取要求。
4. `semantic_conve_model.py` 复用同一个 Bi-GRU 分别编码题目和关系描述；`relation_embedding()` 按 relation type 取得对应的 100 维 `relation_text`。
5. 因为共享 Bi-GRU 在 SemanticConvE 训练阶段允许反向传播，题目文本必须在模型内动态编码，不能仅读取固定 `.npy` 向量；否则题目与关系不会处于同一个更新后的语义空间。
6. 共享词嵌入、共享 Bi-GRU、`MLP_strength` 与 `MLP_relation` 都由最终的 SemanticConvE 三元组损失继续更新。相关 token 仅在被使用的 batch 中接收梯度。
7. 仅有四条关系模板仍是现实限制。因此必须通过 `relation_type_strength`、`relation_text_strength`、`relation_text_only`、`relation_strength_only` 消融验证关系文本是否带来稳定收益。

### 3.8 关系编码消融

为辨别关系文本与强度分支的作用，新增以下关系编码对照：

```text
relation_type_strength:
    type_embedding_16 + strength_1 -> MLP_relation
    （当前 V-Fin6 基线）

relation_text_strength:
    relation_text_100 + strength_embedding_16 -> MLP_relation
    （V-Fin7 主方案）

relation_text_only:
    relation_text_100 -> MLP_relation
    （去除连续 strength）

relation_strength_only:
    strength_embedding_16 -> MLP_relation
    （去除关系文本）
```

正式论文主模型是否改用 `relation_text_strength`，以该消融在五个数据集上的 ACC 表现和稳定性为准。

### 3.9 原关系表示（V-Fin6 基线）

```text
concat(relation_type_embedding_16, relation_strength_1)
  -> MLP_relation(17 -> 400 -> 200)
  -> LayerNorm(200)
```

- `relation_type_embedding` 区分 `rec/mlkc/pkc/exfr`；
- `relation_strength` 直接保留图中解析出的 `0~1` 数值；`rec` 的强度为 1；
- 不再把类型 embedding 与强度一起输入 `LayerNorm(17)`；
- 强度仍按照 ER 图关系名称的小数精度构建，当前不改变关系数量与图结构。该版本保留为 V-Fin6 的可复现实验基线。

## 4. 前置文件与图构建的影响

本次只改 SemanticConvE 内部的特征融合模块，因此以下内容**不需要重新生成**：

- `stu2know_mastery.json`；
- `stu2know_seq.json`；
- `stu2know_forget.json`、`stu2ex_forget.json`；
- MIRT 参数、除文本 token/checkpoint 外的 `semantic_kg_features`、`triples.txt`、`test_triples.txt`、推荐关系；
- TransE、TransE-adv、RotatE、DistMult、ComplEx、CF 等对比模型。

仅修复 `RawConcatFusion` 时，需要重新训练受其影响的 SemanticConvE 模型。若实施共享 Bi-GRU 关系文本方案，则需要额外重跑 EKTM_mirt 的文本编码训练，以便词表纳入关系模板；随后导出并保存共享文本编码器 checkpoint、题目文本 token 与关系文本 token。删除不再使用的 `concept_text_embeddings.npy`。MIRT、序列模型、遗忘模型、ER 图构建和对比模型不需重跑；使用共享 Bi-GRU 的 SemanticConvE 模型必须重新训练。

## 5. 需要重跑的实验

第一阶段先按“仅修复输入 LayerNorm”重跑以下八个 SemanticConvE 模型，均采用训练图，不加入 `test_triples.txt`：

```text
feature_only
id_only
feature_only_relation_id
feature_only_learner_id
feature_only_exercise_id
feature_only_no_mastery
feature_only_no_forgetting
feature_only_no_seq
```

其中 `id_only` 本身绕过 `RawConcatFusion`，理论上不受本次改动影响；仍可保留旧结果作为严格的固定基线。其余模型建议全部按相同种子重新运行，避免混用不同版本结果。

第二阶段再以修复后的 `feature_only` 为基础，运行第 3.8 节的关系编码消融。不要将“LayerNorm 修复”和“关系文本编码”放进同一次对比，否则无法判断提升来自哪个改动。

## 6. 验证与验收

### 6.1 单元测试

新增或更新测试，至少覆盖：

1. `RawConcatFusion(input_dim=1)` 对输入 `[0.2]` 与 `[0.8]` 产生不同输出；
2. 反向传播后 learner MLP 参数可获得非零梯度；
3. 各实体/关系融合输出形状均为 `[batch, 200]`；
4. 输出无 NaN/Inf；
5. `score_tails()` 的输出形状和 type-aware 习题筛选逻辑不变。

### 6.2 推理检查

对 `feature_only` 的一个 seed 导出 `SemanticConvE_uid_ex_scores.pkl` 后检查：

1. 测试学生 theta 的最小值、最大值和唯一值数正常；
2. 抽取至少 10 位学生，比较其完整 948 维题目分数向量；不应全部逐元素相同；
3. 检查不同学生的 Top-20 题目列表。允许部分重叠，但不应 234 位学生完全相同；
4. 局部证据中的题目 ID、题目文本、Q 矩阵对应知识点、知识点名称保持一致；
5. `SemanticConvE Score` 在每位学生内部按 Rank 严格递减。

### 6.3 结果判读

本次的目标首先是修复个性化表示失效，不预设 ACC 必然提高。重点同时观察：

- ACC@5、@10、@20 是否稳定或提升；
- 三个随机种子的标准差是否收敛；
- 局部推荐是否出现真实的学生差异；
- `feature_only_learner_id` 与 `feature_only` 的差异是否更符合“theta 提供学生侧信息”的预期。

## 7. 后续可选优化

若修复后区分度仍贡献有限，再单独研究 `discrimination_norm` 的偏斜分布，例如采用分位数归一化或截断极端值后的 Min-Max 归一化。该项不纳入本次修改，避免同时改变融合结构和前置特征分布，导致无法归因。
