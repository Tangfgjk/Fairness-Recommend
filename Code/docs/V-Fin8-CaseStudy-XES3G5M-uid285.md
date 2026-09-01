# V-Fin8 Case Study：XES3G5M-sub-small 上的推荐解释示例

本文档整理一个用于论文 case study 的样例。目标是对比不同推荐方法对同一名学生的推荐结果，并突出本文模型能够给出更全面的教育学解释。

## 1. Case 选择

本 case 选择数据集 `XES3G5M-sub-small` 中的测试学生：

| 项目 | 内容 |
|---|---|
| 数据集 | `XES3G5M-sub-small` |
| 目标学生 | `uid285` |
| 本文模型 | `SemanticConvE_feature_only` |
| KGE 对比模型 | `TransE-adv` |
| CF 对比模型 | `SB-CF` |

选择该学生的原因：

1. 三个模型的 Top-1 推荐题不同，适合展示推荐差异。
2. 本文模型推荐题的掌握度和遗忘度都处于较有解释价值的区间。
3. `SB-CF` 能找到一个明确的相似学生，可用于绘制类似 KG4Ex 论文 Fig. 4 的解释图。

## 2. 三个模型的推荐结果

| 方法 | Top-1 推荐题 | 知识点 | 推荐依据 |
|---|---|---|---|
| `SemanticConvE_feature_only` | `ex122` | `kc79`：整数好朋友数 | 综合学生能力、题目文本、题目难度/区分度、知识掌握关系、遗忘关系和推荐关系进行图表示学习 |
| `TransE-adv` | `ex102` | `kc73`：树形图 | 基于知识图谱三元组结构的 KGE 打分，主要体现图结构相似性 |
| `SB-CF` | `ex8` | `kc6`：陷阱型不规则图形求周长 | 基于相似学生的历史答题行为推荐 |

### 2.1 三个模型的 Top-3 推荐

为了更完整地展示不同方法的推荐差异，这里整理三个模型对同一学生 `uid285` 的 Top-3 推荐题。

| 方法 | Rank | 推荐题 | 知识点 | 知识点名称 | 题目简述 | 推荐分数 |
|---|---:|---|---|---|---|---:|
| `2CKG4ER` | 1 | `ex122` | `kc79` | 整数好朋友数 | 计算：`23 × 4 × 25`；`125 × 13 × 8` | `0.987736` |
| `2CKG4ER` | 2 | `ex103` | `kc73` | 树形图 | 甲乙丙三人互相传球，第 3 次传到乙手中的方式数 | `0.919607` |
| `2CKG4ER` | 3 | `ex304` | `kc166` | 特殊乘积 | 计算：`4200 ÷ (25 × 7)` | `0.876236` |
| `TransE-adv` | 1 | `ex102` | `kc73` | 树形图 | A/B/C 三个小朋友互相传球，不同传球方式数 | - |
| `TransE-adv` | 2 | `ex484` | `kc226` | 排列组合 | 蚂蚁沿正四面体棱走遍 4 个顶点再回到 A 的走法数 | - |
| `TransE-adv` | 3 | `ex115` | `kc77` | 数列找规律填数 | 2018 个自然数排列，任意相邻四个数之和等于 30 | - |
| `SB-CF` | 1 | `ex8` | `kc6` | 陷阱型不规则图形求周长 | 不规则图形周长题 | `1.000000` |
| `SB-CF` | 2 | `ex116` | `kc43` | 奇数与偶数的加减规律 | 偶数加上一个偶数，结果是？ | `1.000000` |
| `SB-CF` | 3 | `ex121` | `kc79` | 整数好朋友数 | 计算：`4 × 9 × 25` | `1.000000` |

注意：不同模型的分数含义不同，不能直接横向比较。

- `2CKG4ER` 的分数是 SemanticConvE 输出的推荐概率型分数。
- `TransE-adv` 的原始分数是 KGE 嵌入空间打分，不是概率；论文表格中可以只展示排名和推荐题。
- `SB-CF` 的分数来自相似学生协同过滤得分。这里 Top-3 均为 `1.0`，表示相似学生对这些题的历史表现给出了强推荐信号。

## 3. 本文模型推荐解释

本文模型给 `uid285` 推荐 `ex122`：

| 字段 | 数值 |
|---|---:|
| Student | `uid285` |
| Exercise | `ex122` |
| KC | `kc79` |
| Concept Name | 整数好朋友数 |
| Mastery | `0.7411` |
| Forgetting | `0.7293` |
| Theta | `0.6717` |
| Difficulty | `0.2128` |
| Discrimination | `0.9061` |
| SemanticConvE Score | `0.9877` |

题目文本：

```text
计算：
（1）23 × 4 × 25 =
（2）125 × 13 × 8 =
```

可以这样解释：

> 对于学生 `uid285`，模型推荐 `ex122`。该题对应知识点为“整数好朋友数”。学生对该知识点的掌握度约为 `0.7411`，说明该知识点并非完全掌握，仍有练习价值；同时遗忘度约为 `0.7293`，说明该题涉及内容存在较明显的复习需求。题目难度较低但区分度较高，适合作为巩固型推荐题。因此，该推荐不仅来自图结构打分，还可以从知识掌握、遗忘状态和题目教育学属性三个角度解释。

## 4. TransE-adv 推荐解释

`TransE-adv` 给 `uid285` 的 Top-1 推荐为 `ex102`：

| 字段 | 内容 |
|---|---|
| Exercise | `ex102` |
| KC | `kc73` |
| Concept Name | 树形图 |
| 题目简述 | A/B/C 三个小朋友互相传球，求不同传球方式数 |
| Raw KGE Score | 约 `22.84` |

需要注意：

`TransE-adv` 的分数不是概率，而是 KGE 模型中的原始打分，和 `SemanticConvE` 的 `0~1` 概率型分数不能直接比较。它主要说明该三元组在嵌入空间中更符合模型学习到的图结构。

可以这样表述：

> `TransE-adv` 推荐 `ex102`，说明从知识图谱结构嵌入角度看，学生 `uid285` 与该题之间存在较强的推荐关系。但该模型本身不能直接给出该学生当前掌握度、遗忘度、题目难度和区分度等教育学解释。

## 5. SB-CF 推荐解释

`SB-CF` 给 `uid285` 推荐 `ex8`：

| 字段 | 内容 |
|---|---|
| Target Student | `uid285` |
| Similar Student | `uid261` |
| Similarity | `0.5638` |
| Common Exercises | `186` |
| Recommended Exercise | `ex8` |
| KC | `kc6`：陷阱型不规则图形求周长 |
| 推荐原因 | 相似学生 `uid261` 做过并答对 `ex8`，而 `uid285` 没做过该题 |

`SB-CF` 的基本逻辑是：

1. 将每个学生表示为历史答题向量。
2. 答对记为 `1`，答错记为 `-1`，未做过记为 `0`。
3. 用余弦相似度计算学生之间的相似性。
4. 如果相似学生做过某题且表现较好，则该题可能被推荐给目标学生。

相似度公式为：

```text
sim(u, v) = cosine(x_u, x_v)
```

推荐得分可以理解为：

```text
score(u, e) =
sum(sim(u, v) * response(v, e)) / sum(sim(u, v))
```

其中 `v` 是与目标学生 `u` 相似的学生，`response(v, e)` 表示相似学生在题目 `e` 上的答题结果。

## 6. 用于绘图的历史交互题目

为了画类似 KG4Ex Fig. 4 的图，可以在 SB-CF 模块里放一些 `uid285` 和 `uid261` 都做过且都答对的历史题。黑色箭头表示历史做题记录，蓝色箭头表示推荐。

### 6.1 共同历史题：用于说明两个学生相似

| 习题 | 知识点 | 题目简述 | `uid285` | `uid261` |
|---|---|---|---|---|
| `ex102` | `kc73`：树形图 | A/B/C 三人传球方式数 | 答对 | 答对 |
| `ex105` | `kc74`：图文周期 | 黑白三角形周期规律 | 答对 | 答对 |
| `ex106` | `kc75`：环形操作周期问题 | 8 个队员围圈传球 | 答对 | 答对 |
| `ex107` | `kc16`：基本排列的周期问题 | 黑白小球排列规律 | 答对 | 答对 |
| `ex119` | `kc45`：奇数与偶数的混合计算 | 判断算式奇偶性 | 答对 | 答对 |

这些共同历史题可以画在两个学生之间或上方，并用黑色历史交互箭头分别连接到 `uid285` 和 `uid261`。它们的作用是说明：两个学生在较多历史题上有相似的答题行为，因此 SB-CF 会认为他们具有较高相似性。

### 6.2 目标学生 `uid285` 独有历史题

为了让图不只是展示“共同题”，还可以加入一些 `uid285` 做过但 `uid261` 没做过的题，用来表示目标学生自己的学习轨迹。

| 习题 | 知识点 | 题目简述 | `uid285` 表现 |
|---|---|---|---|
| `ex81` | `kc5`：长方形周长 | 已知长方形长 17cm、宽 13cm，求周长 | 答对 |
| `ex85` | `kc60`：用枚举法找出最值 | 从 5、6、7 中选两个数组成最大两位数 | 答对 |
| `ex91` | `kc21`：单归一问题 | 绿化队种树效率问题 | 答错 |

这些题可以画在 `uid285` 左侧或上方，只从 `uid285` 连黑色箭头过去，表示它们是目标学生自己的历史练习记录。

### 6.3 相似学生 `uid261` 独有历史题

同样，可以加入一些 `uid261` 做过但 `uid285` 没做过的题。SB-CF 的推荐正是从这类相似学生历史行为中产生的。

| 习题 | 知识点 | 题目简述 | `uid261` 表现 |
|---|---|---|---|
| `ex8` | `kc6`：陷阱型不规则图形求周长 | 不规则图形周长题 | 答对 |
| `ex40` | `kc29`：单一对象条形图 | 文体活动满意度条形图 | 答对 |
| `ex121` | `kc79`：整数好朋友数 | 计算 `4 × 9 × 25` | 答对 |

其中 `ex8` 是 SB-CF 最终推荐给 `uid285` 的题目：`uid261` 做过并答对，而 `uid285` 没做过。因此，在图中可以把 `ex8` 标成蓝色推荐题，从 `uid261` 的历史题区域指向 `uid285`。

SB-CF 的推荐题：

| 习题 | 知识点 | 目标学生是否做过 | 相似学生是否做过 | 相似学生表现 |
|---|---|---|---|---|
| `ex8` | `kc6`：陷阱型不规则图形求周长 | 未做过 | 做过 | 答对 |

因此，图中可以这样安排：

1. 左边放目标学生 `uid285`。
2. 右边放相似学生 `uid261`。
3. 中间或上方放共同历史题，例如 `ex102`、`ex105`、`ex106`、`ex107`，用于说明两个学生相似。
4. `uid285` 一侧放目标学生独有历史题，例如 `ex81`、`ex85`、`ex91`。
5. `uid261` 一侧放相似学生独有历史题，例如 `ex40`、`ex121`、`ex8`。
6. 从两个学生分别连黑色箭头到共同历史题，表示他们具有相似答题行为。
7. 从 `uid261` 连黑色箭头到 `ex8`，表示相似学生做过该题。
8. 从 `ex8` 连蓝色推荐箭头到 `uid285`，表示 SB-CF 将该题推荐给目标学生。

## 7. Case Study 论文表述建议

可以在论文中这样组织：

> To illustrate the interpretability of the proposed method, we select a learner `uid285` from the `XES3G5M-sub-small` dataset. The SB-CF baseline recommends exercise `ex8` because a similar learner `uid261` correctly answered this exercise. This explanation is mainly based on learner similarity and historical interactions. The KGE baseline `TransE-adv` recommends exercise `ex102` according to structural proximity in the embedding space, but it does not explicitly explain the recommendation from pedagogical factors. In contrast, our model recommends exercise `ex122`, and the recommendation can be explained by multiple cognitive and pedagogical factors, including the learner's mastery level, forgetting degree, ability parameter, exercise difficulty, and discrimination. This demonstrates that the proposed model provides a more comprehensive explanation for exercise recommendation.

中文汇报时可以说：

> 我们选择 `uid285` 作为案例。协同过滤模型 `SB-CF` 推荐 `ex8`，原因是它找到一个相似学生 `uid261`，该学生做过并答对了 `ex8`。这种解释主要依赖“相似学生做过什么”。`TransE-adv` 推荐 `ex102`，它能说明图结构上这个推荐关系得分较高，但缺少教育学含义。相比之下，我们的模型推荐 `ex122`，不仅能给出推荐分数，还能解释该题对应知识点的掌握度、遗忘度，以及题目的难度和区分度，因此推荐理由更加全面。
