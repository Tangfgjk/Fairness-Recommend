# MAED正式训练前两项工程验证：中途续跑与Logit排序（2026-10-23）

## 1. 结论与修改范围

对`c6f253c`之后提出的两项P1风险，已新增真实训练循环回归测试和实际评价链路测试，两项均通过。没有发现需要修改生产训练或评价算法的错误。

本次只改测试和文档，不改MAED数学方法、正式参数、训练循环、评分导出行为或原项目代码。测试不进入生产源码摘要，现有默认run-id及已锁定的预检合同继续有效。

默认仍为三个数据集、seed2024、M3/M4六组训练；beta=0.5、lambdaE=0.01、原25轮及其他正式设置保持。正式训练尚未启动。

## 2. 正权重M4在epoch中途中断

测试：`test_real_Eedi_M4_mid_epoch_failure_replays_batches_negatives_and_updates`。

使用真实Eedi教育状态、原模型特征、原ConvE结构和真实训练图三元组。仅工程测试缩短为64条rec边及64条非rec边、batch size=64、两轮；保留negative ratio=5、beta=0.5、lambdaE=0.01，每次曝光计算用16位训练学生及完整948题。测试因此每轮7批，所有批次实际执行正权重曝光损失。

在本机CPU上比较：

1. 连续完成两轮。
2. 第2轮第3批已完成原Adam更新后，主动抛出异常。
3. 验证磁盘`last.pt`仍是第1轮，且没有训练完成标记。
4. 重新调用真实`train()`，重新构造DataLoader并从第1轮checkpoint恢复，完整重做第2轮。

精确相等断言通过：

- 中断前的批次与连续运行前缀一致；续跑7批与连续运行第2轮一致，包括负采样尾实体。
- 独立曝光采样器的每批前后状态及曝光损失一致。
- best/last的模型完整state、Adam状态、轮次、监督损失及最佳轮次一致。
- Python、NumPy、Torch及CUDA RNG状态（可用时）的保存值一致。
- 两个last checkpoint导出的全部测试学生推荐JSON和logit数组一致。

这是实际Eedi模型及多批次训练循环的工程子集验证，不是完整数据25轮训练，也不是CUDA长时间训练复现。另有既有真实Eedi单批CUDA联合反向检查，不能将其写成“CUDA中途续跑已完成独立复现”。

## 3. 导出logit的评价语义

测试：`test_M3_M4_export_evaluation_uses_order_not_probability_threshold`。

先在临时小图上实际训练正beta的M3和正曝光权重M4，分别调用真实`export_scores()`；验证导出JSON的`scores`等于配套`.logits.npy`，且元数据`score_kind=Q_fused_logit`。

对两种模型各自的导出分数，执行真实`debias.evaluation.evaluate_file()`以及统一指标、Ep分数读取、排序、`evaluate_entry()`和`simulate()`。工程小图只用4题及N=1/2/3；仅缩小Top-N设置及提供KT状态/答题概率夹具，不替换推荐排序和指标计算代码。

### 已验证

- 无浮点并列时，logit和其float64 sigmoid的完整排序一致。
- 同排序的两种输入，Ada、KG4Ex-NOV-Set、曝光/覆盖/DED、推荐列表、Ep及逐学生结果一致。
- 把全部推荐分数移到负数区间，不改变上述指标，不会因推荐分数低于0.5而漏选习题。
- 构造float32 sigmoid饱和例子`[0,100,102,101]`，sigmoid制造并列；正式logit输入仍按`[2,3,1,0]`排序。Ep实际模拟收到的前三题也是`[2,3,1]`。
- 将KT答对概率夹具设为0.49时，Ep@0.5的模拟答对率为0；该阈值作用于KT预测，不作用于推荐logit。

没有把小图/KT夹具下的Ep数值解释为真实Eedi的学习效果，也没有正式M3/M4成绩可供此时比较。

## 4. 下游源码核对与重要例外

|链路|推荐分数的用途|
|---|---|
|`debias.context.load_scores`|读取有限数值及检查学生/题目维度，不要求范围为[0,1]|
|`debias.metrics.evaluate`|降序稳定排序，原题目编号升序处理相同分数；Ada/NOV/DED/曝光指标使用选中列表|
|`debias.evaluation.bootstrap_exposure`|先排序，重采样学生并重新计算曝光|
|`mastery_ep_v2.evaluate.evaluate_entry`|按`(-score, exercise_id)`排序，再将题目序列传给模拟|
|`mastery_ep_v2.evaluate.simulate`|不接收推荐分数；0.5比较对象是`answer_probability`的KT答对概率|

**CFP的附带`mean_score_gap`及逐学生`score_gap`是例外**：它们直接计算推荐分数差。正beta的Q-fixed/M3/M4使用logit差值；历史概率分数实验使用概率差值，两种尺度不能直接比较大小。CFP偏好比例本身只依赖正负/并列关系，严格单调变换下保持，sigmoid浮点饱和并列可使其发生变化。

因此，准确说法是：正式教育/曝光指标和Ep以推荐排序为依据；数值分差是依赖评分尺度的机制诊断。当前主结果不把CFP分差作为跨logit/概率模型的优劣标准。

不需要另建概率推荐接口。训练BCE继续调用`QScorer.probabilities()`；正beta的推荐导出保持logit用于排序。现有`score_kind`元数据保留，不将其解释为校准后的答对概率。

## 5. 执行记录与运行命令

本次训练测试命令：

```powershell
& $py -m unittest discover -s tests -p 'test_maed_training.py' -v
```

14项训练/监测/评价测试全部通过（其中新增以上两项），本次耗时约35秒。临时训练checkpoint和小图评价随测试临时目录清理，日志保存在本地`tmp_maed_two_checks.log`，不混入正式结果。

正式启动时可执行：

```powershell
$repo = 'C:\Users\21709\Desktop\Fairness-Recommend\Fairness-Recommend-New'
$py = 'C:\Users\21709\.conda\envs\rtx5060\python.exe'
Set-Location -LiteralPath $repo

& $py -u .\scripts\run_maed_training.py --stage prepare
if ($LASTEXITCODE -ne 0) { throw '预检失败，请先检查日志' }
& $py -u .\scripts\run_maed_training.py --stage run
if ($LASTEXITCODE -ne 0) { throw '训练或评价中断，请查看日志并用同一命令续跑' }
```

默认目录仍为`runs/maed_controlled_monitored_seed2024_20261023/`；报告在`runs/论文格式文档统计/maed_controlled_monitored_seed2024_20261023/`。

Ada绝对下降≤0.01、KG4Ex-NOV-Set绝对下降≤0.03继续按每个N相对E0验收。工程验证不证明M4一定达标，M4−M3仍是曝光目标的主要增量证据。历史末端快照及原预处理来源限制继续保留。
