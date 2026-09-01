# quota_ratio_hybrid 参数敏感性实验流水线实现记录

生成时间：2026-08-11 20:17:29

## 1. 实现目标

本次新增一个专门用于 `quota_ratio_hybrid` 参数敏感性实验的脚本，用于回答后续论文中的参数选择问题：

1. 长尾习题比例设为多少合适？
2. 知识点覆盖比例设为多少合适？
3. 候选池大小是否会影响公平性提升和 Ada/NOV 损失？

新增脚本：

`E:\Fairness-aware\2CKG4ER\Code\codes-New-ConvE\run_fairness_sensitivity_experiments.py`

新增测试：

`E:\Fairness-aware\2CKG4ER\Code\codes-New-ConvE\test_run_fairness_sensitivity_experiments.py`

## 2. 默认实验组

脚本默认运行三组单变量敏感性实验。

### 2.1 长尾习题比例敏感性

固定：

- `kc-coverage-ratio=0.2`
- `candidate-multiplier=1.5`
- `lambda-item=0.8`
- `lambda-kc=0.3`
- `beta-item=0.3`
- `beta-kc=0.1`

变化：

| 方法名 | long-tail-item-ratio |
| --- | ---: |
| `quota_ratio_hybrid_lt005` | 0.05 |
| `quota_ratio_hybrid_lt010` | 0.10 |
| `quota_ratio_hybrid_lt015` | 0.15 |
| `quota_ratio_hybrid_lt020` | 0.20 |

### 2.2 知识点覆盖比例敏感性

固定：

- `long-tail-item-ratio=0.1`
- `candidate-multiplier=1.5`
- `lambda-item=0.8`
- `lambda-kc=0.3`
- `beta-item=0.3`
- `beta-kc=0.1`

变化：

| 方法名 | kc-coverage-ratio |
| --- | ---: |
| `quota_ratio_hybrid_kc010` | 0.10 |
| `quota_ratio_hybrid_kc020` | 0.20 |
| `quota_ratio_hybrid_kc025` | 0.25 |
| `quota_ratio_hybrid_kc030` | 0.30 |

### 2.3 候选池大小敏感性

固定：

- `long-tail-item-ratio=0.1`
- `kc-coverage-ratio=0.2`
- `lambda-item=0.8`
- `lambda-kc=0.3`
- `beta-item=0.3`
- `beta-kc=0.1`

变化：

| 方法名 | candidate-multiplier |
| --- | ---: |
| `quota_ratio_hybrid_cand10` | 1.0 |
| `quota_ratio_hybrid_cand15` | 1.5 |
| `quota_ratio_hybrid_cand20` | 2.0 |
| `quota_ratio_hybrid_cand30` | 3.0 |

## 3. 输出结构

每个实验会生成：

1. 重排分数文件：

`2CKG4ER_uid_ex_scores_quota_ratio_hybrid_*.pkl`

2. 评估目录：

`eval_quota_ratio_hybrid_*`

3. 参数敏感性总表：

`comparison_sensitivity`

其中包含：

- `fairness_comparison.csv`
- `fairness_comparison_delta.csv`
- `fairness_comparison.md`

4. 流水线 manifest：

`fairness_sensitivity_pipeline.json`

## 4. 使用命令

建议先 dry-run：

```powershell
Set-Location "E:\Fairness-aware\2CKG4ER\Code\codes-New-ConvE"

python .\run_fairness_sensitivity_experiments.py `
  --data-dir "E:\Fairness-aware\2CKG4ER\Code\Data_Fin\Eedi\er_graph" `
  --run-dir "E:\Fairness-aware\2CKG4ER\Code\runs\Eedi\Eedi_2CKG4ER_seed2024_baseline\2CKG4ER\seed2024" `
  --dataset-name Eedi `
  --seed 2024 `
  --dry-run
```

正式运行：

```powershell
Set-Location "E:\Fairness-aware\2CKG4ER\Code\codes-New-ConvE"

python .\run_fairness_sensitivity_experiments.py `
  --data-dir "E:\Fairness-aware\2CKG4ER\Code\Data_Fin\Eedi\er_graph" `
  --run-dir "E:\Fairness-aware\2CKG4ER\Code\runs\Eedi\Eedi_2CKG4ER_seed2024_baseline\2CKG4ER\seed2024" `
  --dataset-name Eedi `
  --seed 2024
```

只跑某一组，例如只跑长尾比例：

```powershell
python .\run_fairness_sensitivity_experiments.py `
  --data-dir "E:\Fairness-aware\2CKG4ER\Code\Data_Fin\Eedi\er_graph" `
  --run-dir "E:\Fairness-aware\2CKG4ER\Code\runs\Eedi\Eedi_2CKG4ER_seed2024_baseline\2CKG4ER\seed2024" `
  --dataset-name Eedi `
  --seed 2024 `
  --suites lt
```

只跑知识点覆盖比例：

```powershell
python .\run_fairness_sensitivity_experiments.py `
  --data-dir "E:\Fairness-aware\2CKG4ER\Code\Data_Fin\Eedi\er_graph" `
  --run-dir "E:\Fairness-aware\2CKG4ER\Code\runs\Eedi\Eedi_2CKG4ER_seed2024_baseline\2CKG4ER\seed2024" `
  --dataset-name Eedi `
  --seed 2024 `
  --suites kc
```

只跑候选池大小：

```powershell
python .\run_fairness_sensitivity_experiments.py `
  --data-dir "E:\Fairness-aware\2CKG4ER\Code\Data_Fin\Eedi\er_graph" `
  --run-dir "E:\Fairness-aware\2CKG4ER\Code\runs\Eedi\Eedi_2CKG4ER_seed2024_baseline\2CKG4ER\seed2024" `
  --dataset-name Eedi `
  --seed 2024 `
  --suites candidate
```

## 5. 验证情况

已通过：

```powershell
python -m py_compile "E:\Fairness-aware\2CKG4ER\Code\codes-New-ConvE\run_fairness_sensitivity_experiments.py"
python -m unittest "E:\Fairness-aware\2CKG4ER\Code\codes-New-ConvE\test_run_fairness_sensitivity_experiments.py"
```

单测结果：

```text
Ran 4 tests
OK
```

全量测试本轮尝试时，代码断言没有暴露失败，但当前 Python 环境加载 `numpy/torch` 时出现内存不足：

```text
MemoryError
OSError: [WinError 1455] 页面文件太小，无法完成操作
```

因此本轮全量测试未作为有效验证结果记录。新增脚本自身的编译和单元测试已经通过。

## 6. 后续分析方式

实验跑完后，优先查看：

`E:\Fairness-aware\2CKG4ER\Code\runs\Eedi\Eedi_2CKG4ER_seed2024_baseline\2CKG4ER\seed2024\comparison_sensitivity\fairness_comparison.md`

建议重点分析 `K=10,20,50,100`：

1. `Ada@K` 是否随着公平约束增强明显下降。
2. `LongTailItemExposureShare@K` 是否随着 `long-tail-item-ratio` 上升。
3. `KCCoverage@K` 和 `KCExposureGini@K` 是否随着 `kc-coverage-ratio` 改善。
4. `candidate-multiplier` 变大是否带来更好的公平性，还是收益很快饱和。

