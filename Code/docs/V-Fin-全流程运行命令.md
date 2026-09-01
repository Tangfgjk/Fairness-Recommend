# KG4ER-New V-Fin 完整运行命令

本文件覆盖从拉取仓库、生成前置文件，到五台电脑并行运行
SemanticConvE 消融实验和对比模型、续跑与统计结果的完整流程。

## 1. 实验协议

- 前置文件与 ER 图均基于 `Data_Fin/<dataset>/raw/` 的同一批学生。
- SemanticConvE 完整模型和消融模型训练时加入 `test_triples.txt` 的认知状态边，使用 `triples.txt` only。该文件不包含 `rec` 关系。
- 对比模型训练只读取 `triples.txt`，不加入 `test_triples.txt`。
- 每台电脑负责一个数据集，并在该数据集上完成五个随机种子和全部模型。
- 不要在同一张 GPU 上同时启动 SemanticConvE 与对比模型；先完成一个再运行另一个。

正式 SemanticConvE 实验包含：

```text
feature_only
id_only
feature_only_relation_id
feature_only_learner_id
feature_only_exercise_id
feature_only_no_mastery
feature_only_no_forgetting
```

对比模型包括：

```text
TransE, TransE-adv, RotatE, DistMult, ComplEx,
EB-CF, SB-CF, CBF, KCP-ER
```

## 2. 拉取代码与环境

首次在新电脑运行：

```powershell
git clone -b V-Fin https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New

conda create -n kg4er_fin python=3.10 -y
conda activate kg4er_fin

# 先按本机 CUDA 环境安装对应的 PyTorch；下列示例适用于 CUDA 12.1。
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

仓库已经存在时更新到 V-Fin：

```powershell
cd KG4ER-New
git fetch origin
git checkout V-Fin
git pull --ff-only origin V-Fin
```

## 3. 校验已拉取的数据

`Data_Fin/` 会随 V-Fin 拉取。先检查五个 raw 数据集：

```powershell
python raw_pipeline\validate_raw_datasets.py `
  --output-root Data_Fin `
  --datasets all `
  --report Data_Fin\raw_validation_report.json
```

前置文件与 ER 图已生成时，可检查完整性：

```powershell
python front_pipeline\validate_front_pipeline.py `
  --datasets all `
  --data-fin-root Data_Fin `
  --require-graph `
  --output-file Data_Fin\front_validation_report.json
```

## 4. 生成一个数据集的前置文件

每台电脑只替换 `$ds` 为自己负责的数据集。该命令依次训练 Q-constrained MIRT、多标签序列模型、EKTM-MIRT，生成遗忘与语义特征，并构建 ER 图。

```powershell
$ds = "Eedi"

python front_pipeline\run_front_pipeline.py `
  --datasets $ds `
  --data-fin-root Data_Fin `
  --mirt-epochs 70 `
  --mirt-batch-size 1024 `
  --ektm-epochs 30 `
  --ektm-batch-size 16 `
  --device cuda `
  --force
```

若前置流程中断，直接使用相同命令重跑；不需要 `--force` 时，已存在的阶段会被跳过：

```powershell
python front_pipeline\run_front_pipeline.py `
  --datasets $ds `
  --data-fin-root Data_Fin `
  --mirt-epochs 70 `
  --ektm-epochs 30 `
  --device cuda `
```

生成结果位于：

```text
Data_Fin/<dataset>/front_features/
Data_Fin/<dataset>/er_graph/
```

## 5. 五台电脑的数据集分工

| 电脑 | `$ds` |
| --- | --- |
| 电脑 1 | `Eedi` |
| 电脑 2 | `algebra2005` |
| 电脑 3 | `assist2009-sub` |
| 电脑 4 | `statics2011` |
| 电脑 5 | `XES3G5M-sub-small` |

以下所有训练命令均只需将 `$ds` 改为本机对应数据集。

```powershell
$seeds = "2024,2025,2026,2027,2028"
$semanticAblations = "feature_only,id_only,feature_only_relation_id,feature_only_learner_id,feature_only_exercise_id,feature_only_no_mastery,feature_only_no_forgetting"
```

## 6. SemanticConvE 完整模型与消融实验

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset $ds `
  --data-root Data_Fin `
  --graph-subdir er_graph `
  --run-id "${ds}_vfin_semantic_5seeds" `
  --seeds $seeds `
  --ablations $semanticAblations `
  --epochs 25 `
  --bs 1024 `
  --learning-rate 0.001 `
  --negative-ratio 5 `
  --cuda auto
```

续跑：

```powershell
python codes-New-ConvE\run_semantic_experiments.py `
  --dataset $ds `
  --data-root Data_Fin `
  --graph-subdir er_graph `
  --run-id "${ds}_vfin_semantic_5seeds" `
  --seeds $seeds `
  --ablations $semanticAblations `
  --epochs 25 `
  --bs 1024 `
  --learning-rate 0.001 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

## 7. 所有对比模型

对比模型只用 `triples.txt` 训练；命令中不传递任何 include-test 参数。

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset $ds `
  --data-root Data_Fin `
  --graph-subdir er_graph `
  --run-id "${ds}_vfin_comparison_5seeds" `
  --seeds $seeds `
  --models all `
  --cuda auto
```

续跑：

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset $ds `
  --data-root Data_Fin `
  --graph-subdir er_graph `
  --run-id "${ds}_vfin_comparison_5seeds" `
  --seeds $seeds `
  --models all `
  --cuda auto `
  --resume
```

`RotatE` 训练时间通常远长于其余对比模型。若需单独运行或续跑它：

```powershell
python comparison_models\run_v9_comparison_experiments.py `
  --dataset $ds `
  --data-root Data_Fin `
  --graph-subdir er_graph `
  --run-id "${ds}_vfin_comparison_5seeds" `
  --seeds $seeds `
  --models RotatE `
  --cuda auto `
  --resume
```

## 8. 汇总结果

SemanticConvE：

```powershell
python codes-New-ConvE\summarize_semantic_results.py `
  --dataset $ds `
  --run-id "${ds}_vfin_semantic_5seeds" `
  --runs-root runs `
  --seeds $seeds `
  --ablations $semanticAblations
```

对比模型：

```powershell
python comparison_models\summarize_dataset_results.py `
  --dataset $ds `
  --run-dir "runs\$ds\${ds}_vfin_comparison_5seeds"
```

汇总文件位于各自 run 目录下的 `summaries/`。建议从五台电脑分别取回：

```text
runs/<dataset>/<dataset>_vfin_semantic_5seeds/summaries/
runs/<dataset>/<dataset>_vfin_comparison_5seeds/summaries/
```

## 9. 单个随机种子重跑

以 `2026` 为例：

```powershell
$oneSeed = "2026"

python codes-New-ConvE\run_semantic_experiments.py `
  --dataset $ds `
  --data-root Data_Fin `
  --graph-subdir er_graph `
  --run-id "${ds}_vfin_semantic_5seeds" `
  --seeds $oneSeed `
  --ablations $semanticAblations `
  --epochs 25 `
  --bs 1024 `
  --negative-ratio 5 `
  --cuda auto `
  --resume
```

同理，将上节对比模型命令中的 `--seeds $seeds` 改为 `--seeds $oneSeed`，即可重跑一个种子。
