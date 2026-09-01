# V-Fin6 AutoDL 三随机种子全实验命令

## 实验协议

- 数据目录：`Data_Fin/<dataset>/er_graph`。
- 随机种子：`2024,2025,2026`。
- 所有模型训练仅使用 `triples.txt`；`test_triples.txt` 仅用于推荐评估。
- SemanticConvE 主模型是 `feature_only`，不再运行旧 `full`。
- 推荐长度：`N=5,10,15,...,100`。
- 每个数据集运行八个 SemanticConvE 设置与八个对比模型；完成后自动统计、绘图，并按 `ACC@20` 自动选择最优 `feature_only` seed 导出本地可解释性文件。

## SemanticConvE 设置

1. `feature_only`：主模型。学习者仅 theta；题目仅文本、难度、区分度；关系仅类型和强度；知识点保留 ID。
2. `id_only`：实体和关系都只使用 ID embedding。
3. `feature_only_relation_id`：在主模型基础上，关系改为仅 ID。
4. `feature_only_learner_id`：在主模型基础上，学习者改为仅 ID。
5. `feature_only_exercise_id`：在主模型基础上，题目改为仅 ID。
6. `feature_only_no_mastery`：在主模型基础上删除 `mlkc`，重建推荐关系。
7. `feature_only_no_forgetting`：在主模型基础上删除 `exfr`，重建推荐关系。
8. `feature_only_no_seq`：在主模型基础上删除 `pkc`，重建推荐关系。

## 对比模型

`TransE, TransE-adv, RotatE, DistMult, ComplEx, EB-CF, SB-CF, CBF`。

## 每台 AutoDL 的准备

```bash
cd /root/autodl-tmp
git clone -b V-Fin6 --single-branch https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New

# 上传对应数据集目录到 Data_Fin/<dataset>/，其中必须包含 er_graph/。
# 若 Windows 上传的脚本为 CRLF，执行一次：
sed -i 's/\r$//' scripts/run_autodl_vfin_dataset.sh
chmod +x scripts/run_autodl_vfin_dataset.sh

# 保留镜像中带 CUDA 的 torch，只安装其余依赖。
sed '/^torch$/d' requirements.txt > requirements_no_torch.txt
pip install -r requirements_no_torch.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 五台 GPU 命令

分别在五台实例的仓库根目录运行：

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi all --seeds 2024,2025,2026
```

```bash
bash scripts/run_autodl_vfin_dataset.sh algebra2005 all --seeds 2024,2025,2026
```

```bash
bash scripts/run_autodl_vfin_dataset.sh assist2009-sub all --seeds 2024,2025,2026
```

```bash
bash scripts/run_autodl_vfin_dataset.sh statics2011 all --seeds 2024,2025,2026
```

```bash
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small all --seeds 2024,2025,2026
```

## 断点续跑

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi all --seeds 2024,2025,2026 --resume
```

将 `Eedi` 替换为对应数据集即可。`--resume` 会跳过已经完成且存在结果文件的阶段。

## 输出位置

以 Eedi 为例：

```text
runs/Eedi/Eedi_vfin6_semantic_3seeds/summaries/
runs/Eedi/Eedi_vfin6_comparison_3seeds/summaries/full_metrics/
runs/Eedi/Eedi_vfin6_report_3seeds/
runs/Eedi/Eedi_vfin6_semantic_3seeds/SemanticConvE_feature_only/local_explanation_best_seed<seed>/
```

最后一个目录包含 `local_evidence_top20.csv`、`local_evidence_top20.json`、`local_evidence_top20.md` 与 `primary_model_selection.json`。后者记录被选中的 seed、三次 `ACC@20` 以及评分文件路径。
