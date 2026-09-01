# V-Fin4 AutoDL 六 GPU 正式实验命令

## 1. 实验协议

- 数据根目录：`Data_Fin/<dataset>/er_graph`。
- 五个随机种子：`2024,2025,2026,2027,2028`。
- SemanticConvE 与所有 KGE 对比模型训练时均只使用 `triples.txt`。
- `test_triples.txt` 仅用于最终推荐评价；不向训练图加入测试认知关系。
- 已删除 `KCP-ER`。对比模型为 `TransE`、`TransE-adv`、`RotatE`、`DistMult`、`ComplEx`、`EB-CF`、`SB-CF`、`CBF`。
- `EB-CF`、`SB-CF`、`CBF` 是确定性方法，因此只需运行一次；KGE 与 SemanticConvE 运行五个随机种子。

SemanticConvE 的九个设置：

`full,id_only,feature_only,feature_only_relation_id,feature_only_learner_id,feature_only_exercise_id,no_forgetting,no_mastery,no_seq`

其中后三种特征消融均以 `feature_only` 为起点，只恢复一类 ID：

- `feature_only_relation_id`：关系仅 relation ID；学生仍仅 theta，题目仍仅文本、难度、区分度。
- `feature_only_learner_id`：学生仅 learner ID；题目和关系保持 feature-only。
- `feature_only_exercise_id`：题目仅 exercise ID；学生和关系保持 feature-only。

`no_mastery`、`no_forgetting`、`no_seq` 会各自重建推荐表：被删除项既不会作为状态关系进入图，也不会进入推荐距离公式。重建排序使用完整浮点精度。

## 2. 每台 AutoDL 实例的初始准备

```bash
cd /root/autodl-tmp
git clone -b V-Fin4 --single-branch https://github.com/Tangfgjk/KG4ER-New.git
cd /root/autodl-tmp/KG4ER-New

# Windows 上传的脚本可能带 CRLF；执行一次即可。
sed -i 's/\r$//' scripts/run_autodl_vfin_dataset.sh
chmod +x scripts/run_autodl_vfin_dataset.sh

# 保留镜像内带 CUDA 的 PyTorch，不重复安装 torch。
sed '/^torch$/d' requirements.txt > requirements_no_torch.txt
conda install -y -c conda-forge pandas scikit-learn tqdm wandb einops matplotlib
pip install -r requirements_no_torch.txt

python -c "import numpy,pandas,sklearn,torch,tqdm,wandb,einops,matplotlib; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

将本地对应数据集的 `Data_Fin/<dataset>` 上传到：

```text
/root/autodl-tmp/KG4ER-New/Data_Fin/<dataset>
```

至少应包含 `er_graph/`；其中必须有 `triples.txt`、`test_triples.txt`、`entities.dict`、`relations.dict`、`Q.txt`、四个状态 JSON，以及 `semantic_kg_features/`。

更新已有实例上的代码：

```bash
cd /root/autodl-tmp/KG4ER-New
git fetch origin
git checkout V-Fin4
git pull --ff-only origin V-Fin4
sed -i 's/\r$//' scripts/run_autodl_vfin_dataset.sh
```

## 3. 六台 GPU 的分配命令

四个常规数据集各用一台 GPU，运行全部 SemanticConvE 消融与全部对比模型：

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi all
bash scripts/run_autodl_vfin_dataset.sh algebra2005 all
bash scripts/run_autodl_vfin_dataset.sh assist2009-sub all
bash scripts/run_autodl_vfin_dataset.sh statics2011 all
```

XES3G5M-sub-small 分到两台 GPU：

```bash
# GPU-5：只运行九个 SemanticConvE 设置。
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small semantic

# GPU-6：只运行全部对比模型。
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small comparison
```

脚本会先校验 ER 图，再执行训练、测试、统计。前四个数据集使用 `all` 时还会自动生成三张分组折线图。

## 4. 断点续跑

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi all --resume
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small semantic --resume
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small comparison --resume
```

`--resume` 会跳过已完成的 `seed/model` 目录。不要手工删除 `run_status.json`、`*_stage.json` 或同名结果目录。

## 5. XES 合并后生成总报告

从两台 XES 实例分别下载：

```text
runs/XES3G5M-sub-small/XES3G5M-sub-small_vfin4_semantic_5seeds/
runs/XES3G5M-sub-small/XES3G5M-sub-small_vfin4_comparison_5seeds/
```

将二者放到同一台机器的相同 `runs/XES3G5M-sub-small/` 下，再运行：

```bash
cd /root/autodl-tmp/KG4ER-New

python codes-New-ConvE/summarize_semantic_results.py \
  --dataset XES3G5M-sub-small \
  --run-id XES3G5M-sub-small_vfin4_semantic_5seeds \
  --runs-root runs \
  --seeds 2024,2025,2026,2027,2028 \
  --ablations full,id_only,feature_only,feature_only_relation_id,feature_only_learner_id,feature_only_exercise_id,no_forgetting,no_mastery,no_seq

python comparison_models/summarize_dataset_results.py \
  --dataset XES3G5M-sub-small \
  --run-dir runs/XES3G5M-sub-small/XES3G5M-sub-small_vfin4_comparison_5seeds

python scripts/report_topk_comparison.py \
  --run-dir SemanticConvE=runs/XES3G5M-sub-small/XES3G5M-sub-small_vfin4_semantic_5seeds \
  --run-dir Comparison=runs/XES3G5M-sub-small/XES3G5M-sub-small_vfin4_comparison_5seeds \
  --output-dir runs/XES3G5M-sub-small/XES3G5M-sub-small_vfin4_report
```

## 6. 结果位置与图形含义

每个数据集的结果目录：

```text
runs/<dataset>/<dataset>_vfin4_semantic_5seeds/summaries/
runs/<dataset>/<dataset>_vfin4_comparison_5seeds/summaries/
runs/<dataset>/<dataset>_vfin4_report/
```

`vfin4_report` 包含：

- `topk_comparison.csv`：所有模型、全部 N 的可处理统计表。
- `topk_comparison.md`：可直接阅读的 ACC/NOV 表。
- `topk_representation.png`：表示消融图，最多 6 条曲线。
- `topk_cognitive.png`：掌握度、遗忘、顺序关系消融图，4 条曲线。
- `topk_baselines.png`：`full` 与 8 个对比模型，9 条曲线。

同一模型在全部数据集及全部图中始终使用同一颜色与点形；`full` 用更粗的深蓝线突出。主结论以 ACC 为准，NOV 作为辅助指标。
