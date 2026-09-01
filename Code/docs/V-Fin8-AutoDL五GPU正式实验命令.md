# V-Fin8 AutoDL 五 GPU 正式实验命令

本文档只覆盖 ER 阶段训练、测试、统计、绘图和可解释性导出。`Data_Fin` 不上传 GitHub，需要单独拷贝到 AutoDL。

## 1. 实验设置

每个数据集在一台 GPU 上运行，随机种子为：

```bash
2024,2025,2026
```

SemanticConvE 训练时只使用 `triples.txt`，不加入 `test_triples.txt`。`test_triples.txt` 只用于最终评价。

SemanticConvE 正式消融模型共 7 个：

```text
feature_only
id_only
feature_only_relation_id
feature_only_learner_id
feature_only_exercise_id
feature_only_no_mastery
feature_only_no_forgetting
```

其中 `feature_only` 是主模型，并会自动导出可解释性推荐列表。

对比模型共 8 个，均为 ID-only 或传统基线，不读取 `semantic_kg_features`：

```text
TransE
TransE-adv
RotatE
DistMult
ComplEx
EB-CF
SB-CF
CBF
```

Top-K 评价范围为：

```text
5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100
```

## 2. AutoDL 准备

建议实例：RTX 4090 / 4090D，PyTorch CUDA 镜像，Python 3.10。

进入数据盘：

```bash
cd /root/autodl-tmp
```

拉取代码：

```bash
git clone https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New
git fetch origin
git checkout V-Fin8
git pull origin V-Fin8
```

如果已经上传 zip 代码包并解压后出现 `KG4ER-New/KG4ER-New` 双层目录，则进入内层：

```bash
cd /root/autodl-tmp/KG4ER-New/KG4ER-New
```

安装依赖。保留镜像自带 CUDA 版 PyTorch，不重复安装 `torch`：

```bash
sed -i 's/\r$//' scripts/run_autodl_vfin_dataset.sh
chmod +x scripts/run_autodl_vfin_dataset.sh

sed '/^torch$/d' requirements.txt > requirements_no_torch.txt
pip install -r requirements_no_torch.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import numpy,pandas,sklearn,torch,tqdm,wandb,einops,matplotlib; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果清华源临时不可用，可改用：

```bash
pip install -r requirements_no_torch.txt -i https://pypi.org/simple
pip install matplotlib -i https://pypi.org/simple
```

## 3. 拷贝 Data_Fin

每台 AutoDL 至少需要拷贝对应数据集：

```text
Data_Fin/<dataset>/er_graph/
```

推荐目录结构：

```text
/root/autodl-tmp/KG4ER-New/
  Data_Fin/
    Eedi/
      er_graph/
    algebra2005/
      er_graph/
    assist2009-sub/
      er_graph/
    statics2011/
      er_graph/
    XES3G5M-sub-small/
      er_graph/
```

验证某个数据集是否可用：

```bash
python codes-New-ConvE/validate_semantic_ready.py \
  --datasets Eedi \
  --data-root Data_Fin \
  --graph-subdir er_graph
```

## 4. 五台 GPU 分别运行

每台机器运行一个数据集即可。

GPU 1:

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin_dataset.sh Eedi all --seeds 2024,2025,2026
```

GPU 2:

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin_dataset.sh algebra2005 all --seeds 2024,2025,2026
```

GPU 3:

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin_dataset.sh assist2009-sub all --seeds 2024,2025,2026
```

GPU 4:

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin_dataset.sh statics2011 all --seeds 2024,2025,2026
```

GPU 5:

```bash
cd /root/autodl-tmp/KG4ER-New
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small all --seeds 2024,2025,2026
```

如果 XES 仍然太慢，可以拆成两台或两次运行：

```bash
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small semantic --seeds 2024,2025,2026
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small comparison --seeds 2024,2025,2026
```

## 5. 续跑

同一个数据集续跑：

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi all --seeds 2024,2025,2026 --resume
```

只续跑 SemanticConvE：

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi semantic --seeds 2024,2025,2026 --resume
```

只续跑对比模型：

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi comparison --seeds 2024,2025,2026 --resume
```

## 6. 输出位置

以 Eedi 为例：

SemanticConvE 结果：

```text
runs/Eedi/Eedi_vfin8_semantic_3seeds/
```

对比模型结果：

```text
runs/Eedi/Eedi_vfin8_comparison_3seeds/
```

总表和折线图：

```text
runs/Eedi/Eedi_vfin8_report_3seeds/
```

其中包括：

```text
topk_comparison.csv
topk_comparison.md
topk_representation.png
topk_cognitive.png
topk_baselines.png
```

可解释性推荐列表：

```text
runs/Eedi/Eedi_vfin8_semantic_3seeds/SemanticConvE_feature_only/local_explanation_best_seed<seed>/
```

其中包括：

```text
local_evidence_top20.csv
local_evidence_top20.json
local_evidence_top20.md
primary_model_selection.json
```

## 7. 手动重新统计和绘图

如果训练已经完成，只想重新统计：

```bash
DATASET=Eedi
TOP_KS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"

python codes-New-ConvE/summarize_semantic_results.py \
  --dataset "$DATASET" \
  --run-id "${DATASET}_vfin8_semantic_3seeds" \
  --runs-root runs \
  --seeds 2024,2025,2026 \
  --ablations id_only,feature_only,feature_only_relation_id,feature_only_learner_id,feature_only_exercise_id,feature_only_no_mastery,feature_only_no_forgetting \
  --top-ks "$TOP_KS"

python comparison_models/summarize_dataset_results.py \
  --dataset "$DATASET" \
  --run-dir "runs/${DATASET}/${DATASET}_vfin8_comparison_3seeds"

python scripts/report_topk_comparison.py \
  --run-dir "SemanticConvE=runs/${DATASET}/${DATASET}_vfin8_semantic_3seeds" \
  --run-dir "Comparison=runs/${DATASET}/${DATASET}_vfin8_comparison_3seeds" \
  --output-dir "runs/${DATASET}/${DATASET}_vfin8_report_3seeds" \
  --top-ks "$TOP_KS"
```

手动重新生成可解释性推荐列表：

```bash
python scripts/export_primary_model_evidence.py \
  --data-dir "Data_Fin/${DATASET}/er_graph" \
  --semantic-run-dir "runs/${DATASET}/${DATASET}_vfin8_semantic_3seeds" \
  --seeds 2024,2025,2026 \
  --selection-top-k 20 \
  --evidence-top-k 20
```

## 8. 压缩下载结果

压缩时不保存大模型和中间大文件：

```bash
DATASET=Eedi
cd /root/autodl-tmp/KG4ER-New/runs/${DATASET}

tar -czf /root/autodl-tmp/${DATASET}_vfin8_results_slim.tar.gz \
  --exclude='*/best.pt' \
  --exclude='*/last.pt' \
  --exclude='*/checkpoint*' \
  --exclude='*.ckpt' \
  --exclude='*.npy' \
  --exclude='*.pkl' \
  --exclude='*/scores/*' \
  "${DATASET}_vfin8_semantic_3seeds" \
  "${DATASET}_vfin8_comparison_3seeds" \
  "${DATASET}_vfin8_report_3seeds"
```

如果需要额外保存主模型 `feature_only` 的一个最佳 seed 模型用于后续预测，请先运行自动选择脚本生成 `primary_model_selection.json`，再根据其中的 `best_seed` 单独打包对应目录下的 `best.pt`。
