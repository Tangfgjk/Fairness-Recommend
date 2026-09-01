# V-Fin4: 已保存分数重评估 Top-K=5,10,...,100 与下载命令

本说明适用于已经完成训练、且每个模型目录仍保存 `*_uid_ex_scores.pkl` 的 AutoDL 实例。

目标是将原来的推荐列表长度：

```text
10, 20, 30, ..., 100
```

改为：

```text
5, 10, 15, 20, ..., 100
```

整个过程**不训练、不重新推理、不需要 checkpoint**。它只读取每位学生对所有习题的已保存预测分数，重新取排序前 `N` 道题并计算 ACC/NOV。

## 1. 前提与输出

每个已完成模型必须存在如下任一分数文件：

```text
<seed>/scores/<Model>_uid_ex_scores.pkl       # KGE 对比模型
<seed>/SemanticConvE_uid_ex_scores.pkl        # SemanticConvE
traditional_baselines/outputs/<Model>_uid_ex_scores.pkl  # CBF / SB-CF / EB-CF
```

重评估完成后：

- 原始 `eval/` 会移动为 `eval_old_10step/`（仅首次）；
- 新 `eval/metrics.json`、`metrics.csv` 含 `@5,@10,@15,...,@100`；
- 所有分数排序保持不变，因此结果只是在更多 Top-K 截断点上计算。

## 2. 进入 AutoDL 项目根目录

本项目在 AutoDL 可能有一层或两层 `KG4ER-New` 目录。先定位：

```bash
find /root/autodl-tmp -path '*/codes-New-ConvE/evaluate_recommendations.py' -print
```

常见目录为：

```bash
cd /root/autodl-tmp/KG4ER-New/KG4ER-New
```

后续命令均从项目根目录执行。

## 3. 只需执行一次：让汇总与绘图脚本读取 5 间隔 Top-K

```bash
sed -i 's/TOP_KS = tuple(range(10, 101, 10))/TOP_KS = tuple(range(5, 101, 5))/' \
  comparison_models/summarize_dataset_results.py \
  scripts/report_topk_comparison.py
```

## 4. 一键重评估一个数据集的全部已保存分数

将以下整段复制到 AutoDL 终端。只改第一行的 `DATASET` 即可。

```bash
DATASET="Eedi"
TOP_KS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"

SEM_RUN="runs/${DATASET}/${DATASET}_vfin4_semantic_5seeds"
CMP_RUN="runs/${DATASET}/${DATASET}_vfin4_comparison_5seeds"
DATA_DIR="Data_Fin/${DATASET}/er_graph"

RUN_DIRS=()
[ -d "$SEM_RUN" ] && RUN_DIRS+=("$SEM_RUN")
[ -d "$CMP_RUN" ] && RUN_DIRS+=("$CMP_RUN")

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
  echo "No completed run directory found for ${DATASET}."
  exit 1
fi

for RUN_DIR in "${RUN_DIRS[@]}"; do
  while IFS= read -r -d '' SCORES_FILE; do
    SCORE_PARENT="$(dirname "$SCORES_FILE")"
    PARENT_NAME="$(basename "$SCORE_PARENT")"
    MODEL_NAME="$(basename "$SCORES_FILE" _uid_ex_scores.pkl)"

    # KGE: seedXXXX/scores/*.pkl
    # SemanticConvE: seedXXXX/*.pkl
    # Traditional CF: traditional_baselines/outputs/*.pkl,
    # whose evaluation destination is traditional_baselines/<Model>/eval.
    if [ "$PARENT_NAME" = "scores" ]; then
      MODEL_DIR="$(dirname "$SCORE_PARENT")"
    elif [ "$PARENT_NAME" = "outputs" ]; then
      MODEL_DIR="$(dirname "$SCORE_PARENT")/${MODEL_NAME}"
    else
      MODEL_DIR="$SCORE_PARENT"
    fi

    SEED_NAME="$(basename "$MODEL_DIR")"
    if [ -d "$MODEL_DIR/eval" ] && [ ! -d "$MODEL_DIR/eval_old_10step" ]; then
      mv "$MODEL_DIR/eval" "$MODEL_DIR/eval_old_10step"
    fi
    rm -rf "$MODEL_DIR/eval"

    SEED_ARGS=()
    if [[ "$SEED_NAME" =~ ^seed([0-9]+)$ ]]; then
      SEED_ARGS=(--seed "${BASH_REMATCH[1]}")
    fi

    echo "Re-evaluating: ${MODEL_NAME} | ${MODEL_DIR}"
    python codes-New-ConvE/evaluate_recommendations.py \
      --data-dir "$DATA_DIR" \
      --scores-file "$SCORES_FILE" \
      --output-dir "$MODEL_DIR/eval" \
      --dataset-name "$DATASET" \
      --model-name "$MODEL_NAME" \
      --top-ks "$TOP_KS" \
      --ep-top-k 10 \
      --target-mastery 0.8 \
      "${SEED_ARGS[@]}"
  done < <(find "$RUN_DIR" -type f -name '*_uid_ex_scores.pkl' -print0)
done
```

### 五个数据集的调用

对每个数据集，只需将上面 `DATASET="Eedi"` 改为下列其中一个，然后完整执行第 4 节：

```bash
DATASET="Eedi"
DATASET="algebra2005"
DATASET="assist2009-sub"
DATASET="statics2011"
DATASET="XES3G5M-sub-small"
```

如果 XES 当前只跑完对比模型，脚本会自动跳过不存在的 SemanticConvE 运行目录，只重评估对比模型。

## 5. 重建汇总表与折线图

设置当前数据集：

```bash
DATASET="Eedi"
SEM_RUN="runs/${DATASET}/${DATASET}_vfin4_semantic_5seeds"
CMP_RUN="runs/${DATASET}/${DATASET}_vfin4_comparison_5seeds"
```

### 5.1 SemanticConvE 汇总

仅当该目录存在时运行：

```bash
python codes-New-ConvE/summarize_semantic_results.py \
  --dataset "$DATASET" \
  --runs-root runs \
  --run-id "${DATASET}_vfin4_semantic_5seeds" \
  --seeds 2024,2025,2026,2027,2028 \
  --ablations full,id_only,feature_only,feature_only_relation_id,feature_only_learner_id,feature_only_exercise_id,no_mastery,no_forgetting,no_seq \
  --top-ks "5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"
```

输出目录：

```text
runs/<dataset>/<dataset>_vfin4_semantic_5seeds/summaries/
```

### 5.2 对比模型汇总

```bash
python comparison_models/summarize_dataset_results.py \
  --dataset "$DATASET" \
  --run-dir "$CMP_RUN"
```

输出目录：

```text
runs/<dataset>/<dataset>_vfin4_comparison_5seeds/summaries/full_metrics/
```

### 5.3 统一生成 CSV、Markdown 与三张分类折线图

如果 SemanticConvE 和对比模型都已完成：

```bash
python scripts/report_topk_comparison.py \
  --run-dir "SemanticConvE=${SEM_RUN}" \
  --run-dir "Comparison=${CMP_RUN}" \
  --output-dir "runs/${DATASET}/${DATASET}_vfin4_report_topk5"
```

如果只有对比模型（例如暂未完成 XES 的 SemanticConvE）：

```bash
python scripts/report_topk_comparison.py \
  --run-dir "Comparison=${CMP_RUN}" \
  --output-dir "runs/${DATASET}/${DATASET}_vfin4_report_topk5"
```

输出包括：

```text
runs/<dataset>/<dataset>_vfin4_report_topk5/
├── topk_comparison.csv
├── topk_comparison.md
├── topk_representation.png
├── topk_cognitive.png
└── topk_baselines.png
```

## 6. 精简压缩并下载结果

以下压缩包保留评估指标、表格、图和日志，但不保留模型 checkpoint、embedding、分数 pkl/json 与 `scores/` 目录。

```bash
DATASET="Eedi"
cd "runs/${DATASET}"

tar -czf "/root/autodl-tmp/${DATASET}_vfin4_topk5_results_slim.tar.gz" \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.npy' \
  --exclude='*.pkl' \
  --exclude='*/scores/*' \
  --exclude='__pycache__' \
  "${DATASET}_vfin4_semantic_5seeds" \
  "${DATASET}_vfin4_comparison_5seeds" \
  "${DATASET}_vfin4_report_topk5" 2>/dev/null || \
tar -czf "/root/autodl-tmp/${DATASET}_vfin4_topk5_results_slim.tar.gz" \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.npy' \
  --exclude='*.pkl' \
  --exclude='*/scores/*' \
  --exclude='__pycache__' \
  "${DATASET}_vfin4_comparison_5seeds" \
  "${DATASET}_vfin4_report_topk5"
```

压缩包路径：

```text
/root/autodl-tmp/<dataset>_vfin4_topk5_results_slim.tar.gz
```

在 AutoDL 文件管理器中进入 `/root/autodl-tmp/` 下载该 `.tar.gz` 文件。

> 若后续还要重新计算其他 Top-K，必须保留 AutoDL 上原始 `*_uid_ex_scores.pkl`；精简压缩包不含这些大文件。
