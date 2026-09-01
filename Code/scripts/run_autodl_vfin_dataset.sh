#!/usr/bin/env bash
# Run V-Fin8 no-sequence experiments for one dataset on one AutoDL GPU.
#
# Examples:
#   bash scripts/run_autodl_vfin_dataset.sh Eedi all
#   bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small semantic
#   bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small comparison --resume
#   bash scripts/run_autodl_vfin_dataset.sh Eedi all --seeds 2024,2025,2026

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_autodl_vfin_dataset.sh <dataset> [all|semantic|comparison] [--seeds 2024,2025,2026] [--resume]"
  exit 2
fi

DATASET="$1"
shift
MODE="all"
RESUME=0
SEEDS="2024,2025,2026"
TOP_KS="5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"

if [[ $# -gt 0 && "$1" != "--resume" && "$1" != "--seeds" ]]; then
  MODE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=1 ;;
    --seeds)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--seeds requires a comma-separated value"
        exit 2
      fi
      SEEDS="$1"
      ;;
    *) echo "Unknown argument: $1"; exit 2 ;;
  esac
  shift
done

case "$DATASET" in
  Eedi|algebra2005|assist2009-sub|statics2011|XES3G5M-sub-small) ;;
  *) echo "Unsupported dataset: $DATASET"; exit 2 ;;
esac

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "No seeds were supplied"
  exit 2
fi
SEED_COUNT=${#SEED_LIST[@]}
case "$MODE" in
  all|semantic|comparison) ;;
  *) echo "Mode must be one of: all, semantic, comparison"; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WANDB_MODE="${WANDB_MODE:-disabled}"
ABLATIONS="id_only,2CKG4ER,2CKG4ER_relation_id,2CKG4ER_learner_id,2CKG4ER_exercise_id,2CKG4ER_no_mastery,2CKG4ER_no_forgetting"
COMPARISON_MODELS="TransE,TransE-adv,RotatE,DistMult,ComplEx,EB-CF,SB-CF,CBF"
SEMANTIC_RUN_ID="${DATASET}_vfin8_2ckg4er_${SEED_COUNT}seeds"
COMPARISON_RUN_ID="${DATASET}_vfin8_comparison_${SEED_COUNT}seeds"
REPORT_DIR="runs/${DATASET}/${DATASET}_vfin8_report_${SEED_COUNT}seeds"

mkdir -p logs
LOG_FILE="logs/${DATASET}_vfin8_${MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== V-Fin8 no-sequence workflow ====="
echo "dataset=$DATASET mode=$MODE"
echo "seeds=$SEEDS top_ks=$TOP_KS"
echo "All training uses triples.txt only; test_triples.txt is evaluation-only."
echo "2CKG4ER ablations=$ABLATIONS"
echo "Comparison models=$COMPARISON_MODELS"
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "===== Validate uploaded ER graph ====="
python codes-New-ConvE/validate_semantic_ready.py \
  --datasets "$DATASET" \
  --data-root Data_Fin \
  --graph-subdir er_graph

if [[ "$MODE" == "all" || "$MODE" == "semantic" ]]; then
  SEMANTIC_RESUME=()
  if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${SEMANTIC_RUN_ID}" ]]; then
    SEMANTIC_RESUME=(--resume)
  fi
  echo "===== 2CKG4ER: primary model plus six ablations ====="
  python codes-New-ConvE/run_semantic_experiments.py \
    --dataset "$DATASET" \
    --data-root Data_Fin \
    --graph-subdir er_graph \
    --run-id "$SEMANTIC_RUN_ID" \
    --seeds "$SEEDS" \
    --ablations "$ABLATIONS" \
    --epochs 25 \
    --bs 1024 \
    --learning-rate 0.001 \
    --negative-ratio 5 \
    --top-ks "$TOP_KS" \
    --cuda auto \
    "${SEMANTIC_RESUME[@]}"

  python codes-New-ConvE/summarize_semantic_results.py \
    --dataset "$DATASET" \
    --run-id "$SEMANTIC_RUN_ID" \
    --runs-root runs \
    --seeds "$SEEDS" \
    --ablations "$ABLATIONS" \
    --top-ks "$TOP_KS"

  echo "===== Export local explanation evidence for the primary 2CKG4ER model ====="
  python scripts/export_primary_model_evidence.py \
    --data-dir "Data_Fin/${DATASET}/er_graph" \
    --semantic-run-dir "runs/${DATASET}/${SEMANTIC_RUN_ID}" \
    --seeds "$SEEDS" \
    --selection-top-k 20 \
    --evidence-top-k 20
fi

if [[ "$MODE" == "all" || "$MODE" == "comparison" ]]; then
  COMPARISON_RESUME=()
  if [[ "$RESUME" -eq 1 && -d "runs/${DATASET}/${COMPARISON_RUN_ID}" ]]; then
    COMPARISON_RESUME=(--resume)
  fi
  echo "===== Comparison models: explicit ID-only baselines, no KCP-ER ====="
  python comparison_models/run_v9_comparison_experiments.py \
    --dataset "$DATASET" \
    --data-root Data_Fin \
    --graph-subdir er_graph \
    --run-id "$COMPARISON_RUN_ID" \
    --seeds "$SEEDS" \
    --models "$COMPARISON_MODELS" \
    --top-ks "$TOP_KS" \
    --cuda auto \
    "${COMPARISON_RESUME[@]}"

  python comparison_models/summarize_dataset_results.py \
    --dataset "$DATASET" \
    --run-dir "runs/${DATASET}/${COMPARISON_RUN_ID}"
fi

if [[ -d "runs/${DATASET}/${SEMANTIC_RUN_ID}" && -d "runs/${DATASET}/${COMPARISON_RUN_ID}" ]]; then
  echo "===== Write split top-K tables and readable figures ====="
  python scripts/report_topk_comparison.py \
    --run-dir "2CKG4ER=runs/${DATASET}/${SEMANTIC_RUN_ID}" \
    --run-dir "Comparison=runs/${DATASET}/${COMPARISON_RUN_ID}" \
    --output-dir "$REPORT_DIR" \
    --top-ks "$TOP_KS"
else
  echo "Combined report deferred: run the other mode on this machine or merge its results first."
fi

echo "===== Completed: $DATASET / $MODE ====="
echo "Semantic summary: runs/${DATASET}/${SEMANTIC_RUN_ID}/summaries"
echo "Comparison summary: runs/${DATASET}/${COMPARISON_RUN_ID}/summaries"
echo "Report: ${REPORT_DIR}"
