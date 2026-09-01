# V-Fin3 AutoDL Five-GPU ER Workflow

This guide covers only the ER stage: SemanticConvE training, testing, ablations, comparison models, and result summaries. Generate all front files locally before uploading data to AutoDL.

## 1. Instance Assignment

| GPU instance | Dataset argument |
| --- | --- |
| GPU 1 | `Eedi` |
| GPU 2 | `algebra2005` |
| GPU 3 | `assist2009-sub` |
| GPU 4 | `statics2011` |
| GPU 5 | `XES3G5M-sub-small` |

Use one GPU per dataset. An RTX 4090 24 GB instance with Python 3.10 and a PyTorch CUDA 12.1 image is sufficient. Run SemanticConvE and comparison models sequentially on the same GPU.

## 2. Upload Only the ER Graph

After local front-file generation and validation, upload the entire directory:

```text
Data_Fin/<dataset>/er_graph/
```

Its destination on AutoDL must be:

```text
/root/autodl-tmp/KG4ER-New/Data_Fin/<dataset>/er_graph/
```

Keep the entire directory, including:

```text
entities.dict
relations.dict
triples.txt
test_triples.txt
Q.txt
stu2know_mastery.json
stu2know_forget.json
stu2ex_forget.json
stu2ex_recommend.json
stu2ex_recommend_full_precision.json
semantic_kg_features/
```

Do not upload `raw/`, `front_features/`, or MIRT/LSTM/EKTM checkpoints. They are not used by ER training, testing, or result aggregation.

## 3. Clone and Install

Run these commands in the AutoDL JupyterLab terminal:

```bash
cd /root/autodl-tmp
git clone -b V-Fin7 --single-branch https://github.com/Tangfgjk/KG4ER-New.git
cd KG4ER-New

# Keep the CUDA-enabled PyTorch supplied by the image.
sed '/^torch$/d' requirements.txt > requirements_no_torch.txt
pip install -r requirements_no_torch.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The last command must display `True` and the GPU name. To update an existing clone:

```bash
cd /root/autodl-tmp/KG4ER-New
git fetch origin
git checkout V-Fin7
git pull --ff-only origin V-Fin7
```

## 4. Validate the Uploaded Data

For Eedi:

```bash
python codes-New-ConvE/validate_semantic_ready.py \
  --datasets Eedi \
  --data-root Data_Fin \
  --graph-subdir er_graph
```

The JSON output must report `"status": "passed"`. If it fails, upload the complete `er_graph/` directory again rather than skipping validation.

## 5. Keep Jobs Running After Disconnecting

Install tmux if necessary:

```bash
apt-get update && apt-get install -y tmux
```

Start a session, using Eedi as an example:

```bash
cd /root/autodl-tmp/KG4ER-New
tmux new -s kg4er-eedi
```

After starting the job, press `Ctrl+B`, release, then press `D` to detach. The work continues after the browser or SSH connection closes. Reattach with:

```bash
tmux attach -t kg4er-eedi
```

Do not shut down or release the AutoDL instance while the job is running.

## 6. One Command Per Dataset

The script `scripts/run_autodl_vfin_dataset.sh` performs:

1. validation of the uploaded ER graph;
2. seven formal 2CKG4ER variants: `2CKG4ER`, `2CKG4ER_id_only`, `2CKG4ER_relation_id`, `2CKG4ER_learner_id`, `2CKG4ER_exercise_id`, `2CKG4ER_no_mastery`, and `2CKG4ER_no_forgetting`;
3. ER training uses `triples.txt` only; `test_triples.txt` is reserved for evaluation;
4. one pilot random seed: `2024`;
5. only the two comparison models: `TransE` and `TransE-adv`;
6. result summaries, a combined top-K table, and Ada/NOV line charts for `N=10,20,...,100`.

All models train from `triples.txt` only. `test_triples.txt` contributes evaluation-time cognitive-state edges and contains no `rec` labels.

GPU 1:

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi
```

GPU 2:

```bash
bash scripts/run_autodl_vfin_dataset.sh algebra2005
```

GPU 3:

```bash
bash scripts/run_autodl_vfin_dataset.sh assist2009-sub
```

GPU 4:

```bash
bash scripts/run_autodl_vfin_dataset.sh statics2011
```

GPU 5:

```bash
bash scripts/run_autodl_vfin_dataset.sh XES3G5M-sub-small
```

Console output is also written to `logs/<dataset>_vfin_<timestamp>.log`.

## 7. Resume

If an ER run stops, resume only the unfinished ER tasks:

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi --resume
```

Replace `Eedi` with the instance dataset. This does not rebuild or overwrite the uploaded ER graph.

## 8. Manual Summaries

The script automatically generates summaries. To run them again for Eedi:

```bash
DS=Eedi
SEEDS=2024,2025,2026,2027,2028
ABLATIONS=feature_only,id_only,feature_only_relation_id,feature_only_learner_id,feature_only_exercise_id,feature_only_no_mastery,feature_only_no_forgetting

for PROTOCOL in include_test exclude_test; do
  python codes-New-ConvE/summarize_semantic_results.py \
    --dataset "$DS" \
    --run-id "${DS}_vfin3_semantic_${PROTOCOL}_1seed" \
    --runs-root runs \
    --seeds 2024 \
    --ablations "$ABLATIONS"
done

python comparison_models/summarize_dataset_results.py \
  --dataset "$DS" \
  --run-dir "runs/${DS}/${DS}_vfin3_comparison_1seed"
```

Main output files:

```text
runs/<dataset>/<dataset>_vfin3_semantic_include_test_1seed/summaries/paper_table.md
runs/<dataset>/<dataset>_vfin3_semantic_exclude_test_1seed/summaries/paper_table.md
runs/<dataset>/<dataset>_vfin3_comparison_1seed/summaries/
runs/<dataset>/<dataset>_vfin3_pilot_report/topk_comparison.md
runs/<dataset>/<dataset>_vfin3_pilot_report/topk_comparison.png
```

## 9. Download Results

```bash
DS=Eedi
tar -czf "/root/autodl-tmp/${DS}_vfin_results.tar.gz" "runs/${DS}"
```

Download `/root/autodl-tmp/Eedi_vfin_results.tar.gz` from the JupyterLab file browser.
