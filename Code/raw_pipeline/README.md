# Canonical Raw Dataset Layer

`build_raw_datasets.py` creates the only input layer for future front-model
training and ER graph construction. It reads immutable legacy data and writes
to a separate `Data_Fin/<dataset>/raw` directory.

The generated raw layer contains:

- `interactions_all.csv`: all selected learner interactions, with canonical
  learner and exercise IDs plus a fixed train/test label.
- `student_split.csv` and `student_id_map.csv`: the cohort and its fixed split.
- `Q.txt`, exercise/concept metadata, and ID maps aligned with Q rows/columns.
- `evaluation_uid_kc_response.txt`: evaluation records remapped to canonical
  test learner IDs.
- `raw_manifest.json`: source paths, protocol, and count audit trail.

It intentionally excludes all learned or derived artifacts, including IRT
parameters, embeddings, cognitive-state JSON files, graph triples, and model
checkpoints.

Example:

```powershell
python raw_pipeline\build_raw_datasets.py `
  --source-root "C:\path\to\KG4ER\data" `
  --output-root "C:\Users\29694\Desktop\Data_Fin" `
  --datasets all `
  --force

python raw_pipeline\validate_raw_datasets.py `
  --output-root "C:\Users\29694\Desktop\Data_Fin" `
  --datasets all
```
