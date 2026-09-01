# V-Fin7 No-Sequence Protocol

## Purpose

The formal V-Fin7 pipeline removes the PKC/sequence relation because prior
experiments showed that it did not improve recommendation ACC. The front
pipeline and the ER graph must therefore be regenerated together; mixing a
new model with an old graph is not valid.

## Front Files Used

The formal pipeline reads only `Data_Fin/<dataset>/raw/` and regenerates:

- Q-constrained MIRT: item discrimination, item difficulty, and learner theta;
- EKTM-MIRT: learner knowledge mastery and shared exercise/relation text encoder;
- forgetting features: `stu2know_forget.json` and `stu2ex_forget.json`;
- semantic features and the final `er_graph/`.

`stu2know_seq.json` is no longer generated, copied into the ER graph, read by
SemanticConvE, or used in the recommendation distance.

## Graph Protocol

The graph keeps three relation families:

- `rec`;
- `mlkcXX`: learner mastery of a knowledge concept;
- `exfrXX`: learner forgetting level for an exercise.

Relation names use two-decimal strengths. There are `1 + 101 + 101 = 203`
relations. Recommendation candidates are ranked by the full-precision distance:

```text
sqrt((delta_1 - mastery_product)^2 + (delta_2 - exercise_forgetting)^2)
```

The persisted rounded JSON is inspection-only and never used for ranking.

## Formal SemanticConvE Models

```text
feature_only
id_only
feature_only_relation_id
feature_only_learner_id
feature_only_exercise_id
feature_only_no_mastery
feature_only_no_forgetting
```

`feature_only` is the primary model. The last two variants remove the named
edge family and rebuild recommendation edges using the remaining distance term.
There is deliberately no `no_seq` variant because no formal graph contains a
sequence edge.

## Regenerate One Dataset

```powershell
python front_pipeline\run_front_pipeline.py `
  --datasets Eedi `
  --data-fin-root Data_Fin `
  --mirt-epochs 70 `
  --mirt-batch-size 1024 `
  --ektm-epochs 30 `
  --ektm-batch-size 16 `
  --device cuda `
  --force
```

Then validate:

```powershell
python front_pipeline\validate_front_pipeline.py `
  --datasets Eedi `
  --data-fin-root Data_Fin `
  --require-graph
```

## ER Training Rule

SemanticConvE and comparison models train only from `triples.txt`.
`test_triples.txt` is evaluation-time cognitive state for held-out learners;
it contains no `rec` labels. Use the AutoDL runner for all formal models:

```bash
bash scripts/run_autodl_vfin_dataset.sh Eedi all --seeds 2024,2025,2026
```
