# Data_Fin Front Pipeline

`front_pipeline` is the current entrypoint for front-feature generation. It does
not read legacy `data/`, `v10_pipeline/`, or `v11_pipeline` code.

## Raw Cohort Rule

The formal workflow uses a fixed raw directory for each dataset:

```text
assist2009-sub -> raw_compact
XES3G5M-sub-small -> raw_compact
all other datasets -> raw
```

`run_front_pipeline.py` chooses this automatically. Single-stage scripts keep
`--raw-name auto` for debugging, but `common.py` rejects any value that does not
match the formal rule.

## Stages

| Stage | Script | Main output |
| --- | --- | --- |
| protocol | `prepare_front_protocol.py` | train/test protocol and MIRT inputs |
| MIRT | `train_q_mirt.py` | Q-constrained item parameters and learner theta |
| EKTM-MIRT | `train_ektm_mirt.py` | mastery and shared text encoder exports |
| forgetting | `generate_forgetting.py` | learner-KC and learner-exercise forgetting |
| semantic features | `build_semantic_features.py` | 2CKG4ER feature bundle metadata |
| ER graph | `build_er_graph.py` | `triples.txt`, `test_triples.txt`, dictionaries |
| validation | `validate_front_pipeline.py` | front/graph validation report |

## One-Command Run

```powershell
python front_pipeline\run_front_pipeline.py `
  --datasets Eedi,algebra2005,assist2009-sub,statics2011,XES3G5M-sub-small `
  --data-fin-root Data_Fin `
  --mirt-epochs 70 `
  --ektm-epochs 30 `
  --device cuda `
  --force
```

Outputs are written under:

```text
Data_Fin/<dataset>/front_features/
Data_Fin/<dataset>/er_graph/
```

2CKG4ER and comparison models train from `er_graph/triples.txt` only.
`test_triples.txt` is evaluation-only.

