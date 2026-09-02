# Branch History

This file records what each branch is for, what changed in it, and how it should be used in later experiments. Keep it updated whenever a new experimental branch is created or completed.

## Branch Naming Rule

Use:

```text
v{number}-{stage-or-topic}-{short-purpose}
```

Examples:

```text
v1-fairness-foundation
v2-stage4-fairloss-scale
v3-topk-aware-fairloss
v4-multidataset-validation
v5-paper-table-export
```

Avoid names such as `V2` or `V3` alone, because they do not explain what changed.

## Status Values

| Status | Meaning |
|---|---|
| `planning` | Designed but not implemented |
| `running` | Code or experiments are in progress |
| `completed` | Code and core validation are complete |
| `abandoned` | Stopped because the idea was not useful or was replaced |

## Branch Records

| Branch | Previous Name | Base | Created | Status | Main Purpose | Latest Commit |
|---|---|---|---|---|---|---|
| `v1-fairness-foundation` | `V1` | initial import | 2026-09-02 | completed | Fairness-aware recommendation foundation: code cleanup, fairness metrics, post-processing rerank, model-in fairness regularization, fair KG graph, experiment scripts, progress documents. `Code/runs/`, `Code/tmp/`, and `Code/Data_Fin/` are excluded from later uploads. | `d543904e8bcaf83ffec377965785fa89c3423a36` |

## v1-fairness-foundation

### Scope

This branch is the current stable foundation for later fairness-aware recommendation improvements.

It contains:

- The cleaned and renamed main model flow using `2CKG4ER` as the primary model name.
- Fairness evaluation metrics for item exposure, KC exposure, coverage, and long-tail exposure.
- Post-processing fairness reranking methods, including soft rerank and quota-ratio hybrid rerank.
- Stage 1 experiment scripts and table export utilities.
- Model-in fairness regularization modules for item/KC fairness.
- Fairness context construction and fair KG graph preparation.
- Stage 2 alpha/gamma sensitivity experiment scripts.
- Stage 4 exploratory support for `fairness_loss_scale` and `candidate_mode`.
- Global explanation documents and progress reports.

### Excluded From Git

The following local directories are intentionally not tracked in later versions:

- `Code/runs/`: experiment outputs, checkpoints, scores, summaries.
- `Code/tmp/`: temporary and historical intermediate files.
- `Code/Data_Fin/`: local datasets and generated front files.

### Notes

- The first upload originally used branch name `V1`.
- The semantic branch name should be `v1-fairness-foundation`.
- Later branches should be created from this branch unless a newer branch is explicitly promoted as stable.

## Template For Future Branches

Copy this section when creating a new branch:

```markdown
## vX-topic-purpose

### Metadata

- Previous branch:
- Base branch:
- Created:
- Status:
- Latest commit:

### Goal

Describe the research or engineering question this branch answers.

### Code Changes

- 

### Experiments

- Dataset:
- Seeds:
- Commands:
- Result paths:

### Result Summary

- 

### Conclusion

- Keep / revise / abandon:
- Reason:
```
