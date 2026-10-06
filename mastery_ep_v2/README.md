# Data_Fin_New and Ep evaluation

`Data_Fin_New` contains the three rebuilt datasets. `front_features/stu2know_mastery.json`
and `er_graph/stu2know_mastery.json` contain the calibrated concept estimates. The
semantic features, recommendation distances, and graph recommendation edges were
rebuilt from them. The frozen EKTM checkpoint supplies exercise representations,
response probabilities, and the GRU update; its original untrained `mastery_head`
is never read as a mastery estimate. Its parameter remains in the model class only
to load the old checkpoint with `strict=True`.

The separately trained `ConceptReadout` is stored under
`Data_Fin_New/_ep_readouts/<dataset>/`. It converts the full-history GRU state
into per-concept mastery, both before and after each simulated response. The
simulation answers correctly when EKTM response probability is at least the
configured threshold. The evaluation uses 0.4, 0.5, 0.6, and 0.7 on every model's
same recommended list. Ep at N is the sum of mastery changes over
all concepts divided by the sum of initial mastery deficits, after the first N
recommended exercises. The evaluator records `Ep_t0.4`, `Ep_t0.5`, `Ep_t0.6`,
and `Ep_t0.7` at N=5,10,...,50 alongside ACC and NOV in every run's
`eval/metrics.json` and `eval/metrics.csv`. These are sensitivity variants of
one simulated metric, so the threshold should not be selected solely by which
variant produces the largest Ep value.

Use `scripts/check_data_fin_new.py` to verify the rebuilt inputs. Use
`scripts/run_data_fin_new.ps1` to retrain the main model, ablations, and comparison
models on all three datasets and create tables. The script writes new outputs under
`runs_Data_Fin_New_main` and `runs_Data_Fin_New_comparison`, leaving prior runs
separate. The cross-model report has one panel per metric, N in rows, and models
in columns. Seeded methods show mean ± sample standard deviation.

`scripts/summarize_paper_tables.py` updates the LaTeX and Markdown comparison
and ablation tables after each completed seed. Missing models and seeds leave
blank cells; `aggregate.json` records per-model seed coverage and whether each
table is complete. Highlights in incomplete tables are provisional. Re-running
the summarizer after later seeds replaces the same tables with updated means
and standard deviations.

The Ep result is a model-estimated simulated gain. It does not use observed
answers to the newly recommended exercises.
