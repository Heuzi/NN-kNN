# Handoff

## Current Status

The regression workflow has been refactored so the repeated logic from
`nnknn_sample_regression.ipynb` now lives in `model/regression_workflow.py`.
The notebook still supports the original interactive flow for training one
model and then using the explanation/sanity-check cells afterward.

## Main Entry Points

NN-kNN-only workflow:

- `train_nnknn_regression_state(...)`
- `evaluate_nnknn_regression_state(...)`
- `evaluate_nnknn_pre_post_adaptation_state(...)`
- `run_nnknn_regression_workflow(...)`
- `run_single_nnknn_regression_experiment(...)`
- `run_repeated_nnknn_regression_experiments(...)`

All-model benchmark workflow:

- `run_regression_benchmark_methods_on_state(...)`
- `run_repeated_regression_model_benchmarks(...)`

Supported benchmark methods:

- `nnknn`
- `knn_x`
- `oracle_knn_y`
- `mlkr_knn`
- `mlp`

Backward-compatible aliases are still present in
`model/regression_workflow.py`, so older notebook code should continue to work.

## Logic Check Against Older Git History

Compared with git commit:

- `ff3d58416bbf23a56e47d79e00614e3bec6cd052`

Summary:

- The training/evaluation/testing logic remains the same for:
  - NN-kNN
  - KNN baseline
  - Oracle KNN baseline
  - MLKR + KNN baseline
  - MLP baseline
- The main real behavior change is a bugfix in `model/nnknn_model.py`:
  - `best_metric` now correctly stores the validation metric associated with
    the best-loss checkpoint.
  - Checkpoint selection logic itself is still based on validation loss, same
    as before.
- The repeated benchmark runner is new orchestration:
  - same split per method within a run
  - explicit reseeding for reproducibility
  - this improves consistency/fairness, but is not a change to the individual
    model formulas

## Notebook Usage

Single interactive workflow:

1. Run `Basic Setup`
2. Choose/load dataset
3. Standardize if needed
4. Configure NN-kNN
5. Train
6. Evaluate / visualize
7. Run explanation blocks on the trained model if desired

Batch table workflow:

- Use the batch runner cell in `nnknn_sample_regression.ipynb`
- That cell now calls `run_repeated_regression_model_benchmarks(...)`

## Suggested Next Step

Run 10 rounds per dataset per model for the table results.

If you want "10 independent reruns" of the current train/val/test workflow,
use:

- `mode="holdout"`
- `num_runs=10`

Use `mode="kfold"` only if you specifically want 10-fold cross-validation,
which is a different protocol.

## Ready-to-Run Example

```python
table_dataset_names = ["car"]  # or a list of datasets
table_methods = ["nnknn", "knn_x", "oracle_knn_y", "mlkr_knn", "mlp"]

table_summary_df, table_runs_df, table_artifacts = run_repeated_regression_model_benchmarks(
    dataset_names=table_dataset_names,
    nnknn_cfg=cfg,
    feature_extractor=feature_extractor,
    methods=table_methods,
    method_cfgs=table_method_cfgs,
    num_runs=10,
    mode="holdout",
    base_seed=42,
    standardize="auto",
    dataset_kwargs_map=table_dataset_kwargs,
)

display(table_summary_df)
display(table_runs_df)
display(table_summary_df[["dataset", "method", "mode", "rmse_raw_table"]])
```

## Outputs

- `table_summary_df`
  - one row per dataset/method
  - includes mean/std summary fields
  - includes `rmse_raw_table` for paper-ready formatting
- `table_runs_df`
  - one row per run/fold
  - useful for significance testing or debugging
- `table_artifacts`
  - detailed per-run objects if you need to inspect a specific trial later

## Notes / Caveats

- `MLKR` depends on `metric-learn` and `scikit-learn`
- There may still be temporary runtime artifacts from smoke tests, such as:
  - `tmpgm85naf7/`
  - temporary checkpoint files under `checkpoints/`
- Root checkpoint files like `nnknn_regression_best.pth` and
  `nnknn_regression_best_retr.pth` may change when experiments are run

## Recommended Follow-Up

After the 10-run benchmark finishes:

1. Save/export `table_summary_df`
2. Use `rmse_raw_table` for the paper tables
3. Optionally use `table_runs_df` for statistical testing between methods
4. If the codebase feels too large again, consider splitting
   `model/regression_workflow.py` into:
   - dataset utilities
   - NN-kNN workflow
   - baseline benchmarks
   - repeated experiment runners
