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

Table 1 reporting workflow:

- `tools/table1_nnknn_kfold.py`
- `run_table1_nnknn_kfold(...)`
- `run_table1_kfold(...)`

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
- For the paper-style Table 1 rerun with 5-fold CV and std reporting, use
  `run_table1_kfold(...)` from `tools/table1_nnknn_kfold.py`

## Suggested Next Step

Use the dedicated Table 1 k-fold runner for paper reporting.

Current Table 1 protocol in this repo:

- `num_folds=5`
- datasets:
  - `califonia_housing`
  - `diabets`
  - `abalone`
  - `body_fat`
  - `airfoil`
  - `car`
  - `student_performance`
  - `yacht`
  - `energy_efficiency`
  - `bike_sharing`
  - `wine`
- exclude `UTKFace` for now
- baselines:
  - `kNN(X)`
  - `MLKR+kNN`
  - `MLP`
- NN-kNN rows per dataset:
  - `pure`
  - `adaptation`
  - `locality`
  - `locality + adaptation`
  across
  - `softmax`
  - `softmax + locality`
  - `sparsemax`
  - `sparsemax + locality`

Meaning of the NN-kNN result labels:

- `pure`: retrieval before adaptation with locality disabled
- `adaptation`: post-adaptation result with locality disabled
- `locality`: retrieval before adaptation with locality enabled
- `locality + adaptation`: post-adaptation result with locality enabled

## Ready-to-Run Example

```python
from tools.table1_nnknn_kfold import run_table1_kfold

table_dataset_names = [
    "califonia_housing",
    "diabets",
    "abalone",
    "body_fat",
    "airfoil",
    "car",
    "student_performance",
    "yacht",
    "energy_efficiency",
    "bike_sharing",
    "wine",
]

table_summary_df, table_runs_df, table_artifacts = run_table1_kfold(
    dataset_names=table_dataset_names,
    num_folds=5,
    base_seed=42,
)

display(table_summary_df)
display(table_runs_df)
display(table_summary_df[["dataset", "entry_label", "rmse_raw_table"]])
```

## Outputs

- `table_summary_df`
  - one row per dataset/table entry
  - includes mean/std summary fields
  - includes `rmse_raw_table` for paper-ready formatting
  - includes baseline rows and all NN-kNN variant/result rows
- `table_runs_df`
  - one row per run/fold
  - useful for significance testing or debugging
- `table_artifacts`
  - nested as `{"baselines": ..., "nnknn": ...}`
  - detailed per-run objects if you need to inspect a specific trial later

Recommended export files after a full run:

- `summary_long.csv`
- `runs_long.csv`
- `table1_like.csv`
- `done.json`

These are produced by the current long-running launch under:

- `results/table1_kfold_20260418_213251/`

Progress/error logs for that run:

- `results/table1_kfold_20260418_213251/stdout.log`
- `results/table1_kfold_20260418_213251/stderr.log`

## Resume / Restart

How to check whether the current Table 1 sweep is still running:

- In PowerShell:
  - `Get-Process python`
- Tail the logs:
  - `Get-Content .\results\table1_kfold_20260418_213251\stdout.log -Wait`
  - `Get-Content .\results\table1_kfold_20260418_213251\stderr.log -Wait`

How to know the run finished:

- `done.json` appears in the run folder
- and the export files exist:
  - `summary_long.csv`
  - `runs_long.csv`
  - `table1_like.csv`

If the current run fails or needs to be restarted:

1. Do not overwrite the existing run folder; create a fresh timestamped folder
   under `results/`
2. Re-run the same `run_table1_kfold(...)` call from a new launcher script or
   notebook cell
3. Redirect stdout/stderr to log files in that new folder
4. Keep the old failed folder for postmortem debugging

Recommended restart pattern:

```python
from pathlib import Path
from tools.table1_nnknn_kfold import run_table1_kfold

outdir = Path("results/table1_kfold_YYYYMMDD_HHMMSS")
outdir.mkdir(parents=True, exist_ok=True)

summary_df, runs_df, artifacts = run_table1_kfold(
    dataset_names=[
        "califonia_housing",
        "diabets",
        "abalone",
        "body_fat",
        "airfoil",
        "car",
        "student_performance",
        "yacht",
        "energy_efficiency",
        "bike_sharing",
        "wine",
    ],
    num_folds=5,
    base_seed=42,
)

summary_df.to_csv(outdir / "summary_long.csv", index=False)
runs_df.to_csv(outdir / "runs_long.csv", index=False)
summary_df.pivot(index="dataset", columns="entry_label", values="rmse_raw_table").reset_index().to_csv(
    outdir / "table1_like.csv",
    index=False,
)
```

Important restart note:

- Future runs will have clearer high-level progress messages in `stdout.log`
  because progress logging was added after the current long-running sweep
  started
- The current sweep will continue using the older logging format until it is
  restarted

## Notes / Caveats

- `MLKR` depends on `metric-learn` and `scikit-learn`
- `bike_sharing` and `wine` are now loaded from `datasets/reg_data.py`
- `datasets/reg_data_yu.py` is now only a compatibility wrapper
- Table 1 baseline defaults are encoded in `tools/table1_nnknn_kfold.py`
  to match the notebook cell:
  - `kNN(X)`: `k=5`
  - `MLKR+kNN`: `k=25`, `max_iter=1000`
  - `MLP`: hidden `(64, 32)`, dropout `0.10`, `lr=1e-3`
- For deterministic runs on this machine, the Table 1 helper pins
  `kNN(X)` and `MLP` baselines to CPU to avoid CUDA deterministic errors
- Progress logging was added for future runs:
  - `[table1] ...` messages for the Table 1 orchestrator
  - `[table1][nnknn] ...` messages for dataset/fold/family NN-kNN progress
  - `[benchmark] ...` messages for dataset/fold/method baseline progress
- There may still be temporary runtime artifacts from smoke tests, such as:
  - `tmpgm85naf7/`
  - temporary checkpoint files under `checkpoints/`
- Root checkpoint files like `nnknn_regression_best.pth` and
  `nnknn_regression_best_retr.pth` may change when experiments are run

## Recommended Follow-Up

After the 5-fold Table 1 benchmark finishes:

1. Save/export `table_summary_df`
2. Use `rmse_raw_table` for the paper tables
3. If needed, pivot by `entry_label` to produce the final wide table layout
4. Optionally use `table_runs_df` for statistical testing between methods
5. If the codebase feels too large again, consider splitting
   `model/regression_workflow.py` into:
   - dataset utilities
   - NN-kNN workflow
   - baseline benchmarks
   - repeated experiment runners
