# NN-kNN Agent Guide

## Purpose

This file is a fast orientation guide for future AI agents working in this
repository. It is not a full paper summary. Its goal is to answer:

- what the NN-kNN model is in this codebase
- where the important code lives
- which workflow entry points matter for classification, regression, and RL
  reporting
- which files are legacy versus current

## What NN-kNN Is Here

At a high level, NN-kNN is a neural case-based model.

- It keeps the training examples as the case base.
- A feature extractor or raw tabular features define the representation space.
- Distances from a query to all stored cases are computed in that space.
- A glocal feature-weighting module reweights feature dimensions when computing
  those distances.
- Case activations are normalized, usually by `softmax` or `sparsemax`.
- The retrieved cases are aggregated into a prediction.
- For classification, normalized case activation is summed into class
  probability mass and optimized with NLL on that mass.
- For regression, an optional NN-CDH adapter can modify the retrieved solution
  after retrieval.

The config key for enabling case normalization is `normalize_over_cases`
because the selected normalizer may be `softmax` or `sparsemax`.

The key practical distinction in the current Table 1 work is:

- retrieval-only output before adaptation
- post-adaptation output after NN-CDH

## Core Model Files

- [model/nnknn_model.py](model/nnknn_model.py)
  This is the main NN-kNN implementation.
- [model/nn_cdh.py](model/nn_cdh.py)
  This contains the regression adaptation module used after retrieval.
- [model/regression_workflow.py](model/regression_workflow.py)
  This is the main orchestration layer for regression experiments and repeated
  benchmarks.
- [model/classification_workflow.py](model/classification_workflow.py)
  This is the maintained orchestration layer for classification experiments,
  image comparisons, and small-dataset benchmark checks.
- [model/rl_workflow.py](model/rl_workflow.py)
  This is the maintained DQN baseline workflow for CartPole and future
  RL-family extensions.
- [datasets/classification_data.py](datasets/classification_data.py)
  This contains portable classification loaders with train-only preprocessing.
- [datasets/rl_tasks.py](datasets/rl_tasks.py)
  This contains the RL task registry and task metadata.

Important anchors:

- `GlocalFeatureWeight` at [model/nnknn_model.py](model/nnknn_model.py#L97)
- `MLPFeatureProjector` at [model/nnknn_model.py](model/nnknn_model.py#L185)
- `NN_KNN_Model` at [model/nnknn_model.py](model/nnknn_model.py#L517)
- `train_model(...)` at [model/nnknn_model.py](model/nnknn_model.py#L1126)
- `add_to_pair_list(...)` at [model/nn_cdh.py](model/nn_cdh.py#L43)
- `NNCDHAdapter` at [model/nn_cdh.py](model/nn_cdh.py#L112)

## RL / DQN / NEC Workflow Files

The current RL path includes repo-native DQN and NEC baselines on
`CartPole-v1`. They are engineering baselines before adding NN-kNN-RL.
Do not use `Outdated NN-kNN Reinforcement Learning.ipynb` or
`OutdatedNewCartpole.ipynb` as implementation references.

Important entry points in `model/rl_workflow.py`:

- `RLTaskSpec`
- `DQNConfig`
- `make_dqn_config(...)`
- `train_dqn(...)`
- `evaluate_dqn(...)`
- `load_dqn_checkpoint(...)`
- `list_supported_rl_tasks(...)`

Important entry points in `model/nec_workflow.py`:

- `NECConfig`
- `make_nec_config(...)`
- `train_nec(...)`
- `evaluate_nec(...)`
- `load_nec_checkpoint(...)`

Task metadata lives in `datasets/rl_tasks.py`; currently supported:

- `cartpole` -> `CartPole-v1`

CLI and notebook surfaces:

- `tools/run_rl_dqn.py` is the source-of-truth CLI.
- `tools/run_rl_nec.py` is the source-of-truth NEC CLI.
- `dqn_cartpole_demo.ipynb` is a thin debugging notebook that calls the Python
  workflow functions.

Routine RL checks:

```bash
.venv/bin/python codex/smoke_test.py --mode rl
.venv/bin/python codex/smoke_test.py --mode nec
.venv/bin/python tools/run_rl_dqn.py cartpole --profile smoke --seed 0
.venv/bin/python tools/run_rl_dqn.py cartpole --profile fast --seed 0
.venv/bin/python tools/run_rl_dqn.py cartpole --eval-only --checkpoint <checkpoint.pt>
.venv/bin/python tools/run_rl_nec.py cartpole --profile smoke --seed 0
.venv/bin/python tools/run_rl_nec.py cartpole --profile debug --seed 0
.venv/bin/python tools/run_rl_nec.py cartpole --profile fast --seed 0
.venv/bin/python tools/run_rl_nec.py cartpole --eval-only --checkpoint <checkpoint.pt>
```

RL protocol for DQN, NEC, and NN-kNN-RL comparisons:

- Train through the fixed environment-step budget; do not early-stop by
  default.
- Evaluate periodically and save the best evaluated checkpoint.
- Keep the final/end-of-budget evaluation for diagnosis.
- Use `training_efficiency` in `summary.json` before interpreting results.
- Treat `best_model_step` and `first_success_step` as sample-efficiency proxies.
- If `budget_interpretation` is `unsolved_or_underfit`, or if the best model is
  the final model and the curve is still rising, increase the budget or tune
  before making paper-style comparisons.

Current CartPole DQN reference result:

- run folder: `results/rl/dqn_cartpole_20260615_145845_570666/`
- selected checkpoint step: `125000`
- selected eval mean return: `500.0` over 20 episodes
- last/end-of-budget eval mean return: `87.7`
- interpretation: the fixed budget exposed regression after the best model, so
  best-checkpoint selection was useful and the selected step is an efficiency
  signal.

Current CartPole NEC reference result:

- run folder: `results/rl/nec_cartpole_20260615_191908_236354/`
- selected checkpoint step: `90000`
- selected eval mean return: `371.45` over 20 episodes
- last/end-of-budget eval mean return: `199.60`
- interpretation: the 150k-step NEC `fast` profile improved over the 25k
  debug-sized run (`292.2` best mean return), but it remains below the `475.0`
  success threshold and below the DQN reference; treat it as unsolved/underfit
  before paper-style claims.

## Classification Workflow Files

Classification now uses the IJCAI-26 shared core rather than restoring the
older IJCAI-25 implementation. The retrieval pipeline stays common to both
tasks; only the final aggregation and loss are task-specific.

Important entry points in `model/classification_workflow.py`:

- `make_classification_cfg(...)`
- `load_classification_dataset_state(...)`
- `split_classification_state(...)`
- `train_nnknn_classification_state(...)`
- `evaluate_nnknn_classification_state(...)`
- `run_single_nnknn_classification_experiment(...)`
- `run_repeated_classification_model_benchmarks(...)`

Supported small datasets are `iris`, `zebra`, `zebra_special`, `wine`,
`breast_cancer`, `balance`, and `digits`. Supported image workflows are
`mnist`, `cifar10`, and `svhn`.
SST-5 and SST-2 remain deferred; do not add text-loader dependencies or
restructure the local `datasets` package as part of routine classification
work.

Small-data baselines are NN-kNN, kNN, and a four-hidden-layer MLP. Image
comparisons include a ConvNet, pixel kNN, pretrained-ConvNet-feature kNN,
trainable-CNN NN-kNN, and frozen-pretrained-CNN NN-kNN. The legacy `NN-kNNO`
case-weight path is intentionally unsupported because case weights were
removed from the maintained core.

Routine classification check:

```bash
.venv/bin/python codex/smoke_test.py --mode classification
.venv/bin/python tools/run_classification_benchmarks.py iris --mode kfold --runs 3 --epochs 20
```

The benchmark CLI creates a fresh timestamped folder under `results/` with
`summary.csv`, `runs.csv`, and `manifest.json`. The notebook keeps repeated
tabular and image benchmark sections opt-in so its default path stays quick.

For manual small-data classification debugging, use
[nnknn_sample_classification.ipynb](nnknn_sample_classification.ipynb). Run the
single-experiment sanity section first and change the dataset passed to
`run_single_nnknn_classification_experiment(...)`:

- use `"iris"` for the Iris sanity baseline
- use `"zebra"` for Zebra (a)
- use `"zebra_special"` for Zebra (b)

Inspect `accuracy`, `class_probabilities`, `most_activated_cases`,
`most_activated_class_ids`, `most_activated_activations`, and
`glocal_weightor.get_feature_weights_display()`. Misclassified validation
queries are more informative than always inspecting query zero. Zebra (a) is
the alternating vertical-band generator in `datasets/classification_data.py`;
debug whether retrieved cases cross those boundaries before expanding to full
benchmark comparisons. Use the opt-in repeated benchmark section only when a
single-run hypothesis is ready for aggregate validation.

## Regression Workflow Files

The regression work has been refactored so repeated experiment logic lives in
`model/regression_workflow.py` instead of being trapped in the notebook.

Important anchors:

- `load_regression_dataset_state(...)` at [model/regression_workflow.py](model/regression_workflow.py#L259)
- `train_nnknn_regression_state(...)` at [model/regression_workflow.py](model/regression_workflow.py#L903)
- `evaluate_nnknn_pre_post_adaptation_state(...)` at [model/regression_workflow.py](model/regression_workflow.py#L1452)
- `run_repeated_regression_model_benchmarks(...)` at [model/regression_workflow.py](model/regression_workflow.py#L1875)

NN-kNN-only workflow entry points:

- `train_nnknn_regression_state(...)`
- `evaluate_nnknn_regression_state(...)`
- `evaluate_nnknn_pre_post_adaptation_state(...)`
- `run_nnknn_regression_workflow(...)`
- `run_single_nnknn_regression_experiment(...)`
- `run_repeated_nnknn_regression_experiments(...)`

All-model benchmark workflow entry points:

- `run_regression_benchmark_methods_on_state(...)`
- `run_repeated_regression_model_benchmarks(...)`

This file also contains the repeated baseline implementations used in the
regression notebook and Table 1 work:

- `kNN(X)`
- `Oracle kNN(y_true)`
- `MLKR+kNN`
- `MLP`

Backward-compatible aliases are still present in `model/regression_workflow.py`,
so older notebook code should continue to work.

## Logic Check Against Older Git History

Compared with git commit `ff3d58416bbf23a56e47d79e00614e3bec6cd052`, the
training, evaluation, and testing logic remains the same for:

- NN-kNN
- KNN baseline
- Oracle KNN baseline
- MLKR + KNN baseline
- MLP baseline

The main real behavior change is a bugfix in `model/nnknn_model.py`:
`best_metric` now correctly stores the validation metric associated with the
best-loss checkpoint. Checkpoint selection logic itself is still based on
validation loss, same as before.

The repeated benchmark runner is new orchestration:

- same split per method within a run
- explicit reseeding for reproducibility
- improves consistency and fairness without changing individual model formulas

## Table 1 Reporting Files

Current Table 1 reruns are driven by:

- [tools/table1_nnknn_kfold.py](tools/table1_nnknn_kfold.py)

Important anchors:

- `build_table1_paper_base_cfg(...)` at [tools/table1_nnknn_kfold.py](tools/table1_nnknn_kfold.py#L42)
- `build_table1_baseline_method_cfgs(...)` at [tools/table1_nnknn_kfold.py](tools/table1_nnknn_kfold.py#L193)
- `run_table1_nnknn_kfold(...)` at [tools/table1_nnknn_kfold.py](tools/table1_nnknn_kfold.py#L479)
- `run_table1_kfold(...)` at [tools/table1_nnknn_kfold.py](tools/table1_nnknn_kfold.py#L713)

This helper currently:

- runs 5-fold CV for Table 1 style reporting
- combines baselines plus NN-kNN family rows
- reports mean/std using `rmse_raw_table`
- emits long-form output that can be pivoted into a paper-style table

Current Table 1 protocol:

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
  across:
  - `softmax`
  - `softmax + locality`
  - `sparsemax`
  - `sparsemax + locality`

NN-kNN row semantics in Table 1:

- `pure`: retrieval before adaptation, no locality
- `adaptation`: post-adaptation, no locality
- `locality`: retrieval before adaptation, locality enabled
- `locality + adaptation`: post-adaptation, locality enabled

Table 1 workflow entry points:

- `tools/table1_nnknn_kfold.py`
- `run_table1_nnknn_kfold(...)`
- `run_table1_kfold(...)`

Ready-to-run example:

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

Output objects:

- `table_summary_df`: one row per dataset/table entry; includes mean/std
  summary fields, `rmse_raw_table`, baseline rows, and all NN-kNN
  variant/result rows
- `table_runs_df`: one row per run/fold; useful for significance testing or
  debugging
- `table_artifacts`: nested as `{"baselines": ..., "nnknn": ...}` with
  detailed per-run objects

Recommended export files after a full run:

- `summary_long.csv`
- `runs_long.csv`
- `table1_like.csv`
- `transposed.csv`
- `done.json`

Restart pattern for fresh Table 1 runs:

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
summary_df.pivot(
    index="dataset",
    columns="entry_label",
    values="rmse_raw_table",
).reset_index().to_csv(outdir / "table1_like.csv", index=False)
```

Do not overwrite an existing run folder. Create a fresh timestamped folder
under `results/`, redirect stdout/stderr into that folder, and keep failed or
older folders for postmortem comparison.

## Dataset Files

Primary regression dataset loader:

- [datasets/reg_data.py](datasets/reg_data.py)

Important anchors:

- `Bike_Sharing()` at [datasets/reg_data.py](datasets/reg_data.py#L219)
- `Wine()` at [datasets/reg_data.py](datasets/reg_data.py#L236)
- `DATATYPES` at [datasets/reg_data.py](datasets/reg_data.py#L303)
- `Reg_data(...)` at [datasets/reg_data.py](datasets/reg_data.py#L323)

Important status note:

- `bike_sharing` and `wine` are integrated into `reg_data.py`
- `datasets/reg_data_yu.py` is now a compatibility wrapper, not the canonical
  loader

## Notebook vs Script Status

Primary notebooks:

- [nnknn_sample_classification.ipynb](nnknn_sample_classification.ipynb)
  for classification single-run inspection and opt-in benchmark checks.
- [nnknn_sample_regression.ipynb](nnknn_sample_regression.ipynb)
  for regression single-run inspection and workflow calls.
- [dqn_cartpole_demo.ipynb](dqn_cartpole_demo.ipynb)
  for DQN CartPole inspection while keeping implementation logic in
  `model/rl_workflow.py`.

The notebooks are still useful for interactive debugging and single-run
inspection, but serious repeated reporting now belongs in the workflow/helper
Python files.

Legacy or specialized files:

- [nnknn_reg_yu.py](nnknn_reg_yu.py)
  Older standalone script with dataset-specific experimentation history.
- `older versions/`
  Historical code snapshots.
- `Copy_nnknn_model.py`, `nnknn_model_1_4.py`
  Older variants, usually not the first place to edit.
- `Outdated NN-kNN Reinforcement Learning.ipynb` and
  `OutdatedNewCartpole.ipynb`
  Archival RL notebooks. Do not use these as current implementation
  references.

## Current Practical Conventions

- For regression Table 1 reruns, prefer `run_table1_kfold(...)`.
- For general repeated regression benchmarks, prefer
  `run_repeated_regression_model_benchmarks(...)`.
- For single interactive notebook work:
  1. Run `Basic Setup`
  2. Choose/load dataset
  3. Standardize if needed
  4. Configure NN-kNN
  5. Train
  6. Evaluate/visualize
  7. Run explanation blocks on the trained model if desired
- For notebook batch table work, use the batch runner cell in
  `nnknn_sample_regression.ipynb`; it calls
  `run_repeated_regression_model_benchmarks(...)`.
- For paper-style Table 1 reruns with 5-fold CV and std reporting, use
  `run_table1_kfold(...)` from `tools/table1_nnknn_kfold.py`.
- For RL/DQN runs, use `tools/run_rl_dqn.py` and inspect `summary.json`,
  `eval_metrics.csv`, and `training_efficiency` before comparing methods.
- Use `datasets/reg_data.py` as the source of truth for tabular regression
  datasets.
- Treat `HANDOFF.md` as the current experiment-status document.

## Machine-Specific Caveats

- On this machine, deterministic CUDA settings conflict with some baseline
  operations.
- The current Table 1 helper therefore pins `kNN(X)` and `MLP` baselines to
  CPU for reproducible repeated runs.
- Long experiments write progress to `stdout.log` and errors to `stderr.log`
  under timestamped folders in `results/`.
- `MLKR` depends on `metric-learn` and `scikit-learn`.
- RL/DQN depends on `gymnasium[classic_control]`.
- RL outputs are timestamped under `results/rl/`.
- `bike_sharing` depends on `ucimlrepo`; verify the active environment before
  launching a full Table 1 run that includes it.
- Table 1 baseline defaults are encoded in `tools/table1_nnknn_kfold.py` to
  match the notebook cell:
  - `kNN(X)`: `k=5`
  - `MLKR+kNN`: `k=25`, `max_iter=1000`
  - `MLP`: hidden `(64, 32)`, dropout `0.10`, `lr=1e-3`
- Progress logging uses:
  - `[table1] ...` messages for the Table 1 orchestrator
  - `[table1][nnknn] ...` messages for dataset/fold/family NN-kNN progress
  - `[benchmark] ...` messages for dataset/fold/method baseline progress

## If You Are Picking Up Mid-Run

- Check [HANDOFF.md](HANDOFF.md) first.
- Then check the newest `results/table1_kfold_*` folder.
- For RL work, also check the newest `results/rl/dqn_*` folder and read
  `summary.json` before trusting a checkpoint.
- Tail:
  - `stdout.log`
  - `stderr.log`
- If `done.json` exists, the run finished and exports should already be there.

## Default Reading Order

If you need to understand the repo quickly, read in this order:

1. [HANDOFF.md](HANDOFF.md)
2. [tools/table1_nnknn_kfold.py](tools/table1_nnknn_kfold.py)
3. [model/regression_workflow.py](model/regression_workflow.py)
4. [model/nnknn_model.py](model/nnknn_model.py)
5. [model/nn_cdh.py](model/nn_cdh.py)
6. [datasets/reg_data.py](datasets/reg_data.py)

For RL work, read these immediately after `HANDOFF.md`:

1. [model/rl_workflow.py](model/rl_workflow.py)
2. [datasets/rl_tasks.py](datasets/rl_tasks.py)
3. [tools/run_rl_dqn.py](tools/run_rl_dqn.py)
