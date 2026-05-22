# NN-kNN Agent Guide

## Purpose

This file is a fast orientation guide for future AI agents working in this
repository. It is not a full paper summary. Its goal is to answer:

- what the NN-kNN model is in this codebase
- where the important code lives
- which workflow entry points matter for regression/table reporting
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
- For regression, an optional NN-CDH adapter can modify the retrieved solution
  after retrieval.

The key practical distinction in the current Table 1 work is:

- retrieval-only output before adaptation
- post-adaptation output after NN-CDH

## Core Model Files

- [model/nnknn_model.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nnknn_model.py)
  This is the main NN-kNN implementation.
- [model/nn_cdh.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nn_cdh.py)
  This contains the regression adaptation module used after retrieval.
- [model/regression_workflow.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/regression_workflow.py)
  This is the main orchestration layer for regression experiments and repeated
  benchmarks.

Important anchors:

- `GlocalFeatureWeight` at [model/nnknn_model.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nnknn_model.py:97)
- `MLPFeatureProjector` at [model/nnknn_model.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nnknn_model.py:185)
- `NN_KNN_Model` at [model/nnknn_model.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nnknn_model.py:517)
- `train_model(...)` at [model/nnknn_model.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nnknn_model.py:1126)
- `add_to_pair_list(...)` at [model/nn_cdh.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nn_cdh.py:43)
- `NNCDHAdapter` at [model/nn_cdh.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nn_cdh.py:112)

## Regression Workflow Files

The regression work has been refactored so repeated experiment logic lives in
`model/regression_workflow.py` instead of being trapped in the notebook.

Important anchors:

- `load_regression_dataset_state(...)` at [model/regression_workflow.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/regression_workflow.py:259)
- `train_nnknn_regression_state(...)` at [model/regression_workflow.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/regression_workflow.py:903)
- `evaluate_nnknn_pre_post_adaptation_state(...)` at [model/regression_workflow.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/regression_workflow.py:1452)
- `run_repeated_regression_model_benchmarks(...)` at [model/regression_workflow.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/regression_workflow.py:1875)

This file also contains the repeated baseline implementations used in the
regression notebook and Table 1 work:

- `kNN(X)`
- `Oracle kNN(y_true)`
- `MLKR+kNN`
- `MLP`

## Table 1 Reporting Files

Current Table 1 reruns are driven by:

- [tools/table1_nnknn_kfold.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/tools/table1_nnknn_kfold.py)

Important anchors:

- `build_table1_paper_base_cfg(...)` at [tools/table1_nnknn_kfold.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/tools/table1_nnknn_kfold.py:22)
- `build_table1_baseline_method_cfgs(...)` at [tools/table1_nnknn_kfold.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/tools/table1_nnknn_kfold.py:131)
- `run_table1_nnknn_kfold(...)` at [tools/table1_nnknn_kfold.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/tools/table1_nnknn_kfold.py:276)
- `run_table1_kfold(...)` at [tools/table1_nnknn_kfold.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/tools/table1_nnknn_kfold.py:412)

This helper currently:

- runs 5-fold CV for Table 1 style reporting
- combines baselines plus NN-kNN family rows
- reports mean/std using `rmse_raw_table`
- emits long-form output that can be pivoted into a paper-style table

NN-kNN row semantics in Table 1:

- `pure`: retrieval before adaptation, no locality
- `adaptation`: post-adaptation, no locality
- `locality`: retrieval before adaptation, locality enabled
- `locality + adaptation`: post-adaptation, locality enabled

## Dataset Files

Primary regression dataset loader:

- [datasets/reg_data.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/datasets/reg_data.py)

Important anchors:

- `Bike_Sharing()` at [datasets/reg_data.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/datasets/reg_data.py:219)
- `Wine()` at [datasets/reg_data.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/datasets/reg_data.py:236)
- `DATATYPES` at [datasets/reg_data.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/datasets/reg_data.py:303)
- `Reg_data(...)` at [datasets/reg_data.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/datasets/reg_data.py:323)

Important status note:

- `bike_sharing` and `wine` are integrated into `reg_data.py`
- `datasets/reg_data_yu.py` is now a compatibility wrapper, not the canonical
  loader

## Notebook vs Script Status

Primary notebook:

- [nnknn_sample_regression.ipynb](/c:/Users/yexia/Documents/GitHub/NN-kNN/nnknn_sample_regression.ipynb)

The notebook is still useful for interactive debugging and single-run
inspection, but serious repeated reporting now belongs in the workflow/helper
Python files.

Legacy or specialized files:

- [nnknn_reg_yu.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/nnknn_reg_yu.py)
  Older standalone script with dataset-specific experimentation history.
- `older versions/`
  Historical code snapshots.
- `Copy_nnknn_model.py`, `nnknn_model_1_4.py`
  Older variants, usually not the first place to edit.

## Current Practical Conventions

- For regression Table 1 reruns, prefer `run_table1_kfold(...)`.
- For general repeated regression benchmarks, prefer
  `run_repeated_regression_model_benchmarks(...)`.
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

## If You Are Picking Up Mid-Run

- Check [HANDOFF.md](/c:/Users/yexia/Documents/GitHub/NN-kNN/HANDOFF.md) first.
- Then check the newest `results/table1_kfold_*` folder.
- Tail:
  - `stdout.log`
  - `stderr.log`
- If `done.json` exists, the run finished and exports should already be there.

## Default Reading Order

If you need to understand the repo quickly, read in this order:

1. [HANDOFF.md](/c:/Users/yexia/Documents/GitHub/NN-kNN/HANDOFF.md)
2. [tools/table1_nnknn_kfold.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/tools/table1_nnknn_kfold.py)
3. [model/regression_workflow.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/regression_workflow.py)
4. [model/nnknn_model.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nnknn_model.py)
5. [model/nn_cdh.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/model/nn_cdh.py)
6. [datasets/reg_data.py](/c:/Users/yexia/Documents/GitHub/NN-kNN/datasets/reg_data.py)
