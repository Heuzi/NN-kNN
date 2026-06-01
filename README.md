# NN-kNN

NN-kNN is a neural case-based model for classification and regression. The
IJCAI-26 codebase is the maintained implementation: it learns retrieval over
cases with feature weighting and normalized case activation, then aggregates
retrieved labels for the target task.

- IJCAI-25 classification paper: [paper link](https://www.ijcai.org/proceedings/2025/763)
- IJCAI-26 regression paper: publication link will be added after release.

## Maintained Workflows

- Regression orchestration: `model/regression_workflow.py`
- Classification orchestration: `model/classification_workflow.py`
- Classification loaders: `datasets/classification_data.py`
- Shared NN-kNN core: `model/nnknn_model.py`

Classification uses the current model rather than the old classification
implementation. Normalized case activation is aggregated into class
probability mass and optimized with negative log likelihood. Supported
classification datasets include the IJCAI-25 small tabular set
(`iris`, `zebra`, `zebra_special`, `wine`, `breast_cancer`, `balance`,
`digits`) and image tasks (`mnist`, `cifar10`, `svhn`).
Image runs use an inner validation slice of the official training set for
checkpoint selection and report accuracy on the untouched official test set.
SST-5 and SST-2 remain deferred; text-loader dependencies are intentionally
outside the current classification workflow.

## Notebooks

- `nnknn_sample_classification.ipynb`: maintained classification workflow,
  explanations, representative baselines, and optional image subset runs.
- `nnknn_sample_regression.ipynb`: maintained regression workflow.

## Quick Checks

```bash
bash codex/setup.sh
.venv/bin/python codex/smoke_test.py --mode imports
.venv/bin/python codex/smoke_test.py --mode train
.venv/bin/python codex/smoke_test.py --mode classification
```

For a small classification benchmark run:

```bash
.venv/bin/python tools/run_classification_benchmarks.py iris --mode kfold --runs 3 --epochs 20
```

Each CLI run creates a fresh timestamped folder under `results/` containing
`summary.csv`, `runs.csv`, and `manifest.json`.

Use representative small-data or subset image checks during development;
full multi-run paper benchmarks are intentionally expensive.
