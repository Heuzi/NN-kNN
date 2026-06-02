# Handoff

## Current Status

- The maintained NN-kNN core supports both regression and classification.
- Classification uses normalized case activation as class probability mass
  with NLL loss; it does not restore the older class-weight formulation.
- `model/classification_workflow.py` and `datasets/classification_data.py`
  provide tabular and image classification workflows and representative
  baselines.
- `nnknn_sample_classification.ipynb` is the maintained classification
  notebook entry point.
- Representative checks completed: one-epoch regression and classification
  smokes, sparsemax Iris folds, portable small/image loaders, and a tiny
  all-image-method MNIST subset comparison.
- A fresh June 1 NN-kNN-only 10-fold rerun confirmed the recorded Iris and
  Zebra representative results exactly. These are current-core functionality
  checks, not exact reproductions of IJCAI-25 results from the older
  classification formulation.

## Current Work Focus

- Classification support is now functional; current work is focused on
  improving scores on the earlier IJCAI-25 classification tasks while keeping
  the maintained IJCAI-26-based NN-kNN retrieval/core implementation.
- Do not restore the retired legacy case-weight classification path solely to
  reproduce old numbers. Improvements should come from the current workflow,
  dataset protocol checks, hyperparameter tuning, or appropriate current-core
  feature extractors.
- A June 1 rerun of the 10-fold stratified CV checks using `softmax`, `tau=0.5`,
  50 epochs, and seed 42 reproduced:
  - `iris`: `0.9600 +/- 0.0562`
  - `zebra` / Zebra (a): `0.5182 +/- 0.1488`
  - `zebra_special` / Zebra (b): `0.5227 +/- 0.0890`
- Zebra (a) and (b) are currently close to chance, so they are the immediate
  debugging priority. Inspect their alternating-boundary structure,
  preprocessing/splitting protocol, retrieved cases, feature weights,
  temperature, normalizer choice, and possible current-core feature mapping.
- Use `nnknn_sample_classification.ipynb` for interactive inspection. Use
  `"zebra"` for Zebra (a) and `"zebra_special"` for Zebra (b).
- Before expanding to full prior-paper comparisons, establish better
  representative results on Zebra and then rerun the remaining small
  classification suite (`wine`, `breast_cancer`, `balance`, `digits`).

## Current Local Artifacts

- `checkpoints/` and transient smoke-test/runtime artifacts are expected and
  gitignored.
- Root checkpoint files such as `nnknn_regression_best.pth` and
  `nnknn_regression_best_retr.pth` may change when experiments are run.
- Classification runs write checkpoints beneath `checkpoints/` by default.
  The CLI creates a fresh `results/classification_<suite>_<timestamp>/` folder
  with summary, per-run, and manifest files for each benchmark invocation.
- The June 1 representative rerun is in
  `results/classification_nnknn_rerun_20260601_103353/`. Its completed
  `manifest.json`, `summary.csv`, and `runs.csv` record 30 NN-kNN fold rows.
- `results/` contains completed Table 1 output folders. Keep old run folders
  for comparison and postmortem context rather than overwriting them.

## Where Durable Guidance Lives

Read `NNKNN_AGENT_GUIDE.md` for model descriptions, workflow entry points,
Table 1 protocol details, output schemas, restart/resume patterns, and
machine-specific caveats.
