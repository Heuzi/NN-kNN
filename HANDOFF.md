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
- A lightweight 10-fold Iris check with the new class-mass formulation reached
  `0.9600 +/- 0.0562` at `tau=0.5`. This is a functionality sanity check, not
  an exact reproduction of IJCAI-25 results from the older classification
  formulation.

## Current Local Artifacts

- `checkpoints/` and transient smoke-test/runtime artifacts are expected and
  gitignored.
- Root checkpoint files such as `nnknn_regression_best.pth` and
  `nnknn_regression_best_retr.pth` may change when experiments are run.
- Classification runs write checkpoints beneath `checkpoints/` by default and
  benchmark CSVs beneath `results/classification/` when the CLI is used.
- `results/` contains completed Table 1 output folders. Keep old run folders
  for comparison and postmortem context rather than overwriting them.

## Where Durable Guidance Lives

Read `NNKNN_AGENT_GUIDE.md` for model descriptions, workflow entry points,
Table 1 protocol details, output schemas, restart/resume patterns, and
machine-specific caveats.
