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
- RL/DQN orchestration: `model/rl_workflow.py`
- RL/NEC orchestration: `model/nec_workflow.py`
- Classification loaders: `datasets/classification_data.py`
- RL task registry: `datasets/rl_tasks.py`
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
- `dqn_cartpole_demo.ipynb`: thin debugging surface for the repo-native DQN
  workflow; the DQN implementation lives in Python files.
- `nec_cartpole_demo.ipynb`: thin debugging surface for the repo-native NEC
  workflow; the NEC implementation lives in Python files.

## RL Baseline Protocol

The current RL baselines are a CleanRL-style DQN and a repo-native NEC on
`CartPole-v1`:

```bash
python tools/run_rl_dqn.py cartpole --profile fast --seed 0
python tools/run_rl_dqn.py cartpole --eval-only --checkpoint <checkpoint.pt>
python tools/run_rl_nec.py cartpole --profile fast --seed 0
python tools/run_rl_nec.py cartpole --eval-only --checkpoint <checkpoint.pt>
```

Current DQN reference run:

- `results/rl/dqn_cartpole_20260615_145845_570666/`
- selected checkpoint step: `125000`
- selected eval mean return: `500.0` over 20 episodes
- last/end-of-budget eval mean return: `87.7`

Current NEC reference run:

- `results/rl/nec_cartpole_20260615_191908_236354/`
- selected checkpoint step: `90000`
- selected eval mean return: `371.45` over 20 episodes
- last/end-of-budget eval mean return: `199.60`
- interpretation: NEC improved over the earlier 25k-step debug-sized run but
  remains below the `475.0` success threshold and below DQN.

For DQN, NEC, and future NN-kNN-RL comparisons, prefer a fixed environment-step
budget with periodic evaluation and best-checkpoint selection. Do not treat the
best checkpoint alone as proof that the budget was adequate. A fixed budget is
most informative when the model has reached a good policy and later regressed or
plateaued; if the best model is the final model, or the success threshold is
never reached, treat the run as possible underfit/unsolved and increase the
budget or tune before making paper-style comparisons.

Each RL run records `training_efficiency` in `summary.json`, including the step
where the selected best model was found, the first step that crossed the success
threshold, the training fraction used, the best-vs-last return gap, and a budget
interpretation such as `regressed_after_best`, `final_is_best_check_for_underfit`,
or `unsolved_or_underfit`. Use these fields as sample-efficiency proxies across
DQN, NEC, and NN-kNN-RL.

## Quick Checks

```bash
bash codex/setup.sh
.venv/bin/python codex/smoke_test.py --mode imports
.venv/bin/python codex/smoke_test.py --mode train
.venv/bin/python codex/smoke_test.py --mode classification
.venv/bin/python codex/smoke_test.py --mode rl
.venv/bin/python codex/smoke_test.py --mode nec
```

For a small classification benchmark run:

```bash
.venv/bin/python tools/run_classification_benchmarks.py iris --mode kfold --runs 3 --epochs 20
```

Each CLI run creates a fresh timestamped folder under `results/` containing
`summary.csv`, `runs.csv`, and `manifest.json`.

Use representative small-data or subset image checks during development;
full multi-run paper benchmarks are intentionally expensive.
