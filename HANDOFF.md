# Handoff

## Current Status

- The maintained NN-kNN core supports both regression and classification.
- A repo-native RL/DQN baseline now exists for `CartPole-v1`; it is the first
  engineering step toward DQN, NEC, and NN-kNN-RL comparisons.
- Classification uses normalized case activation as class probability mass
  with NLL loss; it does not restore the older class-weight formulation.
- `model/classification_workflow.py` and `datasets/classification_data.py`
  provide tabular and image classification workflows and representative
  baselines.
- `model/rl_workflow.py`, `datasets/rl_tasks.py`, and `tools/run_rl_dqn.py`
  provide the maintained DQN baseline workflow. The implementation is
  CleanRL-style, but repo-native so NN-kNN-RL can be swapped in later.
- `nnknn_sample_classification.ipynb` is the maintained classification
  notebook entry point.
- `dqn_cartpole_demo.ipynb` is the maintained RL notebook entry point; it
  should call Python workflow functions rather than duplicating DQN logic.
- `Outdated NN-kNN Reinforcement Learning.ipynb` and `OutdatedNewCartpole.ipynb`
  are archival only. Do not use them as RL implementation references.
- Representative checks completed: one-epoch regression and classification
  smokes, RL smoke, sparsemax Iris folds, portable small/image loaders, and a
  tiny all-image-method MNIST subset comparison.
- The current DQN fast CartPole run solved the task via best-eval checkpoint
  selection:
  - run folder: `results/rl/dqn_cartpole_20260615_145845_570666/`
  - selected checkpoint step: `125000`
  - selected eval mean return: `500.0` over 20 episodes
  - last/end-of-budget eval mean return: `87.7`
  - interpretation: the fixed budget exposed policy regression after a solved
    checkpoint, so the best-checkpoint protocol was meaningful for this run.
- A fresh June 1 NN-kNN-only 10-fold rerun confirmed the recorded Iris and
  Zebra representative results exactly. These are current-core functionality
  checks, not exact reproductions of IJCAI-25 results from the older
  classification formulation.

## Current Work Focus

- Classification support is now functional; current work is focused on
  improving scores on the earlier IJCAI-25 classification tasks while keeping
  the maintained IJCAI-26-based NN-kNN retrieval/core implementation.
- RL work is in a baseline-establishment phase. The next RL step should be to
  keep the CartPole DQN pipeline stable and then add paper-aligned Atari/ALE
  tasks such as Pong or Breakout before implementing NEC or NN-kNN-RL.
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
- For DQN/NEC/NN-kNN-RL comparisons, use fixed environment-step budgets with
  periodic evaluation and best-checkpoint selection. This is not early stopping:
  train through the configured budget, then save the best evaluated checkpoint.
  Always inspect `training_efficiency` in `summary.json`.
- Treat fixed-budget results as paper-useful only when the budget appears
  adequate. If `budget_interpretation` is `unsolved_or_underfit`, or if the
  best model is the final model and the curve is still rising, increase the
  budget or tune before comparing methods. Use `best_model_step` and
  `first_success_step` as sample-efficiency proxies.

## Current Local Artifacts

- `checkpoints/` and transient smoke-test/runtime artifacts are expected and
  gitignored.
- Root checkpoint files such as `nnknn_regression_best.pth` and
  `nnknn_regression_best_retr.pth` may change when experiments are run.
- Classification runs write checkpoints beneath `checkpoints/` by default.
  The CLI creates a fresh `results/classification_<suite>_<timestamp>/` folder
  with summary, per-run, and manifest files for each benchmark invocation.
- RL runs write timestamped folders under `results/rl/`. A DQN run folder
  contains `config.json`, `training_metrics.csv`, `loss_metrics.csv`,
  `eval_metrics.csv`, `final_eval_episodes.csv`, `last_eval_episodes.csv`,
  `summary.json`, `manifest.json`, and `checkpoint.pt`.
- `summary.json` records both the selected best checkpoint evaluation
  (`final_eval`) and the end-of-budget model evaluation (`last_eval`). It also
  records `training_efficiency`, including `best_model_step`,
  `first_success_step`, training fractions, best-vs-last return gap, and
  `budget_interpretation`.
- The June 1 representative rerun is in
  `results/classification_nnknn_rerun_20260601_103353/`. Its completed
  `manifest.json`, `summary.csv`, and `runs.csv` record 30 NN-kNN fold rows.
- `results/` contains completed Table 1 output folders. Keep old run folders
  for comparison and postmortem context rather than overwriting them.

## Where Durable Guidance Lives

Read `NNKNN_AGENT_GUIDE.md` for model descriptions, workflow entry points,
RL/DQN protocol details, Table 1 protocol details, output schemas,
restart/resume patterns, and machine-specific caveats.
