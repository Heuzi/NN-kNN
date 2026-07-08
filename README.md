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
- RL/NN-kNN orchestration: `model/nnknn_rl_workflow.py`
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
- `nnknn_cartpole_demo.ipynb`: thin debugging surface for the repo-native
  NN-kNN-RL workflow; the RL implementation lives in Python files.

## RL Baseline Protocol

The current RL baselines are a CleanRL-style DQN and a repo-native NEC on
`CartPole-v1`, plus an experimental NN-kNN actor-critic workflow:

```bash
python tools/run_rl_dqn.py cartpole --profile fast --seed 0
python tools/run_rl_dqn.py cartpole --eval-only --checkpoint <checkpoint.pt>
python tools/run_rl_nec.py cartpole --profile fast --seed 0
python tools/run_rl_nec.py cartpole --eval-only --checkpoint <checkpoint.pt>
python tools/run_rl_nnknn.py cartpole --profile fast --seed 0
python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --critic-type nnknn
python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --actor-type mlp --critic-type mlp
python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --actor-type mlp --critic-type nnknn
python tools/run_rl_nnknn.py cartpole --profile debug --gamma 0.99 --gae-lambda 0.95 --critic-learning-rate 1e-3 --critic-update-epochs 1
python tools/run_rl_nnknn.py cartpole --eval-only --checkpoint <checkpoint.pt>
```

The actor-critic workflow defaults to NN-kNN as the actor, supports
`actor_type="mlp"` for MLP-actor baselines, supports selectable MLP or NN-kNN
value critics, and uses GAE advantages. Checkpoints record
`algorithm="nnknn_actor_mlp_value_gae"` for compatibility; use
`actor_type` and `critic_type` in the config or summary to distinguish
comparison variants.
Older reward-to-go NN-kNN-RL checkpoints should be retrained.

### NN-kNN Actor-Critic Structure

The NN-kNN actor-critic path is still experimental, but its structure is now:

- An NN-kNN actor should be an optimizable policy network even when the critic
  is an MLP.
- An NN-kNN critic should be an optimizable value network even when the actor is
  an MLP. It appends value-target cases and also trains its NN-kNN retrieval
  parameters with the value loss.
- When both actor and critic are NN-kNN, they use one shared NN-kNN
  actor-critic model with one case base and one retrieval backbone. The shared
  model has separate policy and value label stores, so action probabilities and
  scalar value estimates remain distinct heads over the same retrieved cases.
- When only one side is NN-kNN, that standalone actor or critic should still
  train its NN-kNN parameters directly through the relevant actor or value loss.

The shared NN-kNN actor-critic reuses the case base, similarity computation,
case biases, glocal case weights, and glocal feature weighting. Separate MLP
actor/critic variants remain available for ablations.

Training behavior by variant:

- Rollout is on-policy. The workflow collects complete episodes, stores raw
  environment rewards, and updates after `policy_update_episodes` episodes with
  GAE advantages from the selected critic.
- If `actor_type="nnknn"` and `critic_type="mlp"`, the NN-kNN actor stores each
  sampled state-action pair in its policy case base during rollout. After the
  rollout batch closes, the MLP critic predicts values, GAE produces
  advantages/value targets, and the NN-kNN actor parameters are optimized by the
  policy loss plus entropy and case-bias regularization.
- If `actor_type="mlp"` and `critic_type="nnknn"`, the MLP actor is updated by
  policy loss from GAE, while the NN-kNN critic appends state/value-target cases
  and trains its retrieval parameters directly with the value loss.
- If `actor_type="nnknn"` and `critic_type="nnknn"`, the workflow uses one
  shared NN-kNN actor-critic case base. Rollout inserts state-action cases for
  the actor, then the critic writes value targets back onto those same cases
  through stable shared case IDs, so critic labels remain attached even if case
  maintenance compacts the case base.
- In the shared NN-kNN actor-critic path, GAE bootstrap values can use a lagged
  shared target value model. `shared_target_value_mode="hard"` copies the
  trainable retrieval parameters directly on sync; `shared_target_value_mode="ema"`
  smooths those trainable parameters with EMA. In both modes, the structured
  case memory and label buffers are hard-copied on sync rather than averaged.
- Shared and standalone NN-kNN paths both support case pruning/replacement. The
  shared path protects pending rollout cases until their critic targets have
  been written, then later updates can prune or compact them normally.

Current DQN fast run:

- `results/rl/dqn_cartpole_20260702_135146_537913/`
- selected checkpoint step: `110000`
- selected eval mean return: `110.85` over 20 episodes
- last/end-of-budget eval mean return: `89.45`
- interpretation: this run did not reach the `475.0` success threshold, so
  treat it as `unsolved_or_underfit` rather than a solved DQN reference.

Current NEC reference run:

- `results/rl/nec_cartpole_20260703_173129_688189/`
- selected checkpoint step: `150000`
- selected eval mean return: `450.55` over 20 episodes
- last/end-of-budget eval mean return: `450.55`
- interpretation: NEC improved substantially over the earlier debug-sized run
  and reaches individual 500-return episodes, but the mean remains below the
  `475.0` success threshold; treat it as `unsolved_or_underfit` or possible
  under-budget before paper-style claims.

For DQN, NEC, and NN-kNN-RL comparisons, prefer a fixed environment-step
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

Current NN-kNN-RL status: the actor-critic GAE smoke path validated at
`results/rl/nnknn_rl_cartpole_20260625_161005_956241/` with mean return `14.0`
over 2 smoke-eval episodes. This verifies plumbing only, not benchmark quality.
The current fast NN-kNN-critic comparison run is
`results/rl/nnknn_rl_cartpole_20260626_150805_689987/` with selected eval mean
return `369.5` over 20 episodes; it remains below the `475.0` success
threshold and should be treated as `unsolved_or_underfit`.

## Quick Checks

```bash
bash codex/setup.sh
.venv/bin/python codex/smoke_test.py --mode imports
.venv/bin/python codex/smoke_test.py --mode train
.venv/bin/python codex/smoke_test.py --mode classification
.venv/bin/python codex/smoke_test.py --mode rl
.venv/bin/python codex/smoke_test.py --mode nec
.venv/bin/python codex/smoke_test.py --mode nnknn_rl
```

For a small classification benchmark run:

```bash
.venv/bin/python tools/run_classification_benchmarks.py iris --mode kfold --runs 3 --epochs 20
```

Each CLI run creates a fresh timestamped folder under `results/` containing
`summary.csv`, `runs.csv`, and `manifest.json`.

Use representative small-data or subset image checks during development;
full multi-run paper benchmarks are intentionally expensive.
