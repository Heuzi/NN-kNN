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
python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --critic-type nnknn --critic-mutable-value-labels --critic-trainable-value-labels
python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --critic-type nnknn --critic-value-label-activation-threshold 0 --critic-target-sync-interval 4
python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --actor-type mlp --critic-type mlp
python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --actor-type mlp --critic-type nnknn
python tools/run_rl_nnknn.py cartpole --profile debug --gamma 0.99 --gae-lambda 0.95 --critic-learning-rate 1e-3 --critic-update-epochs 1
python tools/run_rl_nnknn.py cartpole --eval-only --checkpoint <checkpoint.pt>
```

The actor-critic workflow defaults to NN-kNN as the actor, supports
`actor_type="mlp"` for MLP-actor baselines, supports selectable MLP or NN-kNN
value critics, and uses GAE advantages. Checkpoints record
`algorithm="nnknn_actor_critic_separate_memory_gae"`; use
`actor_type` and `critic_type` in the config or summary to distinguish
comparison variants.
Older reward-to-go and shared-case-base NN-kNN-RL checkpoints should be
retrained.

### NN-kNN Actor-Critic Structure

The NN-kNN actor-critic path is still experimental, but its structure is now:

- An NN-kNN actor should be an optimizable policy network even when the critic
  is an MLP.
- An NN-kNN critic should be an optimizable value network even when the actor is
  an MLP. It appends value-target cases and also trains its NN-kNN retrieval
  parameters with the value loss.
- NN-kNN critic value labels default to fixed GAE targets. Optional
  `critic_mutable_value_labels` updates cases whose raw
  `case_bias - distance` activation reaches
  `critic_value_label_activation_threshold`. Updates aggregate matching rollout
  targets per case before one EMA label update. Optional
  `critic_trainable_value_labels` makes labels optimizer-trained parameters;
  enabling both gives the hybrid mutable/trainable mode.
- Trainable value labels use the NN-kNN case-level optimizer group, the same
  group used for case biases and per-case glocal weights. Set
  `case_learning_rate` or `--case-learning-rate` when that group should move at
  a different rate from the actor/critic base network.
- The hybrid label mode follows the Neural Episodic Control precedent: NEC uses
  fast memory-value updates for matching keys and slower gradient updates through
  a differentiable key-value memory. For NN-kNN-RL, keep labels tied to
  GAE/TD-style expected value targets rather than max-return memory.
- When both actor and critic are NN-kNN, they use separate case bases and
  separate per-case parameters. The actor stores `(state, recommended_action)`
  cases; the critic stores `(state, V_target)` cases. They may share only the
  state feature extractor and global feature-distance module, analogous to a
  shared neural trunk with separate policy and value heads.
- When only one side is NN-kNN, that standalone actor or critic should still
  train its NN-kNN parameters directly through the relevant actor or value loss.

The former one-case-base actor-critic structure is retired. Sharing policy and
value case priorities caused conflicting memory semantics and made stable target
critic updates difficult.

Training behavior by variant:

- Rollout is on-policy and does not mutate either case base. The workflow stages
  complete episodes, computes GAE, applies actor/critic optimizer steps against
  the frozen rollout representation, and only then updates case memory. If a
  fixed step budget ends mid-episode, the final partial rollout follows the same
  path before final evaluation and checkpointing.
- GAE uses termination to mask value bootstrapping and a separate episode-
  boundary mask to stop lambda recursion across terminated, truncated, and
  final partial rollout boundaries.
- NN-kNN actor policy loss uses every raw GAE advantage, including negative
  advantages. After optimization, only transitions with raw `A(s,a) > 0` are
  inserted as `(state, recommended_action)` cases. Bad actions remain available
  through stochastic/epsilon exploration and train the policy without becoming
  recommendations.
- If `actor_type="mlp"` and `critic_type="nnknn"`, the MLP actor is updated by
  policy loss from GAE, while the NN-kNN critic trains on every rollout state and
  then updates/appends `(state, V_target)` cases. MLP actors are standard
  stochastic policies: training samples directly from `pi(a|s)` with entropy
  regularization and does not apply the NN-kNN epsilon schedule.
- If `actor_type="nnknn"` and `critic_type="nnknn"`, the workflow uses one
  shared representation with separate actor and critic memories. Their losses
  are computed before one joint optimizer step so the shared representation
  remains consistent with the rollout policy.
- Every NN-kNN critic can use a lagged target critic for GAE bootstrap values.
  Online and target critics share the raw state-case tensor and stable structural
  IDs. Target value labels, biases, per-case weights, and encoder/global metric
  remain separate and EMA-update at `critic_target_sync_interval`. Maintenance
  aligns both parameter views by case ID after compaction.
- Training action selection is stochastic (`greedy=False`) and evaluation is
  greedy. NN-kNN uses uniform sampling before case readiness and scheduled
  epsilon mixing afterward. MLP samples directly from its softmax policy with
  effective epsilon `0`. Sampling during training helps exploration and
  on-policy coverage, but it is not a substitute for avoiding max-return labels;
  critic labels should still represent expected GAE/TD value targets.
- Actor and critic maintenance runs only at batch boundaries. Capacity pressure
  and scheduled maintenance use `case_prune_quantile` (default `0.05`), report
  actor/critic counts separately, and clear stale Adam state after per-case
  parameter compaction.
- Critic reporting uses `critic_optimization_mse` for the final optimization
  loss and `critic_train_*` for optional post-update in-sample diagnostics.
  Periodic behavior-policy rollouts excluded from all training and case memory
  updates write `critic_holdout_metrics.csv`, including
  `critic_holdout_mse` and `critic_holdout_explained_variance` against
  discounted Monte Carlo returns with target-critic bootstrap only at time-limit
  truncation. Holdout metrics are reporting-only; policy evaluation returns
  drive checkpoint selection and early stopping.

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

All DQN, NEC, and NN-kNN-RL workflows support the same evaluation-based early
stopping policy. `fast` enables it for every workflow, as does `debug` where
available; `smoke` and `gold` keep strict fixed budgets. By default, training
stops immediately when evaluation
mean return reaches the task maximum (`500` for CartPole), or after 30
non-improving evaluation checkpoints (`min_delta=1.0`) counted only from
25,000 environment steps onward. Use `--no-early-stopping` for controlled
fixed-budget comparisons, or override the patience, minimum delta, minimum
steps, and target score from any RL CLI.

Do not treat the best checkpoint alone as proof that the budget was adequate.
Each run records `configured_total_timesteps`, `actual_timesteps`, and complete
`early_stopping` state in its checkpoint and summary. If the best model is the
final model, or the success threshold is never reached, treat the run as
possible underfit/unsolved and increase the budget or tune before paper-style
comparisons.

Each RL run records `training_efficiency` in `summary.json`, including the step
where the selected best model was found, the first step that crossed the success
threshold, the training fraction used, the best-vs-last return gap, and a budget
interpretation such as `regressed_after_best`, `final_is_best_check_for_underfit`,
or `unsolved_or_underfit`. Use these fields as sample-efficiency proxies across
DQN, NEC, and NN-kNN-RL.

Current NN-kNN-RL status: the expanded `nnknn_rl` smoke check validates all
actor/critic variants, checkpoint reloads, final partial-rollout training,
episode-boundary-aware GAE, staged positive-advantage actor cases, separate
actor/critic memories with a shared representation, EMA target critics,
activation-threshold mutable/trainable critic labels, and maintenance reporting.
This verifies plumbing only, not benchmark quality. Previous NN-kNN-RL artifacts
predate this architecture and should not be used as current comparisons. MLP
checkpoints now record `actor_behavior_policy="standard_stochastic_policy"`;
older MLP artifacts without that marker used epsilon mixing.

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
