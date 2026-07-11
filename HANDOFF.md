# Handoff

## Current Status

- The maintained NN-kNN core supports both regression and classification.
- A repo-native RL/DQN baseline now exists for `CartPole-v1`; it is the first
  engineering step toward DQN, NEC, and NN-kNN-RL comparisons.
- A repo-native RL/NEC baseline now exists for `CartPole-v1`; it uses exact
  per-action kNN dictionaries and the same evaluation, early-stopping, and
  best-checkpoint reporting protocol as DQN.
- An experimental actor-critic workflow now exists for `CartPole-v1`; it
  defaults to NN-kNN as the actor, supports MLP-actor comparison baselines,
  supports MLP or NN-kNN regression value critics with GAE advantages, and
  writes the same budget/early-stopping/best-checkpoint artifacts as DQN and
  NEC. Treat it as a
  research/debug surface, not as a solved or reliable baseline. NN-kNN actor
  and critic memories are separate; when both are selected they share only the
  state feature extractor/global feature-distance representation.
- Classification uses normalized case activation as class probability mass
  with NLL loss; it does not restore the older class-weight formulation.
- `model/classification_workflow.py` and `datasets/classification_data.py`
  provide tabular and image classification workflows and representative
  baselines.
- `model/rl_workflow.py`, `datasets/rl_tasks.py`, and `tools/run_rl_dqn.py`
  provide the maintained DQN baseline workflow. The implementation is
  CleanRL-style, but repo-native so NN-kNN-RL can be swapped in later.
- `model/nec_workflow.py` and `tools/run_rl_nec.py` provide the maintained NEC
  baseline workflow.
- `model/nnknn_rl_workflow.py` and `tools/run_rl_nnknn.py` provide the
  experimental actor-critic workflow for NN-kNN and MLP actor comparisons.
- `nnknn_sample_classification.ipynb` is the maintained classification
  notebook entry point.
- `dqn_cartpole_demo.ipynb` is the maintained RL notebook entry point; it
  should call Python workflow functions rather than duplicating DQN logic.
- `nnknn_cartpole_demo.ipynb` is the maintained NN-kNN-RL notebook entry point;
  it should call Python workflow functions rather than duplicating training
  logic.
- `Outdated NN-kNN Reinforcement Learning.ipynb` and `OutdatedNewCartpole.ipynb`
  are archival only. Do not use them as RL implementation references.
- Representative checks completed: one-epoch regression and classification
  smokes, RL smoke, sparsemax Iris folds, portable small/image loaders, and a
  tiny all-image-method MNIST subset comparison.
- The current DQN fast CartPole notebook/artifact run did not reproduce the
  older solved checkpoint result:
  - run folder: `results/rl/dqn_cartpole_20260702_135146_537913/`
  - selected checkpoint step: `110000`
  - selected eval mean return: `110.85` over 20 episodes
  - last/end-of-budget eval mean return: `89.45`
  - interpretation: the success threshold was never reached, so treat this run
    as `unsolved_or_underfit` rather than a solved DQN reference.
- The current NEC fast CartPole notebook/artifact run is much stronger than the
  older NEC run but remains just below the solve threshold:
  - run folder: `results/rl/nec_cartpole_20260703_173129_688189/`
  - selected checkpoint step: `150000`
  - selected eval mean return: `450.55` over 20 episodes
  - last/end-of-budget eval mean return: `450.55`
  - interpretation: NEC now reaches several 500-return episodes but the mean
    remains below the `475.0` success threshold; because the selected checkpoint
    is the final model, treat it as `unsolved_or_underfit` or possible
    under-budget before making paper-style claims.
- The previous NEC 25k-step debug-sized run selected step `17663` with mean
  return `281.2`, so the larger 150k-step NEC profile substantially improved
  the best evaluation.
- Current NN-kNN-RL smoke runs are functional but not competitive. The expanded
  smoke now checks all actor/critic variants, checkpoint reloads,
  final-partial-rollout training, episode-boundary-aware GAE, staged actor case
  insertion, separate actor/critic memories, EMA target critics, and NN-kNN
  maintenance reporting. Treat smoke as plumbing validation only.
- The previous NN-kNN-RL fast NN-kNN-critic artifact remains unsolved:
  `results/rl/nnknn_rl_cartpole_20260626_150805_689987/`, selected eval mean
  return `369.5` over 20 episodes at step `150000`. It predates the latest
  separate-memory architecture and remains below the `475.0` threshold.
- A full six-variant `fast` CPU sweep launched from
  `results/rl/run_cartpole_variant_fast_sweep.py` was stopped because it was
  still on the first variant after several hours. A smaller CUDA debug sweep
  was launched under `results/rl/cartpole_variant_debug_gpu_sweep_*/` with
  logs in `results/rl/debug_gpu_sweep_logs/`; use that for quick variant
  inspection, not paper-style claims.
- A fresh June 1 NN-kNN-only 10-fold rerun confirmed the recorded Iris and
  Zebra representative results exactly. These are current-core functionality
  checks, not exact reproductions of IJCAI-25 results from the older
  classification formulation.

## Current Work Focus

- Classification support is now functional; current work is focused on
  improving scores on the earlier IJCAI-25 classification tasks while keeping
  the maintained IJCAI-26-based NN-kNN retrieval/core implementation.
- RL work is in a baseline-establishment/debugging phase. DQN currently needs
  reproduction/debugging before serving as the solved CartPole reference; NEC is
  near-solved but still below threshold, and NN-kNN-RL has
  been refactored to actor-critic GAE with selectable NN-kNN or MLP actors and
  selectable MLP or NN-kNN regression value critics. The major NN-kNN-RL focus
  now is the actor/critic model structure:
  - an NN-kNN actor should be an optimizable policy network even when the
    critic is an MLP
  - an NN-kNN critic should be an optimizable value network even when the actor
    is an MLP; it should append value-target cases and train its NN-kNN
    retrieval parameters with the value loss
  - when both actor and critic are NN-kNN, they use separate case bases and
    per-case parameters while sharing the state feature extractor/global
    feature-distance module
  Tuning and diagnostics should support those goals, especially actor
  probabilities, critic value loss, explained variance, case maintenance, and
  selected-checkpoint behavior.
  Current training behavior is:
  - rollout is on-policy and does not mutate case memory; complete or final
    partial batches compute GAE and optimizer losses against the frozen rollout
    representation before any actor/critic cases are updated
  - GAE masks value bootstrapping with `terminated` and stops lambda recursion
    with an episode-boundary mask, so traces do not cross truncated episode or
    final partial-rollout boundaries
  - NN-kNN actor loss uses all raw advantages, including negative advantages;
    only raw-positive-advantage transitions are inserted afterward as
    `(state, recommended_action)` cases
  - MLP actors are standard stochastic-policy baselines: training samples
    directly from `pi(a|s)` with entropy regularization and effective epsilon
    `0`; the readiness/random and scheduled epsilon behavior applies only to
    NN-kNN actors
  - NN-kNN critics train on every rollout target, then update/append
    `(state, V_target)` cases regardless of advantage sign
  - NN-kNN critic value labels default to fixed GAE targets; optional mutable
    labels use raw `case_bias - distance` activation thresholding and aggregate
    matching batch targets before one EMA update; optional trainable labels make
    labels optimizer parameters, and both options form the hybrid mode
  - trainable value labels use the same case-level optimizer group as case
    biases and per-case glocal weights; tune this group with
    `case_learning_rate` / `--case-learning-rate`
  - the hybrid label mode is intentionally NEC-like: fast memory-value updates
    plus slower differentiable training, but labels should remain GAE/TD-style
    expected value targets rather than max-return episodic memory
  - when both sides are NN-kNN, their losses are computed before one joint
    optimizer step so the shared representation remains consistent with the
    rollout policy; their memories and per-case parameters remain separate
  - every NN-kNN critic can use a lagged target critic for GAE bootstrap values;
    online and target critics share raw state cases/stable IDs, while target
    labels, biases, per-case weights, and encoder parameters remain separate and
    EMA-update on the configured synchronization interval
  - training should keep stochastic action selection (`greedy=False`) and reserve
    greedy action selection for evaluation; NN-kNN uses readiness-driven random
    sampling and scheduled epsilon mixing, while MLP samples its policy directly
  - actor and critic maintenance runs at batch boundaries, reports each store
    separately, and clears stale Adam state after per-case compaction
  - every NN-kNN-RL profile defaults to `case_capacity=500`; larger capacities
    are explicit ablations because exact retrieval becomes the dominant runtime
    cost
  - critic reporting separates `critic_optimization_mse` from periodic
    post-update in-sample fields prefixed `critic_train_`; these diagnostics do
    not affect gradients, checkpoint selection, or early stopping and are not a
    held-out critic validation set
  - periodic critic holdout rollouts use the current stochastic behavior policy
    with independent seeds, are excluded from gradients and all case memories,
    and report Monte Carlo-return MSE/explained variance separately in
    `critic_holdout_metrics.csv`; truncations alone bootstrap from the lagged
    target critic
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
- DQN, NEC, and NN-kNN-RL use one configurable early-stopping protocol.
  `fast` enables it for every workflow, as does `debug` where available;
  `smoke` and `gold` disable it. Defaults are 30 stale evaluation checkpoints,
  `min_delta=1.0`, patience counting only after 25,000 environment steps, and
  immediate stopping at task-maximum mean return.
- Use `gold` or `--no-early-stopping` for strict fixed-budget comparisons.
  Always inspect `configured_total_timesteps`, `actual_timesteps`,
  `early_stopping`, and `training_efficiency` in `summary.json`.
- Treat fixed-budget results as paper-useful only when the budget appears
  adequate. If `budget_interpretation` is `unsolved_or_underfit`, or if the
  best model is the final model and the curve is still rising, increase the
  budget or tune before comparing methods. Use `best_model_step` and
  `first_success_step` as sample-efficiency proxies.
- For NN-kNN-RL specifically, do not treat a poor smoke/debug run or an
  unsolved fast run as a final method result. The current design is now an
  on-policy actor-critic workflow with selectable NN-kNN or MLP actors,
  selectable MLP or NN-kNN regression value critics, and GAE.
- MLP artifacts without `actor_behavior_policy="standard_stochastic_policy"`
  predate the standard baseline behavior and should be treated as epsilon-mixed
  MLP variants rather than current MLP actor-critic results.

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
- NEC run folders use the same artifact names as DQN run folders.
- Every RL checkpoint and summary records configured and actual timesteps plus
  the early-stopping configuration, counters, stopping reason, and stopping
  step. A budget-complete run reports `stopping_reason="budget_exhausted"`.
- NN-kNN-RL run folders use the same artifact names as DQN and NEC run folders
  and add `critic_holdout_metrics.csv`, `algorithm`, `gae`, `actor_type`,
  `critic_type`, and comparison diagnostics to the saved summary. NN-kNN actor
  runs record `case_entries` and
  action-count fields; NN-kNN critic runs record `critic_case_entries`.
  Summaries also record total and per-store actor/critic
  `*_cases_pruned`, `*_cases_replaced`, `partial_rollout_segments`,
  `partial_rollout_samples`, `critic_target_value_syncs`,
  `critic_label_updates`, and `critic_label_update_samples` where applicable.
  Current actor-critic checkpoints record
  `algorithm="nnknn_actor_critic_separate_memory_gae"` and include both actor
  and critic state. Older reward-to-go and shared-case-base checkpoints are
  legacy and should be retrained rather than loaded.
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
RL/DQN/NEC protocol details, Table 1 protocol details, output schemas,
restart/resume patterns, and machine-specific caveats.
