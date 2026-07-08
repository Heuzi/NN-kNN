# Handoff

## Current Status

- The maintained NN-kNN core supports both regression and classification.
- A repo-native RL/DQN baseline now exists for `CartPole-v1`; it is the first
  engineering step toward DQN, NEC, and NN-kNN-RL comparisons.
- A repo-native RL/NEC baseline now exists for `CartPole-v1`; it uses exact
  per-action kNN dictionaries and the same fixed-budget/best-checkpoint
  reporting protocol as DQN.
- An experimental actor-critic workflow now exists for `CartPole-v1`; it
  defaults to NN-kNN as the actor, supports MLP-actor comparison baselines,
  supports MLP or NN-kNN regression value critics with GAE advantages, and
  writes the same
  fixed-budget/best-checkpoint artifacts as DQN and NEC. Treat it as a
  research/debug surface, not as a solved or reliable baseline.
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
- `demos/nnknn_sample_classification.ipynb` is the maintained classification
  notebook entry point.
- `demos/dqn_cartpole_demo.ipynb` is the maintained RL notebook entry point; it
  should call Python workflow functions rather than duplicating DQN logic.
- `demos/nec_cartpole_demo.ipynb` is the maintained NEC notebook entry point.
- `demos/nnknn_cartpole_demo.ipynb` is the maintained NN-kNN-RL notebook entry point;
  it should call Python workflow functions rather than duplicating training
  logic.
- `demos/dqn_atari_pong_demo.ipynb` and
  `demos/dqn_atari_breakout_demo.ipynb` are maintained Atari DQN demo entry
  points. Atari is DQN-only for now; NEC and NN-kNN-RL still need
  image-observation workflow support before runnable Atari demos.
- `Outdated NN-kNN Reinforcement Learning.ipynb` and `OutdatedNewCartpole.ipynb`
  are archival only. Do not use them as RL implementation references.
- Representative checks completed: one-epoch regression and classification
  smokes, RL smoke, sparsemax Iris folds, portable small/image loaders, and a
  tiny all-image-method MNIST subset comparison.
- The current saved DQN fast CartPole notebook output did not reproduce the
  older solved checkpoint result:
  - notebook: `demos/dqn_cartpole_demo.ipynb`
  - saved run folder: `results/rl/dqn_cartpole_20260701_183004_199116/`
  - selected checkpoint step: `135000`
  - selected eval mean return: `152.70` over 20 episodes
  - last/end-of-budget eval mean return: `118.75`
  - interpretation: the success threshold was never reached, so treat this
    saved notebook output as `unsolved_or_underfit`.
- The current DQN fast CartPole artifact reference is also unsolved:
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
- The current saved NEC CartPole notebook output is the older debug-sized run:
  - notebook: `demos/nec_cartpole_demo.ipynb`
  - saved run folder: `results/rl/nec_cartpole_20260616_184922_757182/`
  - selected checkpoint step: `17663`
  - selected eval mean return: `281.20` over 20 episodes
  - last/end-of-budget eval mean return: `176.05`
  - interpretation: useful debug history, but superseded by the 150k-step NEC
    artifact above for current CartPole comparisons.
- Current NN-kNN-RL smoke runs are functional but not competitive:
  - run folders: `results/rl/nnknn_rl_cartpole_*`
  - smoke profile: `256` environment steps, `case_capacity=1000`,
    `eval_episodes=2`
  - current actor-critic GAE smoke artifact:
    `results/rl/nnknn_rl_cartpole_20260625_161005_956241/`
  - observed actor-critic smoke eval mean return: `14.0`
  - interpretation: smoke verifies plumbing only. It is not a meaningful
    performance result, and fast/debug runs still need tuning before
    paper-style comparison.
- Current NN-kNN-RL fast NN-kNN-critic run remains unsolved but is a useful
  comparison artifact:
  - current saved notebook run folder:
    `results/rl/nnknn_rl_cartpole_20260629_130136_701190/`
  - critic: `critic_type="nnknn"`
  - selected checkpoint step: `121325`
  - selected eval mean return: `432.35` over 20 episodes
  - last/end-of-budget eval mean return: `418.15`
  - interpretation: this is much stronger than early smoke/debug runs and
    reaches some 500-return episodes, but the mean remains below the `475.0`
    success threshold. Treat it as `unsolved_or_underfit`, not solved.
- Current DQN Atari notebook outputs:
  - Pong gold output from `demos/dqn_atari_pong_demo.ipynb`:
    `results/rl/dqn_pong_20260708_133010_927252/`, selected final checkpoint
    at `1000000` steps, mean return `-21.0` over 10 episodes, interpreted as
    an unsolved learning/debug result rather than benchmark quality.
  - Breakout smoke output:
    `results/rl/dqn_breakout_20260707_203426_373636/`, mean return `1.0` over
    1 episode, plumbing only.
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
  selectable MLP or NN-kNN regression value critics. The next NN-kNN-RL step
  should be tuning and diagnostics of actor probabilities, critic value loss,
  explained variance, and
  selected-checkpoint behavior.
- Atari demo work has begun with DQN support for Pong and Breakout using
  Gymnasium ALE preprocessing and a CNN DQN. The current Pong gold notebook
  output still returns `-21.0`, so the Atari path should be treated as runnable
  but unsolved and in need of DQN diagnostics/tuning before comparison claims.
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
- Use `demos/nnknn_sample_classification.ipynb` for interactive inspection. Use
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
- For NN-kNN-RL specifically, do not treat a poor smoke/debug run or an
  unsolved fast run as a final method result. The current design is now an
  on-policy actor-critic workflow with selectable NN-kNN or MLP actors,
  selectable MLP or NN-kNN regression value critics, and GAE.

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
- NN-kNN-RL run folders use the same artifact names as DQN and NEC run folders
  and add `algorithm`, `gae`, `actor_type`, `critic_type`, and comparison
  diagnostics to the saved summary. NN-kNN actor runs record `case_entries` and
  action-count fields; NN-kNN critic runs record `critic_case_entries`. Current
  actor-critic checkpoints record `algorithm="nnknn_actor_mlp_value_gae"` and
  include both actor and critic state; the algorithm name is retained for
  compatibility even when `actor_type="mlp"` or `critic_type="nnknn"`. Older
  reward-to-go NN-kNN-RL checkpoints are legacy and should be retrained rather
  than loaded.
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
