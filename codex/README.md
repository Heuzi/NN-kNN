# Codex Cloud Environment

Use this repository with a Codex web cloud environment configured as follows:

- Base image: `universal`
- Python version: `3.11.9`
- Setup script: `bash codex/setup.sh`
- Maintenance script: `bash codex/maintenance.sh`

Recommended environment variables:

- `MPLBACKEND=Agg`
- `PYTHONUNBUFFERED=1`
- `NNKNN_DEVICE=cpu`

Quick validation commands inside a task:

- `.venv/bin/python codex/smoke_test.py --mode imports`
- `.venv/bin/python codex/smoke_test.py --mode train`
- `.venv/bin/python codex/smoke_test.py --mode classification`
- `.venv/bin/python codex/smoke_test.py --mode rl`
- `.venv/bin/python codex/smoke_test.py --mode nec`
- `.venv/bin/python codex/smoke_test.py --mode nnknn_rl`

RL baseline commands:

- `.venv/bin/python tools/run_rl_dqn.py cartpole --profile smoke --seed 0`
- `.venv/bin/python tools/run_rl_dqn.py cartpole --profile fast --seed 0`
- `.venv/bin/python tools/run_rl_nec.py cartpole --profile smoke --seed 0`
- `.venv/bin/python tools/run_rl_nec.py cartpole --profile debug --seed 0`
- `.venv/bin/python tools/run_rl_nec.py cartpole --profile fast --seed 0`
- `.venv/bin/python tools/run_rl_nnknn.py cartpole --profile smoke --seed 0`
- `.venv/bin/python tools/run_rl_nnknn.py cartpole --profile debug --gamma 0.99 --gae-lambda 0.95 --critic-learning-rate 1e-3 --critic-update-epochs 1`
- `.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --critic-type nnknn`
- `.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --actor-type mlp --critic-type mlp`
- `.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --actor-type mlp --critic-type nnknn`

RL runs write timestamped artifacts under `results/rl/`. Inspect
`summary.json`, especially `training_efficiency`, before comparing DQN, NEC,
or NN-kNN-RL results. Current NN-kNN-RL defaults to an NN-kNN actor, supports
MLP-actor comparison baselines, supports selectable MLP or NN-kNN value
critics, and uses GAE advantages; checkpoints record
`algorithm="nnknn_actor_critic_separate_memory_gae"`. Use `actor_type`
and `critic_type` in config or summary output to distinguish comparison
variants. NN-kNN actor and critic case bases are separate; when both are used,
they share only the state feature extractor/global feature-distance module.
MLP actors are standard stochastic baselines that sample directly from their
softmax policy with entropy regularization and effective epsilon zero. NN-kNN
actors alone use readiness-driven uniform sampling followed by scheduled
epsilon mixing.
NN-kNN critic value labels default to fixed GAE targets, with optional mutable
relabeling, trainable label parameters, or both as a hybrid mode. NN-kNN-RL
summaries also report partial-rollout training, critic label-update counts, and
per-store actor/critic maintenance counters. Trainable value labels share
the case-level optimizer group with case biases and per-case glocal weights; set
`case_learning_rate` / `--case-learning-rate` for that group. The hybrid mode
follows the NEC pattern of fast memory-value updates plus slower differentiable
training, but critic labels should remain expected GAE/TD targets, not
max-return memory. Mutable labels use raw `case_bias - distance` activation
thresholding, and every NN-kNN critic supports a lagged EMA target critic that
shares structural state cases/IDs but keeps target labels and parameters distinct.
Periodic critic holdout rollouts are excluded from training and case insertion
and write Monte Carlo-return MSE/explained variance to
`critic_holdout_metrics.csv`; truncations bootstrap from the lagged target
critic. Their extra environment steps are reported separately from the training
budget.

All three RL workflows use the same optional early-stopping implementation.
`fast` enables it for every workflow, as does `debug` where available, while
`smoke` and `gold` run their full budgets. Defaults are patience `30`,
`min_delta=1.0`, patience counting after
`25_000` environment steps, and immediate stopping at task-maximum evaluation
mean return. Use `--no-early-stopping` for fixed-budget comparisons. Run
summaries and checkpoints report configured/actual timesteps and the stopping
reason.

Current CartPole references:

- DQN fast: `results/rl/dqn_cartpole_20260702_135146_537913/`, selected eval
  mean return `110.85` at step `110000`; this did not reproduce the older
  solved checkpoint and should be treated as `unsolved_or_underfit`.
- NEC fast: `results/rl/nec_cartpole_20260703_173129_688189/`, selected eval
  mean return `450.55` at step `150000`; this reaches individual 500-return
  episodes but remains below the `475.0` success threshold.
- NN-kNN-RL smoke: run `.venv/bin/python codex/smoke_test.py --mode nnknn_rl`;
  this validates plumbing across actor/critic variants, final partial rollouts,
  boundary-aware GAE, staged actor cases, separate memories, target critics,
  and maintenance reporting.
- NN-kNN-RL fast with NN-kNN critic:
  `results/rl/nnknn_rl_cartpole_20260626_150805_689987/`, selected eval mean
  return `369.5` over 20 episodes; this predates the separate-memory
  architecture and remains below the `475.0` success threshold.
