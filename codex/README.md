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
`algorithm="nnknn_actor_mlp_value_gae"` for compatibility. Use `actor_type`
and `critic_type` in config or summary output to distinguish comparison
variants. In the shared NN-kNN actor-critic path, one case base is used and
each active shared case has both an action label and a scalar value label.
NN-kNN-RL summaries also report partial-rollout training and per-store
actor/critic/shared maintenance counters.

Current CartPole references:

- DQN fast: `results/rl/dqn_cartpole_20260702_135146_537913/`, selected eval
  mean return `110.85` at step `110000`; this did not reproduce the older
  solved checkpoint and should be treated as `unsolved_or_underfit`.
- NEC fast: `results/rl/nec_cartpole_20260703_173129_688189/`, selected eval
  mean return `450.55` at step `150000`; this reaches individual 500-return
  episodes but remains below the `475.0` success threshold.
- NN-kNN-RL smoke: run `.venv/bin/python codex/smoke_test.py --mode nnknn_rl`;
  this validates plumbing across actor/critic variants, final partial rollouts,
  boundary-aware GAE, shared value-label writes, and maintenance reporting.
- NN-kNN-RL fast with NN-kNN critic:
  `results/rl/nnknn_rl_cartpole_20260626_150805_689987/`, selected eval mean
  return `369.5` over 20 episodes; this predates the latest shared-case-base
  audit fixes and remains below the `475.0` success threshold.
