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

RL runs write timestamped artifacts under `results/rl/`. Inspect
`summary.json`, especially `training_efficiency`, before comparing DQN, NEC,
or NN-kNN-RL results. Current NN-kNN-RL uses an NN-kNN actor with an MLP value
critic and GAE advantages; checkpoints record
`algorithm="nnknn_actor_mlp_value_gae"`.

Current CartPole references:

- DQN fast: `results/rl/dqn_cartpole_20260615_145845_570666/`, selected eval
  mean return `500.0` at step `125000`.
- NEC fast: `results/rl/nec_cartpole_20260615_191908_236354/`, selected eval
  mean return `371.45` at step `90000`; this is nontrivial but still below the
  `475.0` success threshold.
- NN-kNN-RL smoke:
  `results/rl/nnknn_rl_cartpole_20260625_161005_956241/`, mean return `14.0`
  over 2 smoke-eval episodes; this validates plumbing only.
