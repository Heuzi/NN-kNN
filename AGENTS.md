# AGENTS.md

## Repository expectations

- This repository is primarily a Python regression/classification research workflow centered on `model/` and `datasets/`, with RL/DQN, RL/NEC, and NN-kNN-RL paths for CartPole and future RL work.
- For regression work, prefer the helpers in `model/regression_workflow.py` instead of calling low-level training code directly.
- Regression configs must explicitly set `task_type="regression"`. The notebook does this already; ad hoc scripts should too.
- For RL baseline work, prefer `model/rl_workflow.py`, `model/nec_workflow.py`, `model/nnknn_rl_workflow.py`, `datasets/rl_tasks.py`, `tools/run_rl_dqn.py`, `tools/run_rl_nec.py`, and `tools/run_rl_nnknn.py` instead of notebook-only implementations.
- Maintained notebooks live under `demos/`. Keep archival `Outdated...` notebooks and `older versions/` as historical references only.
- Atari demo support is currently DQN-only for Pong and Breakout. NEC and NN-kNN-RL remain CartPole/flat-observation workflows until image-observation support is added.
- Current NN-kNN-RL is an on-policy actor-critic workflow: the actor is selectable between NN-kNN and MLP, the value critic is selectable between MLP and NN-kNN regression, and GAE supplies actor advantages. Preserve `actor_type="nnknn"` and `critic_type="mlp"` as the default comparison baseline unless explicitly changed.
- Current NN-kNN-RL is an on-policy actor-critic workflow: the actor and critic are selectable between NN-kNN and MLP variants, and GAE supplies actor advantages and value targets.
- When both actor and critic are NN-kNN, use one shared NN-kNN actor-critic case base. Each active shared case has both an action label and a value label over the same retrieval backbone; do not add a separate critic case base for that path.
- NN-kNN critic value labels default to fixed GAE targets; optional mutable labels, trainable labels, or the combined hybrid mode are controlled by `critic_mutable_value_labels` and `critic_trainable_value_labels`.
- Trainable NN-kNN value labels use the same case-level optimizer group as case biases and per-case glocal weights; use `case_learning_rate` rather than a label-only learning rate.
- Treat the hybrid label mode as NEC-like two-timescale memory learning, but keep labels as expected GAE/TD targets rather than max-return episodic memory. Prefer lagged bootstrap targets for NN-kNN critics; the shared path has this now, and standalone NN-kNN critics should get a target-critic option.
- NN-kNN-RL trains the final partial rollout at the fixed step budget, uses episode-boundary-aware GAE, validates `reward_shaping`, and reports actor/critic/shared case maintenance separately.
- Treat notebooks named `Outdated...` as archival only. Do not use them as implementation references for RL work.
- For DQN, NEC, and NN-kNN-RL comparisons, use fixed environment-step budgets with periodic evaluation and best-checkpoint selection. Interpret `training_efficiency` before making paper-style claims; if the best model is the final model or the threshold is never reached, treat the run as possible underfit/unsolved.
- Use synthetic datasets for quick validation unless a task specifically requires the larger real datasets in `datasets/`.

## Setup

- For Codex cloud tasks, use `bash codex/setup.sh` as the setup script.
- For cached Codex cloud environments, use `bash codex/maintenance.sh` as the maintenance script.
- The cloud setup assumes Python `3.11.9`, matching `python_version.txt`.

## Validation

- Fast import check: `python codex/smoke_test.py --mode imports`
- Fast training smoke test: `python codex/smoke_test.py --mode train`
- Fast RL smoke test: `python codex/smoke_test.py --mode rl`
- Fast NEC smoke test: `python codex/smoke_test.py --mode nec`
- Fast NN-kNN-RL smoke test: `python codex/smoke_test.py --mode nnknn_rl`
- Atari DQN smoke commands such as `python tools/run_rl_dqn.py pong --profile smoke --seed 0` require Gymnasium Atari dependencies and accepted/installed ALE ROMs.

## Notes

- `checkpoints/` and transient artifacts are expected and are gitignored.
- RL runs write timestamped artifacts under `results/rl/`.
- `matplotlib` should run headlessly in cloud tasks with `MPLBACKEND=Agg`.
- Avoid full multi-run benchmarks for routine verification; they are intentionally expensive.
