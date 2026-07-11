# AGENTS.md

## Repository expectations

- This repository is primarily a Python regression/classification research workflow centered on `model/` and `datasets/`, with RL/DQN, RL/NEC, and NN-kNN-RL paths for CartPole and future RL work.
- For regression work, prefer the helpers in `model/regression_workflow.py` instead of calling low-level training code directly.
- Regression configs must explicitly set `task_type="regression"`. The notebook does this already; ad hoc scripts should too.
- For RL baseline work, prefer `model/rl_workflow.py`, `model/nec_workflow.py`, `model/nnknn_rl_workflow.py`, `datasets/rl_tasks.py`, `tools/run_rl_dqn.py`, `tools/run_rl_nec.py`, and `tools/run_rl_nnknn.py` instead of notebook-only implementations.
- Current NN-kNN-RL is an on-policy actor-critic workflow: the actor and critic are selectable between NN-kNN and MLP variants, and GAE supplies actor advantages and value targets.
- MLP actors are standard stochastic-policy baselines: sample directly from their softmax policy during training, use entropy regularization, and keep effective epsilon at zero. Reserve readiness-driven uniform sampling and scheduled epsilon mixing for NN-kNN actors.
- When both actor and critic are NN-kNN, use separate actor and critic case bases and per-case parameters. Share only the state feature extractor/global feature-distance module, analogous to a shared neural trunk with separate policy and value heads.
- Keep rollout case insertion staged: compute actor/critic losses against the frozen rollout representation, then insert only raw-positive-advantage actor recommendations and update/append critic value cases after optimization.
- NN-kNN critic value labels default to fixed GAE targets; optional mutable labels, trainable labels, or the combined hybrid mode are controlled by `critic_mutable_value_labels` and `critic_trainable_value_labels`.
- Trainable NN-kNN value labels use the same case-level optimizer group as case biases and per-case glocal weights; use `case_learning_rate` rather than a label-only learning rate.
- Treat the hybrid label mode as NEC-like two-timescale memory learning, but keep labels as expected GAE/TD targets rather than max-return episodic memory. Mutable labels use a raw `case_bias - distance` activation threshold and aggregate matching batch targets before one EMA update.
- NN-kNN critics use a lagged target critic by default. Online and target critics share the raw state-case tensor and aligned stable case IDs, while value labels, biases, per-case weights, and encoder parameters remain distinct and update through target EMA.
- NN-kNN-RL trains the final partial rollout at the fixed step budget, uses episode-boundary-aware GAE, validates `reward_shaping`, and reports actor/critic case maintenance separately.
- NN-kNN-RL defaults every profile to a 500-case actor/critic capacity; larger case bases must be explicit experiment overrides because exact retrieval cost grows materially with memory size.
- NN-kNN-RL critic holdout diagnostics must use separate stochastic behavior-policy rollouts, preserve the training RNG stream, exclude all holdout transitions from gradients and case insertion, and report discounted Monte Carlo metrics separately in `critic_holdout_metrics.csv`. Bootstrap only time-limit truncations, preferably with the lagged target critic.
- Treat notebooks named `Outdated...` as archival only. Do not use them as implementation references for RL work.
- DQN, NEC, and NN-kNN-RL share evaluation-based early stopping: patience 30, minimum improvement 1.0, patience counting after 25,000 environment steps, and immediate stopping at task-maximum mean return. Fast enables it for every workflow, as does debug where available; smoke/gold disable it. Use gold or `--no-early-stopping` for strict fixed-budget comparisons, and interpret actual/configured timesteps plus `training_efficiency` before paper-style claims.
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

## Notes

- `checkpoints/` and transient artifacts are expected and are gitignored.
- RL runs write timestamped artifacts under `results/rl/`.
- `matplotlib` should run headlessly in cloud tasks with `MPLBACKEND=Agg`.
- Avoid full multi-run benchmarks for routine verification; they are intentionally expensive.
