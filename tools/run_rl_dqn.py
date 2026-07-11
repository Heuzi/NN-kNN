from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.rl_workflow import (  # noqa: E402
    DQNConfig,
    evaluate_dqn,
    load_dqn_checkpoint,
    make_dqn_config,
    make_dqn_output_dir,
    train_dqn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the repo-native DQN baseline.")
    parser.add_argument("task", nargs="?", default="cartpole", help="RL task key, such as cartpole.")
    parser.add_argument("--profile", choices=["smoke", "fast", "gold"], default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/rl", help="Parent folder for timestamped run dirs.")
    parser.add_argument("--device", default=None, help="Optional torch device override, such as cpu or cuda.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    early_stopping_group = parser.add_mutually_exclusive_group()
    early_stopping_group.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        help="Enable evaluation-based early stopping.",
    )
    early_stopping_group.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Run through the full configured timestep budget.",
    )
    parser.set_defaults(early_stopping=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--early-stopping-min-steps", type=int, default=None)
    parser.add_argument("--early-stopping-target-score", type=float, default=None)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved checkpoint without training.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for --eval-only.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging during training.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> DQNConfig:
    overrides = {"seed": args.seed}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.eval_frequency is not None:
        overrides["eval_frequency"] = args.eval_frequency
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.eval_seed is not None:
        overrides["eval_seed"] = args.eval_seed
    if args.early_stopping is not None:
        overrides["early_stopping"] = args.early_stopping
    if args.early_stopping_patience is not None:
        overrides["early_stopping_patience"] = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        overrides["early_stopping_min_delta"] = args.early_stopping_min_delta
    if args.early_stopping_min_steps is not None:
        overrides["early_stopping_min_steps"] = args.early_stopping_min_steps
    if args.early_stopping_target_score is not None:
        overrides["early_stopping_target_score"] = args.early_stopping_target_score
    return make_dqn_config(args.profile, **overrides)


def _write_eval_only_summary(outdir: Path, payload: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required with --eval-only")
        loaded = load_dqn_checkpoint(args.checkpoint, device=args.device)
        cfg = loaded["config"]
        task_name = loaded["task"]["name"]
        episodes = args.eval_episodes if args.eval_episodes is not None else cfg.eval_episodes
        eval_seed = args.eval_seed if args.eval_seed is not None else cfg.eval_seed
        metrics = evaluate_dqn(
            task_name,
            loaded["model"],
            episodes=episodes,
            seed=eval_seed,
            device=loaded["device"],
        )
        outdir = make_dqn_output_dir(task_name, parent=args.output_dir, suffix="eval")
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": task_name,
            "checkpoint": str(args.checkpoint),
            "episodes": episodes,
            "eval_seed": eval_seed,
            "metrics": {k: v for k, v in metrics.items() if k != "episode_metrics"},
        }
        _write_eval_only_summary(outdir, payload)
        print(json.dumps(payload["metrics"], indent=2))
        print(f"Wrote eval-only results to {outdir}.")
        return

    cfg = _config_from_args(args)
    state = train_dqn(
        args.task,
        cfg,
        output_dir=make_dqn_output_dir(args.task, parent=args.output_dir),
        device=args.device,
        progress=not args.quiet,
    )
    summary = state["summary"]
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoint_path"}, indent=2))
    print(f"Checkpoint: {state['checkpoint_path']}")


if __name__ == "__main__":
    main()
