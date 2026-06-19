from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.nnknn_rl_workflow import (  # noqa: E402
    NNKNNRLConfig,
    evaluate_nnknn_rl,
    load_nnknn_rl_checkpoint,
    make_nnknn_rl_config,
    make_nnknn_rl_output_dir,
    train_nnknn_rl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the repo-native NN-kNN-RL workflow.")
    parser.add_argument("task", nargs="?", default="cartpole", help="RL task key, such as cartpole.")
    parser.add_argument("--profile", choices=["smoke", "debug", "fast", "gold"], default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/rl", help="Parent folder for timestamped run dirs.")
    parser.add_argument("--device", default=None, help="Optional torch device override, such as cpu or cuda.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--eval-episode-frequency", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved checkpoint without training.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for --eval-only.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging during training.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> NNKNNRLConfig:
    overrides = {"seed": args.seed}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.eval_frequency is not None:
        overrides["eval_frequency"] = args.eval_frequency
    if args.eval_episode_frequency is not None:
        overrides["eval_episode_frequency"] = args.eval_episode_frequency
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.eval_seed is not None:
        overrides["eval_seed"] = args.eval_seed
    return make_nnknn_rl_config(args.profile, **overrides)


def _write_eval_only_summary(outdir: Path, payload: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required with --eval-only")
        loaded = load_nnknn_rl_checkpoint(args.checkpoint, device=args.device)
        cfg = loaded["config"]
        task_name = loaded["task"]["name"]
        episodes = args.eval_episodes if args.eval_episodes is not None else cfg.eval_episodes
        eval_seed = args.eval_seed if args.eval_seed is not None else cfg.eval_seed
        metrics = evaluate_nnknn_rl(
            task_name,
            loaded["model"],
            episodes=episodes,
            seed=eval_seed,
            device=loaded["device"],
        )
        outdir = make_nnknn_rl_output_dir(task_name, parent=args.output_dir, suffix="eval")
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
    state = train_nnknn_rl(
        args.task,
        cfg,
        output_dir=make_nnknn_rl_output_dir(args.task, parent=args.output_dir),
        device=args.device,
        progress=not args.quiet,
    )
    summary = state["summary"]
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoint_path"}, indent=2))
    print(f"Checkpoint: {state['checkpoint_path']}")


if __name__ == "__main__":
    main()
