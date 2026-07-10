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


def _optional_float_arg(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    return float(value)


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
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--exploration-initial-epsilon", type=float, default=None)
    parser.add_argument("--exploration-final-epsilon", type=float, default=None)
    parser.add_argument("--exploration-fraction", type=float, default=None)
    parser.add_argument("--min-case-entries", type=int, default=None)
    parser.add_argument("--min-cases-per-action", type=int, default=None)
    parser.add_argument("--actor-type", choices=["nnknn", "mlp"], default=None)
    parser.add_argument("--critic-type", choices=["mlp", "nnknn"], default=None)
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--critic-update-epochs", type=int, default=None)
    parser.add_argument(
        "--critic-mutable-value-labels",
        action="store_true",
        help="Smooth existing close/high-activation NN-kNN critic value labels toward new GAE targets.",
    )
    parser.add_argument(
        "--critic-trainable-value-labels",
        action="store_true",
        help="Make NN-kNN critic value labels optimizer-trained parameters.",
    )
    parser.add_argument("--critic-value-label-update-alpha", type=float, default=None)
    parser.add_argument("--critic-value-label-min-activation", type=_optional_float_arg, default=argparse.SUPPRESS)
    parser.add_argument("--critic-value-label-distance-threshold", type=_optional_float_arg, default=argparse.SUPPRESS)
    parser.add_argument(
        "--case-learning-rate",
        type=float,
        default=None,
        help="Optional NN-kNN case-parameter LR used for case biases, per-case weights, and trainable value labels.",
    )
    parser.add_argument(
        "--no-critic-value-label-append-on-no-match",
        dest="critic_value_label_append_on_no_match",
        action="store_false",
        default=None,
        help="With mutable standalone NN-kNN critic labels, do not append samples that match no existing case.",
    )
    parser.add_argument("--shared-target-value-mode", choices=["none", "hard", "ema"], default=None)
    parser.add_argument("--shared-target-sync-interval", type=int, default=None)
    parser.add_argument("--shared-target-ema-tau", type=float, default=None)
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
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    if args.gae_lambda is not None:
        overrides["gae_lambda"] = args.gae_lambda
    if args.exploration_initial_epsilon is not None:
        overrides["exploration_initial_epsilon"] = args.exploration_initial_epsilon
    if args.exploration_final_epsilon is not None:
        overrides["exploration_final_epsilon"] = args.exploration_final_epsilon
    if args.exploration_fraction is not None:
        overrides["exploration_fraction"] = args.exploration_fraction
    if args.min_case_entries is not None:
        overrides["min_case_entries"] = args.min_case_entries
    if args.min_cases_per_action is not None:
        overrides["min_cases_per_action"] = args.min_cases_per_action
    if args.actor_type is not None:
        overrides["actor_type"] = args.actor_type
    if args.critic_type is not None:
        overrides["critic_type"] = args.critic_type
    if args.critic_learning_rate is not None:
        overrides["critic_learning_rate"] = args.critic_learning_rate
    if args.critic_update_epochs is not None:
        overrides["critic_update_epochs"] = args.critic_update_epochs
    if args.critic_mutable_value_labels:
        overrides["critic_mutable_value_labels"] = True
    if args.critic_trainable_value_labels:
        overrides["critic_trainable_value_labels"] = True
    if args.critic_value_label_update_alpha is not None:
        overrides["critic_value_label_update_alpha"] = args.critic_value_label_update_alpha
    if hasattr(args, "critic_value_label_min_activation"):
        overrides["critic_value_label_min_activation"] = args.critic_value_label_min_activation
    if hasattr(args, "critic_value_label_distance_threshold"):
        overrides["critic_value_label_distance_threshold"] = args.critic_value_label_distance_threshold
    if args.case_learning_rate is not None:
        overrides["case_learning_rate"] = args.case_learning_rate
    if args.critic_value_label_append_on_no_match is not None:
        overrides["critic_value_label_append_on_no_match"] = args.critic_value_label_append_on_no_match
    if args.shared_target_value_mode is not None:
        overrides["shared_target_value_mode"] = args.shared_target_value_mode
    if args.shared_target_sync_interval is not None:
        overrides["shared_target_sync_interval"] = args.shared_target_sync_interval
    if args.shared_target_ema_tau is not None:
        overrides["shared_target_ema_tau"] = args.shared_target_ema_tau
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
