from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from model.nec_workflow import make_nec_config, train_nec
from model.nnknn_rl_workflow import make_nnknn_rl_config, train_nnknn_rl
from model.rl_workflow import make_dqn_config, train_dqn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every maintained CartPole model variant.")
    parser.add_argument("--profile", choices=["smoke", "fast", "gold"], default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--case-capacity", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="results/rl")
    parser.add_argument(
        "--variants",
        default=None,
        help="Optional comma-separated subset of variant names.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT_DIR,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT_DIR)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _nnknn_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "critic_target_value_mode": "ema",
    }
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.case_capacity is not None:
        overrides["case_capacity"] = args.case_capacity
    return overrides


def _baseline_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {"seed": args.seed, "eval_episodes": args.eval_episodes}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    return overrides


def _label_mode(mode: str) -> dict[str, bool]:
    if mode == "fixed":
        return {}
    if mode == "mutable":
        return {"critic_mutable_value_labels": True}
    if mode == "trainable":
        return {"critic_trainable_value_labels": True}
    if mode == "hybrid":
        return {
            "critic_mutable_value_labels": True,
            "critic_trainable_value_labels": True,
        }
    raise ValueError(f"Unknown critic label mode: {mode}")


def _build_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline = _baseline_overrides(args)
    nnknn = _nnknn_overrides(args)
    variants: list[dict[str, Any]] = [
        {
            "name": "dqn",
            "family": "dqn",
            "description": "Repo-native DQN baseline.",
            "train": lambda out: train_dqn(
                "cartpole",
                make_dqn_config(args.profile, **baseline),
                output_dir=out,
                device=args.device,
                progress=False,
            ),
        },
        {
            "name": "nec",
            "family": "nec",
            "description": "Repo-native Neural Episodic Control baseline.",
            "train": lambda out: train_nec(
                "cartpole",
                make_nec_config(args.profile, **baseline),
                output_dir=out,
                device=args.device,
                progress=False,
            ),
        },
        {
            "name": "mlp_actor_mlp_critic",
            "family": "actor_critic",
            "description": "Standard stochastic MLP actor with MLP V(s) critic.",
            "train": lambda out: train_nnknn_rl(
                "cartpole",
                make_nnknn_rl_config(
                    args.profile,
                    **nnknn,
                    actor_type="mlp",
                    critic_type="mlp",
                ),
                output_dir=out,
                device=args.device,
                progress=False,
            ),
        },
        {
            "name": "nnknn_actor_mlp_critic",
            "family": "actor_critic",
            "description": "NN-kNN actor with MLP V(s) critic.",
            "train": lambda out: train_nnknn_rl(
                "cartpole",
                make_nnknn_rl_config(
                    args.profile,
                    **nnknn,
                    actor_type="nnknn",
                    critic_type="mlp",
                ),
                output_dir=out,
                device=args.device,
                progress=False,
            ),
        },
    ]
    for actor_type in ("mlp", "nnknn"):
        for label_mode in ("fixed", "mutable", "trainable", "hybrid"):
            name = f"{actor_type}_actor_nnknn_{label_mode}_critic"
            mode_overrides = _label_mode(label_mode)

            def train_variant(
                out: Path,
                *,
                actor_type: str = actor_type,
                mode_overrides: dict[str, bool] = mode_overrides,
            ) -> dict[str, Any]:
                return train_nnknn_rl(
                    "cartpole",
                    make_nnknn_rl_config(
                        args.profile,
                        **nnknn,
                        **mode_overrides,
                        actor_type=actor_type,
                        critic_type="nnknn",
                    ),
                    output_dir=out,
                    device=args.device,
                    progress=False,
                )

            variants.append(
                {
                    "name": name,
                    "family": "actor_critic",
                    "description": (
                        f"{actor_type.upper()} actor with NN-kNN V(s) critic and {label_mode} value labels."
                    ),
                    "train": train_variant,
                }
            )
    if args.variants:
        selected = {value.strip() for value in args.variants.split(",") if value.strip()}
        known = {str(item["name"]) for item in variants}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"Unknown variants: {', '.join(unknown)}")
        variants = [item for item in variants if item["name"] in selected]
    return variants


def _summary_row(
    variant: dict[str, Any],
    state: dict[str, Any],
    *,
    run_dir: Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    summary = state["summary"]
    final_eval = summary.get("final_eval") or {}
    last_eval = summary.get("last_eval") or {}
    stopping = summary.get("early_stopping") or {}
    holdout = summary.get("latest_critic_holdout") or {}
    config = state["config"].to_dict()
    return {
        "variant": variant["name"],
        "family": variant["family"],
        "status": "ok",
        "description": variant["description"],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "run_dir": str(run_dir),
        "profile": summary.get("profile"),
        "seed": summary.get("seed"),
        "configured_timesteps": summary.get("configured_total_timesteps", config.get("total_timesteps")),
        "actual_timesteps": summary.get("actual_timesteps", summary.get("total_timesteps")),
        "stopping_reason": stopping.get("stopping_reason"),
        "selected_mean_return": final_eval.get("mean_return"),
        "selected_std_return": final_eval.get("std_return"),
        "last_mean_return": last_eval.get("mean_return"),
        "selected_step": summary.get("selected_step"),
        "selected_source": summary.get("selected_source"),
        "passed": summary.get("passed"),
        "actor_type": summary.get("actor_type"),
        "critic_type": summary.get("critic_type"),
        "critic_mutable_value_labels": config.get("critic_mutable_value_labels"),
        "critic_trainable_value_labels": config.get("critic_trainable_value_labels"),
        "case_entries": summary.get("case_entries"),
        "critic_case_entries": summary.get("critic_case_entries"),
        "cases_pruned": summary.get("cases_pruned"),
        "cases_replaced": summary.get("cases_replaced"),
        "critic_holdout_evaluations": summary.get("critic_holdout_evaluations"),
        "latest_critic_holdout_mse": holdout.get("critic_holdout_mse"),
        "latest_critic_holdout_explained_variance": holdout.get(
            "critic_holdout_explained_variance"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    variants = _build_variants(args)
    created_at = datetime.now(timezone.utc)
    sweep_dir = (
        ROOT_DIR
        / args.output_dir
        / f"cartpole_all_variants_{args.profile}_{created_at.strftime('%Y%m%d_%H%M%S')}"
    )
    sweep_dir.mkdir(parents=True, exist_ok=False)
    source_paths = [
        ROOT_DIR / "model" / "rl_workflow.py",
        ROOT_DIR / "model" / "nec_workflow.py",
        ROOT_DIR / "model" / "nnknn_rl_workflow.py",
        ROOT_DIR / "model" / "nnknn_model.py",
        Path(__file__).resolve(),
    ]
    manifest = {
        "created_at_utc": created_at.isoformat(),
        "profile": args.profile,
        "seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "total_timesteps_override": args.total_timesteps,
        "case_capacity_override": args.case_capacity,
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_head": _git_head(),
        "git_dirty": _git_dirty(),
        "source_sha256": _source_hash(source_paths),
        "variants": [
            {
                "name": item["name"],
                "family": item["family"],
                "description": item["description"],
            }
            for item in variants
        ],
    }
    _write_json(sweep_dir / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants, start=1):
        name = str(variant["name"])
        run_dir = sweep_dir / name
        print(f"[{index}/{len(variants)}] starting {name} on {args.device}", flush=True)
        started = time.perf_counter()
        try:
            train_fn: Callable[[Path], dict[str, Any]] = variant["train"]
            state = train_fn(run_dir)
            row = _summary_row(
                variant,
                state,
                run_dir=run_dir,
                elapsed_seconds=time.perf_counter() - started,
            )
            print(
                f"[{index}/{len(variants)}] finished {name}: "
                f"mean={row['selected_mean_return']} steps={row['actual_timesteps']} "
                f"elapsed={row['elapsed_seconds']}s",
                flush=True,
            )
        except Exception as exc:
            error_path = sweep_dir / f"{name}_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            row = {
                "variant": name,
                "family": variant["family"],
                "status": "error",
                "description": variant["description"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "run_dir": str(run_dir),
                "error": repr(exc),
                "error_path": str(error_path),
            }
            print(f"[{index}/{len(variants)}] failed {name}: {exc!r}", flush=True)
        rows.append(row)
        _write_csv(sweep_dir / "sweep_summary.csv", rows)
        _write_json(sweep_dir / "sweep_summary.json", {"rows": rows})

    print(f"sweep_dir={sweep_dir}", flush=True)


if __name__ == "__main__":
    main()
