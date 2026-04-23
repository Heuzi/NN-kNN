from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.table1_nnknn_kfold import (
    TABLE1_DEFAULT_DATASET_NAMES,
    export_table1_outputs,
    run_table1_kfold_resumable,
)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def _default_outdir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"table1_kfold_resume_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume a Table 1 NN-kNN k-fold run from existing checkpoints.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=_default_outdir(),
        help="Output folder for summary_long.csv, runs_long.csv, table1_like.csv, and done.json.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed used for the k-fold split. Must match the interrupted run to reuse checkpoints.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of CV folds. Must match the interrupted run to reuse checkpoints.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names to run. Defaults to the repository Table 1 dataset list.",
    )
    parser.add_argument(
        "--checkpoint-mtime-after",
        default=None,
        help=(
            "Only reuse checkpoints modified at or after this ISO timestamp, e.g. "
            "2026-04-21T11:48:00. Useful when old checkpoints share the same filenames."
        ),
    )
    parser.add_argument(
        "--no-reuse-checkpoints",
        action="store_true",
        help="Ignore existing NN-kNN checkpoints and train all NN-kNN entries from scratch.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Only run/resume NN-kNN entries; do not rerun Table 1 baselines.",
    )
    parser.add_argument(
        "--scikit-learn-data",
        type=Path,
        default=None,
        help="Writable sklearn dataset cache directory. Defaults to <outdir>/scikit_learn_data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    if not outdir.is_absolute():
        outdir = REPO_ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sklearn_data = args.scikit_learn_data or (outdir / "scikit_learn_data")
    sklearn_data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["SCIKIT_LEARN_DATA"] = str(sklearn_data)

    dataset_names = args.datasets or TABLE1_DEFAULT_DATASET_NAMES
    checkpoint_mtime_after = _parse_datetime(args.checkpoint_mtime_after)

    print("Starting reusable Table 1 resume run", flush=True)
    print("Output folder:", outdir, flush=True)
    print("Datasets:", dataset_names, flush=True)
    print("Base seed:", args.base_seed, flush=True)
    print("Checkpoint mtime filter:", checkpoint_mtime_after, flush=True)

    summary_df, runs_df, _, resume_info = run_table1_kfold_resumable(
        dataset_names=dataset_names,
        num_folds=args.num_folds,
        base_seed=args.base_seed,
        reuse_checkpoints=not args.no_reuse_checkpoints,
        checkpoint_mtime_after=checkpoint_mtime_after,
        run_baselines=not args.skip_baselines,
    )

    manifest_payload = {
        "outdir": str(outdir),
        "dataset_names": list(dataset_names),
        "base_seed": args.base_seed,
        "num_folds": args.num_folds,
        "checkpoint_mtime_after": checkpoint_mtime_after.isoformat() if checkpoint_mtime_after else None,
        "reuse_checkpoints": not args.no_reuse_checkpoints,
        "run_baselines": not args.skip_baselines,
        "scikit_learn_data": str(sklearn_data),
        "resume_info": resume_info,
    }
    manifest_path = outdir / "resume_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8")

    export_table1_outputs(
        summary_df,
        runs_df,
        outdir,
        metadata={
            "base_seed": args.base_seed,
            "num_folds": args.num_folds,
            "resumed": True,
            "run_baselines": not args.skip_baselines,
            "resume_manifest": str(manifest_path),
            "num_reused_checkpoints": resume_info["nnknn"]["num_reused_checkpoints"],
            "num_trained_entries": resume_info["nnknn"]["num_trained_entries"],
        },
    )
    print("Finished reusable Table 1 resume run", flush=True)


if __name__ == "__main__":
    main()
