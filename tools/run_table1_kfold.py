from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tools.table1_nnknn_kfold import export_table1_outputs, run_table1_kfold


DEFAULT_TABLE1_DATASETS = [
    "califonia_housing",
    "diabets",
    "abalone",
    "body_fat",
    "airfoil",
    "car",
    "student_performance",
    "yacht",
    "energy_efficiency",
    "bike_sharing",
    "wine",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Table 1 5-fold regression benchmark suite."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_TABLE1_DATASETS,
        help="Dataset keys to run. Defaults to the current Table 1 subset.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of CV folds. Default: 5.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory. Defaults to results/table1_kfold_<timestamp>.",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        outdir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path("results") / f"table1_kfold_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def main() -> None:
    args = parse_args()
    outdir = resolve_output_dir(args.output_dir)

    print("Starting Table 1 k-fold sweep", flush=True)
    print(f"Output directory: {outdir}", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"num_folds={args.num_folds} base_seed={args.base_seed}", flush=True)

    summary_df, runs_df, _ = run_table1_kfold(
        dataset_names=args.datasets,
        num_folds=args.num_folds,
        base_seed=args.base_seed,
    )

    done_path = outdir / "done.json"
    written_paths = export_table1_outputs(
        summary_df,
        runs_df,
        outdir,
        metadata={
            "output_dir": str(outdir),
            "datasets": list(args.datasets),
            "num_folds": int(args.num_folds),
            "base_seed": int(args.base_seed),
        },
    )

    if written_paths["done"] != done_path:
        raise RuntimeError(f"Expected done path {done_path}, got {written_paths['done']}")

    print("Finished Table 1 k-fold sweep", flush=True)
    print(f"Wrote: {written_paths['summary']}", flush=True)
    print(f"Wrote: {written_paths['runs']}", flush=True)
    print(f"Wrote: {written_paths['table']}", flush=True)
    print(f"Wrote: {written_paths['transposed']}", flush=True)
    print(f"Wrote: {written_paths['done']}", flush=True)


if __name__ == "__main__":
    main()
