from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tools.table1_nnknn_kfold import run_table1_kfold


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

    summary_path = outdir / "summary_long.csv"
    runs_path = outdir / "runs_long.csv"
    table_like_path = outdir / "table1_like.csv"
    done_path = outdir / "done.json"

    summary_df.to_csv(summary_path, index=False)
    runs_df.to_csv(runs_path, index=False)
    pivot_df = summary_df.pivot(index="dataset", columns="entry_label", values="rmse_raw_table").reset_index()
    pivot_df.to_csv(table_like_path, index=False)

    done_path.write_text(
        json.dumps(
            {
                "output_dir": str(outdir),
                "datasets": list(args.datasets),
                "num_folds": int(args.num_folds),
                "base_seed": int(args.base_seed),
                "summary_rows": int(len(summary_df)),
                "run_rows": int(len(runs_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Finished Table 1 k-fold sweep", flush=True)
    print(f"Wrote: {summary_path}", flush=True)
    print(f"Wrote: {runs_path}", flush=True)
    print(f"Wrote: {table_like_path}", flush=True)
    print(f"Wrote: {done_path}", flush=True)


if __name__ == "__main__":
    main()
