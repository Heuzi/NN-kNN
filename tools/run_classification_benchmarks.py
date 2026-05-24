from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.classification_workflow import (
    make_classification_cfg,
    run_repeated_classification_model_benchmarks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run maintained NN-kNN classification benchmarks.")
    parser.add_argument("datasets", nargs="+", help="Dataset names, such as iris digits cifar10.")
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to benchmark.")
    parser.add_argument("--mode", default="holdout", choices=["holdout", "kfold", "official", "preset"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalizer", default="softmax", choices=["softmax", "sparsemax"])
    parser.add_argument("--output-dir", default="results/classification")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = make_classification_cfg(
        {
            "training_epochs": args.epochs,
            "batch_size": args.batch_size,
            "case_normalizer": args.normalizer,
        }
    )
    dataset_kwargs_map = {
        name: {
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "seed": args.seed,
        }
        for name in args.datasets
        if name.lower() in {"mnist", "cifar10", "cifar-10", "svhn"}
    }
    summary, runs, _ = run_repeated_classification_model_benchmarks(
        args.datasets,
        cfg,
        methods=args.methods,
        num_runs=args.runs,
        mode=args.mode,
        base_seed=args.seed,
        dataset_kwargs_map=dataset_kwargs_map,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "classification_summary.csv", index=False)
    runs.to_csv(output_dir / "classification_runs.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote results to {output_dir}.")


if __name__ == "__main__":
    main()
