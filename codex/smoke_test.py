from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.regression_workflow import (
    list_supported_regression_benchmark_methods,
    list_supported_regression_datasets,
    make_regression_cfg,
    run_single_nnknn_regression_experiment,
)


def run_import_smoke() -> None:
    datasets = list_supported_regression_datasets()
    methods = list_supported_regression_benchmark_methods()
    print("import smoke ok")
    print(f"synthetic datasets: {datasets['synthetic']}")
    print(f"benchmark methods: {methods}")


def run_training_smoke() -> None:
    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = make_regression_cfg(
        {
            "task_type": "regression",
            "training_epochs": 1,
            "checkpoint_path": "checkpoints/tmp_codex_smoke.pth",
            "batch_size": 32,
            "top_k": 8,
            "bias_manual_set": True,
            "bias_manual_value": 1.0,
        }
    )

    state = run_single_nnknn_regression_experiment(
        "linear_regression",
        cfg,
        dataset_kwargs={"n": 96, "d": 5, "noise_scale": 0.1},
        run_seed=0,
        dataset_seed=0,
        split_seed=0,
        standardize=False,
        checkpoint_label="codex_smoke",
    )

    rmse_raw = float(state["rmse_raw"])
    best_metric = float(state["best_metric"])
    print("training smoke ok")
    print(f"rmse_raw={rmse_raw:.6f}")
    print(f"best_metric={best_metric:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke checks for Codex cloud environments.")
    parser.add_argument(
        "--mode",
        choices=("imports", "train"),
        default="imports",
        help="Choose a lightweight import check or a tiny one-epoch training run.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training_smoke()
    else:
        run_import_smoke()


if __name__ == "__main__":
    main()
