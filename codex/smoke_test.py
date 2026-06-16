from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_import_smoke() -> None:
    from datasets.rl_tasks import list_supported_rl_tasks
    from model.regression_workflow import (
        list_supported_regression_benchmark_methods,
        list_supported_regression_datasets,
        make_regression_cfg,
    )
    from model.classification_workflow import (
        list_supported_classification_benchmark_methods,
        list_supported_classification_datasets,
    )

    datasets = list_supported_regression_datasets()
    methods = list_supported_regression_benchmark_methods()
    cls_datasets = list_supported_classification_datasets()
    cls_methods = list_supported_classification_benchmark_methods()
    rl_tasks = list_supported_rl_tasks()
    regression_cfg = make_regression_cfg({"task_type": "regression"})
    if regression_cfg.get("case_score_mode") != "bias_minus_distance":
        raise AssertionError("Generic regression runs must use the current bias-minus-distance case score.")
    if regression_cfg.get("normalize_over_cases") is not True:
        raise AssertionError("Generic regression runs must normalize case activations.")
    print("import smoke ok")
    print(f"synthetic datasets: {datasets['synthetic']}")
    print(f"benchmark methods: {methods}")
    print(f"classification small datasets: {cls_datasets['small']}")
    print(f"classification methods: {cls_methods}")
    print(f"rl tasks: {rl_tasks}")


def run_training_smoke() -> None:
    from model.regression_workflow import make_regression_cfg, run_single_nnknn_regression_experiment

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


def run_classification_smoke() -> None:
    from model.classification_workflow import make_classification_cfg, run_single_nnknn_classification_experiment

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = make_classification_cfg(
        {
            "training_epochs": 1,
            "checkpoint_path": "checkpoints/tmp_codex_classification_smoke.pth",
            "batch_size": 32,
            "top_k": 3,
            "explanation_mode": True,
            "case_normalizer": "softmax",
        }
    )
    state = run_single_nnknn_classification_experiment(
        "iris",
        cfg,
        run_seed=0,
        split_seed=0,
        checkpoint_label="codex_smoke",
    )
    probabilities = state["class_probabilities"]
    if not torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.size(0)), atol=1e-5):
        raise AssertionError("Classification outputs do not sum to one.")
    if state.get("most_activated_cases") is None:
        raise AssertionError("Classification explanation outputs were not returned.")
    if state.get("most_activated_class_ids") is None:
        raise AssertionError("Classification explanation labels were not decoded.")
    print("classification training smoke ok")
    print(f"accuracy={float(state['accuracy']):.6f}")


def run_rl_smoke() -> None:
    from model.rl_workflow import make_dqn_config, train_dqn

    cfg = make_dqn_config("smoke", seed=0)
    state = train_dqn("cartpole", cfg, progress=False)
    final_eval = state["final_eval"]
    if final_eval["episodes"] != cfg.eval_episodes:
        raise AssertionError("RL smoke evaluation did not run the configured number of episodes.")
    if not state["checkpoint_path"].exists():
        raise AssertionError("RL smoke did not write a checkpoint.")
    print("rl smoke ok")
    print(f"run_dir={state['run_dir']}")
    print(f"mean_return={float(final_eval['mean_return']):.6f}")


def run_nec_smoke() -> None:
    from model.nec_workflow import make_nec_config, train_nec

    cfg = make_nec_config("smoke", seed=0)
    state = train_nec("cartpole", cfg, progress=False)
    final_eval = state["final_eval"]
    if final_eval["episodes"] != cfg.eval_episodes:
        raise AssertionError("NEC smoke evaluation did not run the configured number of episodes.")
    if not state["checkpoint_path"].exists():
        raise AssertionError("NEC smoke did not write a checkpoint.")
    print("nec smoke ok")
    print(f"run_dir={state['run_dir']}")
    print(f"mean_return={float(final_eval['mean_return']):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke checks for Codex cloud environments.")
    parser.add_argument(
        "--mode",
        default="imports",
        choices=("imports", "train", "classification", "rl", "nec"),
        help="Choose a lightweight import check or a tiny training run.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training_smoke()
    elif args.mode == "classification":
        run_classification_smoke()
    elif args.mode == "rl":
        run_rl_smoke()
    elif args.mode == "nec":
        run_nec_smoke()
    else:
        run_import_smoke()


if __name__ == "__main__":
    main()
