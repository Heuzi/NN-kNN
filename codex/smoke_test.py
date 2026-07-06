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


def run_nnknn_rl_smoke() -> None:
    from model.nnknn_rl_workflow import (
        ALGORITHM_NAME,
        MLPPolicyNetwork,
        NNKNNQNetwork,
        compute_gae,
        evaluate_nnknn_rl,
        load_nnknn_rl_checkpoint,
        make_nnknn_rl_config,
        train_nnknn_rl,
    )

    advantages, value_targets = compute_gae(
        rewards=[1.0, 1.0, 1.0],
        values=[0.5, 0.5, 0.5],
        next_values=[0.5, 0.5, 0.0],
        terminated=[False, False, True],
        gamma=0.9,
        gae_lambda=0.8,
    )
    if not torch.allclose(advantages, torch.tensor([1.8932, 1.31, 0.5]), atol=1e-4):
        raise AssertionError("NN-kNN-RL GAE helper returned unexpected advantages.")
    if not torch.allclose(value_targets, torch.tensor([2.3932, 1.81, 1.0]), atol=1e-4):
        raise AssertionError("NN-kNN-RL GAE helper returned unexpected value targets.")

    wrapper = NNKNNQNetwork(4, 2, case_capacity=6, top_k=2, min_cases_per_action=1)
    wrapper.configure_case_maintenance(prune_quantile=1.0, prune_bias_threshold=None)
    wrapper.add_cases(
        torch.zeros(6, 4),
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
    )
    probs = wrapper.policy_probs(torch.zeros(2, 4))
    if probs.shape != (2, 2) or not torch.allclose(
        probs.sum(dim=1), torch.ones(2, device=probs.device), atol=1e-5
    ):
        raise AssertionError("NN-kNN-RL policy wrapper did not return normalized action probabilities.")
    with torch.no_grad():
        wrapper.nnknn_model.biases[:6].copy_(torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]))
    wrapper.prune_cases()
    if torch.any(wrapper.action_counts() < 1):
        raise AssertionError("NN-kNN-RL pruning removed all cases for an action.")

    cfg = make_nnknn_rl_config("smoke", seed=0)
    for removed_field in ("advantage_method", "value_function", "bootstrap_n_steps", "vtrace"):
        if hasattr(cfg, removed_field):
            raise AssertionError(f"NN-kNN-RL config still exposes removed field {removed_field}.")
    if cfg.gae_lambda != 0.95:
        raise AssertionError("NN-kNN-RL should default to GAE lambda 0.95.")
    if cfg.reward_shaping is not None:
        raise AssertionError("NN-kNN-RL should default to raw environment rewards.")
    if cfg.critic_type != "mlp":
        raise AssertionError("NN-kNN-RL should default to the MLP value critic.")
    if cfg.actor_type != "nnknn":
        raise AssertionError("NN-kNN-RL should default to the NN-kNN actor.")

    mlp_wrapper = MLPPolicyNetwork(4, 2, hidden_sizes=(8,))
    mlp_probs = mlp_wrapper.policy_probs(torch.zeros(3, 4))
    if mlp_probs.shape != (3, 2) or not torch.allclose(mlp_probs.sum(dim=1), torch.ones(3), atol=1e-5):
        raise AssertionError("MLP actor did not return normalized action probabilities.")

    state = train_nnknn_rl("cartpole", cfg, progress=False)
    final_eval = state["final_eval"]
    if final_eval["episodes"] != cfg.eval_episodes:
        raise AssertionError("NN-kNN-RL smoke evaluation did not run the configured number of episodes.")
    if not state["checkpoint_path"].exists():
        raise AssertionError("NN-kNN-RL smoke did not write a checkpoint.")
    loaded = load_nnknn_rl_checkpoint(state["checkpoint_path"])
    if loaded["checkpoint"].get("algorithm") != ALGORITHM_NAME:
        raise AssertionError("NN-kNN-RL checkpoint did not preserve the actor-critic algorithm marker.")
    if loaded["config"].actor_type != "nnknn":
        raise AssertionError("NN-kNN actor checkpoint reload did not preserve actor_type.")
    if "actor_state" not in loaded["checkpoint"] or "critic_state_dict" not in loaded["checkpoint"]:
        raise AssertionError("NN-kNN-RL checkpoint did not preserve actor and critic state.")
    if loaded["model"].case_entries != state["model"].case_entries:
        raise AssertionError("NN-kNN-RL checkpoint did not preserve active case count.")
    if "value_model" not in loaded:
        raise AssertionError("NN-kNN-RL checkpoint reload did not return the value critic.")
    legacy_checkpoint = dict(torch.load(state["checkpoint_path"], map_location="cpu", weights_only=False))
    legacy_checkpoint["config"] = dict(legacy_checkpoint["config"])
    legacy_checkpoint["config"].pop("actor_type", None)
    legacy_checkpoint.pop("actor_type", None)
    legacy_path = state["run_dir"] / "legacy_no_actor_type_checkpoint.pt"
    torch.save(legacy_checkpoint, legacy_path)
    legacy_loaded = load_nnknn_rl_checkpoint(legacy_path)
    if legacy_loaded["config"].actor_type != "nnknn":
        raise AssertionError("Legacy checkpoint without actor_type should reload as NN-kNN actor.")

    nnknn_cfg = make_nnknn_rl_config("smoke", seed=1, critic_type="nnknn")
    nnknn_state = train_nnknn_rl("cartpole", nnknn_cfg, progress=False)
    if not nnknn_state["checkpoint_path"].exists():
        raise AssertionError("NN-kNN critic smoke did not write a checkpoint.")
    nnknn_loaded = load_nnknn_rl_checkpoint(nnknn_state["checkpoint_path"])
    if nnknn_loaded["config"].critic_type != "nnknn":
        raise AssertionError("NN-kNN critic checkpoint reload did not preserve critic_type.")
    if "value_model" not in nnknn_loaded:
        raise AssertionError("NN-kNN critic checkpoint reload did not return the value critic.")
    if getattr(nnknn_loaded["value_model"], "case_entries", 0) <= 0:
        raise AssertionError("NN-kNN critic checkpoint did not preserve critic value cases.")

    mlp_cfg = make_nnknn_rl_config("smoke", seed=2, actor_type="mlp", critic_type="mlp")
    mlp_state = train_nnknn_rl("cartpole", mlp_cfg, progress=False)
    mlp_loaded = load_nnknn_rl_checkpoint(mlp_state["checkpoint_path"])
    if mlp_loaded["config"].actor_type != "mlp" or mlp_loaded["config"].critic_type != "mlp":
        raise AssertionError("MLP actor + MLP critic checkpoint reload did not preserve actor/critic types.")
    if mlp_state["summary"]["case_entries"] is not None or mlp_state["summary"]["action_counts"] is not None:
        raise AssertionError("MLP actor summary should not report NN-kNN actor cases.")
    mlp_eval = evaluate_nnknn_rl("cartpole", mlp_loaded["model"], episodes=mlp_cfg.eval_episodes, seed=mlp_cfg.eval_seed)
    if mlp_eval["episodes"] != mlp_cfg.eval_episodes:
        raise AssertionError("MLP actor checkpoint did not evaluate with the configured episode count.")

    mlp_nnknn_cfg = make_nnknn_rl_config("smoke", seed=3, actor_type="mlp", critic_type="nnknn")
    mlp_nnknn_state = train_nnknn_rl("cartpole", mlp_nnknn_cfg, progress=False)
    mlp_nnknn_loaded = load_nnknn_rl_checkpoint(mlp_nnknn_state["checkpoint_path"])
    if mlp_nnknn_loaded["config"].actor_type != "mlp" or mlp_nnknn_loaded["config"].critic_type != "nnknn":
        raise AssertionError("MLP actor + NN-kNN critic checkpoint reload did not preserve actor/critic types.")
    if getattr(mlp_nnknn_loaded["value_model"], "case_entries", 0) <= 0:
        raise AssertionError("MLP actor + NN-kNN critic checkpoint did not preserve critic value cases.")
    print("nnknn rl smoke ok")
    print(f"run_dir={state['run_dir']}")
    print(f"nnknn_critic_run_dir={nnknn_state['run_dir']}")
    print(f"mlp_actor_run_dir={mlp_state['run_dir']}")
    print(f"mlp_actor_nnknn_critic_run_dir={mlp_nnknn_state['run_dir']}")
    print(f"mean_return={float(final_eval['mean_return']):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke checks for Codex cloud environments.")
    parser.add_argument(
        "--mode",
        default="imports",
        choices=("imports", "train", "classification", "rl", "nec", "nnknn_rl"),
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
    elif args.mode == "nnknn_rl":
        run_nnknn_rl_smoke()
    else:
        run_import_smoke()


if __name__ == "__main__":
    main()
