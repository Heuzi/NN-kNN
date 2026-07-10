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
        NNKNNPolicyNetwork,
        NNKNNValueNetwork,
        SharedNNKNNActorCriticNetwork,
        _build_nnknn_rl_optimizer,
        _epsilon_mixed_policy_probs,
        _exploration_epsilon,
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

    boundary_advantages, _boundary_targets = compute_gae(
        rewards=[1.0, 1.0],
        values=[0.0, 0.0],
        next_values=[0.0, 0.0],
        terminated=[False, True],
        episode_boundaries=[True, True],
        gamma=0.99,
        gae_lambda=0.95,
    )
    if not torch.allclose(boundary_advantages, torch.tensor([1.0, 1.0]), atol=1e-5):
        raise AssertionError("NN-kNN-RL GAE leaked advantage across an episode boundary.")

    wrapper = NNKNNPolicyNetwork(4, 2, case_capacity=6, top_k=2, min_cases_per_action=1)
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

    actor_probe = NNKNNPolicyNetwork(2, 2, case_capacity=4, top_k=2, min_cases_per_action=1)
    actor_probe_device = next(actor_probe.parameters()).device
    actor_probe.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=actor_probe_device),
        torch.tensor([0, 1], dtype=torch.long, device=actor_probe_device),
    )
    actor_probe_optim = torch.optim.Adam(actor_probe.parameters(), lr=0.05)
    actor_bias_before = actor_probe.nnknn_model.biases[: actor_probe.case_entries].detach().clone()
    actor_probe_loss = -torch.log(
        actor_probe.policy_probs(torch.tensor([[0.2, 0.2]], dtype=torch.float32, device=actor_probe_device))[0, 1].clamp_min(1e-8)
    )
    actor_probe_optim.zero_grad()
    actor_probe_loss.backward()
    actor_probe_optim.step()
    actor_bias_after = actor_probe.nnknn_model.biases[: actor_probe.case_entries].detach().clone()
    if torch.allclose(actor_bias_before, actor_bias_after):
        raise AssertionError("NN-kNN actor parameters did not update under policy loss with an MLP critic path.")
    readiness_probe = NNKNNPolicyNetwork(2, 2, case_capacity=6, top_k=2, min_cases_per_action=2)
    readiness_device = next(readiness_probe.parameters()).device
    readiness_probe.add_cases(
        torch.tensor([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]], dtype=torch.float32, device=readiness_device),
        torch.tensor([0, 0, 1], dtype=torch.long, device=readiness_device),
    )
    if readiness_probe.is_policy_ready(min_case_entries=3):
        raise AssertionError("NN-kNN policy should wait for the configured per-action case floor.")
    readiness_probe.add_cases(
        torch.tensor([[1.1, 1.1]], dtype=torch.float32, device=readiness_device),
        torch.tensor([1], dtype=torch.long, device=readiness_device),
    )
    if not readiness_probe.is_policy_ready(min_case_entries=4):
        raise AssertionError("NN-kNN policy did not become ready after each action reached its case floor.")

    critic_probe = NNKNNValueNetwork(2, case_capacity=4, top_k=2)
    critic_probe_device = next(critic_probe.parameters()).device
    critic_probe.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=critic_probe_device),
        torch.tensor([0.0, 1.0], dtype=torch.float32, device=critic_probe_device),
    )
    critic_probe_optim = torch.optim.Adam(critic_probe.parameters(), lr=0.05)
    critic_bias_before = critic_probe.nnknn_model.biases[: critic_probe.case_entries].detach().clone()
    critic_probe_loss = torch.nn.functional.mse_loss(
        critic_probe(torch.tensor([[0.1, 0.1], [0.9, 0.9]], dtype=torch.float32, device=critic_probe_device)),
        torch.tensor([1.0, 0.0], dtype=torch.float32, device=critic_probe_device),
    )
    critic_probe_optim.zero_grad()
    critic_probe_loss.backward()
    critic_probe_optim.step()
    critic_bias_after = critic_probe.nnknn_model.biases[: critic_probe.case_entries].detach().clone()
    if torch.allclose(critic_bias_before, critic_bias_after):
        raise AssertionError("NN-kNN critic parameters did not update under value loss with an MLP actor path.")

    mutable_critic = NNKNNValueNetwork(
        2,
        case_capacity=4,
        top_k=1,
        mutable_value_labels=True,
        value_label_update_alpha=0.5,
        value_label_min_activation=None,
        value_label_distance_threshold=1e-8,
    )
    mutable_critic_device = next(mutable_critic.parameters()).device
    mutable_critic.add_cases(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=mutable_critic_device),
        torch.tensor([0.0], dtype=torch.float32, device=mutable_critic_device),
    )
    mutable_stats = mutable_critic.add_cases(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=mutable_critic_device),
        torch.tensor([10.0], dtype=torch.float32, device=mutable_critic_device),
    )
    if mutable_critic.case_entries != 1:
        raise AssertionError("Mutable NN-kNN critic should relabel an identical active case instead of appending it.")
    if mutable_stats["label_updates"] != 1 or mutable_stats["label_update_samples"] != 1:
        raise AssertionError("Mutable NN-kNN critic did not report the expected value-label relabel.")
    if not torch.allclose(
        mutable_critic.nnknn_model.labels[0, 0],
        torch.tensor(5.0, dtype=torch.float32, device=mutable_critic_device),
        atol=1e-5,
    ):
        raise AssertionError("Mutable NN-kNN critic did not smooth the existing value label toward the new target.")

    trainable_critic = NNKNNValueNetwork(2, case_capacity=4, top_k=1, trainable_value_labels=True)
    trainable_critic_device = next(trainable_critic.parameters()).device
    if not isinstance(trainable_critic.nnknn_model.labels, torch.nn.Parameter):
        raise AssertionError("Trainable NN-kNN critic labels should be registered as a Parameter.")
    trainable_critic.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=trainable_critic_device),
        torch.tensor([0.0, 1.0], dtype=torch.float32, device=trainable_critic_device),
    )
    trainable_critic_optim = torch.optim.Adam(trainable_critic.parameters(), lr=0.1)
    trainable_labels_before = trainable_critic.nnknn_model.labels[: trainable_critic.case_entries].detach().clone()
    trainable_loss = torch.nn.functional.mse_loss(
        trainable_critic(
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=trainable_critic_device)
        ),
        torch.tensor([10.0, -10.0], dtype=torch.float32, device=trainable_critic_device),
    )
    trainable_critic_optim.zero_grad()
    trainable_loss.backward()
    trainable_critic_optim.step()
    trainable_labels_after = trainable_critic.nnknn_model.labels[: trainable_critic.case_entries].detach().clone()
    if torch.allclose(trainable_labels_before, trainable_labels_after):
        raise AssertionError("Trainable NN-kNN critic labels did not update under value loss.")
    grouped_critic_optim = _build_nnknn_rl_optimizer(
        trainable_critic,
        base_lr=0.03,
        case_lr=0.007,
    )
    label_group_lr = None
    bias_group_lr = None
    for group in grouped_critic_optim.param_groups:
        if any(param is trainable_critic.nnknn_model.labels for param in group["params"]):
            label_group_lr = float(group["lr"])
        if any(param is trainable_critic.nnknn_model.biases for param in group["params"]):
            bias_group_lr = float(group["lr"])
    if label_group_lr != 0.007 or bias_group_lr != 0.007:
        raise AssertionError("Trainable critic labels should share the case-level learning rate with case biases.")

    shared_probe = SharedNNKNNActorCriticNetwork(2, 2, case_capacity=3, top_k=2, min_cases_per_action=1)
    shared_probe_device = next(shared_probe.parameters()).device
    first_insert = shared_probe.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32, device=shared_probe_device),
        torch.tensor([0, 1, 0], dtype=torch.long, device=shared_probe_device),
    )
    first_case_ids = first_insert["case_ids"]
    with torch.no_grad():
        shared_probe.biases[: shared_probe.case_entries].copy_(torch.tensor([-3.0, -2.0, -1.0]))
    second_insert = shared_probe.add_cases(
        torch.tensor([[3.0, 3.0]], dtype=torch.float32, device=shared_probe_device),
        torch.tensor([0], dtype=torch.long, device=shared_probe_device),
        protected_case_ids=first_case_ids[1:],
    )
    active_case_ids = shared_probe.case_ids[: shared_probe.case_entries].detach().cpu().tolist()
    expected_case_ids = [first_case_ids[1], first_case_ids[2], second_insert["case_ids"][0]]
    if active_case_ids != expected_case_ids:
        raise AssertionError("Shared NN-kNN actor-critic did not preserve protected shared cases across compaction.")
    shared_probe.update_value_labels(
        expected_case_ids,
        torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32, device=shared_probe_device),
    )
    if not torch.allclose(
        shared_probe.value_labels[: shared_probe.case_entries],
        torch.tensor([1.1, 2.2, 3.3], dtype=torch.float32, device=shared_probe_device),
    ):
        raise AssertionError("Shared NN-kNN actor-critic did not keep separate value labels over the shared case base.")

    shared_hybrid_probe = SharedNNKNNActorCriticNetwork(
        2,
        2,
        case_capacity=4,
        top_k=2,
        min_cases_per_action=1,
        mutable_value_labels=True,
        trainable_value_labels=True,
        value_label_update_alpha=0.5,
        value_label_min_activation=None,
        value_label_distance_threshold=1e-8,
    )
    shared_hybrid_device = next(shared_hybrid_probe.parameters()).device
    if not isinstance(shared_hybrid_probe.value_labels, torch.nn.Parameter):
        raise AssertionError("Shared trainable value labels should be registered as a Parameter.")
    hybrid_first = shared_hybrid_probe.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=shared_hybrid_device),
        torch.tensor([0, 1], dtype=torch.long, device=shared_hybrid_device),
    )
    shared_hybrid_probe.update_value_labels(
        hybrid_first["case_ids"],
        torch.tensor([0.0, 1.0], dtype=torch.float32, device=shared_hybrid_device),
    )
    hybrid_second = shared_hybrid_probe.add_cases(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=shared_hybrid_device),
        torch.tensor([1], dtype=torch.long, device=shared_hybrid_device),
    )
    hybrid_update_stats = shared_hybrid_probe.update_value_labels(
        hybrid_second["case_ids"],
        torch.tensor([10.0], dtype=torch.float32, device=shared_hybrid_device),
        observations=torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=shared_hybrid_device),
    )
    if hybrid_update_stats["label_updates"] < 1 or hybrid_update_stats["label_update_samples"] != 1:
        raise AssertionError("Shared hybrid NN-kNN did not relabel a similar existing value label.")
    if not torch.allclose(
        shared_hybrid_probe.value_labels[0],
        torch.tensor(5.0, dtype=torch.float32, device=shared_hybrid_device),
        atol=1e-5,
    ):
        raise AssertionError("Shared hybrid NN-kNN did not smooth a similar existing value label.")
    if not torch.allclose(
        shared_hybrid_probe.value_labels[2],
        torch.tensor(10.0, dtype=torch.float32, device=shared_hybrid_device),
        atol=1e-5,
    ):
        raise AssertionError("Shared hybrid NN-kNN did not preserve the direct value target for the rollout case.")

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
    if cfg.exploration_initial_epsilon != 1.0 or cfg.exploration_final_epsilon != 0.05:
        raise AssertionError("NN-kNN-RL should default to epsilon-mixed stochastic exploration.")
    if abs(_exploration_epsilon(cfg, 0) - cfg.exploration_initial_epsilon) > 1e-12:
        raise AssertionError("NN-kNN-RL exploration schedule should start at initial epsilon.")
    if abs(_exploration_epsilon(cfg, cfg.total_timesteps) - cfg.exploration_final_epsilon) > 1e-12:
        raise AssertionError("NN-kNN-RL exploration schedule should end at final epsilon.")
    if cfg.shared_target_value_mode != "hard":
        raise AssertionError("NN-kNN-RL should default shared NN-kNN target value updates to hard sync.")
    if cfg.critic_mutable_value_labels or cfg.critic_trainable_value_labels:
        raise AssertionError("NN-kNN-RL should default to fixed critic value labels.")
    hybrid_cfg = make_nnknn_rl_config(
        "smoke",
        critic_type="nnknn",
        critic_mutable_value_labels="true",
        critic_trainable_value_labels=True,
        critic_value_label_min_activation=None,
        case_learning_rate=7e-4,
    )
    if not hybrid_cfg.critic_mutable_value_labels or not hybrid_cfg.critic_trainable_value_labels:
        raise AssertionError("NN-kNN-RL config did not enable the requested hybrid value-label mode.")
    if hybrid_cfg.critic_value_label_min_activation is not None:
        raise AssertionError("NN-kNN-RL config should allow disabling activation-based mutable label matching.")
    if hybrid_cfg.case_learning_rate != 7e-4:
        raise AssertionError("NN-kNN-RL config did not preserve the case-level learning rate.")
    for invalid_overrides in (
        {"critic_value_label_update_alpha": 0.0},
        {"critic_value_label_min_activation": -1.0},
        {"critic_value_label_distance_threshold": -1.0},
        {"case_learning_rate": 0.0},
        {"exploration_initial_epsilon": -0.1},
        {"exploration_final_epsilon": 1.1},
        {"exploration_initial_epsilon": 0.1, "exploration_final_epsilon": 0.2},
        {"exploration_fraction": 1.1},
        {"min_case_entries": 0},
        {"min_cases_per_action": 0},
    ):
        try:
            make_nnknn_rl_config("smoke", **invalid_overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f"NN-kNN-RL should reject invalid value-label config {invalid_overrides}.")
    try:
        make_nnknn_rl_config("smoke", reward_shaping="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("NN-kNN-RL should reject unknown reward_shaping values.")

    mlp_wrapper = MLPPolicyNetwork(4, 2, hidden_sizes=(8,))
    mlp_probs = mlp_wrapper.policy_probs(torch.zeros(3, 4))
    if mlp_probs.shape != (3, 2) or not torch.allclose(mlp_probs.sum(dim=1), torch.ones(3), atol=1e-5):
        raise AssertionError("MLP actor did not return normalized action probabilities.")
    mixed_probs = _epsilon_mixed_policy_probs(torch.tensor([[0.9, 0.1]], dtype=torch.float32), 0.2)
    if not torch.allclose(mixed_probs, torch.tensor([[0.82, 0.18]], dtype=torch.float32), atol=1e-6):
        raise AssertionError("NN-kNN-RL epsilon-mixed behavior probabilities are incorrect.")

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
    exploration_summary = state["summary"].get("exploration", {})
    if exploration_summary.get("initial_epsilon") != cfg.exploration_initial_epsilon:
        raise AssertionError("NN-kNN actor smoke summary did not record exploration configuration.")
    if not state["training_metrics"] or "behavior_epsilon" not in state["training_metrics"][0]:
        raise AssertionError("NN-kNN actor training metrics did not record behavior epsilon.")
    final_loss_row = state["loss_metrics"][-1]
    if state["summary"]["partial_rollout_samples"] > 0 and (
        int(final_loss_row["global_step"]) != cfg.total_timesteps
        or int(final_loss_row["partial_rollout_samples"]) != state["summary"]["partial_rollout_samples"]
    ):
        raise AssertionError("NN-kNN actor final partial rollout cases were not included in policy updates.")
    partial_cfg = make_nnknn_rl_config(
        "smoke",
        seed=42,
        total_timesteps=1,
        eval_episodes=1,
        policy_update_episodes=10,
    )
    partial_state = train_nnknn_rl("cartpole", partial_cfg, progress=False)
    if partial_state["summary"]["partial_rollout_samples"] != 1:
        raise AssertionError("NN-kNN actor one-step run should train exactly one final partial rollout sample.")
    if int(partial_state["loss_metrics"][-1]["global_step"]) != partial_cfg.total_timesteps:
        raise AssertionError("NN-kNN actor one-step partial rollout was not updated at the fixed step budget.")
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
    if not isinstance(nnknn_state["model"], SharedNNKNNActorCriticNetwork):
        raise AssertionError("NN-kNN actor + NN-kNN critic should use the shared actor-critic model.")
    if nnknn_state["model"] is not nnknn_state["value_model"]:
        raise AssertionError("Shared NN-kNN actor-critic should be returned as both actor and critic.")
    nnknn_loaded = load_nnknn_rl_checkpoint(nnknn_state["checkpoint_path"])
    if nnknn_loaded["config"].critic_type != "nnknn":
        raise AssertionError("NN-kNN critic checkpoint reload did not preserve critic_type.")
    if "value_model" not in nnknn_loaded:
        raise AssertionError("NN-kNN critic checkpoint reload did not return the value critic.")
    if getattr(nnknn_loaded["value_model"], "case_entries", 0) <= 0:
        raise AssertionError("NN-kNN critic checkpoint did not preserve critic value cases.")
    shared_loss_samples = sum(int(row["samples"]) for row in nnknn_state["loss_metrics"])
    if nnknn_state["summary"]["partial_rollout_samples"] <= 0:
        raise AssertionError("Shared NN-kNN smoke should train a final partial rollout.")
    if nnknn_state["summary"]["shared_value_labels_written"] != shared_loss_samples:
        raise AssertionError("Shared NN-kNN did not write one value label per updated shared rollout case.")
    if nnknn_state["summary"]["shared_target_value_mode"] != "hard":
        raise AssertionError("Shared NN-kNN actor-critic summary should record hard target sync mode by default.")
    if nnknn_state["summary"]["shared_target_value_syncs"] <= 0:
        raise AssertionError("Shared NN-kNN actor-critic should record at least one target value sync.")

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

    mlp_nnknn_capacity_cfg = make_nnknn_rl_config(
        "smoke",
        seed=6,
        actor_type="mlp",
        critic_type="nnknn",
        case_capacity=32,
        total_timesteps=96,
        policy_update_episodes=1,
        eval_episodes=1,
    )
    mlp_nnknn_capacity_state = train_nnknn_rl("cartpole", mlp_nnknn_capacity_cfg, progress=False)
    if mlp_nnknn_capacity_state["summary"]["critic_case_entries"] != 32:
        raise AssertionError("Small-capacity NN-kNN critic should fill to its configured capacity.")
    if (
        mlp_nnknn_capacity_state["summary"]["critic_cases_pruned"]
        + mlp_nnknn_capacity_state["summary"]["critic_cases_replaced"]
        <= 0
    ):
        raise AssertionError("Small-capacity NN-kNN critic maintenance was not reported.")

    shared_capacity_cfg = make_nnknn_rl_config(
        "smoke",
        seed=4,
        critic_type="nnknn",
        case_capacity=64,
        total_timesteps=128,
        policy_update_episodes=2,
    )
    shared_capacity_state = train_nnknn_rl("cartpole", shared_capacity_cfg, progress=False)
    if not isinstance(shared_capacity_state["model"], SharedNNKNNActorCriticNetwork):
        raise AssertionError("Small-capacity shared NN-kNN smoke should still build the shared actor-critic path.")
    if shared_capacity_state["summary"]["critic_case_entries"] <= 0:
        raise AssertionError("Shared NN-kNN actor-critic small-capacity smoke did not keep critic value labels active.")
    if (
        shared_capacity_state["summary"]["shared_cases_pruned"]
        + shared_capacity_state["summary"]["shared_cases_replaced"]
        <= 0
    ):
        raise AssertionError("Small-capacity shared NN-kNN actor-critic maintenance was not reported.")
    shared_capacity_loss_samples = sum(int(row["samples"]) for row in shared_capacity_state["loss_metrics"])
    if shared_capacity_state["summary"]["shared_value_labels_written"] != shared_capacity_loss_samples:
        raise AssertionError("Small-capacity shared NN-kNN did not write value labels for updated cases.")

    shared_ema_cfg = make_nnknn_rl_config(
        "smoke",
        seed=5,
        critic_type="nnknn",
        case_capacity=128,
        total_timesteps=96,
        policy_update_episodes=2,
        shared_target_value_mode="ema",
        shared_target_sync_interval=1,
        shared_target_ema_tau=0.5,
    )
    shared_ema_state = train_nnknn_rl("cartpole", shared_ema_cfg, progress=False)
    if shared_ema_state["summary"]["shared_target_value_mode"] != "ema":
        raise AssertionError("Shared NN-kNN EMA smoke should record EMA target value mode.")
    if shared_ema_state["summary"]["shared_target_value_syncs"] <= 0:
        raise AssertionError("Shared NN-kNN EMA smoke should perform target value syncs.")
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
