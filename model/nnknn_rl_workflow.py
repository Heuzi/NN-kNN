from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets.rl_tasks import get_rl_task_spec
from model.nnknn_model import GlocalFeatureWeight, NN_KNN_Model, normalize_cases
from model.rl_workflow import (
    _build_training_efficiency,
    _first_threshold_step,
    _json_default,
    _linear_schedule,
    _make_env,
    _resolve_device_arg,
    _validate_env_spaces,
    seed_everything,
)

ALGORITHM_NAME = "nnknn_actor_mlp_value_gae"


@dataclass(frozen=True)
class NNKNNRLConfig:
    """Training configuration for the repo-native NN-kNN actor-critic workflow."""

    profile: str = "fast"
    seed: int = 0
    total_timesteps: int = 150_000
    learning_rate: float = 5e-4
    actor_type: str = "nnknn"
    actor_hidden_sizes: tuple[int, ...] = (128, 128)
    critic_learning_rate: float = 1e-3
    critic_type: str = "mlp"
    critic_hidden_sizes: tuple[int, ...] = (128, 128)
    critic_update_epochs: int = 1
    critic_diagnostic_episode_frequency: int = 10
    critic_nnknn_config: dict[str, Any] = field(default_factory=dict)
    critic_case_capacity: int | None = None
    critic_mutable_value_labels: bool = False
    critic_trainable_value_labels: bool = False
    critic_value_label_update_alpha: float = 0.25
    critic_value_label_min_activation: float | None = 0.5
    critic_value_label_distance_threshold: float | None = 1e-6
    critic_value_label_append_on_no_match: bool = True
    case_capacity: int = 10_000
    max_grad_norm: float = 10.0
    min_case_entries: int = 32
    eval_frequency: int = 0
    eval_episode_frequency: int = 100
    eval_episodes: int = 20
    eval_seed: int = 10_000
    success_threshold: float | None = 475.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_update_episodes: int = 4
    exploration_initial_epsilon: float = 1.0
    exploration_final_epsilon: float = 0.05
    exploration_fraction: float = 0.5
    entropy_coef: float = 0.01
    advantage_epsilon: float = 1e-8
    advantage_clip: float = 5.0
    value_loss_coef: float = 1.0
    reward_shaping: str | None = None
    nnknn_config: dict[str, Any] = field(default_factory=dict)
    use_glocal_weightor: bool = True
    glocal_fw_set_num: int = 1
    tau: float = 1.0
    top_k: int = 10
    case_default_bias: float = 0.0
    case_bias_l2: float = 1e-4
    case_learning_rate: float | None = None
    case_maintenance_frequency: int = 1_000
    case_prune_quantile: float = 0.05
    case_prune_bias_threshold: float | None = None
    min_cases_per_action: int = 8
    shared_target_value_mode: str = "hard"
    shared_target_sync_interval: int = 1
    shared_target_ema_tau: float = 0.05
    source_reference: str = "NN-kNN actor with selectable value critic and GAE advantages"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NNKNNPolicyNetwork(nn.Module):
    """NN-kNN policy actor for discrete-action actor-critic RL.

    The inner NN-kNN model stores cases as state -> one-hot(action). Its
    classification output is interpreted as pi(a | s), not as a Q-value.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        case_capacity: int,
        tau: float = 1.0,
        top_k: int = 10,
        case_default_bias: float = 0.0,
        min_cases_per_action: int = 1,
        nnknn_config: dict[str, Any] | None = None,
        use_glocal_weightor: bool = True,
        glocal_fw_set_num: int = 1,
        mutable_value_labels: bool = False,
        trainable_value_labels: bool = False,
        value_label_update_alpha: float = 0.25,
        value_label_min_activation: float | None = 0.5,
        value_label_distance_threshold: float | None = 1e-6,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.case_capacity = int(case_capacity)
        self.min_cases_per_action = int(min_cases_per_action)
        self.tau = float(tau)
        self.top_k = int(top_k)
        self.case_default_bias = float(case_default_bias)
        self.use_glocal_weightor = bool(use_glocal_weightor)
        self.glocal_fw_set_num = int(glocal_fw_set_num)
        self._prune_quantile = 0.0
        self._prune_bias_threshold: float | None = None

        cases = torch.zeros(self.case_capacity, self.obs_dim, dtype=torch.float32)
        labels = torch.zeros(self.case_capacity, self.action_dim, dtype=torch.float32)
        glocal_weightor = (
            GlocalFeatureWeight(self.obs_dim, self.glocal_fw_set_num) if self.use_glocal_weightor else None
        )
        model_config = {
            "task_type": "classification",
            "normalize_over_cases": True,
            "case_score_mode": "bias_minus_distance",
            "case_normalizer": "softmax",
            "pre_topk_mask": True,
            "top_k": self.top_k,
            "tau": self.tau,
            "bias_manual_set": True,
            "bias_manual_value": self.case_default_bias,
            "ignore_identical_in_training": False,
            "explanation_mode": True,
            "active_case_count": 0,
            "glocal_fw_set_num": self.glocal_fw_set_num,
        }
        model_config.update(nnknn_config or {})
        model_config["task_type"] = "classification"
        model_config["normalize_over_cases"] = True
        model_config["case_score_mode"] = "bias_minus_distance"
        model_config["case_normalizer"] = "softmax"
        model_config["pre_topk_mask"] = True
        model_config["top_k"] = self.top_k
        model_config["tau"] = self.tau
        model_config["bias_manual_set"] = True
        model_config["bias_manual_value"] = self.case_default_bias
        model_config["ignore_identical_in_training"] = False
        model_config["explanation_mode"] = True
        model_config["glocal_fw_set_num"] = self.glocal_fw_set_num
        model_config["active_case_count"] = 0
        self.nnknn_config = dict(model_config)
        self.nnknn_model = NN_KNN_Model(
            cases,
            labels,
            feature_extractor=None,
            glocal_weightor=glocal_weightor,
            **model_config,
        )
        self.nnknn_model.to(self.nnknn_model.cases.device)

    @property
    def case_entries(self) -> int:
        return self.nnknn_model.case_count()

    def case_state(self) -> dict[str, Any]:
        return {
            "case_entries": int(self.case_entries),
            "action_counts": self.action_counts().detach().cpu().tolist(),
        }

    def load_case_state(self, state: dict[str, Any]) -> None:
        self.nnknn_model.set_active_case_count(int(state.get("case_entries", 0)))

    def action_tensor(self) -> torch.Tensor:
        if self.case_entries <= 0:
            return torch.zeros(0, dtype=torch.long, device=self.nnknn_model.labels.device)
        return self.nnknn_model.labels[: self.case_entries].argmax(dim=1)

    def action_counts(self) -> torch.Tensor:
        actions = self.action_tensor()
        return torch.bincount(actions, minlength=self.action_dim)

    def is_policy_ready(self, min_case_entries: int = 1) -> bool:
        if self.case_entries < int(min_case_entries):
            return False
        return bool(torch.all(self.action_counts() >= self.min_cases_per_action).detach().cpu().item())

    def policy_probs(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        observations = observations.to(next(self.parameters()).device, dtype=torch.float32)
        batch_size = observations.shape[0]
        if self.case_entries <= 0:
            return torch.full(
                (batch_size, self.action_dim),
                1.0 / float(self.action_dim),
                dtype=observations.dtype,
                device=observations.device,
            )
        final_predictions, _predicted, *_ = self.nnknn_model(observations)
        probs = final_predictions[:, : self.action_dim].clamp_min(0.0)
        denom = probs.sum(dim=1, keepdim=True)
        uniform = torch.full_like(probs, 1.0 / float(self.action_dim))
        return torch.where(denom > 0, probs / denom.clamp_min(1e-12), uniform)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy_probs(observations)

    def add_cases(self, observations: torch.Tensor | np.ndarray, actions: torch.Tensor | np.ndarray) -> dict[str, int]:
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.nnknn_model.cases.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.nnknn_model.labels.device).view(-1)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.shape[0] != actions_t.shape[0]:
            raise ValueError("observations and actions must have the same batch size")
        if torch.any(actions_t < 0) or torch.any(actions_t >= self.action_dim):
            raise ValueError(f"actions must be integer ids in [0, {self.action_dim})")
        available = self.case_capacity - self.case_entries
        if obs_t.shape[0] <= available:
            labels = F.one_hot(actions_t, num_classes=self.action_dim).to(dtype=torch.float32)
            start = self.case_entries
            added = self.nnknn_model.append_cases(obs_t, labels)
            return {"added": added, "pruned": 0, "replaced": 0, "start": start, "end": start + added}
        added = 0
        replaced = 0
        pruned = 0
        first_added_index: int | None = None
        for idx in range(obs_t.shape[0]):
            if self.case_entries >= self.case_capacity:
                pruned += self.prune_cases(force=True)
            if self.case_entries >= self.case_capacity:
                replace_idx = self._lowest_replaceable_case_index()
                if replace_idx is None:
                    continue
                keep_indices = [i for i in range(self.case_entries) if i != replace_idx]
                self.nnknn_model.compact_cases(keep_indices)
                replaced += 1
            label = F.one_hot(actions_t[idx], num_classes=self.action_dim).to(dtype=torch.float32).unsqueeze(0)
            start = self.case_entries
            self.nnknn_model.append_cases(obs_t[idx].unsqueeze(0), label)
            if first_added_index is None:
                first_added_index = start
            added += 1
        start = self.case_entries - added if first_added_index is None else first_added_index
        return {"added": added, "pruned": pruned, "replaced": replaced, "start": start, "end": start + added}

    def _lowest_replaceable_case_index(self) -> int | None:
        if self.case_entries <= 0:
            return None
        actions = self.action_tensor()
        counts = self.action_counts()
        biases = self.nnknn_model.biases[: self.case_entries].detach()
        sorted_indices = torch.argsort(biases)
        for idx_t in sorted_indices:
            idx = int(idx_t.item())
            action = int(actions[idx].item())
            if int(counts[action].item()) > self.min_cases_per_action:
                return idx
        return None

    def prune_cases(self, *, force: bool = False) -> int:
        active_count = self.case_entries
        if active_count <= 0:
            return 0
        biases = self.nnknn_model.biases[:active_count].detach()
        thresholds: list[torch.Tensor] = []
        if self._prune_quantile > 0.0:
            thresholds.append(torch.quantile(biases, min(max(self._prune_quantile, 0.0), 1.0)))
        if self._prune_bias_threshold is not None:
            thresholds.append(torch.as_tensor(self._prune_bias_threshold, device=biases.device, dtype=biases.dtype))
        remove_candidates: torch.Tensor
        if thresholds:
            threshold = torch.stack(thresholds).max()
            remove_candidates = torch.nonzero(biases < threshold, as_tuple=False).view(-1)
        elif force:
            remove_candidates = torch.argsort(biases)[:1]
        else:
            return 0
        if remove_candidates.numel() == 0:
            return 0

        actions = self.action_tensor()
        counts = self.action_counts().clone()
        candidate_biases = biases[remove_candidates]
        candidate_order = remove_candidates[torch.argsort(candidate_biases)]
        remove: list[int] = []
        for idx_t in candidate_order:
            idx = int(idx_t.item())
            action = int(actions[idx].item())
            if int(counts[action].item()) <= self.min_cases_per_action:
                continue
            remove.append(idx)
            counts[action] -= 1
            if force and not thresholds:
                break
        if not remove:
            return 0
        remove_set = set(remove)
        keep_indices = [idx for idx in range(active_count) if idx not in remove_set]
        return self.nnknn_model.compact_cases(keep_indices)

    def configure_case_maintenance(self, *, prune_quantile: float, prune_bias_threshold: float | None) -> None:
        self._prune_quantile = float(prune_quantile)
        self._prune_bias_threshold = None if prune_bias_threshold is None else float(prune_bias_threshold)

    def case_bias_stats(self) -> dict[str, float | None]:
        if self.case_entries <= 0:
            return {"bias_min": None, "bias_mean": None, "bias_max": None}
        biases = self.nnknn_model.biases[: self.case_entries].detach()
        return {
            "bias_min": float(biases.min().cpu().item()),
            "bias_mean": float(biases.mean().cpu().item()),
            "bias_max": float(biases.max().cpu().item()),
        }

    def explain(self, observation: torch.Tensor | np.ndarray, action: int, *, k: int | None = None) -> dict[str, Any]:
        if self.case_entries <= 0:
            return {"case_entries": 0, "neighbors": []}
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).view(1, self.obs_dim)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            probs, _pred, _pre, top_cases, top_labels, top_weights = self.nnknn_model(obs_t)
        if was_training:
            self.train()
        neighbors = []
        if top_cases is not None and top_labels is not None and top_weights is not None:
            k_eff = min(int(k or self.top_k), top_cases.shape[1])
            for rank in range(k_eff):
                label = top_labels[0, rank]
                label_action = int(label.argmax().detach().cpu().item()) if label.dim() > 0 else int(label.item())
                neighbors.append(
                    {
                        "rank": rank + 1,
                        "action": label_action,
                        "weight": float(top_weights[0, rank].detach().cpu().item()),
                        "observation": top_cases[0, rank].detach().cpu().numpy().tolist(),
                    }
                )
        return {
            "case_entries": int(self.case_entries),
            "query_action": int(action),
            "probability": float(probs[0, int(action)].detach().cpu().item()),
            "neighbors": neighbors,
        }


class MLPPolicyNetwork(nn.Module):
    """Small MLP policy actor for discrete-action actor-critic baselines."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        layers: list[nn.Module] = []
        in_features = int(obs_dim)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, int(hidden_size)))
            layers.append(nn.ReLU())
            in_features = int(hidden_size)
        layers.append(nn.Linear(in_features, int(action_dim)))
        self.net = nn.Sequential(*layers)

    @property
    def case_entries(self) -> None:
        return None

    def is_policy_ready(self, min_case_entries: int = 1) -> bool:
        return True

    def policy_probs(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        logits = self.net(observations.float())
        return F.softmax(logits, dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy_probs(observations)


class ValueNetwork(nn.Module):
    """Small MLP critic that predicts V(s)."""

    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = int(obs_dim)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, int(hidden_size)))
            layers.append(nn.ReLU())
            in_features = int(hidden_size)
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        values = self.net(observations.float())
        return values.squeeze(-1)


class NNKNNValueNetwork(nn.Module):
    """NN-kNN regression critic that predicts V(s)."""

    def __init__(
        self,
        obs_dim: int,
        *,
        case_capacity: int,
        tau: float = 1.0,
        top_k: int = 10,
        case_default_bias: float = 0.0,
        nnknn_config: dict[str, Any] | None = None,
        use_glocal_weightor: bool = True,
        glocal_fw_set_num: int = 1,
        mutable_value_labels: bool = False,
        trainable_value_labels: bool = False,
        value_label_update_alpha: float = 0.25,
        value_label_min_activation: float | None = 0.5,
        value_label_distance_threshold: float | None = 1e-6,
        value_label_append_on_no_match: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.case_capacity = int(case_capacity)
        self.tau = float(tau)
        self.top_k = int(top_k)
        self.case_default_bias = float(case_default_bias)
        self.use_glocal_weightor = bool(use_glocal_weightor)
        self.glocal_fw_set_num = int(glocal_fw_set_num)
        self.mutable_value_labels = bool(mutable_value_labels)
        self.trainable_value_labels = bool(trainable_value_labels)
        self.value_label_update_alpha = float(value_label_update_alpha)
        self.value_label_min_activation = (
            None if value_label_min_activation is None else float(value_label_min_activation)
        )
        self.value_label_distance_threshold = (
            None if value_label_distance_threshold is None else float(value_label_distance_threshold)
        )
        self.value_label_append_on_no_match = bool(value_label_append_on_no_match)
        if not (0.0 < self.value_label_update_alpha <= 1.0):
            raise ValueError("value_label_update_alpha must be in (0, 1]")
        if self.value_label_min_activation is not None and self.value_label_min_activation < 0.0:
            raise ValueError("value_label_min_activation must be None or non-negative")
        if self.value_label_distance_threshold is not None and self.value_label_distance_threshold < 0.0:
            raise ValueError("value_label_distance_threshold must be None or non-negative")
        self._prune_quantile = 0.0
        self._prune_bias_threshold: float | None = None

        cases = torch.zeros(self.case_capacity, self.obs_dim, dtype=torch.float32)
        labels = torch.zeros(self.case_capacity, 1, dtype=torch.float32)
        glocal_weightor = (
            GlocalFeatureWeight(self.obs_dim, self.glocal_fw_set_num) if self.use_glocal_weightor else None
        )
        model_config = {
            "task_type": "regression",
            "normalize_over_cases": True,
            "case_score_mode": "bias_minus_distance",
            "case_normalizer": "softmax",
            "pre_topk_mask": True,
            "top_k": self.top_k,
            "tau": self.tau,
            "bias_manual_set": True,
            "bias_manual_value": self.case_default_bias,
            "ignore_identical_in_training": False,
            "explanation_mode": False,
            "active_case_count": 0,
            "glocal_fw_set_num": self.glocal_fw_set_num,
        }
        model_config.update(nnknn_config or {})
        model_config["task_type"] = "regression"
        model_config["normalize_over_cases"] = True
        model_config["case_score_mode"] = "bias_minus_distance"
        model_config["case_normalizer"] = "softmax"
        model_config["pre_topk_mask"] = True
        model_config["top_k"] = self.top_k
        model_config["tau"] = self.tau
        model_config["bias_manual_set"] = True
        model_config["bias_manual_value"] = self.case_default_bias
        model_config["ignore_identical_in_training"] = False
        model_config["glocal_fw_set_num"] = self.glocal_fw_set_num
        model_config["active_case_count"] = 0
        self.nnknn_config = dict(model_config)
        self.nnknn_model = NN_KNN_Model(
            cases,
            labels,
            feature_extractor=None,
            glocal_weightor=glocal_weightor,
            **model_config,
        )
        if self.trainable_value_labels:
            labels_param = nn.Parameter(self.nnknn_model.labels.detach().clone())
            del self.nnknn_model._buffers["labels"]
            self.nnknn_model.register_parameter("labels", labels_param)
        self.nnknn_model.to(self.nnknn_model.cases.device)

    @property
    def case_entries(self) -> int:
        return self.nnknn_model.case_count()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        observations = observations.to(next(self.parameters()).device, dtype=torch.float32)
        if self.case_entries <= 0:
            return torch.zeros(observations.shape[0], dtype=observations.dtype, device=observations.device)
        final_predictions, _predicted, *_ = self.nnknn_model(observations)
        return final_predictions.view(observations.shape[0], -1)[:, 0]

    def _case_distances_and_weights(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        observations = observations.to(self.nnknn_model.cases.device, dtype=torch.float32)
        active_count = self.case_entries
        if active_count <= 0:
            raise ValueError("NN-kNN value critic requires at least one active case")
        case_features = self.nnknn_model.cases[:active_count]
        query_expanded = observations.unsqueeze(1).expand(-1, active_count, -1)
        case_expanded = case_features.unsqueeze(0).expand(observations.shape[0], -1, -1)
        elementwise_distance = (query_expanded - case_expanded) ** 2
        if self.nnknn_model.glocal_weightor is not None:
            glocal_weights = self.nnknn_model.glocal_weights[:active_count]
            elementwise_distance = self.nnknn_model.glocal_weightor(elementwise_distance, glocal_weights)
        distances = torch.sqrt(torch.relu(elementwise_distance.sum(dim=-1)))
        mode = str(self.nnknn_model.config.get("case_score_mode", "bias_minus_distance"))
        if mode == "hard_knn":
            k_eff = min(int(self.top_k), active_count)
            _, top_idx = torch.topk(-distances, k=k_eff, dim=1)
            weights = torch.zeros_like(distances)
            weights.scatter_(1, top_idx, 1.0 / float(k_eff))
            return distances, weights
        scores = -distances if mode == "neg_distance" else self.nnknn_model.biases[:active_count].unsqueeze(0) - distances
        if bool(self.nnknn_model.config.get("pre_topk_mask", False)):
            k_eff = min(int(self.top_k), active_count)
            top_vals, top_idx = torch.topk(scores, k=k_eff, dim=1)
            fill_val = float("-inf") if self.nnknn_model.case_normalizer == "softmax" else -1e9
            masked_scores = torch.full_like(scores, fill_val)
            scores = masked_scores.scatter(1, top_idx, top_vals)
        weights = normalize_cases(scores, normalizer=self.nnknn_model.case_normalizer, tau=self.tau, dim=1)
        return distances, weights

    def update_similar_value_labels(
        self,
        observations: torch.Tensor,
        values: torch.Tensor,
        *,
        exclude_positions: torch.Tensor | None = None,
    ) -> dict[str, int]:
        if self.case_entries <= 0:
            return {"label_updates": 0, "label_update_samples": 0}
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.nnknn_model.labels.device).view(-1)
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.nnknn_model.cases.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.shape[0] != values_t.numel():
            raise ValueError("observations and values must have the same batch size")
        with torch.no_grad():
            distances, weights = self._case_distances_and_weights(obs_t)
            match_mask = torch.zeros_like(weights, dtype=torch.bool)
            if self.value_label_min_activation is not None:
                match_mask |= weights >= float(self.value_label_min_activation)
            if self.value_label_distance_threshold is not None:
                match_mask |= distances <= float(self.value_label_distance_threshold)
            if exclude_positions is not None and exclude_positions.numel():
                exclude_t = exclude_positions.to(match_mask.device, dtype=torch.long).view(-1)
                if exclude_t.numel() == obs_t.shape[0]:
                    match_mask[torch.arange(obs_t.shape[0], device=match_mask.device), exclude_t] = False
                else:
                    match_mask[:, exclude_t] = False

            labels = self.nnknn_model.labels[: self.case_entries].view(-1)
            alpha = float(self.value_label_update_alpha)
            label_updates = 0
            label_update_samples = 0
            for row_idx in range(obs_t.shape[0]):
                positions = torch.nonzero(match_mask[row_idx], as_tuple=False).view(-1)
                if positions.numel() == 0:
                    continue
                label_update_samples += 1
                labels[positions] = labels[positions] * (1.0 - alpha) + values_t[row_idx] * alpha
                label_updates += int(positions.numel())
        return {"label_updates": label_updates, "label_update_samples": label_update_samples}

    def add_cases(self, observations: torch.Tensor | np.ndarray, values: torch.Tensor | np.ndarray) -> dict[str, int]:
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.nnknn_model.cases.device)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.nnknn_model.labels.device).view(-1, 1)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.shape[0] != values_t.shape[0]:
            raise ValueError("observations and values must have the same batch size")
        mutable_stats = {"label_updates": 0, "label_update_samples": 0}
        if self.mutable_value_labels and self.case_entries > 0:
            mutable_stats = self.update_similar_value_labels(obs_t, values_t.view(-1))
            if self.value_label_append_on_no_match:
                with torch.no_grad():
                    distances, weights = self._case_distances_and_weights(obs_t)
                    match_mask = torch.zeros_like(weights, dtype=torch.bool)
                    if self.value_label_min_activation is not None:
                        match_mask |= weights >= float(self.value_label_min_activation)
                    if self.value_label_distance_threshold is not None:
                        match_mask |= distances <= float(self.value_label_distance_threshold)
                    append_mask = ~match_mask.any(dim=1)
                obs_t = obs_t[append_mask]
                values_t = values_t[append_mask]
            else:
                obs_t = obs_t[:0]
                values_t = values_t[:0]
        if obs_t.shape[0] == 0:
            return {
                "added": 0,
                "pruned": 0,
                "replaced": 0,
                "start": self.case_entries,
                "end": self.case_entries,
                **mutable_stats,
            }
        available = self.case_capacity - self.case_entries
        if obs_t.shape[0] <= available:
            start = self.case_entries
            added = self.nnknn_model.append_cases(obs_t, values_t)
            return {
                "added": added,
                "pruned": 0,
                "replaced": 0,
                "start": start,
                "end": start + added,
                **mutable_stats,
            }
        added = 0
        replaced = 0
        pruned = 0
        first_added_index: int | None = None
        for idx in range(obs_t.shape[0]):
            if self.case_entries >= self.case_capacity:
                pruned += self.prune_cases(force=True)
            if self.case_entries >= self.case_capacity:
                replace_idx = self._lowest_case_index()
                if replace_idx is None:
                    continue
                keep_indices = [i for i in range(self.case_entries) if i != replace_idx]
                self.nnknn_model.compact_cases(keep_indices)
                replaced += 1
            start = self.case_entries
            self.nnknn_model.append_cases(obs_t[idx].unsqueeze(0), values_t[idx].view(1, 1))
            if first_added_index is None:
                first_added_index = start
            added += 1
        start = self.case_entries - added if first_added_index is None else first_added_index
        return {
            "added": added,
            "pruned": pruned,
            "replaced": replaced,
            "start": start,
            "end": start + added,
            **mutable_stats,
        }

    def _lowest_case_index(self) -> int | None:
        if self.case_entries <= 0:
            return None
        biases = self.nnknn_model.biases[: self.case_entries].detach()
        return int(torch.argmin(biases).item())

    def prune_cases(self, *, force: bool = False) -> int:
        active_count = self.case_entries
        if active_count <= 0:
            return 0
        biases = self.nnknn_model.biases[:active_count].detach()
        thresholds: list[torch.Tensor] = []
        if self._prune_quantile > 0.0:
            thresholds.append(torch.quantile(biases, min(max(self._prune_quantile, 0.0), 1.0)))
        if self._prune_bias_threshold is not None:
            thresholds.append(torch.as_tensor(self._prune_bias_threshold, device=biases.device, dtype=biases.dtype))
        if thresholds:
            threshold = torch.stack(thresholds).max()
            remove_candidates = torch.nonzero(biases < threshold, as_tuple=False).view(-1)
        elif force:
            remove_candidates = torch.argsort(biases)[:1]
        else:
            return 0
        if remove_candidates.numel() == 0:
            return 0
        remove_set = {int(idx.item()) for idx in remove_candidates}
        keep_indices = [idx for idx in range(active_count) if idx not in remove_set]
        return self.nnknn_model.compact_cases(keep_indices)

    def configure_case_maintenance(self, *, prune_quantile: float, prune_bias_threshold: float | None) -> None:
        self._prune_quantile = float(prune_quantile)
        self._prune_bias_threshold = None if prune_bias_threshold is None else float(prune_bias_threshold)

    def case_bias_stats(self) -> dict[str, float | None]:
        if self.case_entries <= 0:
            return {"bias_min": None, "bias_mean": None, "bias_max": None}
        biases = self.nnknn_model.biases[: self.case_entries].detach()
        return {
            "bias_min": float(biases.min().cpu().item()),
            "bias_mean": float(biases.mean().cpu().item()),
            "bias_max": float(biases.max().cpu().item()),
        }

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        state = super().state_dict(*args, **kwargs)
        state["_critic_case_entries"] = torch.tensor(int(self.case_entries), dtype=torch.long)
        return state

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        state_copy = dict(state_dict)
        case_entries_t = state_copy.pop("_critic_case_entries", None)
        result = super().load_state_dict(state_copy, strict=strict)
        if case_entries_t is not None:
            self.nnknn_model.set_active_case_count(int(case_entries_t.item()))
        return result


class SharedNNKNNActorCriticNetwork(nn.Module):
    """Shared NN-kNN retrieval model with separate policy and value label heads."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        case_capacity: int,
        tau: float = 1.0,
        top_k: int = 10,
        case_default_bias: float = 0.0,
        min_cases_per_action: int = 1,
        nnknn_config: dict[str, Any] | None = None,
        use_glocal_weightor: bool = True,
        glocal_fw_set_num: int = 1,
        mutable_value_labels: bool = False,
        trainable_value_labels: bool = False,
        value_label_update_alpha: float = 0.25,
        value_label_min_activation: float | None = 0.5,
        value_label_distance_threshold: float | None = 1e-6,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.case_capacity = int(case_capacity)
        self.min_cases_per_action = int(min_cases_per_action)
        self.tau = float(tau)
        self.top_k = int(top_k)
        self.case_default_bias = float(case_default_bias)
        self.use_glocal_weightor = bool(use_glocal_weightor)
        self.glocal_fw_set_num = int(glocal_fw_set_num)
        self.mutable_value_labels = bool(mutable_value_labels)
        self.trainable_value_labels = bool(trainable_value_labels)
        self.value_label_update_alpha = float(value_label_update_alpha)
        self.value_label_min_activation = (
            None if value_label_min_activation is None else float(value_label_min_activation)
        )
        self.value_label_distance_threshold = (
            None if value_label_distance_threshold is None else float(value_label_distance_threshold)
        )
        if not (0.0 < self.value_label_update_alpha <= 1.0):
            raise ValueError("value_label_update_alpha must be in (0, 1]")
        if self.value_label_min_activation is not None and self.value_label_min_activation < 0.0:
            raise ValueError("value_label_min_activation must be None or non-negative")
        if self.value_label_distance_threshold is not None and self.value_label_distance_threshold < 0.0:
            raise ValueError("value_label_distance_threshold must be None or non-negative")
        self._active_case_count = 0
        self._prune_quantile = 0.0
        self._prune_bias_threshold: float | None = None

        self.register_buffer("cases", torch.zeros(self.case_capacity, self.obs_dim, dtype=torch.float32))
        self.register_buffer("action_labels", torch.zeros(self.case_capacity, self.action_dim, dtype=torch.float32))
        if self.trainable_value_labels:
            self.value_labels = nn.Parameter(torch.zeros(self.case_capacity, dtype=torch.float32))
        else:
            self.register_buffer("value_labels", torch.zeros(self.case_capacity, dtype=torch.float32))
        self.register_buffer("case_ids", torch.full((self.case_capacity,), -1, dtype=torch.long))
        self.register_buffer("next_case_id", torch.zeros((), dtype=torch.long))
        self.biases = nn.Parameter(torch.full((self.case_capacity,), self.case_default_bias, dtype=torch.float32))
        self.glocal_weights = nn.Parameter(
            torch.softmax(torch.ones(self.case_capacity, self.glocal_fw_set_num, dtype=torch.float32), dim=-1)
        )
        self.glocal_weightor = (
            GlocalFeatureWeight(self.obs_dim, self.glocal_fw_set_num) if self.use_glocal_weightor else None
        )
        self.nnknn_config = dict(nnknn_config or {})
        self.case_normalizer = str(self.nnknn_config.get("case_normalizer", "softmax"))
        self.case_score_mode = str(self.nnknn_config.get("case_score_mode", "bias_minus_distance"))
        if self.case_score_mode not in {"bias_minus_distance", "neg_distance", "hard_knn"}:
            raise ValueError("Shared NN-kNN actor-critic supports bias_minus_distance, neg_distance, or hard_knn")
        self.pre_topk_mask = bool(self.nnknn_config.get("pre_topk_mask", True))

    @property
    def case_entries(self) -> int:
        return int(self._active_case_count)

    def case_state(self) -> dict[str, Any]:
        return {
            "case_entries": int(self.case_entries),
            "action_counts": self.action_counts().detach().cpu().tolist(),
        }

    def load_case_state(self, state: dict[str, Any]) -> None:
        self._active_case_count = int(state.get("case_entries", self._active_case_count))

    def action_tensor(self) -> torch.Tensor:
        if self.case_entries <= 0:
            return torch.zeros(0, dtype=torch.long, device=self.action_labels.device)
        return self.action_labels[: self.case_entries].argmax(dim=1)

    def action_counts(self) -> torch.Tensor:
        actions = self.action_tensor()
        return torch.bincount(actions, minlength=self.action_dim)

    def is_policy_ready(self, min_case_entries: int = 1) -> bool:
        if self.case_entries < int(min_case_entries):
            return False
        return bool(torch.all(self.action_counts() >= self.min_cases_per_action).detach().cpu().item())

    def _protected_case_id_set(self, protected_case_ids: torch.Tensor | list[int] | None) -> set[int]:
        if protected_case_ids is None:
            return set()
        if isinstance(protected_case_ids, set):
            return {int(case_id) for case_id in protected_case_ids if int(case_id) >= 0}
        protected_ids_t = torch.as_tensor(protected_case_ids, dtype=torch.long).view(-1)
        return {int(case_id) for case_id in protected_ids_t.detach().cpu().tolist() if int(case_id) >= 0}

    def _case_distances_and_weights(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        observations = observations.to(self.cases.device, dtype=torch.float32)
        active_count = self.case_entries
        if active_count <= 0:
            raise ValueError("Shared NN-kNN actor-critic requires at least one active case")
        case_features = self.cases[:active_count]
        query_expanded = observations.unsqueeze(1).expand(-1, active_count, -1)
        case_expanded = case_features.unsqueeze(0).expand(observations.shape[0], -1, -1)
        elementwise_distance = (query_expanded - case_expanded) ** 2
        if self.glocal_weightor is not None:
            glocal_weights = self.glocal_weights[:active_count]
            elementwise_distance = self.glocal_weightor(elementwise_distance, glocal_weights)
        distances = torch.sqrt(torch.relu(elementwise_distance.sum(dim=-1)))
        if self.case_score_mode == "hard_knn":
            k_eff = min(int(self.top_k), active_count)
            _, top_idx = torch.topk(-distances, k=k_eff, dim=1)
            weights = torch.zeros_like(distances)
            weights.scatter_(1, top_idx, 1.0 / float(k_eff))
            return distances, weights
        scores = -distances if self.case_score_mode == "neg_distance" else self.biases[:active_count].unsqueeze(0) - distances
        if self.pre_topk_mask:
            k_eff = min(int(self.top_k), active_count)
            top_vals, top_idx = torch.topk(scores, k=k_eff, dim=1)
            fill_val = float("-inf") if self.case_normalizer == "softmax" else -1e9
            masked_scores = torch.full_like(scores, fill_val)
            scores = masked_scores.scatter(1, top_idx, top_vals)
        weights = normalize_cases(scores, normalizer=self.case_normalizer, tau=self.tau, dim=1)
        return distances, weights

    def _case_weights(self, observations: torch.Tensor) -> torch.Tensor:
        _distances, weights = self._case_distances_and_weights(observations)
        return weights

    def policy_probs(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        observations = observations.to(self.cases.device, dtype=torch.float32)
        batch_size = observations.shape[0]
        if self.case_entries <= 0:
            return torch.full(
                (batch_size, self.action_dim),
                1.0 / float(self.action_dim),
                dtype=observations.dtype,
                device=observations.device,
            )
        weights = self._case_weights(observations)
        probs = torch.matmul(weights, self.action_labels[: self.case_entries].to(weights.dtype)).clamp_min(0.0)
        denom = probs.sum(dim=1, keepdim=True)
        uniform = torch.full_like(probs, 1.0 / float(self.action_dim))
        return torch.where(denom > 0, probs / denom.clamp_min(1e-12), uniform)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        observations = observations.to(self.cases.device, dtype=torch.float32)
        if self.case_entries <= 0:
            return torch.zeros(observations.shape[0], dtype=observations.dtype, device=observations.device)
        weights = self._case_weights(observations)
        return torch.matmul(weights, self.value_labels[: self.case_entries].to(weights.dtype))

    def add_cases(
        self,
        observations: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
        *,
        protected_case_ids: torch.Tensor | list[int] | None = None,
    ) -> dict[str, Any]:
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.cases.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.action_labels.device).view(-1)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.shape[0] != actions_t.shape[0]:
            raise ValueError("observations and actions must have the same batch size")
        if torch.any(actions_t < 0) or torch.any(actions_t >= self.action_dim):
            raise ValueError(f"actions must be integer ids in [0, {self.action_dim})")
        protected_ids = self._protected_case_id_set(protected_case_ids)
        added = 0
        replaced = 0
        pruned = 0
        first_added_index: int | None = None
        assigned_case_ids: list[int] = []
        with torch.no_grad():
            for idx in range(obs_t.shape[0]):
                if self.case_entries >= self.case_capacity:
                    pruned += self.prune_cases(force=True, protected_case_ids=protected_ids)
                if self.case_entries >= self.case_capacity:
                    replace_idx = self._lowest_replaceable_case_index(protected_case_ids=protected_ids)
                    if replace_idx is None:
                        continue
                    keep_indices = [i for i in range(self.case_entries) if i != replace_idx]
                    self.compact_cases(keep_indices)
                    replaced += 1
                insert_idx = self.case_entries
                case_id = int(self.next_case_id.item())
                self.cases[insert_idx].copy_(obs_t[idx])
                self.action_labels[insert_idx].copy_(
                    F.one_hot(actions_t[idx], num_classes=self.action_dim).to(dtype=torch.float32)
                )
                self.value_labels[insert_idx].zero_()
                self.case_ids[insert_idx].fill_(case_id)
                self.biases[insert_idx].fill_(self.case_default_bias)
                self.glocal_weights[insert_idx].copy_(
                    torch.softmax(torch.ones(self.glocal_fw_set_num, device=self.glocal_weights.device), dim=-1)
                )
                self._active_case_count += 1
                self.next_case_id.add_(1)
                if first_added_index is None:
                    first_added_index = insert_idx
                assigned_case_ids.append(case_id)
                added += 1
        start = self.case_entries - added if first_added_index is None else first_added_index
        return {
            "added": added,
            "pruned": pruned,
            "replaced": replaced,
            "start": start,
            "end": start + added,
            "case_ids": assigned_case_ids,
        }

    def update_similar_value_labels(
        self,
        observations: torch.Tensor,
        values: torch.Tensor,
        *,
        exclude_positions: torch.Tensor | None = None,
    ) -> dict[str, int]:
        if self.case_entries <= 0:
            return {"label_updates": 0, "label_update_samples": 0}
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.value_labels.device).view(-1)
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.cases.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.shape[0] != values_t.numel():
            raise ValueError("observations and values must have the same batch size")
        with torch.no_grad():
            distances, weights = self._case_distances_and_weights(obs_t)
            match_mask = torch.zeros_like(weights, dtype=torch.bool)
            if self.value_label_min_activation is not None:
                match_mask |= weights >= float(self.value_label_min_activation)
            if self.value_label_distance_threshold is not None:
                match_mask |= distances <= float(self.value_label_distance_threshold)
            if exclude_positions is not None and exclude_positions.numel():
                exclude_t = exclude_positions.to(match_mask.device, dtype=torch.long).view(-1)
                if exclude_t.numel() == obs_t.shape[0]:
                    match_mask[torch.arange(obs_t.shape[0], device=match_mask.device), exclude_t] = False
                else:
                    match_mask[:, exclude_t] = False

            labels = self.value_labels[: self.case_entries]
            alpha = float(self.value_label_update_alpha)
            label_updates = 0
            label_update_samples = 0
            for row_idx in range(obs_t.shape[0]):
                positions = torch.nonzero(match_mask[row_idx], as_tuple=False).view(-1)
                if positions.numel() == 0:
                    continue
                label_update_samples += 1
                labels[positions] = labels[positions] * (1.0 - alpha) + values_t[row_idx] * alpha
                label_updates += int(positions.numel())
        return {"label_updates": label_updates, "label_update_samples": label_update_samples}

    def update_value_labels(
        self,
        case_ids: torch.Tensor | list[int],
        values: torch.Tensor,
        observations: torch.Tensor | None = None,
    ) -> dict[str, int]:
        case_ids_t = torch.as_tensor(case_ids, dtype=torch.long, device=self.value_labels.device).view(-1)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.value_labels.device).view(-1)
        if case_ids_t.numel() != values_t.numel():
            raise ValueError("case_ids and values must have the same length")
        if case_ids_t.numel() == 0:
            return {"label_updates": 0, "label_update_samples": 0}
        active_case_ids = self.case_ids[: self.case_entries]
        if active_case_ids.numel() == 0:
            raise ValueError("Shared NN-kNN actor-critic has no active cases to label")
        positions = torch.searchsorted(active_case_ids, case_ids_t)
        clamped_positions = positions.clamp(max=max(active_case_ids.numel() - 1, 0))
        valid = (positions < active_case_ids.numel()) & (active_case_ids[clamped_positions] == case_ids_t)
        if not bool(torch.all(valid).detach().cpu().item()):
            raise ValueError("case_ids contain inactive or unknown shared NN-kNN cases")
        with torch.no_grad():
            self.value_labels[positions] = values_t
        mutable_stats = {"label_updates": 0, "label_update_samples": 0}
        if self.mutable_value_labels and observations is not None:
            mutable_stats = self.update_similar_value_labels(
                observations,
                values_t,
                exclude_positions=positions,
            )
        return mutable_stats

    def _lowest_replaceable_case_index(self, *, protected_case_ids: torch.Tensor | list[int] | None = None) -> int | None:
        if self.case_entries <= 0:
            return None
        actions = self.action_tensor()
        counts = self.action_counts()
        biases = self.biases[: self.case_entries].detach()
        active_case_ids = self.case_ids[: self.case_entries].detach()
        protected_ids = self._protected_case_id_set(protected_case_ids)
        sorted_indices = torch.argsort(biases)
        for idx_t in sorted_indices:
            idx = int(idx_t.item())
            if int(active_case_ids[idx].item()) in protected_ids:
                continue
            action = int(actions[idx].item())
            if int(counts[action].item()) > self.min_cases_per_action:
                return idx
        return None

    def compact_cases(self, keep_indices: torch.Tensor | list[int]) -> int:
        active_count = self.case_entries
        keep_t = torch.as_tensor(keep_indices, dtype=torch.long, device=self.cases.device).view(-1)
        if keep_t.numel() and (int(keep_t.min().item()) < 0 or int(keep_t.max().item()) >= active_count):
            raise ValueError("keep_indices must refer to active cases")
        new_count = int(keep_t.numel())
        with torch.no_grad():
            if new_count:
                self.cases[:new_count].copy_(self.cases[keep_t].clone())
                self.action_labels[:new_count].copy_(self.action_labels[keep_t].clone())
                self.value_labels[:new_count].copy_(self.value_labels[keep_t].clone())
                self.case_ids[:new_count].copy_(self.case_ids[keep_t].clone())
                self.biases[:new_count].copy_(self.biases[keep_t].clone())
                self.glocal_weights[:new_count].copy_(self.glocal_weights[keep_t].clone())
            if new_count < active_count:
                self.cases[new_count:active_count].zero_()
                self.action_labels[new_count:active_count].zero_()
                self.value_labels[new_count:active_count].zero_()
                self.case_ids[new_count:active_count].fill_(-1)
                self.biases[new_count:active_count].fill_(self.case_default_bias)
                self.glocal_weights[new_count:active_count].copy_(
                    torch.softmax(
                        torch.ones(active_count - new_count, self.glocal_fw_set_num, device=self.glocal_weights.device),
                        dim=-1,
                    )
                )
            self._active_case_count = new_count
        return active_count - new_count

    def prune_cases(self, *, force: bool = False, protected_case_ids: torch.Tensor | list[int] | None = None) -> int:
        active_count = self.case_entries
        if active_count <= 0:
            return 0
        biases = self.biases[:active_count].detach()
        active_case_ids = self.case_ids[:active_count].detach()
        protected_ids = self._protected_case_id_set(protected_case_ids)
        thresholds: list[torch.Tensor] = []
        if self._prune_quantile > 0.0:
            thresholds.append(torch.quantile(biases, min(max(self._prune_quantile, 0.0), 1.0)))
        if self._prune_bias_threshold is not None:
            thresholds.append(torch.as_tensor(self._prune_bias_threshold, device=biases.device, dtype=biases.dtype))
        if thresholds:
            threshold = torch.stack(thresholds).max()
            remove_candidates = torch.nonzero(biases < threshold, as_tuple=False).view(-1)
        elif force:
            remove_candidates = torch.argsort(biases)[:1]
        else:
            return 0
        if remove_candidates.numel() == 0:
            return 0
        actions = self.action_tensor()
        counts = self.action_counts().clone()
        candidate_order = remove_candidates[torch.argsort(biases[remove_candidates])]
        remove: list[int] = []
        for idx_t in candidate_order:
            idx = int(idx_t.item())
            if int(active_case_ids[idx].item()) in protected_ids:
                continue
            action = int(actions[idx].item())
            if int(counts[action].item()) <= self.min_cases_per_action:
                continue
            remove.append(idx)
            counts[action] -= 1
            if force and not thresholds:
                break
        if not remove:
            return 0
        remove_set = set(remove)
        keep_indices = [idx for idx in range(active_count) if idx not in remove_set]
        return self.compact_cases(keep_indices)

    def configure_case_maintenance(self, *, prune_quantile: float, prune_bias_threshold: float | None) -> None:
        self._prune_quantile = float(prune_quantile)
        self._prune_bias_threshold = None if prune_bias_threshold is None else float(prune_bias_threshold)

    def case_bias_stats(self) -> dict[str, float | None]:
        if self.case_entries <= 0:
            return {"bias_min": None, "bias_mean": None, "bias_max": None}
        biases = self.biases[: self.case_entries].detach()
        return {
            "bias_min": float(biases.min().cpu().item()),
            "bias_mean": float(biases.mean().cpu().item()),
            "bias_max": float(biases.max().cpu().item()),
        }

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        state = super().state_dict(*args, **kwargs)
        state["_shared_case_entries"] = torch.tensor(int(self.case_entries), dtype=torch.long)
        return state

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        state_copy = dict(state_dict)
        case_entries_t = state_copy.pop("_shared_case_entries", None)
        result = super().load_state_dict(state_copy, strict=strict)
        if case_entries_t is not None:
            self._active_case_count = int(case_entries_t.item())
        return result


PolicyActor = NNKNNPolicyNetwork | MLPPolicyNetwork | SharedNNKNNActorCriticNetwork
ValueCritic = ValueNetwork | NNKNNValueNetwork | SharedNNKNNActorCriticNetwork


@dataclass(frozen=True)
class ActionSelection:
    action: int
    policy_probs: list[float]
    behavior_probs: list[float]
    behavior_epsilon: float
    policy_ready: bool


def make_nnknn_rl_config(profile: str = "fast", **overrides: Any) -> NNKNNRLConfig:
    def _coerce_bool_field(field_name: str, default: bool) -> bool:
        value = data.get(field_name, default)
        if isinstance(value, str):
            normalized_value = value.strip().lower()
            if normalized_value in {"1", "true", "yes", "on"}:
                return True
            if normalized_value in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{field_name} must be a boolean")
        return bool(value)

    profiles: dict[str, dict[str, Any]] = {
        "smoke": {
            "profile": "smoke",
            "total_timesteps": 256,
            "case_capacity": 1_000,
            "learning_rate": 1e-3,
            "policy_update_episodes": 1,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 2,
            "min_case_entries": 8,
            "top_k": 16,
            "case_maintenance_frequency": 64,
            "case_prune_quantile": 0.05,
            "min_cases_per_action": 2,
            "success_threshold": None,
        },
        "debug": {
            "profile": "debug",
            "total_timesteps": 25_000,
            "case_capacity": 5_000,
            "learning_rate": 5e-4,
            "policy_update_episodes": 4,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "min_case_entries": 32,
            "top_k": 64,
            "case_maintenance_frequency": 1_000,
            "case_prune_quantile": 0.05,
            "min_cases_per_action": 8,
            "success_threshold": 475.0,
        },
        "fast": {
            "profile": "fast",
            "total_timesteps": 150_000,
            "learning_rate": 5e-4,
            "case_capacity": 10_000,
            "policy_update_episodes": 4,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "min_case_entries": 32,
            "top_k": 10,
            "case_maintenance_frequency": 1_000,
            "case_prune_quantile": 0.05,
            "min_cases_per_action": 8,
            "success_threshold": 475.0,
        },
        "gold": {
            "profile": "gold",
            "total_timesteps": 500_000,
            "learning_rate": 5e-4,
            "case_capacity": 25_000,
            "policy_update_episodes": 8,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "min_case_entries": 32,
            "top_k": 10,
            "case_maintenance_frequency": 1_000,
            "case_prune_quantile": 0.05,
            "min_cases_per_action": 8,
            "success_threshold": 475.0,
        },
    }
    normalized = profile.strip().lower()
    if normalized not in profiles:
        raise ValueError(f"Unknown NN-kNN-RL profile '{profile}'. Choose one of: {', '.join(sorted(profiles))}")
    data = {**profiles[normalized], **overrides}
    actor_type = str(data.get("actor_type", "nnknn")).strip().lower()
    if actor_type not in {"nnknn", "mlp"}:
        raise ValueError("actor_type must be either 'nnknn' or 'mlp'")
    data["actor_type"] = actor_type
    data["actor_hidden_sizes"] = tuple(data.get("actor_hidden_sizes", (128, 128)))
    critic_type = str(data.get("critic_type", "mlp")).strip().lower()
    if critic_type not in {"mlp", "nnknn"}:
        raise ValueError("critic_type must be either 'mlp' or 'nnknn'")
    data["critic_type"] = critic_type
    data["critic_hidden_sizes"] = tuple(data.get("critic_hidden_sizes", (128, 128)))
    reward_shaping = data.get("reward_shaping", None)
    if reward_shaping is not None:
        reward_shaping = str(reward_shaping).strip()
        if reward_shaping not in {"cartpole_potential"}:
            raise ValueError("reward_shaping must be None or 'cartpole_potential'")
    data["reward_shaping"] = reward_shaping
    shared_target_value_mode = str(data.get("shared_target_value_mode", "hard")).strip().lower()
    if shared_target_value_mode not in {"none", "hard", "ema"}:
        raise ValueError("shared_target_value_mode must be one of: none, hard, ema")
    data["shared_target_value_mode"] = shared_target_value_mode
    shared_target_sync_interval = int(data.get("shared_target_sync_interval", 1))
    if shared_target_sync_interval <= 0:
        raise ValueError("shared_target_sync_interval must be a positive integer")
    data["shared_target_sync_interval"] = shared_target_sync_interval
    shared_target_ema_tau = float(data.get("shared_target_ema_tau", 0.05))
    if not (0.0 < shared_target_ema_tau <= 1.0):
        raise ValueError("shared_target_ema_tau must be in (0, 1]")
    data["shared_target_ema_tau"] = shared_target_ema_tau
    min_case_entries = int(data.get("min_case_entries", 32))
    if min_case_entries <= 0:
        raise ValueError("min_case_entries must be a positive integer")
    data["min_case_entries"] = min_case_entries
    min_cases_per_action = int(data.get("min_cases_per_action", 1))
    if min_cases_per_action <= 0:
        raise ValueError("min_cases_per_action must be a positive integer")
    data["min_cases_per_action"] = min_cases_per_action
    exploration_initial_epsilon = float(data.get("exploration_initial_epsilon", 1.0))
    exploration_final_epsilon = float(data.get("exploration_final_epsilon", 0.05))
    if not (0.0 <= exploration_initial_epsilon <= 1.0):
        raise ValueError("exploration_initial_epsilon must be in [0, 1]")
    if not (0.0 <= exploration_final_epsilon <= 1.0):
        raise ValueError("exploration_final_epsilon must be in [0, 1]")
    if exploration_initial_epsilon < exploration_final_epsilon:
        raise ValueError("exploration_initial_epsilon must be greater than or equal to exploration_final_epsilon")
    data["exploration_initial_epsilon"] = exploration_initial_epsilon
    data["exploration_final_epsilon"] = exploration_final_epsilon
    exploration_fraction = float(data.get("exploration_fraction", 0.5))
    if not (0.0 <= exploration_fraction <= 1.0):
        raise ValueError("exploration_fraction must be in [0, 1]")
    data["exploration_fraction"] = exploration_fraction
    data["critic_mutable_value_labels"] = _coerce_bool_field("critic_mutable_value_labels", False)
    data["critic_trainable_value_labels"] = _coerce_bool_field("critic_trainable_value_labels", False)
    data["critic_value_label_append_on_no_match"] = _coerce_bool_field(
        "critic_value_label_append_on_no_match",
        True,
    )
    critic_value_label_update_alpha = float(data.get("critic_value_label_update_alpha", 0.25))
    if not (0.0 < critic_value_label_update_alpha <= 1.0):
        raise ValueError("critic_value_label_update_alpha must be in (0, 1]")
    data["critic_value_label_update_alpha"] = critic_value_label_update_alpha
    critic_value_label_min_activation = data.get("critic_value_label_min_activation", 0.5)
    if critic_value_label_min_activation is not None:
        critic_value_label_min_activation = float(critic_value_label_min_activation)
        if critic_value_label_min_activation < 0.0:
            raise ValueError("critic_value_label_min_activation must be None or non-negative")
    data["critic_value_label_min_activation"] = critic_value_label_min_activation
    critic_value_label_distance_threshold = data.get("critic_value_label_distance_threshold", 1e-6)
    if critic_value_label_distance_threshold is not None:
        critic_value_label_distance_threshold = float(critic_value_label_distance_threshold)
        if critic_value_label_distance_threshold < 0.0:
            raise ValueError("critic_value_label_distance_threshold must be None or non-negative")
    data["critic_value_label_distance_threshold"] = critic_value_label_distance_threshold
    case_learning_rate = data.get("case_learning_rate", None)
    if case_learning_rate is not None:
        case_learning_rate = float(case_learning_rate)
        if case_learning_rate <= 0.0:
            raise ValueError("case_learning_rate must be None or positive")
    data["case_learning_rate"] = case_learning_rate
    return NNKNNRLConfig(**data)


def make_nnknn_rl_output_dir(
    task_name: str,
    *,
    parent: str | Path = "results/rl",
    suffix: str | None = None,
) -> Path:
    created_at = datetime.now(timezone.utc)
    stem = f"nnknn_rl_{task_name}_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
    if suffix:
        stem = f"{stem}_{suffix}"
    parent_path = Path(parent)
    for attempt in range(100):
        candidate = parent_path / (stem if attempt == 0 else f"{stem}_{attempt + 1}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Could not create a unique NN-kNN-RL output directory under {parent_path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


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


def _model_state(model: PolicyActor) -> dict[str, Any]:
    state: dict[str, Any] = {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }
    if _is_nnknn_policy_actor(model):
        state["case_state"] = model.case_state()
    return state


def _load_model_state(model: PolicyActor, state: dict[str, Any]) -> None:
    model.load_state_dict(state["state_dict"])
    if _is_nnknn_policy_actor(model):
        model.load_case_state(state.get("case_state", {}))


def _build_shared_actor_critic_model(
    obs_dim: int,
    action_dim: int,
    cfg: NNKNNRLConfig,
    device: torch.device,
) -> SharedNNKNNActorCriticNetwork:
    model = SharedNNKNNActorCriticNetwork(
        obs_dim,
        action_dim,
        case_capacity=cfg.case_capacity,
        tau=cfg.tau,
        top_k=cfg.top_k,
        case_default_bias=cfg.case_default_bias,
        min_cases_per_action=cfg.min_cases_per_action,
        nnknn_config={**cfg.nnknn_config, **cfg.critic_nnknn_config},
        use_glocal_weightor=cfg.use_glocal_weightor,
        glocal_fw_set_num=cfg.glocal_fw_set_num,
        mutable_value_labels=cfg.critic_mutable_value_labels,
        trainable_value_labels=cfg.critic_trainable_value_labels,
        value_label_update_alpha=cfg.critic_value_label_update_alpha,
        value_label_min_activation=cfg.critic_value_label_min_activation,
        value_label_distance_threshold=cfg.critic_value_label_distance_threshold,
    ).to(device)
    model.configure_case_maintenance(
        prune_quantile=cfg.case_prune_quantile,
        prune_bias_threshold=cfg.case_prune_bias_threshold,
    )
    return model


def _build_actor_model(obs_dim: int, action_dim: int, cfg: NNKNNRLConfig, device: torch.device) -> PolicyActor:
    actor_type = cfg.actor_type.strip().lower()
    if actor_type == "mlp":
        return MLPPolicyNetwork(obs_dim, action_dim, tuple(cfg.actor_hidden_sizes)).to(device)
    if actor_type == "nnknn":
        model = NNKNNPolicyNetwork(
            obs_dim,
            action_dim,
            case_capacity=cfg.case_capacity,
            tau=cfg.tau,
            top_k=cfg.top_k,
            case_default_bias=cfg.case_default_bias,
            min_cases_per_action=cfg.min_cases_per_action,
            nnknn_config=cfg.nnknn_config,
            use_glocal_weightor=cfg.use_glocal_weightor,
            glocal_fw_set_num=cfg.glocal_fw_set_num,
        ).to(device)
        model.configure_case_maintenance(
            prune_quantile=cfg.case_prune_quantile,
            prune_bias_threshold=cfg.case_prune_bias_threshold,
        )
        return model
    raise ValueError("actor_type must be either 'nnknn' or 'mlp'")


def _build_value_model(obs_dim: int, cfg: NNKNNRLConfig, device: torch.device) -> ValueCritic:
    critic_type = cfg.critic_type.strip().lower()
    if critic_type == "mlp":
        return ValueNetwork(obs_dim, tuple(cfg.critic_hidden_sizes)).to(device)
    if critic_type == "nnknn":
        model = NNKNNValueNetwork(
            obs_dim,
            case_capacity=cfg.critic_case_capacity or cfg.case_capacity,
            tau=cfg.tau,
            top_k=cfg.top_k,
            case_default_bias=cfg.case_default_bias,
            nnknn_config=cfg.critic_nnknn_config,
            use_glocal_weightor=cfg.use_glocal_weightor,
            glocal_fw_set_num=cfg.glocal_fw_set_num,
            mutable_value_labels=cfg.critic_mutable_value_labels,
            trainable_value_labels=cfg.critic_trainable_value_labels,
            value_label_update_alpha=cfg.critic_value_label_update_alpha,
            value_label_min_activation=cfg.critic_value_label_min_activation,
            value_label_distance_threshold=cfg.critic_value_label_distance_threshold,
            value_label_append_on_no_match=cfg.critic_value_label_append_on_no_match,
        ).to(device)
        model.configure_case_maintenance(
            prune_quantile=cfg.case_prune_quantile,
            prune_bias_threshold=cfg.case_prune_bias_threshold,
        )
        return model
    raise ValueError("critic_type must be either 'mlp' or 'nnknn'")


def _build_actor_critic_models(
    obs_dim: int,
    action_dim: int,
    cfg: NNKNNRLConfig,
    device: torch.device,
) -> tuple[PolicyActor, ValueCritic]:
    if cfg.actor_type.strip().lower() == "nnknn" and cfg.critic_type.strip().lower() == "nnknn":
        model = _build_shared_actor_critic_model(obs_dim, action_dim, cfg, device)
        return model, model
    return _build_actor_model(obs_dim, action_dim, cfg, device), _build_value_model(obs_dim, cfg, device)


def _nnknn_case_parameter_names(model: nn.Module) -> set[str]:
    if isinstance(model, SharedNNKNNActorCriticNetwork):
        case_names = {"biases", "glocal_weights"}
        if isinstance(model.value_labels, nn.Parameter):
            case_names.add("value_labels")
        return case_names
    if isinstance(model, (NNKNNPolicyNetwork, NNKNNValueNetwork)):
        case_names: set[str] = set()
        for attr_name in ("biases", "negative_weights", "glocal_weights"):
            if isinstance(getattr(model.nnknn_model, attr_name, None), nn.Parameter):
                case_names.add(f"nnknn_model.{attr_name}")
        if isinstance(getattr(model.nnknn_model, "labels", None), nn.Parameter):
            case_names.add("nnknn_model.labels")
        return case_names
    return set()


def _build_nnknn_rl_optimizer(
    model: nn.Module,
    *,
    base_lr: float,
    case_lr: float | None,
) -> optim.Optimizer:
    case_parameter_names = _nnknn_case_parameter_names(model)
    base_params: list[nn.Parameter] = []
    case_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in case_parameter_names:
            case_params.append(parameter)
        else:
            base_params.append(parameter)

    param_groups: list[dict[str, Any]] = []
    if base_params:
        param_groups.append({"params": base_params, "lr": float(base_lr)})
    if case_params:
        param_groups.append({"params": case_params, "lr": float(case_lr if case_lr is not None else base_lr)})
    if not param_groups:
        raise ValueError("Cannot build an optimizer for a model with no trainable parameters")
    return optim.Adam(param_groups)


def _copy_state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _sync_shared_target_value_model(
    source: SharedNNKNNActorCriticNetwork,
    target: SharedNNKNNActorCriticNetwork,
    *,
    mode: str,
    ema_tau: float,
) -> None:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"hard", "ema"}:
        raise ValueError("shared target value sync mode must be 'hard' or 'ema'")
    if normalized_mode == "ema" and not (0.0 < float(ema_tau) <= 1.0):
        raise ValueError("shared_target_ema_tau must be in (0, 1] when EMA sync is enabled")

    source_params = dict(source.named_parameters())
    source_buffers = dict(source.named_buffers())
    with torch.no_grad():
        for name, target_param in target.named_parameters():
            source_param = source_params[name]
            if normalized_mode == "hard":
                target_param.copy_(source_param)
            else:
                target_param.lerp_(source_param, float(ema_tau))
        for name, target_buffer in target.named_buffers():
            target_buffer.copy_(source_buffers[name])
        target._active_case_count = source.case_entries
    target.eval()


def _build_shared_target_value_model(
    source: SharedNNKNNActorCriticNetwork,
    obs_dim: int,
    action_dim: int,
    cfg: NNKNNRLConfig,
    device: torch.device,
) -> SharedNNKNNActorCriticNetwork:
    target = _build_shared_actor_critic_model(obs_dim, action_dim, cfg, device)
    target.requires_grad_(False)
    _sync_shared_target_value_model(source, target, mode="hard", ema_tau=1.0)
    return target


def _is_nnknn_policy_actor(actor: PolicyActor) -> bool:
    return isinstance(actor, (NNKNNPolicyNetwork, SharedNNKNNActorCriticNetwork))


def _actor_case_entries(actor: PolicyActor) -> int | None:
    return int(actor.case_entries) if _is_nnknn_policy_actor(actor) else None


def _actor_action_counts(actor: PolicyActor) -> list[int] | None:
    return actor.action_counts().detach().cpu().tolist() if _is_nnknn_policy_actor(actor) else None


def _actor_action_counts_json(actor: PolicyActor) -> str | None:
    counts = _actor_action_counts(actor)
    return json.dumps(counts) if counts is not None else None


def _actor_case_bias_stats(actor: PolicyActor) -> dict[str, float | None]:
    if _is_nnknn_policy_actor(actor):
        return actor.case_bias_stats()
    return {"bias_min": None, "bias_mean": None, "bias_max": None}


def _actor_bias_loss(actor: PolicyActor, device: torch.device) -> torch.Tensor:
    if isinstance(actor, SharedNNKNNActorCriticNetwork):
        active_biases = actor.biases[: actor.case_entries]
        return active_biases.pow(2).mean() if active_biases.numel() else torch.zeros((), device=device)
    if not isinstance(actor, NNKNNPolicyNetwork):
        return torch.zeros((), dtype=torch.float32, device=device)
    active_biases = actor.nnknn_model.biases[: actor.case_entries]
    return active_biases.pow(2).mean() if active_biases.numel() else torch.zeros((), device=device)


def _case_store_entries(model: Any) -> int | None:
    if isinstance(model, (NNKNNPolicyNetwork, NNKNNValueNetwork, SharedNNKNNActorCriticNetwork)):
        return int(model.case_entries)
    return None


def _case_store_bias_stats(model: Any) -> dict[str, float | None]:
    if isinstance(model, (NNKNNPolicyNetwork, NNKNNValueNetwork, SharedNNKNNActorCriticNetwork)):
        return model.case_bias_stats()
    return {"bias_min": None, "bias_mean": None, "bias_max": None}


def _case_store_action_counts_json(model: Any) -> str | None:
    if isinstance(model, (NNKNNPolicyNetwork, SharedNNKNNActorCriticNetwork)):
        return json.dumps(model.action_counts().detach().cpu().tolist())
    return None


def _maintenance_row(
    *,
    global_step: int,
    case_store: str,
    cases_pruned: int,
    cases_replaced: int,
    model: Any,
    source: str,
) -> dict[str, Any] | None:
    if int(cases_pruned) <= 0 and int(cases_replaced) <= 0:
        return None
    return {
        "global_step": int(global_step),
        "case_store": case_store,
        "source": source,
        "cases_pruned": int(cases_pruned),
        "cases_replaced": int(cases_replaced),
        "case_entries": _case_store_entries(model),
        "action_counts": _case_store_action_counts_json(model),
        **_case_store_bias_stats(model),
    }


def _should_run_critic_diagnostics(
    *,
    completed_episodes: int,
    batch_episodes: int,
    frequency: int,
) -> bool:
    if frequency <= 1:
        return True
    previous_completed = max(0, int(completed_episodes) - int(batch_episodes))
    return int(completed_episodes) // int(frequency) != previous_completed // int(frequency)


def compute_gae(
    rewards: list[float] | torch.Tensor,
    values: list[float] | torch.Tensor,
    next_values: list[float] | torch.Tensor,
    terminated: list[bool] | torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    episode_boundaries: list[bool] | torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1)
    values_t = torch.as_tensor(values, dtype=torch.float32, device=device).view(-1)
    next_values_t = torch.as_tensor(next_values, dtype=torch.float32, device=device).view(-1)
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=device).view(-1)
    if episode_boundaries is None:
        episode_boundaries_t = terminated_t
    else:
        episode_boundaries_t = torch.as_tensor(episode_boundaries, dtype=torch.bool, device=device).view(-1)
    if not (
        rewards_t.numel()
        == values_t.numel()
        == next_values_t.numel()
        == terminated_t.numel()
        == episode_boundaries_t.numel()
    ):
        raise ValueError("GAE inputs must have the same length")

    advantages = torch.zeros_like(rewards_t)
    running_advantage = torch.zeros((), dtype=torch.float32, device=rewards_t.device)
    for idx in range(rewards_t.numel() - 1, -1, -1):
        not_done = (~terminated_t[idx]).to(dtype=torch.float32)
        continue_trace = (~episode_boundaries_t[idx]).to(dtype=torch.float32)
        delta = rewards_t[idx] + float(gamma) * next_values_t[idx] * not_done - values_t[idx]
        running_advantage = delta + float(gamma) * float(gae_lambda) * continue_trace * running_advantage
        advantages[idx] = running_advantage
    value_targets = advantages + values_t
    return advantages, value_targets


def normalize_advantages(
    advantages: torch.Tensor,
    *,
    epsilon: float,
    clip: float | None,
) -> torch.Tensor:
    if advantages.numel() == 0:
        return advantages
    normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(float(epsilon))
    if clip is not None and clip > 0:
        normalized = normalized.clamp(-float(clip), float(clip))
    return normalized


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    targets = targets.detach().float()
    predictions = predictions.detach().float()
    target_var = torch.var(targets, unbiased=False)
    if float(target_var.cpu().item()) <= 1e-12:
        return 0.0
    residual_var = torch.var(targets - predictions, unbiased=False)
    return float((1.0 - residual_var / target_var).cpu().item())


def shape_cartpole_rewards(
    observations: list[np.ndarray],
    next_observations: list[np.ndarray],
    terminated: list[bool],
    rewards: list[float],
    gamma: float,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    if rewards_t.numel() == 0:
        return rewards_t

    obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=device)
    next_obs_t = torch.as_tensor(np.asarray(next_observations, dtype=np.float32), dtype=torch.float32, device=device)
    if (
        obs_t.ndim != 2
        or next_obs_t.ndim != 2
        or obs_t.shape[0] != rewards_t.numel()
        or next_obs_t.shape[0] != rewards_t.numel()
        or obs_t.shape[1] < 3
        or next_obs_t.shape[1] < 3
    ):
        raise ValueError("CartPole reward shaping requires one observation per reward with at least 3 features")

    x_threshold = 2.4
    theta_threshold = 12.0 * np.pi / 180.0
    x = (obs_t[:, 0] / x_threshold).abs().clamp_max(1.0)
    theta = (obs_t[:, 2] / theta_threshold).abs().clamp_max(1.0)
    next_x = (next_obs_t[:, 0] / x_threshold).abs().clamp_max(1.0)
    next_theta = (next_obs_t[:, 2] / theta_threshold).abs().clamp_max(1.0)
    potential = -(x.square() + theta.square())
    next_potential = -(next_x.square() + next_theta.square())
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=device)
    if terminated_t.numel() == next_potential.numel():
        next_potential = next_potential.masked_fill(terminated_t, 0.0)
    return rewards_t + float(gamma) * next_potential - potential


def episode_policy_rewards(
    episode: dict[str, Any],
    cfg: NNKNNRLConfig,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return the per-step rewards used for policy updates.

    Environment rewards are stored unchanged during rollout and summed for
    episode returns. Policy updates use raw environment rewards by default.
    Set reward_shaping="cartpole_potential" to use CartPole potential-based
    shaping for policy updates.
    """

    if cfg.reward_shaping not in {None, "cartpole_potential"}:
        raise ValueError("reward_shaping must be None or 'cartpole_potential'")
    if cfg.reward_shaping == "cartpole_potential":
        return shape_cartpole_rewards(
            episode["observations"],
            episode["next_observations"],
            episode["terminated"],
            episode["rewards"],
            cfg.gamma,
            device=device,
        )
    return torch.as_tensor(episode["rewards"], dtype=torch.float32, device=device)


def _select_action(
    model: PolicyActor,
    obs: np.ndarray,
    *,
    env: Any,
    device: torch.device,
    min_case_entries: int,
    exploration_epsilon: float = 0.0,
    greedy: bool = False,
) -> ActionSelection:
    action_dim = int(model.action_dim)
    if not model.is_policy_ready(min_case_entries=min_case_entries):
        action = int(env.action_space.sample())
        uniform_probs = [1.0 / float(action_dim) for _ in range(action_dim)]
        return ActionSelection(
            action=action,
            policy_probs=uniform_probs,
            behavior_probs=uniform_probs,
            behavior_epsilon=1.0,
            policy_ready=False,
        )
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        probs_tensor = model.policy_probs(obs_tensor).squeeze(0)
        probs = [float(value) for value in probs_tensor.detach().cpu().tolist()]
        if greedy:
            return ActionSelection(
                action=int(probs_tensor.argmax().item()),
                policy_probs=probs,
                behavior_probs=probs,
                behavior_epsilon=0.0,
                policy_ready=True,
            )
        behavior_epsilon = min(max(float(exploration_epsilon), 0.0), 1.0)
        behavior_probs_tensor = _epsilon_mixed_policy_probs(probs_tensor.unsqueeze(0), behavior_epsilon).squeeze(0)
        behavior_probs = [float(value) for value in behavior_probs_tensor.detach().cpu().tolist()]
        action = int(torch.distributions.Categorical(probs=behavior_probs_tensor).sample().item())
        return ActionSelection(
            action=action,
            policy_probs=probs,
            behavior_probs=behavior_probs,
            behavior_epsilon=behavior_epsilon,
            policy_ready=True,
        )


def _exploration_epsilon(cfg: NNKNNRLConfig, global_step: int) -> float:
    duration = int(float(cfg.exploration_fraction) * int(cfg.total_timesteps))
    return float(
        _linear_schedule(
            float(cfg.exploration_initial_epsilon),
            float(cfg.exploration_final_epsilon),
            duration,
            max(0, int(global_step)),
        )
    )


def _epsilon_mixed_policy_probs(
    policy_probs: torch.Tensor,
    epsilon: float | torch.Tensor,
) -> torch.Tensor:
    if policy_probs.dim() == 1:
        policy_probs = policy_probs.unsqueeze(0)
    action_dim = int(policy_probs.shape[1])
    epsilon_t = torch.as_tensor(epsilon, dtype=policy_probs.dtype, device=policy_probs.device)
    epsilon_t = epsilon_t.clamp(0.0, 1.0)
    if epsilon_t.dim() == 0:
        epsilon_t = epsilon_t.view(1, 1)
    elif epsilon_t.dim() == 1:
        epsilon_t = epsilon_t.view(-1, 1)
    uniform_prob = 1.0 / float(action_dim)
    return ((1.0 - epsilon_t) * policy_probs) + (epsilon_t * uniform_prob)


def _protected_shared_case_ids(
    pending_update_episodes: list[dict[str, Any]],
    current_episode_case_ids: list[int],
) -> list[int]:
    protected_case_ids: list[int] = []
    for episode in pending_update_episodes:
        protected_case_ids.extend(int(case_id) for case_id in episode.get("case_ids", []))
    protected_case_ids.extend(int(case_id) for case_id in current_episode_case_ids)
    return protected_case_ids


def _train_actor_critic_batch(
    actor: PolicyActor,
    value_model: ValueCritic,
    target_value_model: ValueCritic | None,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer | None,
    episodes: list[dict[str, Any]],
    cfg: NNKNNRLConfig,
    *,
    device: torch.device,
    global_step: int,
    completed_episodes: int,
) -> dict[str, Any] | None:
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[int] = []
    behavior_epsilons: list[float] = []
    case_ids: list[int] = []
    rewards: list[torch.Tensor] = []
    terminated: list[bool] = []
    episode_boundaries: list[bool] = []
    episode_returns: list[float] = []
    partial_rollout_segments = 0
    partial_rollout_samples = 0
    reward_variation_flags: list[torch.Tensor] = []
    reward_checked_episodes = 0
    for episode in episodes:
        episode_length = len(episode["observations"])
        if episode_length <= 0:
            continue
        observations.extend(episode["observations"])
        next_observations.extend(episode["next_observations"])
        actions.extend(episode["actions"])
        behavior_epsilons.extend(
            float(value)
            for value in episode.get("behavior_epsilons", [0.0] * episode_length)
        )
        case_ids.extend(int(case_id) for case_id in episode.get("case_ids", []))
        policy_rewards = episode_policy_rewards(episode, cfg, device=device)
        if policy_rewards.numel() > 1:
            reward_checked_episodes += 1
            reward_variation_flags.append((policy_rewards != policy_rewards[0]).any())
        rewards.append(policy_rewards)
        terminated.extend(bool(done) for done in episode["terminated"])
        episode_boundaries.extend([False] * (episode_length - 1) + [True])
        if bool(episode.get("partial", False)):
            partial_rollout_segments += 1
            partial_rollout_samples += episode_length
        episode_returns.append(float(sum(episode["rewards"])))
    if not observations:
        return None
    if len(behavior_epsilons) != len(actions):
        raise ValueError("Each NN-kNN-RL action must have a matching behavior epsilon")

    obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=device)
    next_obs_t = torch.as_tensor(np.asarray(next_observations, dtype=np.float32), dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    behavior_epsilons_t = torch.as_tensor(behavior_epsilons, dtype=torch.float32, device=device).view(-1, 1)
    rewards_t = torch.cat(rewards).to(device)
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=device)
    episode_boundaries_t = torch.as_tensor(episode_boundaries, dtype=torch.bool, device=device)
    case_ids_t = torch.as_tensor(case_ids, dtype=torch.long, device=device) if case_ids else None

    bootstrap_value_model = target_value_model if target_value_model is not None else value_model
    bootstrap_value_model.eval()
    with torch.no_grad():
        critic_inputs = torch.cat([obs_t, next_obs_t], dim=0)
        critic_values = bootstrap_value_model(critic_inputs)
        values_t, next_values_t = critic_values.chunk(2)
        raw_advantages, value_targets = compute_gae(
            rewards_t,
            values_t,
            next_values_t,
            terminated_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            episode_boundaries=episode_boundaries_t,
            device=device,
        )
        normalized_advantages = normalize_advantages(
            raw_advantages,
            epsilon=cfg.advantage_epsilon,
            clip=cfg.advantage_clip,
        )

    critic_loss = torch.zeros((), dtype=torch.float32, device=device)
    shared_actor_critic = actor is value_model and isinstance(actor, SharedNNKNNActorCriticNetwork)
    critic_add_stats = {"pruned": 0, "replaced": 0, "label_updates": 0, "label_update_samples": 0}
    critic_label_stats = {"label_updates": 0, "label_update_samples": 0}
    if shared_actor_critic:
        if case_ids_t is None or case_ids_t.numel() != obs_t.shape[0]:
            raise ValueError("Shared NN-kNN actor-critic updates require one case id per sample")
        critic_label_stats = value_model.update_value_labels(
            case_ids_t,
            value_targets.detach(),
            observations=obs_t,
        )
        value_model.train()
        critic_loss = F.mse_loss(value_model(obs_t), value_targets.detach())
    elif isinstance(value_model, NNKNNValueNetwork):
        critic_add_stats = value_model.add_cases(obs_t, value_targets.detach())
        critic_label_stats = {
            "label_updates": int(critic_add_stats.get("label_updates", 0)),
            "label_update_samples": int(critic_add_stats.get("label_update_samples", 0)),
        }
        if critic_optimizer is None:
            raise ValueError("critic_optimizer is required for the NN-kNN critic")
        value_model.train()
        for _epoch in range(max(1, int(cfg.critic_update_epochs))):
            value_predictions = value_model(obs_t)
            critic_loss = F.mse_loss(value_predictions, value_targets.detach())
            critic_optimizer.zero_grad()
            (float(cfg.value_loss_coef) * critic_loss).backward()
            nn.utils.clip_grad_norm_(value_model.parameters(), cfg.max_grad_norm)
            critic_optimizer.step()
    else:
        if critic_optimizer is None:
            raise ValueError("critic_optimizer is required for the MLP critic")
        value_model.train()
        for _epoch in range(max(1, int(cfg.critic_update_epochs))):
            value_predictions = value_model(obs_t)
            critic_loss = F.mse_loss(value_predictions, value_targets.detach())
            critic_optimizer.zero_grad()
            (float(cfg.value_loss_coef) * critic_loss).backward()
            nn.utils.clip_grad_norm_(value_model.parameters(), cfg.max_grad_norm)
            critic_optimizer.step()

    value_model.eval()
    with torch.no_grad():
        run_critic_diagnostics = not isinstance(value_model, NNKNNValueNetwork) or _should_run_critic_diagnostics(
            completed_episodes=completed_episodes,
            batch_episodes=len(episodes),
            frequency=cfg.critic_diagnostic_episode_frequency,
        )
        if run_critic_diagnostics:
            post_value_predictions = value_model(obs_t)
            if isinstance(value_model, NNKNNValueNetwork):
                critic_loss = F.mse_loss(post_value_predictions, value_targets.detach())
        else:
            post_value_predictions = values_t
            critic_loss = F.mse_loss(post_value_predictions, value_targets.detach())

    actor.train()
    probs = actor.policy_probs(obs_t)
    behavior_probs = _epsilon_mixed_policy_probs(probs, behavior_epsilons_t)
    chosen_probs = behavior_probs.gather(1, actions_t.view(-1, 1)).squeeze(1).clamp_min(cfg.advantage_epsilon)
    log_probs = torch.log(chosen_probs)
    policy_loss = -(log_probs * normalized_advantages.detach()).mean()
    probs_clamped = probs.clamp_min(cfg.advantage_epsilon)
    entropy = -(probs_clamped * probs_clamped.log()).sum(dim=1).mean()
    behavior_probs_clamped = behavior_probs.clamp_min(cfg.advantage_epsilon)
    behavior_entropy = -(behavior_probs_clamped * behavior_probs_clamped.log()).sum(dim=1).mean()
    bias_loss = _actor_bias_loss(actor, device)
    actor_loss = policy_loss - cfg.entropy_coef * entropy + cfg.case_bias_l2 * bias_loss

    actor_optimizer.zero_grad()
    if shared_actor_critic:
        total_train_loss = actor_loss + float(cfg.value_loss_coef) * critic_loss
        total_train_loss.backward()
    else:
        actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
    actor_optimizer.step()

    stats = _actor_case_bias_stats(actor)
    reward_varying_episodes = (
        int(torch.stack(reward_variation_flags).sum().detach().cpu().item()) if reward_variation_flags else 0
    )
    total_loss = actor_loss.detach() + float(cfg.value_loss_coef) * critic_loss.detach()
    return {
        "global_step": global_step,
        "episodes": len(episodes),
        "samples": int(obs_t.shape[0]),
        "loss": float(total_loss.cpu().item()),
        "actor_loss": float(actor_loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "critic_diagnostics_post_update": run_critic_diagnostics,
        "entropy": float(entropy.detach().cpu().item()),
        "behavior_entropy": float(behavior_entropy.detach().cpu().item()),
        "mean_behavior_epsilon": float(behavior_epsilons_t.mean().detach().cpu().item()),
        "min_behavior_epsilon": float(behavior_epsilons_t.min().detach().cpu().item()),
        "max_behavior_epsilon": float(behavior_epsilons_t.max().detach().cpu().item()),
        "bias_loss": float(bias_loss.detach().cpu().item()),
        "mean_reward": float(rewards_t.mean().detach().cpu().item()),
        "mean_advantage": float(raw_advantages.mean().detach().cpu().item()),
        "mean_normalized_advantage": float(normalized_advantages.mean().detach().cpu().item()),
        "value_mean": float(post_value_predictions.mean().detach().cpu().item()),
        "value_target_mean": float(value_targets.mean().detach().cpu().item()),
        "explained_variance": explained_variance(post_value_predictions, value_targets),
        "bootstrap_value_source": (
            "shared_target_value_model"
            if target_value_model is not None
            else "online_value_model"
        ),
        "partial_rollout_segments": partial_rollout_segments,
        "partial_rollout_samples": partial_rollout_samples,
        "critic_cases_pruned": int(critic_add_stats["pruned"]) if isinstance(value_model, NNKNNValueNetwork) else 0,
        "critic_cases_replaced": (
            int(critic_add_stats["replaced"]) if isinstance(value_model, NNKNNValueNetwork) else 0
        ),
        "critic_label_updates": int(critic_label_stats.get("label_updates", 0)),
        "critic_label_update_samples": int(critic_label_stats.get("label_update_samples", 0)),
        "shared_value_labels_written": int(case_ids_t.numel()) if shared_actor_critic and case_ids_t is not None else 0,
        "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "reward_varying_episodes": reward_varying_episodes,
        "reward_checked_episodes": reward_checked_episodes,
        "actor_type": cfg.actor_type,
        "case_entries": _actor_case_entries(actor),
        "critic_type": cfg.critic_type,
        "critic_case_entries": (
            value_model.case_entries
            if isinstance(value_model, (NNKNNValueNetwork, SharedNNKNNActorCriticNetwork))
            else None
        ),
        "action_counts": _actor_action_counts_json(actor),
        **stats,
    }


def evaluate_nnknn_rl(
    task_name: str,
    model: PolicyActor,
    *,
    episodes: int = 20,
    seed: int = 10_000,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run greedy-policy evaluation for the actor-critic workflow."""

    spec = get_rl_task_spec(task_name)
    if device is None:
        run_device = next(model.parameters()).device
    else:
        run_device = device if isinstance(device, torch.device) else torch.device(device)
    env = _make_env(spec, seed=seed)
    was_training = model.training
    model.eval()
    returns: list[float] = []
    lengths: list[int] = []
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            episode_length = 0
            while not done:
                selection = _select_action(
                    model,
                    np.asarray(obs, dtype=np.float32),
                    env=env,
                    device=run_device,
                    min_case_entries=1,
                    greedy=True,
                )
                action = selection.action
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)
            returns.append(episode_return)
            lengths.append(episode_length)
            rows.append(
                {
                    "episode": episode + 1,
                    "return": episode_return,
                    "length": episode_length,
                    "seed": seed + episode,
                }
            )
    finally:
        env.close()
        if was_training:
            model.train()
    returns_arr = np.asarray(returns, dtype=np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.float32)
    return {
        "episodes": episodes,
        "seed": seed,
        "mean_return": float(returns_arr.mean()) if len(returns_arr) else 0.0,
        "std_return": float(returns_arr.std()) if len(returns_arr) else 0.0,
        "min_return": float(returns_arr.min()) if len(returns_arr) else 0.0,
        "max_return": float(returns_arr.max()) if len(returns_arr) else 0.0,
        "mean_length": float(lengths_arr.mean()) if len(lengths_arr) else 0.0,
        "episode_metrics": rows,
    }


def train_nnknn_rl(
    task_name: str = "cartpole",
    config: NNKNNRLConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train a selectable actor with a selectable value critic and GAE advantages."""

    spec = get_rl_task_spec(task_name)
    cfg = config or make_nnknn_rl_config(spec.default_profile)
    seed_everything(cfg.seed)
    run_device = _resolve_device_arg(device)

    env = _make_env(spec, seed=cfg.seed)
    obs_dim, action_dim = _validate_env_spaces(env, spec)
    actor, value_network = _build_actor_critic_models(obs_dim, action_dim, cfg, run_device)
    shared_target_value_model: SharedNNKNNActorCriticNetwork | None = None
    if (
        isinstance(actor, SharedNNKNNActorCriticNetwork)
        and actor is value_network
        and cfg.shared_target_value_mode != "none"
    ):
        shared_target_value_model = _build_shared_target_value_model(actor, obs_dim, action_dim, cfg, run_device)
    actor_optimizer = _build_nnknn_rl_optimizer(
        actor,
        base_lr=cfg.learning_rate,
        case_lr=cfg.case_learning_rate,
    )
    critic_optimizer = (
        _build_nnknn_rl_optimizer(
            value_network,
            base_lr=cfg.critic_learning_rate,
            case_lr=cfg.case_learning_rate,
        )
        if isinstance(value_network, (ValueNetwork, NNKNNValueNetwork))
        else None
    )

    run_dir = Path(output_dir) if output_dir is not None else make_nnknn_rl_output_dir(spec.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    created_at = datetime.now(timezone.utc)

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    maintenance_rows: list[dict[str, Any]] = []
    best_eval: dict[str, Any] | None = None
    best_eval_step: int | None = None
    best_actor_state: dict[str, Any] | None = None
    best_critic_state: dict[str, torch.Tensor] | None = None
    actor_cases_pruned = 0
    actor_cases_replaced = 0
    critic_cases_pruned = 0
    critic_cases_replaced = 0
    shared_cases_pruned = 0
    shared_cases_replaced = 0
    shared_value_labels_written = 0
    critic_label_updates = 0
    critic_label_update_samples = 0
    partial_rollout_segments = 0
    partial_rollout_samples = 0
    shared_target_syncs = 1 if shared_target_value_model is not None else 0
    shared_update_batches = 0

    def total_cases_pruned() -> int:
        return actor_cases_pruned + critic_cases_pruned + shared_cases_pruned

    def total_cases_replaced() -> int:
        return actor_cases_replaced + critic_cases_replaced + shared_cases_replaced

    obs, _ = env.reset(seed=cfg.seed)
    episode_observations: list[np.ndarray] = []
    episode_next_observations: list[np.ndarray] = []
    episode_actions: list[int] = []
    episode_behavior_epsilons: list[float] = []
    episode_case_ids: list[int] = []
    episode_rewards: list[float] = []
    episode_terminated: list[bool] = []
    episode_truncated: list[bool] = []
    episode_return = 0.0
    episode_length = 0
    episode_index = 0
    pending_update_episodes: list[dict[str, Any]] = []

    try:
        for global_step in range(cfg.total_timesteps):
            obs_array = np.asarray(obs, dtype=np.float32)
            exploration_epsilon = _exploration_epsilon(cfg, global_step)
            selection = _select_action(
                actor,
                obs_array,
                env=env,
                device=run_device,
                min_case_entries=cfg.min_case_entries,
                exploration_epsilon=exploration_epsilon,
                greedy=False,
            )
            action = selection.action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if _is_nnknn_policy_actor(actor):
                action_case = np.asarray([int(action)], dtype=np.int64)
                if isinstance(actor, SharedNNKNNActorCriticNetwork):
                    add_stats = actor.add_cases(
                        obs_array[np.newaxis, :],
                        action_case,
                        protected_case_ids=_protected_shared_case_ids(pending_update_episodes, episode_case_ids),
                    )
                    assigned_case_ids = add_stats.get("case_ids", [])
                    if len(assigned_case_ids) != 1:
                        raise RuntimeError("Shared NN-kNN actor-critic failed to store the selected state-action case.")
                    episode_case_ids.append(int(assigned_case_ids[0]))
                else:
                    add_stats = actor.add_cases(obs_array[np.newaxis, :], action_case)
                if add_stats["added"] != 1:
                    raise RuntimeError("NN-kNN actor failed to store the selected state-action case.")
                add_pruned = int(add_stats["pruned"])
                add_replaced = int(add_stats["replaced"])
                if isinstance(actor, SharedNNKNNActorCriticNetwork):
                    shared_cases_pruned += add_pruned
                    shared_cases_replaced += add_replaced
                    row = _maintenance_row(
                        global_step=global_step + 1,
                        case_store="shared",
                        cases_pruned=add_pruned,
                        cases_replaced=add_replaced,
                        model=actor,
                        source="capacity_insert",
                    )
                else:
                    actor_cases_pruned += add_pruned
                    actor_cases_replaced += add_replaced
                    row = _maintenance_row(
                        global_step=global_step + 1,
                        case_store="actor",
                        cases_pruned=add_pruned,
                        cases_replaced=add_replaced,
                        model=actor,
                        source="capacity_insert",
                    )
                if row is not None:
                    maintenance_rows.append(row)

            episode_observations.append(obs_array)
            episode_next_observations.append(np.asarray(next_obs, dtype=np.float32))
            episode_actions.append(action)
            episode_behavior_epsilons.append(selection.behavior_epsilon)
            episode_rewards.append(float(reward))
            episode_terminated.append(bool(terminated))
            episode_truncated.append(bool(truncated))
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1
            episode_done = bool(terminated or truncated)

            completed_step = global_step + 1
            if (
                cfg.case_maintenance_frequency > 0
                and completed_step % cfg.case_maintenance_frequency == 0
            ):
                if isinstance(actor, SharedNNKNNActorCriticNetwork):
                    pruned = actor.prune_cases(
                        protected_case_ids=_protected_shared_case_ids(pending_update_episodes, episode_case_ids)
                    )
                    shared_cases_pruned += pruned
                    row = _maintenance_row(
                        global_step=completed_step,
                        case_store="shared",
                        cases_pruned=pruned,
                        cases_replaced=0,
                        model=actor,
                        source="scheduled",
                    )
                    if row is not None:
                        maintenance_rows.append(row)
                elif isinstance(actor, NNKNNPolicyNetwork):
                    pruned = actor.prune_cases()
                    actor_cases_pruned += pruned
                    row = _maintenance_row(
                        global_step=completed_step,
                        case_store="actor",
                        cases_pruned=pruned,
                        cases_replaced=0,
                        model=actor,
                        source="scheduled",
                    )
                    if row is not None:
                        maintenance_rows.append(row)
                if isinstance(value_network, NNKNNValueNetwork):
                    pruned = value_network.prune_cases()
                    critic_cases_pruned += pruned
                    row = _maintenance_row(
                        global_step=completed_step,
                        case_store="critic",
                        cases_pruned=pruned,
                        cases_replaced=0,
                        model=value_network,
                        source="scheduled",
                    )
                    if row is not None:
                        maintenance_rows.append(row)

            if episode_done:
                episode_index += 1
                pending_update_episodes.append(
                    {
                        "observations": episode_observations,
                        "next_observations": episode_next_observations,
                        "actions": episode_actions,
                        "behavior_epsilons": episode_behavior_epsilons,
                        "case_ids": episode_case_ids,
                        "rewards": episode_rewards,
                        "terminated": episode_terminated,
                        "truncated": episode_truncated,
                    }
                )
                latest_loss: float | None = None
                if len(pending_update_episodes) >= cfg.policy_update_episodes:
                    loss_row = _train_actor_critic_batch(
                        actor,
                        value_network,
                        shared_target_value_model,
                        actor_optimizer,
                        critic_optimizer,
                        pending_update_episodes,
                        cfg,
                        device=run_device,
                        global_step=completed_step,
                        completed_episodes=episode_index,
                    )
                    if loss_row is not None:
                        if shared_target_value_model is not None:
                            shared_update_batches += 1
                            if shared_update_batches % cfg.shared_target_sync_interval == 0:
                                _sync_shared_target_value_model(
                                    actor,
                                    shared_target_value_model,
                                    mode=cfg.shared_target_value_mode,
                                    ema_tau=cfg.shared_target_ema_tau,
                                )
                                shared_target_syncs += 1
                        critic_pruned = int(loss_row.get("critic_cases_pruned", 0))
                        critic_replaced = int(loss_row.get("critic_cases_replaced", 0))
                        critic_cases_pruned += critic_pruned
                        critic_cases_replaced += critic_replaced
                        shared_value_labels_written += int(loss_row.get("shared_value_labels_written", 0))
                        critic_label_updates += int(loss_row.get("critic_label_updates", 0))
                        critic_label_update_samples += int(loss_row.get("critic_label_update_samples", 0))
                        row = _maintenance_row(
                            global_step=completed_step,
                            case_store="critic",
                            cases_pruned=critic_pruned,
                            cases_replaced=critic_replaced,
                            model=value_network,
                            source="capacity_insert",
                        )
                        if row is not None:
                            maintenance_rows.append(row)
                        partial_rollout_segments += int(loss_row.get("partial_rollout_segments", 0))
                        partial_rollout_samples += int(loss_row.get("partial_rollout_samples", 0))
                        loss_rows.append(loss_row)
                        latest_loss = float(loss_row["loss"])
                    pending_update_episodes = []

                stats = _actor_case_bias_stats(actor)
                training_rows.append(
                    {
                        "global_step": completed_step,
                        "episode": episode_index,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "loss": latest_loss,
                        "actor_type": cfg.actor_type,
                        "case_entries": _actor_case_entries(actor),
                        "cases_pruned": total_cases_pruned(),
                        "cases_replaced": total_cases_replaced(),
                        "actor_cases_pruned": actor_cases_pruned,
                        "actor_cases_replaced": actor_cases_replaced,
                        "critic_cases_pruned": critic_cases_pruned,
                        "critic_cases_replaced": critic_cases_replaced,
                        "shared_cases_pruned": shared_cases_pruned,
                        "shared_cases_replaced": shared_cases_replaced,
                        "critic_label_updates": critic_label_updates,
                        "critic_label_update_samples": critic_label_update_samples,
                        "exploration_epsilon": exploration_epsilon,
                        "behavior_epsilon": selection.behavior_epsilon,
                        "policy_ready": selection.policy_ready,
                        "action_probs": json.dumps(selection.policy_probs),
                        "behavior_action_probs": json.dumps(selection.behavior_probs),
                        "action_counts": _actor_action_counts_json(actor),
                        **stats,
                    }
                )
                if progress and (episode_index <= 5 or episode_index % 10 == 0):
                    print(
                        "[nnknn-rl] "
                        f"step={completed_step} episode={episode_index} "
                        f"return={episode_return:.1f} length={episode_length} "
                        f"loss={latest_loss} cases={_actor_case_entries(actor)}",
                        flush=True,
                    )

                if (
                    cfg.eval_episode_frequency > 0
                    and episode_index > 0
                    and episode_index % cfg.eval_episode_frequency == 0
                ):
                    eval_metrics = evaluate_nnknn_rl(
                        spec.name,
                        actor,
                        episodes=cfg.eval_episodes,
                        seed=cfg.eval_seed,
                        device=run_device,
                    )
                    eval_row = {
                        "global_step": completed_step,
                        "episode": episode_index,
                        "mean_return": eval_metrics["mean_return"],
                        "std_return": eval_metrics["std_return"],
                        "min_return": eval_metrics["min_return"],
                        "max_return": eval_metrics["max_return"],
                        "mean_length": eval_metrics["mean_length"],
                        "episodes": cfg.eval_episodes,
                        "actor_type": cfg.actor_type,
                        "case_entries": _actor_case_entries(actor),
                    }
                    eval_rows.append(eval_row)
                    if best_eval is None or eval_metrics["mean_return"] > best_eval["mean_return"]:
                        best_eval = eval_metrics
                        best_eval_step = completed_step
                        best_actor_state = _model_state(actor)
                        best_critic_state = _copy_state_dict_to_cpu(value_network)
                    if progress:
                        print(
                            "[nnknn-rl][eval] "
                            f"step={completed_step} mean_return={eval_row['mean_return']:.2f} "
                            f"max_return={eval_row['max_return']:.2f}",
                            flush=True,
                        )

                obs, _ = env.reset(seed=cfg.seed + episode_index)
                episode_observations = []
                episode_next_observations = []
                episode_actions = []
                episode_behavior_epsilons = []
                episode_case_ids = []
                episode_rewards = []
                episode_terminated = []
                episode_truncated = []
                episode_return = 0.0
                episode_length = 0

            if cfg.eval_frequency > 0 and global_step > 0 and global_step % cfg.eval_frequency == 0:
                eval_metrics = evaluate_nnknn_rl(
                    spec.name,
                    actor,
                    episodes=cfg.eval_episodes,
                    seed=cfg.eval_seed,
                    device=run_device,
                )
                eval_row = {
                    "global_step": global_step,
                    "episode": episode_index,
                    "mean_return": eval_metrics["mean_return"],
                    "std_return": eval_metrics["std_return"],
                    "min_return": eval_metrics["min_return"],
                    "max_return": eval_metrics["max_return"],
                    "mean_length": eval_metrics["mean_length"],
                    "episodes": cfg.eval_episodes,
                    "actor_type": cfg.actor_type,
                    "case_entries": _actor_case_entries(actor),
                }
                eval_rows.append(eval_row)
                if best_eval is None or eval_metrics["mean_return"] > best_eval["mean_return"]:
                    best_eval = eval_metrics
                    best_eval_step = global_step
                    best_actor_state = _model_state(actor)
                    best_critic_state = _copy_state_dict_to_cpu(value_network)

        if episode_observations:
            partial_rollout = {
                "observations": episode_observations,
                "next_observations": episode_next_observations,
                "actions": episode_actions,
                "behavior_epsilons": episode_behavior_epsilons,
                "case_ids": episode_case_ids,
                "rewards": episode_rewards,
                "terminated": episode_terminated,
                "truncated": episode_truncated,
                "partial": True,
            }
            pending_update_episodes.append(partial_rollout)

        if pending_update_episodes:
            loss_row = _train_actor_critic_batch(
                actor,
                value_network,
                shared_target_value_model,
                actor_optimizer,
                critic_optimizer,
                pending_update_episodes,
                cfg,
                device=run_device,
                global_step=cfg.total_timesteps,
                completed_episodes=episode_index,
            )
            if loss_row is not None:
                if shared_target_value_model is not None:
                    shared_update_batches += 1
                    if shared_update_batches % cfg.shared_target_sync_interval == 0:
                        _sync_shared_target_value_model(
                            actor,
                            shared_target_value_model,
                            mode=cfg.shared_target_value_mode,
                            ema_tau=cfg.shared_target_ema_tau,
                        )
                        shared_target_syncs += 1
                critic_pruned = int(loss_row.get("critic_cases_pruned", 0))
                critic_replaced = int(loss_row.get("critic_cases_replaced", 0))
                critic_cases_pruned += critic_pruned
                critic_cases_replaced += critic_replaced
                shared_value_labels_written += int(loss_row.get("shared_value_labels_written", 0))
                critic_label_updates += int(loss_row.get("critic_label_updates", 0))
                critic_label_update_samples += int(loss_row.get("critic_label_update_samples", 0))
                row = _maintenance_row(
                    global_step=cfg.total_timesteps,
                    case_store="critic",
                    cases_pruned=critic_pruned,
                    cases_replaced=critic_replaced,
                    model=value_network,
                    source="capacity_insert",
                )
                if row is not None:
                    maintenance_rows.append(row)
                partial_rollout_segments += int(loss_row.get("partial_rollout_segments", 0))
                partial_rollout_samples += int(loss_row.get("partial_rollout_samples", 0))
                loss_rows.append(loss_row)
    finally:
        env.close()

    last_eval = evaluate_nnknn_rl(
        spec.name,
        actor,
        episodes=cfg.eval_episodes,
        seed=cfg.eval_seed,
        device=run_device,
    )
    if best_eval is None or last_eval["mean_return"] >= best_eval["mean_return"]:
        selected_eval = last_eval
        selected_step = cfg.total_timesteps
        selected_source = "final"
        selected_actor_state = _model_state(actor)
        selected_critic_state = _copy_state_dict_to_cpu(value_network)
    else:
        selected_eval = best_eval
        selected_step = int(best_eval_step or 0)
        selected_source = "best_eval"
        selected_actor_state = best_actor_state or _model_state(actor)
        selected_critic_state = best_critic_state or _copy_state_dict_to_cpu(value_network)
        _load_model_state(actor, selected_actor_state)
        value_network.load_state_dict(selected_critic_state)

    passed = True if cfg.success_threshold is None else selected_eval["mean_return"] >= cfg.success_threshold
    first_success_step = _first_threshold_step(eval_rows, cfg.success_threshold)
    if (
        first_success_step is None
        and cfg.success_threshold is not None
        and last_eval["mean_return"] >= cfg.success_threshold
    ):
        first_success_step = cfg.total_timesteps
    training_efficiency = _build_training_efficiency(
        selected_eval=selected_eval,
        last_eval=last_eval,
        selected_step=selected_step,
        selected_source=selected_source,
        total_timesteps=cfg.total_timesteps,
        success_threshold=cfg.success_threshold,
        first_success_step=first_success_step,
    )
    checkpoint = {
        "algorithm": ALGORITHM_NAME,
        "actor_type": cfg.actor_type,
        "critic_type": cfg.critic_type,
        "actor_state": selected_actor_state,
        "critic_state_dict": selected_critic_state,
        "task": spec.to_dict(),
        "config": cfg.to_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "gae": {
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "advantage_clip": cfg.advantage_clip,
        },
        "selected_eval": {k: v for k, v in selected_eval.items() if k != "episode_metrics"},
        "last_eval": {k: v for k, v in last_eval.items() if k != "episode_metrics"},
        "selected_step": selected_step,
        "selected_source": selected_source,
        "training_efficiency": training_efficiency,
        "case_entries": _actor_case_entries(actor),
        "action_counts": _actor_action_counts(actor),
        "cases_pruned": total_cases_pruned(),
        "cases_replaced": total_cases_replaced(),
        "actor_cases_pruned": actor_cases_pruned,
        "actor_cases_replaced": actor_cases_replaced,
        "critic_cases_pruned": critic_cases_pruned,
        "critic_cases_replaced": critic_cases_replaced,
        "shared_cases_pruned": shared_cases_pruned,
        "shared_cases_replaced": shared_cases_replaced,
        "critic_label_updates": critic_label_updates,
        "critic_label_update_samples": critic_label_update_samples,
        "partial_rollout_segments": partial_rollout_segments,
        "partial_rollout_samples": partial_rollout_samples,
        "shared_value_labels_written": shared_value_labels_written,
        "passed": passed,
    }
    torch.save(checkpoint, checkpoint_path)

    _write_json(
        run_dir / "config.json",
        {
            "created_at_utc": created_at.isoformat(),
            "algorithm": ALGORITHM_NAME,
            "task": spec.to_dict(),
            "config": cfg.to_dict(),
            "device": str(run_device),
            "source_reference": cfg.source_reference,
        },
    )
    _write_csv(run_dir / "training_metrics.csv", training_rows)
    _write_csv(run_dir / "loss_metrics.csv", loss_rows)
    _write_csv(run_dir / "eval_metrics.csv", eval_rows)
    _write_csv(run_dir / "case_maintenance.csv", maintenance_rows)
    _write_csv(run_dir / "final_eval_episodes.csv", selected_eval["episode_metrics"])
    _write_csv(run_dir / "last_eval_episodes.csv", last_eval["episode_metrics"])
    summary = {
        "task": spec.name,
        "env_id": spec.env_id,
        "profile": cfg.profile,
        "seed": cfg.seed,
        "total_timesteps": cfg.total_timesteps,
        "eval_episodes": cfg.eval_episodes,
        "success_threshold": cfg.success_threshold,
        "passed": passed,
        "algorithm": ALGORITHM_NAME,
        "actor_type": cfg.actor_type,
        "critic_type": cfg.critic_type,
        "gae": {
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "advantage_clip": cfg.advantage_clip,
        },
        "exploration": {
            "initial_epsilon": cfg.exploration_initial_epsilon,
            "final_epsilon": cfg.exploration_final_epsilon,
            "fraction": cfg.exploration_fraction,
            "final_step_epsilon": _exploration_epsilon(cfg, cfg.total_timesteps),
            "readiness": {
                "min_case_entries": cfg.min_case_entries,
                "min_cases_per_action": cfg.min_cases_per_action,
            },
        },
        "final_eval": {k: v for k, v in selected_eval.items() if k != "episode_metrics"},
        "last_eval": {k: v for k, v in last_eval.items() if k != "episode_metrics"},
        "selected_step": selected_step,
        "selected_source": selected_source,
        "training_efficiency": training_efficiency,
        "case_entries": _actor_case_entries(actor),
        "critic_case_entries": (
            value_network.case_entries
            if isinstance(value_network, (NNKNNValueNetwork, SharedNNKNNActorCriticNetwork))
            else None
        ),
        "action_counts": _actor_action_counts(actor),
        "cases_pruned": total_cases_pruned(),
        "cases_replaced": total_cases_replaced(),
        "actor_cases_pruned": actor_cases_pruned,
        "actor_cases_replaced": actor_cases_replaced,
        "critic_cases_pruned": critic_cases_pruned,
        "critic_cases_replaced": critic_cases_replaced,
        "shared_cases_pruned": shared_cases_pruned,
        "shared_cases_replaced": shared_cases_replaced,
        "critic_label_updates": critic_label_updates,
        "critic_label_update_samples": critic_label_update_samples,
        "partial_rollout_segments": partial_rollout_segments,
        "partial_rollout_samples": partial_rollout_samples,
        "shared_value_labels_written": shared_value_labels_written,
        "shared_target_value_mode": cfg.shared_target_value_mode,
        "shared_target_sync_interval": cfg.shared_target_sync_interval,
        "shared_target_ema_tau": cfg.shared_target_ema_tau,
        "shared_target_value_syncs": shared_target_syncs,
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "manifest.json",
        {
            "created_at_utc": created_at.isoformat(),
            "algorithm": ALGORITHM_NAME,
            "task": spec.name,
            "env_id": spec.env_id,
            "profile": cfg.profile,
            "outputs": [
                "config.json",
                "training_metrics.csv",
                "loss_metrics.csv",
                "eval_metrics.csv",
                "case_maintenance.csv",
                "final_eval_episodes.csv",
                "last_eval_episodes.csv",
                "summary.json",
                "manifest.json",
                "checkpoint.pt",
            ],
        },
    )
    if progress:
        print(
            "[nnknn-rl] finished "
            f"profile={cfg.profile} mean_eval_return={selected_eval['mean_return']:.2f} "
            f"passed={passed} run_dir={run_dir}",
            flush=True,
        )
    return {
        "model": actor,
        "value_model": value_network,
        "target_model": None,
        "task": spec,
        "config": cfg,
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "training_metrics": training_rows,
        "loss_metrics": loss_rows,
        "eval_metrics": eval_rows,
        "final_eval": selected_eval,
        "last_eval": last_eval,
        "passed": passed,
        "summary": summary,
    }


def load_nnknn_rl_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    run_device = _resolve_device_arg(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=run_device, weights_only=False)
    algorithm = checkpoint.get("algorithm")
    if algorithm != ALGORITHM_NAME:
        raise ValueError(
            f"Unsupported NN-kNN-RL checkpoint algorithm {algorithm!r} at {checkpoint_path}. "
            f"Expected {ALGORITHM_NAME!r}; retrain with the actor-critic GAE workflow."
        )
    config_data = dict(checkpoint["config"])
    config_data["actor_type"] = str(config_data.get("actor_type", checkpoint.get("actor_type", "nnknn"))).lower()
    config_data["actor_hidden_sizes"] = tuple(config_data.get("actor_hidden_sizes", (128, 128)))
    config_data["critic_type"] = str(config_data.get("critic_type", checkpoint.get("critic_type", "mlp"))).lower()
    config_data["critic_hidden_sizes"] = tuple(config_data.get("critic_hidden_sizes", (128, 128)))
    config_data.setdefault("critic_nnknn_config", {})
    config_data.setdefault("critic_case_capacity", None)
    cfg = NNKNNRLConfig(**config_data)
    model, value_model = _build_actor_critic_models(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        cfg,
        run_device,
    )
    _load_model_state(model, checkpoint["actor_state"])
    value_model.load_state_dict(checkpoint["critic_state_dict"])
    model.eval()
    value_model.eval()
    return {
        "model": model,
        "value_model": value_model,
        "target_model": None,
        "config": cfg,
        "task": checkpoint["task"],
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "device": run_device,
    }
