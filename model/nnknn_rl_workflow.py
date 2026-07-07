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
    case_maintenance_frequency: int = 1_000
    case_prune_quantile: float = 0.05
    case_prune_bias_threshold: float | None = None
    min_cases_per_action: int = 8
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
        return bool(torch.all(self.action_counts() > 0).detach().cpu().item())

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
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.case_capacity = int(case_capacity)
        self.tau = float(tau)
        self.top_k = int(top_k)
        self.case_default_bias = float(case_default_bias)
        self.use_glocal_weightor = bool(use_glocal_weightor)
        self.glocal_fw_set_num = int(glocal_fw_set_num)
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

    def add_cases(self, observations: torch.Tensor | np.ndarray, values: torch.Tensor | np.ndarray) -> dict[str, int]:
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.nnknn_model.cases.device)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.nnknn_model.labels.device).view(-1, 1)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.shape[0] != values_t.shape[0]:
            raise ValueError("observations and values must have the same batch size")
        available = self.case_capacity - self.case_entries
        if obs_t.shape[0] <= available:
            start = self.case_entries
            added = self.nnknn_model.append_cases(obs_t, values_t)
            return {"added": added, "pruned": 0, "replaced": 0, "start": start, "end": start + added}
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
        return {"added": added, "pruned": pruned, "replaced": replaced, "start": start, "end": start + added}

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
        self._active_case_count = 0
        self._prune_quantile = 0.0
        self._prune_bias_threshold: float | None = None

        self.register_buffer("cases", torch.zeros(self.case_capacity, self.obs_dim, dtype=torch.float32))
        self.register_buffer("action_labels", torch.zeros(self.case_capacity, self.action_dim, dtype=torch.float32))
        self.register_buffer("value_labels", torch.zeros(self.case_capacity, dtype=torch.float32))
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
        return bool(torch.all(self.action_counts() > 0).detach().cpu().item())

    def _case_weights(self, observations: torch.Tensor) -> torch.Tensor:
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
            return weights
        scores = -distances if self.case_score_mode == "neg_distance" else self.biases[:active_count].unsqueeze(0) - distances
        if self.pre_topk_mask:
            k_eff = min(int(self.top_k), active_count)
            top_vals, top_idx = torch.topk(scores, k=k_eff, dim=1)
            fill_val = float("-inf") if self.case_normalizer == "softmax" else -1e9
            masked_scores = torch.full_like(scores, fill_val)
            scores = masked_scores.scatter(1, top_idx, top_vals)
        return normalize_cases(scores, normalizer=self.case_normalizer, tau=self.tau, dim=1)

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

    def add_cases(self, observations: torch.Tensor | np.ndarray, actions: torch.Tensor | np.ndarray) -> dict[str, int]:
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.cases.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.action_labels.device).view(-1)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.shape[0] != actions_t.shape[0]:
            raise ValueError("observations and actions must have the same batch size")
        if torch.any(actions_t < 0) or torch.any(actions_t >= self.action_dim):
            raise ValueError(f"actions must be integer ids in [0, {self.action_dim})")
        added = 0
        replaced = 0
        pruned = 0
        first_added_index: int | None = None
        with torch.no_grad():
            for idx in range(obs_t.shape[0]):
                if self.case_entries >= self.case_capacity:
                    pruned += self.prune_cases(force=True)
                if self.case_entries >= self.case_capacity:
                    replace_idx = self._lowest_replaceable_case_index()
                    if replace_idx is None:
                        continue
                    keep_indices = [i for i in range(self.case_entries) if i != replace_idx]
                    self.compact_cases(keep_indices)
                    replaced += 1
                insert_idx = self.case_entries
                self.cases[insert_idx].copy_(obs_t[idx])
                self.action_labels[insert_idx].copy_(
                    F.one_hot(actions_t[idx], num_classes=self.action_dim).to(dtype=torch.float32)
                )
                self.value_labels[insert_idx].zero_()
                self.biases[insert_idx].fill_(self.case_default_bias)
                self.glocal_weights[insert_idx].copy_(
                    torch.softmax(torch.ones(self.glocal_fw_set_num, device=self.glocal_weights.device), dim=-1)
                )
                self._active_case_count += 1
                if first_added_index is None:
                    first_added_index = insert_idx
                added += 1
        start = self.case_entries - added if first_added_index is None else first_added_index
        return {"added": added, "pruned": pruned, "replaced": replaced, "start": start, "end": start + added}

    def update_value_labels(self, case_indices: torch.Tensor | list[int], values: torch.Tensor) -> None:
        indices_t = torch.as_tensor(case_indices, dtype=torch.long, device=self.value_labels.device).view(-1)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.value_labels.device).view(-1)
        if indices_t.numel() != values_t.numel():
            raise ValueError("case_indices and values must have the same length")
        if indices_t.numel() == 0:
            return
        valid = (indices_t >= 0) & (indices_t < self.case_entries)
        if not bool(torch.all(valid).detach().cpu().item()):
            raise ValueError("case_indices contain inactive or out-of-range entries")
        with torch.no_grad():
            self.value_labels[indices_t] = values_t

    def _lowest_replaceable_case_index(self) -> int | None:
        if self.case_entries <= 0:
            return None
        actions = self.action_tensor()
        counts = self.action_counts()
        biases = self.biases[: self.case_entries].detach()
        sorted_indices = torch.argsort(biases)
        for idx_t in sorted_indices:
            idx = int(idx_t.item())
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
                self.biases[:new_count].copy_(self.biases[keep_t].clone())
                self.glocal_weights[:new_count].copy_(self.glocal_weights[keep_t].clone())
            if new_count < active_count:
                self.cases[new_count:active_count].zero_()
                self.action_labels[new_count:active_count].zero_()
                self.value_labels[new_count:active_count].zero_()
                self.biases[new_count:active_count].fill_(self.case_default_bias)
                self.glocal_weights[new_count:active_count].copy_(
                    torch.softmax(
                        torch.ones(active_count - new_count, self.glocal_fw_set_num, device=self.glocal_weights.device),
                        dim=-1,
                    )
                )
            self._active_case_count = new_count
        return active_count - new_count

    def prune_cases(self, *, force: bool = False) -> int:
        active_count = self.case_entries
        if active_count <= 0:
            return 0
        biases = self.biases[:active_count].detach()
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


def make_nnknn_rl_config(profile: str = "fast", **overrides: Any) -> NNKNNRLConfig:
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
        ).to(device)
        model.configure_case_maintenance(
            prune_quantile=cfg.case_prune_quantile,
            prune_bias_threshold=cfg.case_prune_bias_threshold,
        )
        return model, model
    return _build_actor_model(obs_dim, action_dim, cfg, device), _build_value_model(obs_dim, cfg, device)


def _copy_state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


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
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1)
    values_t = torch.as_tensor(values, dtype=torch.float32, device=device).view(-1)
    next_values_t = torch.as_tensor(next_values, dtype=torch.float32, device=device).view(-1)
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=device).view(-1)
    if not (rewards_t.numel() == values_t.numel() == next_values_t.numel() == terminated_t.numel()):
        raise ValueError("GAE inputs must have the same length")

    advantages = torch.zeros_like(rewards_t)
    running_advantage = torch.zeros((), dtype=torch.float32, device=rewards_t.device)
    for idx in range(rewards_t.numel() - 1, -1, -1):
        not_done = (~terminated_t[idx]).to(dtype=torch.float32)
        delta = rewards_t[idx] + float(gamma) * next_values_t[idx] * not_done - values_t[idx]
        running_advantage = delta + float(gamma) * float(gae_lambda) * not_done * running_advantage
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
    greedy: bool = False,
) -> tuple[int, list[float]]:
    if not model.is_policy_ready(min_case_entries=min_case_entries):
        action = int(env.action_space.sample())
        probs = [1.0 / float(model.action_dim) for _ in range(model.action_dim)]
        return action, probs
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        probs_tensor = model.policy_probs(obs_tensor).squeeze(0)
        probs = [float(value) for value in probs_tensor.detach().cpu().tolist()]
        if greedy:
            return int(probs_tensor.argmax().item()), probs
        action = int(torch.distributions.Categorical(probs=probs_tensor).sample().item())
        return action, probs


def _train_actor_critic_batch(
    actor: PolicyActor,
    value_model: ValueCritic,
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
    case_indices: list[int] = []
    rewards: list[torch.Tensor] = []
    terminated: list[bool] = []
    episode_returns: list[float] = []
    reward_variation_flags: list[torch.Tensor] = []
    reward_checked_episodes = 0
    for episode in episodes:
        observations.extend(episode["observations"])
        next_observations.extend(episode["next_observations"])
        actions.extend(episode["actions"])
        case_indices.extend(int(idx) for idx in episode.get("case_indices", []))
        policy_rewards = episode_policy_rewards(episode, cfg, device=device)
        if policy_rewards.numel() > 1:
            reward_checked_episodes += 1
            reward_variation_flags.append((policy_rewards != policy_rewards[0]).any())
        rewards.append(policy_rewards)
        terminated.extend(bool(done) for done in episode["terminated"])
        episode_returns.append(float(sum(episode["rewards"])))
    if not observations:
        return None

    obs_t = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=device)
    next_obs_t = torch.as_tensor(np.asarray(next_observations, dtype=np.float32), dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.cat(rewards).to(device)
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=device)
    case_indices_t = torch.as_tensor(case_indices, dtype=torch.long, device=device) if case_indices else None

    value_model.eval()
    with torch.no_grad():
        critic_inputs = torch.cat([obs_t, next_obs_t], dim=0)
        critic_values = value_model(critic_inputs)
        values_t, next_values_t = critic_values.chunk(2)
        raw_advantages, value_targets = compute_gae(
            rewards_t,
            values_t,
            next_values_t,
            terminated_t,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=device,
        )
        normalized_advantages = normalize_advantages(
            raw_advantages,
            epsilon=cfg.advantage_epsilon,
            clip=cfg.advantage_clip,
        )

    critic_loss = torch.zeros((), dtype=torch.float32, device=device)
    shared_actor_critic = actor is value_model and isinstance(actor, SharedNNKNNActorCriticNetwork)
    if shared_actor_critic:
        if case_indices_t is None or case_indices_t.numel() != obs_t.shape[0]:
            raise ValueError("Shared NN-kNN actor-critic updates require one case index per sample")
        value_model.update_value_labels(case_indices_t, value_targets.detach())
        value_model.train()
        critic_loss = F.mse_loss(value_model(obs_t), value_targets.detach())
    elif isinstance(value_model, NNKNNValueNetwork):
        value_model.add_cases(obs_t, value_targets.detach())
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
    chosen_probs = probs.gather(1, actions_t.view(-1, 1)).squeeze(1).clamp_min(cfg.advantage_epsilon)
    log_probs = torch.log(chosen_probs)
    policy_loss = -(log_probs * normalized_advantages.detach()).mean()
    probs_clamped = probs.clamp_min(cfg.advantage_epsilon)
    entropy = -(probs_clamped * probs_clamped.log()).sum(dim=1).mean()
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
        "bias_loss": float(bias_loss.detach().cpu().item()),
        "mean_reward": float(rewards_t.mean().detach().cpu().item()),
        "mean_advantage": float(raw_advantages.mean().detach().cpu().item()),
        "mean_normalized_advantage": float(normalized_advantages.mean().detach().cpu().item()),
        "value_mean": float(post_value_predictions.mean().detach().cpu().item()),
        "value_target_mean": float(value_targets.mean().detach().cpu().item()),
        "explained_variance": explained_variance(post_value_predictions, value_targets),
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
                action, _ = _select_action(
                    model,
                    np.asarray(obs, dtype=np.float32),
                    env=env,
                    device=run_device,
                    min_case_entries=1,
                    greedy=True,
                )
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
    actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    critic_optimizer = (
        optim.Adam(value_network.parameters(), lr=cfg.critic_learning_rate)
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
    total_pruned = 0
    total_replaced = 0

    obs, _ = env.reset(seed=cfg.seed)
    episode_observations: list[np.ndarray] = []
    episode_next_observations: list[np.ndarray] = []
    episode_actions: list[int] = []
    episode_case_indices: list[int] = []
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
            action, action_probs = _select_action(
                actor,
                obs_array,
                env=env,
                device=run_device,
                min_case_entries=cfg.min_case_entries,
                greedy=False,
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if _is_nnknn_policy_actor(actor):
                action_case = np.asarray([int(action)], dtype=np.int64)
                add_stats = actor.add_cases(obs_array[np.newaxis, :], action_case)
                if add_stats["added"] != 1:
                    raise RuntimeError("NN-kNN actor failed to store the selected state-action case.")
                episode_case_indices.append(int(add_stats["start"]))
                total_pruned += int(add_stats["pruned"])
                total_replaced += int(add_stats["replaced"])

            episode_observations.append(obs_array)
            episode_next_observations.append(np.asarray(next_obs, dtype=np.float32))
            episode_actions.append(action)
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
                and not isinstance(actor, SharedNNKNNActorCriticNetwork)
            ):
                pruned = actor.prune_cases() if isinstance(actor, NNKNNPolicyNetwork) else 0
                total_pruned += pruned
                if pruned and isinstance(actor, NNKNNPolicyNetwork):
                    maintenance_rows.append(
                        {
                            "global_step": completed_step,
                            "cases_pruned": pruned,
                            "case_entries": _actor_case_entries(actor),
                            "action_counts": _actor_action_counts_json(actor),
                            **_actor_case_bias_stats(actor),
                        }
                    )

            if episode_done:
                episode_index += 1
                pending_update_episodes.append(
                    {
                        "observations": episode_observations,
                        "next_observations": episode_next_observations,
                        "actions": episode_actions,
                        "case_indices": episode_case_indices,
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
                        actor_optimizer,
                        critic_optimizer,
                        pending_update_episodes,
                        cfg,
                        device=run_device,
                        global_step=completed_step,
                        completed_episodes=episode_index,
                    )
                    if loss_row is not None:
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
                        "cases_pruned": total_pruned,
                        "cases_replaced": total_replaced,
                        "action_probs": json.dumps(action_probs),
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
                episode_case_indices = []
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

        if pending_update_episodes:
            loss_row = _train_actor_critic_batch(
                actor,
                value_network,
                actor_optimizer,
                critic_optimizer,
                pending_update_episodes,
                cfg,
                device=run_device,
                global_step=cfg.total_timesteps,
                completed_episodes=episode_index,
            )
            if loss_row is not None:
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
        "cases_pruned": total_pruned,
        "cases_replaced": total_replaced,
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
