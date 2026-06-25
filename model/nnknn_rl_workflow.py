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
from model.nnknn_model import GlocalFeatureWeight, NN_KNN_Model
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
    critic_learning_rate: float = 1e-3
    critic_hidden_sizes: tuple[int, ...] = (128, 128)
    critic_update_epochs: int = 1
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
    source_reference: str = "NN-kNN actor with MLP value critic and GAE advantages"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NNKNNQNetwork(nn.Module):
    """Actor wrapper around NN_KNN_Model for discrete-action RL.

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

    def q_values(self, observations: torch.Tensor) -> torch.Tensor:
        """Compatibility alias: returns action probabilities, not Q-values."""

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
        added = 0
        replaced = 0
        pruned = 0
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
            self.nnknn_model.append_cases(obs_t[idx].unsqueeze(0), label)
            added += 1
        return {"added": added, "pruned": pruned, "replaced": replaced}

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


def _model_state(model: NNKNNQNetwork) -> dict[str, Any]:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "case_state": model.case_state(),
    }


def _load_model_state(model: NNKNNQNetwork, state: dict[str, Any]) -> None:
    model.load_state_dict(state["state_dict"])
    model.load_case_state(state.get("case_state", {}))


def _build_model(obs_dim: int, action_dim: int, cfg: NNKNNRLConfig, device: torch.device) -> NNKNNQNetwork:
    model = NNKNNQNetwork(
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


def _build_value_model(obs_dim: int, cfg: NNKNNRLConfig, device: torch.device) -> ValueNetwork:
    return ValueNetwork(obs_dim, tuple(cfg.critic_hidden_sizes)).to(device)


def _copy_state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


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
    model: NNKNNQNetwork,
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
    actor: NNKNNQNetwork,
    value_model: ValueNetwork,
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    episodes: list[dict[str, Any]],
    cfg: NNKNNRLConfig,
    *,
    device: torch.device,
    global_step: int,
) -> dict[str, Any] | None:
    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[torch.Tensor] = []
    terminated: list[bool] = []
    episode_returns: list[float] = []
    reward_variation_flags: list[torch.Tensor] = []
    reward_checked_episodes = 0
    for episode in episodes:
        observations.extend(episode["observations"])
        next_observations.extend(episode["next_observations"])
        actions.extend(episode["actions"])
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

    value_model.eval()
    with torch.no_grad():
        values_t = value_model(obs_t)
        next_values_t = value_model(next_obs_t)
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

    value_model.train()
    critic_loss = torch.zeros((), dtype=torch.float32, device=device)
    value_predictions = value_model(obs_t)
    for _epoch in range(max(1, int(cfg.critic_update_epochs))):
        value_predictions = value_model(obs_t)
        critic_loss = F.mse_loss(value_predictions, value_targets.detach())
        critic_optimizer.zero_grad()
        (float(cfg.value_loss_coef) * critic_loss).backward()
        nn.utils.clip_grad_norm_(value_model.parameters(), cfg.max_grad_norm)
        critic_optimizer.step()

    value_model.eval()
    with torch.no_grad():
        post_value_predictions = value_model(obs_t)

    actor.train()
    probs = actor.policy_probs(obs_t)
    chosen_probs = probs.gather(1, actions_t.view(-1, 1)).squeeze(1).clamp_min(cfg.advantage_epsilon)
    log_probs = torch.log(chosen_probs)
    policy_loss = -(log_probs * normalized_advantages.detach()).mean()
    probs_clamped = probs.clamp_min(cfg.advantage_epsilon)
    entropy = -(probs_clamped * probs_clamped.log()).sum(dim=1).mean()
    active_biases = actor.nnknn_model.biases[: actor.case_entries]
    bias_loss = active_biases.pow(2).mean() if active_biases.numel() else torch.zeros((), device=device)
    actor_loss = policy_loss - cfg.entropy_coef * entropy + cfg.case_bias_l2 * bias_loss

    actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
    actor_optimizer.step()

    stats = actor.case_bias_stats()
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
        "case_entries": actor.case_entries,
        "action_counts": json.dumps(actor.action_counts().detach().cpu().tolist()),
        **stats,
    }


def evaluate_nnknn_rl(
    task_name: str,
    model: NNKNNQNetwork,
    *,
    episodes: int = 20,
    seed: int = 10_000,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run greedy-policy evaluation for the NN-kNN-RL workflow."""

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
    """Train the NN-kNN actor with an MLP value critic and GAE advantages."""

    spec = get_rl_task_spec(task_name)
    cfg = config or make_nnknn_rl_config(spec.default_profile)
    seed_everything(cfg.seed)
    run_device = _resolve_device_arg(device)

    env = _make_env(spec, seed=cfg.seed)
    obs_dim, action_dim = _validate_env_spaces(env, spec)
    q_network = _build_model(obs_dim, action_dim, cfg, run_device)
    value_network = _build_value_model(obs_dim, cfg, run_device)
    actor_optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    critic_optimizer = optim.Adam(value_network.parameters(), lr=cfg.critic_learning_rate)

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
                q_network,
                obs_array,
                env=env,
                device=run_device,
                min_case_entries=cfg.min_case_entries,
                greedy=False,
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ##TODO, Elliot double check this
            add_stats = q_network.add_cases(obs_array, np.asarray([action], dtype=np.int64))
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
            if cfg.case_maintenance_frequency > 0 and completed_step % cfg.case_maintenance_frequency == 0:
                pruned = q_network.prune_cases()
                total_pruned += pruned
                if pruned:
                    maintenance_rows.append(
                        {
                            "global_step": completed_step,
                            "cases_pruned": pruned,
                            "case_entries": q_network.case_entries,
                            "action_counts": json.dumps(q_network.action_counts().detach().cpu().tolist()),
                            **q_network.case_bias_stats(),
                        }
                    )

            if episode_done:
                episode_index += 1
                pending_update_episodes.append(
                    {
                        "observations": episode_observations,
                        "next_observations": episode_next_observations,
                        "actions": episode_actions,
                        "rewards": episode_rewards,
                        "terminated": episode_terminated,
                        "truncated": episode_truncated,
                    }
                )
                latest_loss: float | None = None
                if len(pending_update_episodes) >= cfg.policy_update_episodes:
                    loss_row = _train_actor_critic_batch(
                        q_network,
                        value_network,
                        actor_optimizer,
                        critic_optimizer,
                        pending_update_episodes,
                        cfg,
                        device=run_device,
                        global_step=completed_step,
                    )
                    if loss_row is not None:
                        loss_rows.append(loss_row)
                        latest_loss = float(loss_row["loss"])
                    pending_update_episodes = []

                stats = q_network.case_bias_stats()
                training_rows.append(
                    {
                        "global_step": completed_step,
                        "episode": episode_index,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "loss": latest_loss,
                        "case_entries": q_network.case_entries,
                        "cases_pruned": total_pruned,
                        "cases_replaced": total_replaced,
                        "action_probs": json.dumps(action_probs),
                        "action_counts": json.dumps(q_network.action_counts().detach().cpu().tolist()),
                        **stats,
                    }
                )
                if progress and (episode_index <= 5 or episode_index % 10 == 0):
                    print(
                        "[nnknn-rl] "
                        f"step={completed_step} episode={episode_index} "
                        f"return={episode_return:.1f} length={episode_length} "
                        f"loss={latest_loss} cases={q_network.case_entries}",
                        flush=True,
                    )

                if (
                    cfg.eval_episode_frequency > 0
                    and episode_index > 0
                    and episode_index % cfg.eval_episode_frequency == 0
                ):
                    eval_metrics = evaluate_nnknn_rl(
                        spec.name,
                        q_network,
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
                        "case_entries": q_network.case_entries,
                    }
                    eval_rows.append(eval_row)
                    if best_eval is None or eval_metrics["mean_return"] > best_eval["mean_return"]:
                        best_eval = eval_metrics
                        best_eval_step = completed_step
                        best_actor_state = _model_state(q_network)
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
                episode_rewards = []
                episode_terminated = []
                episode_truncated = []
                episode_return = 0.0
                episode_length = 0

            if cfg.eval_frequency > 0 and global_step > 0 and global_step % cfg.eval_frequency == 0:
                eval_metrics = evaluate_nnknn_rl(
                    spec.name,
                    q_network,
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
                    "case_entries": q_network.case_entries,
                }
                eval_rows.append(eval_row)
                if best_eval is None or eval_metrics["mean_return"] > best_eval["mean_return"]:
                    best_eval = eval_metrics
                    best_eval_step = global_step
                    best_actor_state = _model_state(q_network)
                    best_critic_state = _copy_state_dict_to_cpu(value_network)

        if pending_update_episodes:
            loss_row = _train_actor_critic_batch(
                q_network,
                value_network,
                actor_optimizer,
                critic_optimizer,
                pending_update_episodes,
                cfg,
                device=run_device,
                global_step=cfg.total_timesteps,
            )
            if loss_row is not None:
                loss_rows.append(loss_row)
    finally:
        env.close()

    last_eval = evaluate_nnknn_rl(
        spec.name,
        q_network,
        episodes=cfg.eval_episodes,
        seed=cfg.eval_seed,
        device=run_device,
    )
    if best_eval is None or last_eval["mean_return"] >= best_eval["mean_return"]:
        selected_eval = last_eval
        selected_step = cfg.total_timesteps
        selected_source = "final"
        selected_actor_state = _model_state(q_network)
        selected_critic_state = _copy_state_dict_to_cpu(value_network)
    else:
        selected_eval = best_eval
        selected_step = int(best_eval_step or 0)
        selected_source = "best_eval"
        selected_actor_state = best_actor_state or _model_state(q_network)
        selected_critic_state = best_critic_state or _copy_state_dict_to_cpu(value_network)
        _load_model_state(q_network, selected_actor_state)
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
        "case_entries": q_network.case_entries,
        "action_counts": q_network.action_counts().detach().cpu().tolist(),
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
        "case_entries": q_network.case_entries,
        "action_counts": q_network.action_counts().detach().cpu().tolist(),
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
        "model": q_network,
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
    config_data["critic_hidden_sizes"] = tuple(config_data.get("critic_hidden_sizes", (128, 128)))
    cfg = NNKNNRLConfig(**config_data)
    model = _build_model(int(checkpoint["obs_dim"]), int(checkpoint["action_dim"]), cfg, run_device)
    value_model = _build_value_model(int(checkpoint["obs_dim"]), cfg, run_device)
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
