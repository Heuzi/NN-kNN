from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
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
from model.rl_workflow import (
    ReplayBuffer,
    _build_training_efficiency,
    _first_threshold_step,
    _json_default,
    _linear_schedule,
    _make_env,
    _resolve_device_arg,
    _validate_env_spaces,
    seed_everything,
)


@dataclass(frozen=True)
class NNKNNRLConfig:
    """Training configuration for the repo-native NN-kNN-RL CartPole workflow."""

    profile: str = "fast"
    seed: int = 0
    total_timesteps: int = 150_000
    learning_rate: float = 5e-4
    buffer_size: int = 20_000
    case_capacity: int = 10_000
    gamma: float = 0.99
    target_network_frequency: int = 500
    max_grad_norm: float = 10.0
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 1_000
    train_frequency: int = 1
    eval_frequency: int = 0
    eval_episode_frequency: int = 100
    eval_episodes: int = 20
    eval_seed: int = 10_000
    success_threshold: float | None = 475.0
    feature_dim: int = 32
    hidden_sizes: tuple[int, int] = (128, 128)
    tau: float = 1.0
    top_k: int = 64
    min_case_entries: int = 32
    case_default_bias: float = 0.0
    source_reference: str = "DQN-style TD learning with NN-kNN case retrieval"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


class NNKNNQNetwork(nn.Module):
    """Case-based Q approximator over state/action queries.

    Queries are `[observation, one_hot(action)]`. Stored cases use the same
    representation and carry scalar TD target labels. A small feature projector,
    trainable case biases, and positive feature weights define normalized
    case attention before aggregating case labels into Q values.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        case_capacity: int,
        feature_dim: int = 32,
        hidden_sizes: tuple[int, int] = (128, 128),
        tau: float = 1.0,
        top_k: int = 64,
        case_default_bias: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.case_capacity = int(case_capacity)
        self.feature_dim = int(feature_dim)
        self.tau = float(tau)
        self.top_k = int(top_k)
        self.case_default_bias = float(case_default_bias)
        input_dim = self.obs_dim + self.action_dim
        h1, h2 = hidden_sizes
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, self.feature_dim),
        )
        self.case_biases = nn.Parameter(torch.full((self.case_capacity,), self.case_default_bias))
        self.feature_weights = nn.Parameter(torch.ones(self.feature_dim))
        self.register_buffer("case_inputs", torch.zeros(self.case_capacity, input_dim))
        self.register_buffer("case_observations", torch.zeros(self.case_capacity, self.obs_dim))
        self.register_buffer("case_actions", torch.zeros(self.case_capacity, dtype=torch.long))
        self.register_buffer("case_values", torch.zeros(self.case_capacity))
        self.case_entries = 0
        self.case_pos = 0

    def _one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), num_classes=self.action_dim).to(dtype=torch.float32)

    def _make_inputs(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([observations.float(), self._one_hot(actions)], dim=1)

    def add_cases(
        self,
        observations: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
        values: torch.Tensor | np.ndarray,
    ) -> None:
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.case_inputs.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.case_inputs.device).view(-1)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.case_inputs.device).view(-1)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        inputs = self._make_inputs(obs_t, actions_t)
        with torch.no_grad():
            for idx in range(inputs.shape[0]):
                pos = self.case_pos
                self.case_inputs[pos].copy_(inputs[idx])
                self.case_observations[pos].copy_(obs_t[idx])
                self.case_actions[pos] = actions_t[idx]
                self.case_values[pos] = values_t[idx]
                self.case_pos = (self.case_pos + 1) % self.case_capacity
                self.case_entries = min(self.case_entries + 1, self.case_capacity)

    def _active_case_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = int(self.case_entries)
        return (
            self.case_inputs[:size],
            self.case_values[:size],
            self.case_biases[:size],
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if int(self.case_entries) <= 0:
            empty_features = self.feature_net(self._make_inputs(observations, actions))
            return empty_features.sum(dim=1) * 0.0

        query_inputs = self._make_inputs(observations, actions)
        query_features = self.feature_net(query_inputs)
        case_inputs, case_values, case_biases = self._active_case_tensors()
        case_features = self.feature_net(case_inputs)
        positive_weights = F.softplus(self.feature_weights).view(1, 1, -1)
        distances = torch.sqrt(
            torch.relu(((query_features.unsqueeze(1) - case_features.unsqueeze(0)) ** 2 * positive_weights).sum(dim=2))
            + 1e-12
        )
        scores = case_biases.unsqueeze(0) - distances
        if self.top_k > 0 and self.top_k < scores.shape[1]:
            top_values, top_indices = torch.topk(scores, k=self.top_k, dim=1)
            masked = torch.full_like(scores, float("-inf"))
            scores = masked.scatter(1, top_indices, top_values)
        weights = torch.softmax(scores / max(self.tau, 1e-6), dim=1)
        return (weights * case_values.unsqueeze(0)).sum(dim=1)

    def q_values(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        values = []
        for action in range(self.action_dim):
            actions = torch.full((batch_size,), action, dtype=torch.long, device=observations.device)
            values.append(self.forward(observations, actions))
        return torch.stack(values, dim=1)

    def explain(
        self,
        observation: torch.Tensor | np.ndarray,
        action: int,
        *,
        k: int | None = None,
    ) -> dict[str, Any]:
        if int(self.case_entries) <= 0:
            return {"case_entries": 0, "neighbors": []}
        device = self.case_inputs.device
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=device).view(1, self.obs_dim)
        action_t = torch.tensor([int(action)], dtype=torch.long, device=device)
        query_features = self.feature_net(self._make_inputs(obs_t, action_t))
        case_inputs, case_values, case_biases = self._active_case_tensors()
        case_features = self.feature_net(case_inputs)
        positive_weights = F.softplus(self.feature_weights).view(1, -1)
        distances = torch.sqrt(torch.relu(((case_features - query_features) ** 2 * positive_weights).sum(dim=1)) + 1e-12)
        scores = case_biases - distances
        k_eff = min(int(k or self.top_k), int(self.case_entries))
        top_scores, top_indices = torch.topk(scores, k=k_eff)
        top_weights = torch.softmax(top_scores / max(self.tau, 1e-6), dim=0)
        indices = top_indices.detach().cpu().tolist()
        neighbors = []
        for rank, idx in enumerate(indices):
            neighbors.append(
                {
                    "rank": rank + 1,
                    "index": int(idx),
                    "action": int(self.case_actions[idx].detach().cpu().item()),
                    "value": float(case_values[idx].detach().cpu().item()),
                    "distance": float(distances[idx].detach().cpu().item()),
                    "weight": float(top_weights[rank].detach().cpu().item()),
                    "observation": self.case_observations[idx].detach().cpu().numpy().tolist(),
                }
            )
        return {
            "case_entries": int(self.case_entries),
            "query_action": int(action),
            "q_value": float(self.forward(obs_t, action_t).detach().cpu().item()),
            "neighbors": neighbors,
        }

    def case_state(self) -> dict[str, int]:
        return {"case_entries": int(self.case_entries), "case_pos": int(self.case_pos)}

    def load_case_state(self, state: dict[str, Any]) -> None:
        self.case_entries = int(state.get("case_entries", 0))
        self.case_pos = int(state.get("case_pos", 0))


def make_nnknn_rl_config(profile: str = "fast", **overrides: Any) -> NNKNNRLConfig:
    profiles: dict[str, dict[str, Any]] = {
        "smoke": {
            "profile": "smoke",
            "total_timesteps": 256,
            "buffer_size": 1_000,
            "case_capacity": 1_000,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_network_frequency": 100,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 2,
            "min_case_entries": 8,
            "top_k": 16,
            "success_threshold": None,
        },
        "debug": {
            "profile": "debug",
            "total_timesteps": 25_000,
            "buffer_size": 10_000,
            "case_capacity": 5_000,
            "batch_size": 64,
            "learning_starts": 256,
            "target_network_frequency": 250,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "min_case_entries": 32,
            "top_k": 64,
            "success_threshold": 475.0,
        },
        "fast": {
            "profile": "fast",
            "total_timesteps": 150_000,
            "learning_rate": 5e-4,
            "buffer_size": 20_000,
            "case_capacity": 10_000,
            "batch_size": 128,
            "learning_starts": 1_000,
            "train_frequency": 1,
            "target_network_frequency": 500,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "success_threshold": 475.0,
        },
        "gold": {
            "profile": "gold",
            "total_timesteps": 500_000,
            "learning_rate": 5e-4,
            "buffer_size": 50_000,
            "case_capacity": 25_000,
            "batch_size": 128,
            "learning_starts": 1_000,
            "target_network_frequency": 500,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "success_threshold": 475.0,
        },
    }
    normalized = profile.strip().lower()
    if normalized not in profiles:
        raise ValueError(f"Unknown NN-kNN-RL profile '{profile}'. Choose one of: {', '.join(sorted(profiles))}")
    data = {**profiles[normalized], **overrides}
    if "hidden_sizes" in data and not isinstance(data["hidden_sizes"], tuple):
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
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
    return NNKNNQNetwork(
        obs_dim,
        action_dim,
        case_capacity=cfg.case_capacity,
        feature_dim=cfg.feature_dim,
        hidden_sizes=cfg.hidden_sizes,
        tau=cfg.tau,
        top_k=cfg.top_k,
        case_default_bias=cfg.case_default_bias,
    ).to(device)


def _select_action(
    model: NNKNNQNetwork,
    obs: np.ndarray,
    *,
    epsilon: float,
    env: Any,
    device: torch.device,
) -> tuple[int, list[float]]:
    if random.random() < epsilon or model.case_entries <= 0:
        action = int(env.action_space.sample())
        q_values = [0.0 for _ in range(model.action_dim)]
        return action, q_values
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values_tensor = model.q_values(obs_tensor).squeeze(0)
        q_values = [float(value) for value in q_values_tensor.detach().cpu().tolist()]
        return int(q_values_tensor.argmax().item()), q_values


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
                    epsilon=0.0,
                    env=env,
                    device=run_device,
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
    """Train the DQN-style NN-kNN-RL workflow and write reproducible artifacts."""

    spec = get_rl_task_spec(task_name)
    cfg = config or make_nnknn_rl_config(spec.default_profile)
    seed_everything(cfg.seed)
    run_device = _resolve_device_arg(device)

    env = _make_env(spec, seed=cfg.seed)
    obs_dim, action_dim = _validate_env_spaces(env, spec)
    q_network = _build_model(obs_dim, action_dim, cfg, run_device)
    target_network = _build_model(obs_dim, action_dim, cfg, run_device)
    _load_model_state(target_network, _model_state(q_network))
    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    replay_buffer = ReplayBuffer.create(cfg.buffer_size, obs_dim)

    run_dir = Path(output_dir) if output_dir is not None else make_nnknn_rl_output_dir(spec.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    created_at = datetime.now(timezone.utc)

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    latest_loss: float | None = None
    latest_q_value: float | None = None
    best_eval: dict[str, Any] | None = None
    best_eval_step: int | None = None
    best_model_state: dict[str, Any] | None = None
    best_target_state: dict[str, Any] | None = None

    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0
    episode_index = 0

    try:
        for global_step in range(cfg.total_timesteps):
            epsilon = _linear_schedule(
                cfg.start_e,
                cfg.end_e,
                int(cfg.exploration_fraction * cfg.total_timesteps),
                global_step,
            )
            obs_array = np.asarray(obs, dtype=np.float32)
            action, q_values = _select_action(q_network, obs_array, epsilon=epsilon, env=env, device=run_device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            train_done = bool(terminated)
            episode_done = bool(terminated or truncated)
            next_obs_array = np.asarray(next_obs, dtype=np.float32)
            replay_buffer.add(obs_array, action, float(reward), next_obs_array, train_done)
            q_network.add_cases(obs_array, np.asarray([action]), np.asarray([reward], dtype=np.float32))
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            if (
                replay_buffer.size >= cfg.learning_starts
                and q_network.case_entries >= cfg.min_case_entries
                and global_step % cfg.train_frequency == 0
            ):
                batch = replay_buffer.sample(cfg.batch_size, run_device)
                with torch.no_grad():
                    next_q = target_network.q_values(batch["next_observations"]).max(dim=1).values
                    td_target = batch["rewards"] + cfg.gamma * next_q * (1.0 - batch["dones"])
                pred_q = q_network(batch["observations"], batch["actions"])
                loss = F.smooth_l1_loss(pred_q, td_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), cfg.max_grad_norm)
                optimizer.step()
                q_network.add_cases(batch["observations"], batch["actions"], td_target)
                latest_loss = float(loss.detach().cpu().item())
                latest_q_value = float(pred_q.detach().mean().cpu().item())
                loss_rows.append(
                    {
                        "global_step": global_step,
                        "loss": latest_loss,
                        "mean_q_value": latest_q_value,
                        "epsilon": epsilon,
                        "case_entries": q_network.case_entries,
                    }
                )

            if global_step > 0 and global_step % cfg.target_network_frequency == 0:
                _load_model_state(target_network, _model_state(q_network))

            if episode_done:
                episode_index += 1
                training_rows.append(
                    {
                        "global_step": global_step + 1,
                        "episode": episode_index,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "epsilon": epsilon,
                        "loss": latest_loss,
                        "mean_q_value": latest_q_value,
                        "case_entries": q_network.case_entries,
                        "q_values": json.dumps(q_values),
                    }
                )
                if progress and (episode_index <= 5 or episode_index % 10 == 0):
                    print(
                        "[nnknn-rl] "
                        f"step={global_step + 1} episode={episode_index} "
                        f"return={episode_return:.1f} length={episode_length} "
                        f"epsilon={epsilon:.3f} loss={latest_loss} cases={q_network.case_entries}",
                        flush=True,
                    )
                if (
                    cfg.eval_episode_frequency > 0
                    and episode_index > 0
                    and episode_index % cfg.eval_episode_frequency == 0
                ):
                    completed_step = global_step + 1
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
                        "eval_trigger": "episode",
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
                        best_model_state = _model_state(q_network)
                        best_target_state = _model_state(target_network)
                    if progress:
                        print(
                            "[nnknn-rl][eval] "
                            f"step={completed_step} episode={episode_index} "
                            f"mean_return={eval_row['mean_return']:.2f} "
                            f"max_return={eval_row['max_return']:.2f}",
                            flush=True,
                        )
                obs, _ = env.reset(seed=cfg.seed + episode_index)
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
                    "eval_trigger": "step",
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
                    best_model_state = _model_state(q_network)
                    best_target_state = _model_state(target_network)
                if progress:
                    print(
                        "[nnknn-rl][eval] "
                        f"step={global_step} mean_return={eval_row['mean_return']:.2f} "
                        f"max_return={eval_row['max_return']:.2f}",
                        flush=True,
                    )
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
        selected_model_state = _model_state(q_network)
        selected_target_state = _model_state(target_network)
    else:
        selected_eval = best_eval
        selected_step = int(best_eval_step or 0)
        selected_source = "best_eval"
        selected_model_state = best_model_state or _model_state(q_network)
        selected_target_state = best_target_state or _model_state(target_network)
        _load_model_state(q_network, selected_model_state)
        _load_model_state(target_network, selected_target_state)

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
        "model_state": selected_model_state,
        "target_model_state": selected_target_state,
        "task": spec.to_dict(),
        "config": cfg.to_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "selected_eval": {k: v for k, v in selected_eval.items() if k != "episode_metrics"},
        "last_eval": {k: v for k, v in last_eval.items() if k != "episode_metrics"},
        "selected_step": selected_step,
        "selected_source": selected_source,
        "training_efficiency": training_efficiency,
        "passed": passed,
    }
    torch.save(checkpoint, checkpoint_path)

    _write_json(
        run_dir / "config.json",
        {
            "created_at_utc": created_at.isoformat(),
            "task": spec.to_dict(),
            "config": cfg.to_dict(),
            "device": str(run_device),
            "source_reference": cfg.source_reference,
        },
    )
    _write_csv(run_dir / "training_metrics.csv", training_rows)
    _write_csv(run_dir / "loss_metrics.csv", loss_rows)
    _write_csv(run_dir / "eval_metrics.csv", eval_rows)
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
        "final_eval": {k: v for k, v in selected_eval.items() if k != "episode_metrics"},
        "last_eval": {k: v for k, v in last_eval.items() if k != "episode_metrics"},
        "selected_step": selected_step,
        "selected_source": selected_source,
        "training_efficiency": training_efficiency,
        "case_entries": q_network.case_entries,
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "manifest.json",
        {
            "created_at_utc": created_at.isoformat(),
            "task": spec.name,
            "env_id": spec.env_id,
            "profile": cfg.profile,
            "outputs": [
                "config.json",
                "training_metrics.csv",
                "loss_metrics.csv",
                "eval_metrics.csv",
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
        "target_model": target_network,
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
    config_data = dict(checkpoint["config"])
    config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
    cfg = NNKNNRLConfig(**config_data)
    model = _build_model(int(checkpoint["obs_dim"]), int(checkpoint["action_dim"]), cfg, run_device)
    target_model = _build_model(int(checkpoint["obs_dim"]), int(checkpoint["action_dim"]), cfg, run_device)
    _load_model_state(model, checkpoint["model_state"])
    _load_model_state(target_model, checkpoint["target_model_state"])
    model.eval()
    target_model.eval()
    return {
        "model": model,
        "target_model": target_model,
        "config": cfg,
        "task": checkpoint["task"],
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "device": run_device,
    }
