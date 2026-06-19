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
    _build_training_efficiency,
    _first_threshold_step,
    _json_default,
    _make_env,
    _resolve_device_arg,
    _validate_env_spaces,
    seed_everything,
)


class NECEmbeddingNetwork(nn.Module):
    """Small MLP embedding network for flat observations used by NEC."""

    def __init__(
        self,
        obs_dim: int,
        embedding_dim: int = 32,
        hidden_sizes: tuple[int, int] = (128, 128),
    ):
        super().__init__()
        h1, h2 = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(frozen=True)
class NECConfig:
    """Training configuration for the repo-native NEC baseline."""

    profile: str = "fast"
    seed: int = 0
    total_timesteps: int = 150_000
    learning_rate: float = 1e-3
    replay_size: int = 20_000
    dictionary_size: int = 10_000
    gamma: float = 0.99
    n_step: int = 50
    k_neighbors: int = 10
    kernel_delta: float = 1e-3
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
    embedding_dim: int = 32
    hidden_sizes: tuple[int, int] = (128, 128)
    max_grad_norm: float = 10.0
    source_reference: str = "Neural Episodic Control paper; EndingCredits/Neural-Episodic-Control as reference only"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


@dataclass
class NECReplayBuffer:
    observations: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    terminals: np.ndarray
    pos: int = 0
    size: int = 0

    @classmethod
    def create(cls, replay_size: int, obs_dim: int) -> "NECReplayBuffer":
        return cls(
            observations=np.zeros((replay_size, obs_dim), dtype=np.float32),
            actions=np.zeros(replay_size, dtype=np.int64),
            returns=np.zeros(replay_size, dtype=np.float32),
            terminals=np.zeros(replay_size, dtype=np.float32),
        )

    def add(self, obs: np.ndarray, action: int, return_value: float, terminal: bool) -> None:
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.returns[self.pos] = return_value
        self.terminals[self.pos] = float(terminal)
        self.pos = (self.pos + 1) % self.observations.shape[0]
        self.size = min(self.size + 1, self.observations.shape[0])

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device).long(),
            "returns": torch.as_tensor(self.returns[indices], device=device),
            "terminals": torch.as_tensor(self.terminals[indices], device=device),
        }


class ActionValueDictionary:
    """Exact per-action kNN dictionary with LRU replacement."""

    def __init__(
        self,
        capacity: int,
        embedding_dim: int,
        observation_dim: int | None = None,
    ):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.observation_dim = observation_dim
        self.embeddings = np.zeros((capacity, embedding_dim), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.observations = (
            np.zeros((capacity, observation_dim), dtype=np.float32)
            if observation_dim is not None
            else None
        )
        self.lru = np.zeros(capacity, dtype=np.float64)
        self.size = 0
        self.clock = 0.0

    def queryable(self, k: int) -> bool:
        return self.size >= k

    def add(self, embedding: np.ndarray, value: float, observation: np.ndarray | None = None) -> None:
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            index = int(np.argmin(self.lru))
        self.embeddings[index] = embedding
        self.values[index] = value
        if self.observations is not None:
            if observation is None:
                raise ValueError(
                    "observation is required when the DND was configured to store observations"
                )
            self.observations[index] = np.asarray(observation, dtype=np.float32)
        self.lru[index] = self.clock
        self.clock += 1.0

    def nearest_tensors(
        self,
        query_embeddings: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.queryable(k):
            return None
        device = query_embeddings.device
        stored_embeddings = torch.as_tensor(self.embeddings[: self.size], device=device)
        stored_values = torch.as_tensor(self.values[: self.size], device=device)
        distances = torch.cdist(query_embeddings, stored_embeddings)
        _, indices = torch.topk(distances, k=k, dim=1, largest=False)
        touched = indices.detach().cpu().numpy().reshape(-1)
        for index in touched:
            self.lru[int(index)] = self.clock
            self.clock += 1.0
        return stored_embeddings[indices], stored_values[indices]

    def to_state(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "embedding_dim": self.embedding_dim,
            "observation_dim": self.observation_dim,
            "embeddings": self.embeddings[: self.size].copy(),
            "values": self.values[: self.size].copy(),
            "observations": None
            if self.observations is None
            else self.observations[: self.size].copy(),
            "lru": self.lru[: self.size].copy(),
            "size": self.size,
            "clock": self.clock,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ActionValueDictionary":
        observations = state.get("observations")
        observation_dim = state.get("observation_dim")
        if observation_dim is None and observations is not None:
            observation_dim = int(observations.shape[1])
        dictionary = cls(
            int(state["capacity"]),
            int(state["embedding_dim"]),
            None if observation_dim is None else int(observation_dim),
        )
        size = int(state["size"])
        dictionary.size = size
        dictionary.embeddings[:size] = state["embeddings"]
        dictionary.values[:size] = state["values"]
        if observations is not None and dictionary.observations is not None:
            dictionary.observations[:size] = observations
        dictionary.lru[:size] = state["lru"]
        dictionary.clock = float(state["clock"])
        return dictionary


class DifferentiableNeuralDictionary:
    """Per-action dictionary set used by NEC."""

    def __init__(
        self,
        action_dim: int,
        capacity: int,
        embedding_dim: int,
        observation_dim: int | None = None,
    ):
        self.action_dim = action_dim
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.observation_dim = observation_dim
        self.dicts = [
            ActionValueDictionary(capacity, embedding_dim, observation_dim)
            for _ in range(action_dim)
        ]

    def add(
        self,
        embedding: np.ndarray,
        action: int,
        value: float,
        observation: np.ndarray | None = None,
    ) -> None:
        self.dicts[int(action)].add(embedding.astype(np.float32), float(value), observation)

    def q_values(
        self,
        embeddings: torch.Tensor,
        *,
        k: int,
        delta: float,
    ) -> torch.Tensor:
        values = []
        for action_dictionary in self.dicts:
            nearest = action_dictionary.nearest_tensors(embeddings, k)
            if nearest is None:
                values.append(torch.zeros(embeddings.shape[0], device=embeddings.device) + embeddings.sum(dim=1) * 0.0)
                continue
            neighbor_embeddings, neighbor_values = nearest
            distances = torch.linalg.norm(embeddings.unsqueeze(1) - neighbor_embeddings, dim=2)
            weights = 1.0 / (distances + delta)
            weights = weights / weights.sum(dim=1, keepdim=True)
            values.append((weights * neighbor_values).sum(dim=1))
        return torch.stack(values, dim=1)

    def q_values_for_actions(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor,
        *,
        k: int,
        delta: float,
    ) -> torch.Tensor:
        all_values = self.q_values(embeddings, k=k, delta=delta)
        return all_values.gather(1, actions.view(-1, 1)).squeeze(1)

    def min_size(self) -> int:
        return min(dictionary.size for dictionary in self.dicts)

    def total_size(self) -> int:
        return sum(dictionary.size for dictionary in self.dicts)

    def to_state(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "embedding_dim": self.embedding_dim,
            "observation_dim": self.observation_dim,
            "dicts": [dictionary.to_state() for dictionary in self.dicts],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "DifferentiableNeuralDictionary":
        observation_dim = state.get("observation_dim")
        dnd = cls(
            int(state["action_dim"]),
            int(state["capacity"]),
            int(state["embedding_dim"]),
            None if observation_dim is None else int(observation_dim),
        )
        dnd.dicts = [ActionValueDictionary.from_state(item) for item in state["dicts"]]
        if dnd.observation_dim is None and dnd.dicts and dnd.dicts[0].observation_dim is not None:
            dnd.observation_dim = dnd.dicts[0].observation_dim
        return dnd


def make_nec_config(profile: str = "fast", **overrides: Any) -> NECConfig:
    profiles: dict[str, dict[str, Any]] = {
        "smoke": {
            "profile": "smoke",
            "total_timesteps": 256,
            "replay_size": 1_000,
            "dictionary_size": 1_000,
            "batch_size": 32,
            "learning_starts": 32,
            "n_step": 20,
            "k_neighbors": 3,
            "train_frequency": 1,
            "eval_frequency": 0,
            "eval_episode_frequency": 10,
            "eval_episodes": 2,
            "success_threshold": None,
        },
        "debug": {
            "profile": "debug",
            "total_timesteps": 25_000,
            "learning_rate": 1e-3,
            "replay_size": 5_000,
            "dictionary_size": 2_500,
            "batch_size": 32,
            "learning_starts": 256,
            "train_frequency": 4,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "success_threshold": 475.0,
        },
        "fast": {
            "profile": "fast",
            "total_timesteps": 150_000,
            "learning_rate": 1e-3,
            "replay_size": 20_000,
            "dictionary_size": 10_000,
            "batch_size": 128,
            "learning_starts": 1_000,
            "train_frequency": 1,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "success_threshold": 475.0,
        },
        "gold": {
            "profile": "gold",
            "total_timesteps": 500_000,
            "learning_rate": 1e-3,
            "replay_size": 50_000,
            "dictionary_size": 25_000,
            "batch_size": 64,
            "learning_starts": 1_000,
            "train_frequency": 4,
            "eval_frequency": 0,
            "eval_episode_frequency": 100,
            "eval_episodes": 20,
            "success_threshold": 475.0,
        },
    }
    normalized = profile.strip().lower()
    if normalized not in profiles:
        raise ValueError(f"Unknown NEC profile '{profile}'. Choose one of: {', '.join(sorted(profiles))}")
    data = {**profiles[normalized], **overrides}
    if "hidden_sizes" in data and not isinstance(data["hidden_sizes"], tuple):
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
    return NECConfig(**data)


def make_nec_output_dir(
    task_name: str,
    *,
    parent: str | Path = "results/rl",
    suffix: str | None = None,
) -> Path:
    created_at = datetime.now(timezone.utc)
    stem = f"nec_{task_name}_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
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
    raise FileExistsError(f"Could not create a unique NEC output directory under {parent_path}")


def _linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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


def _copy_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _compute_n_step_returns(rewards: list[float], bootstrap_values: list[float], gamma: float, n_step: int) -> list[float]:
    returns: list[float] = []
    horizon = len(rewards)
    for start in range(horizon):
        end = min(start + n_step, horizon)
        value = bootstrap_values[end] if end < horizon else 0.0
        for idx in range(end - 1, start - 1, -1):
            value = rewards[idx] + gamma * value
        returns.append(float(value))
    return returns


def _select_action(
    model: NECEmbeddingNetwork,
    dnd: DifferentiableNeuralDictionary,
    obs: np.ndarray,
    *,
    k: int,
    delta: float,
    epsilon: float,
    env: Any,
    device: torch.device,
) -> tuple[int, float, np.ndarray, list[float]]:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        embedding = model(obs_tensor)
        q_values = dnd.q_values(embedding, k=k, delta=delta).squeeze(0)
    q_values_list = [float(value) for value in q_values.detach().cpu().tolist()]
    greedy_action = int(q_values.argmax().item())
    greedy_value = float(q_values[greedy_action].detach().cpu().item())
    if random.random() < epsilon:
        action = int(env.action_space.sample())
    else:
        action = greedy_action
    return action, greedy_value, embedding.squeeze(0).detach().cpu().numpy(), q_values_list


def evaluate_nec(
    task_name: str,
    model: NECEmbeddingNetwork,
    dnd: DifferentiableNeuralDictionary,
    config: NECConfig,
    *,
    episodes: int = 20,
    seed: int = 10_000,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run greedy-policy NEC evaluation and return per-episode and aggregate metrics."""

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
                action, _, _, _ = _select_action(
                    model,
                    dnd,
                    np.asarray(obs, dtype=np.float32),
                    k=config.k_neighbors,
                    delta=config.kernel_delta,
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


def train_nec(
    task_name: str = "cartpole",
    config: NECConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train an NEC baseline and write reproducible run artifacts."""

    spec = get_rl_task_spec(task_name)
    cfg = config or make_nec_config(spec.default_profile)
    seed_everything(cfg.seed)
    run_device = _resolve_device_arg(device)

    env = _make_env(spec, seed=cfg.seed)
    obs_dim, action_dim = _validate_env_spaces(env, spec)
    model = NECEmbeddingNetwork(obs_dim, cfg.embedding_dim, hidden_sizes=cfg.hidden_sizes).to(run_device)
    dnd = DifferentiableNeuralDictionary(action_dim, cfg.dictionary_size, cfg.embedding_dim, obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    replay_buffer = NECReplayBuffer.create(cfg.replay_size, obs_dim)

    run_dir = Path(output_dir) if output_dir is not None else make_nec_output_dir(spec.name)
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
    best_model_state: dict[str, torch.Tensor] | None = None
    best_dnd_state: dict[str, Any] | None = None
    candidate_episode_return: float | None = None
    candidate_model_state: dict[str, torch.Tensor] | None = None
    candidate_dnd_state: dict[str, Any] | None = None
    candidate_step: int | None = None
    candidate_episode: int | None = None

    obs, _ = env.reset(seed=cfg.seed)
    episode_observations: list[np.ndarray] = []
    episode_embeddings: list[np.ndarray] = []
    episode_actions: list[int] = []
    episode_rewards: list[float] = []
    episode_bootstrap_values: list[float] = []
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
            action, greedy_value, embedding, q_values = _select_action(
                model,
                dnd,
                obs_array,
                k=cfg.k_neighbors,
                delta=cfg.kernel_delta,
                epsilon=epsilon,
                env=env,
                device=run_device,
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_done = bool(terminated or truncated)

            episode_observations.append(obs_array)
            episode_embeddings.append(embedding)
            episode_actions.append(action)
            episode_rewards.append(float(reward))
            episode_bootstrap_values.append(greedy_value)
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            if (
                replay_buffer.size >= cfg.learning_starts
                and dnd.min_size() >= cfg.k_neighbors
                and global_step % cfg.train_frequency == 0
            ):
                batch = replay_buffer.sample(cfg.batch_size, run_device)
                embeddings = model(batch["observations"])
                pred_q = dnd.q_values_for_actions(
                    embeddings,
                    batch["actions"],
                    k=cfg.k_neighbors,
                    delta=cfg.kernel_delta,
                )
                loss = F.smooth_l1_loss(pred_q, batch["returns"])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                latest_loss = float(loss.detach().cpu().item())
                latest_q_value = float(pred_q.detach().mean().cpu().item())
                loss_rows.append(
                    {
                        "global_step": global_step,
                        "loss": latest_loss,
                        "mean_q_value": latest_q_value,
                        "epsilon": epsilon,
                        "dictionary_entries": dnd.total_size(),
                    }
                )

            if episode_done:
                returns = _compute_n_step_returns(
                    episode_rewards,
                    episode_bootstrap_values,
                    cfg.gamma,
                    cfg.n_step,
                )
                last_index = len(returns) - 1
                for idx, return_value in enumerate(returns):
                    terminal_entry = bool(idx == last_index and terminated)
                    replay_buffer.add(
                        episode_observations[idx],
                        episode_actions[idx],
                        return_value,
                        terminal_entry,
                    )
                    dnd.add(
                        episode_embeddings[idx],
                        episode_actions[idx],
                        return_value,
                        episode_observations[idx],
                    )

                episode_index += 1
                completed_step = global_step + 1
                if candidate_episode_return is None or episode_return >= candidate_episode_return:
                    candidate_episode_return = float(episode_return)
                    candidate_model_state = _copy_state_dict_to_cpu(model)
                    candidate_dnd_state = dnd.to_state()
                    candidate_step = completed_step
                    candidate_episode = episode_index
                training_rows.append(
                    {
                        "global_step": completed_step,
                        "episode": episode_index,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "epsilon": epsilon,
                        "loss": latest_loss,
                        "mean_q_value": latest_q_value,
                        "dictionary_entries": dnd.total_size(),
                        "dictionary_min_entries": dnd.min_size(),
                        "q_values": json.dumps(q_values),
                    }
                )
                if progress and (episode_index <= 5 or episode_index % 10 == 0):
                    print(
                        "[nec] "
                        f"step={completed_step} episode={episode_index} "
                        f"return={episode_return:.1f} length={episode_length} "
                        f"epsilon={epsilon:.3f} loss={latest_loss} entries={dnd.total_size()}",
                        flush=True,
                    )
                if (
                    cfg.eval_episode_frequency > 0
                    and episode_index > 0
                    and episode_index % cfg.eval_episode_frequency == 0
                    and candidate_model_state is not None
                    and candidate_dnd_state is not None
                ):
                    current_model_state = _copy_state_dict_to_cpu(model)
                    current_dnd_state = dnd.to_state()
                    model.load_state_dict(candidate_model_state)
                    dnd = DifferentiableNeuralDictionary.from_state(candidate_dnd_state)
                    eval_metrics = evaluate_nec(
                        spec.name,
                        model,
                        dnd,
                        cfg,
                        episodes=cfg.eval_episodes,
                        seed=cfg.eval_seed,
                        device=run_device,
                    )
                    model.load_state_dict(current_model_state)
                    dnd = DifferentiableNeuralDictionary.from_state(current_dnd_state)
                    eval_row = {
                        "global_step": int(candidate_step or completed_step),
                        "episode": int(candidate_episode or episode_index),
                        "eval_global_step": completed_step,
                        "eval_episode": episode_index,
                        "eval_trigger": "episode",
                        "candidate_episode_return": candidate_episode_return,
                        "mean_return": eval_metrics["mean_return"],
                        "std_return": eval_metrics["std_return"],
                        "min_return": eval_metrics["min_return"],
                        "max_return": eval_metrics["max_return"],
                        "mean_length": eval_metrics["mean_length"],
                        "episodes": cfg.eval_episodes,
                        "dictionary_entries": dnd.total_size(),
                    }
                    eval_rows.append(eval_row)
                    if best_eval is None or eval_metrics["mean_return"] > best_eval["mean_return"]:
                        best_eval = eval_metrics
                        best_eval_step = int(candidate_step or completed_step)
                        best_model_state = candidate_model_state
                        best_dnd_state = candidate_dnd_state
                    if progress:
                        print(
                            "[nec][eval] "
                            f"step={eval_row['global_step']} episode={eval_row['episode']} "
                            f"candidate_return={candidate_episode_return:.1f} "
                            f"mean_return={eval_row['mean_return']:.2f} "
                            f"max_return={eval_row['max_return']:.2f}",
                            flush=True,
                        )
                obs, _ = env.reset(seed=cfg.seed + episode_index)
                episode_observations = []
                episode_embeddings = []
                episode_actions = []
                episode_rewards = []
                episode_bootstrap_values = []
                episode_return = 0.0
                episode_length = 0

            if (
                cfg.eval_frequency > 0
                and global_step > 0
                and global_step % cfg.eval_frequency == 0
            ):
                eval_metrics = evaluate_nec(
                    spec.name,
                    model,
                    dnd,
                    cfg,
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
                    "dictionary_entries": dnd.total_size(),
                }
                eval_rows.append(eval_row)
                if best_eval is None or eval_metrics["mean_return"] > best_eval["mean_return"]:
                    best_eval = eval_metrics
                    best_eval_step = global_step
                    best_model_state = _copy_state_dict_to_cpu(model)
                    best_dnd_state = dnd.to_state()
                if progress:
                    print(
                        "[nec][eval] "
                        f"step={global_step} mean_return={eval_row['mean_return']:.2f} "
                        f"max_return={eval_row['max_return']:.2f}",
                        flush=True,
                    )
    finally:
        env.close()

    last_eval = evaluate_nec(
        spec.name,
        model,
        dnd,
        cfg,
        episodes=cfg.eval_episodes,
        seed=cfg.eval_seed,
        device=run_device,
    )
    if best_eval is None or last_eval["mean_return"] >= best_eval["mean_return"]:
        selected_eval = last_eval
        selected_step = cfg.total_timesteps
        selected_source = "final"
        selected_model_state = _copy_state_dict_to_cpu(model)
        selected_dnd_state = dnd.to_state()
    else:
        selected_eval = best_eval
        selected_step = int(best_eval_step or 0)
        selected_source = "best_eval"
        selected_model_state = best_model_state or _copy_state_dict_to_cpu(model)
        selected_dnd_state = best_dnd_state or dnd.to_state()
        model.load_state_dict(selected_model_state)
        dnd = DifferentiableNeuralDictionary.from_state(selected_dnd_state)

    passed = (
        True
        if cfg.success_threshold is None
        else selected_eval["mean_return"] >= cfg.success_threshold
    )
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
        "model_state_dict": selected_model_state,
        "dnd_state": selected_dnd_state,
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
        "dictionary_entries": dnd.total_size(),
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
            "[nec] finished "
            f"profile={cfg.profile} mean_eval_return={selected_eval['mean_return']:.2f} "
            f"passed={passed} run_dir={run_dir}",
            flush=True,
        )
    return {
        "model": model,
        "dnd": dnd,
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


def load_nec_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    run_device = _resolve_device_arg(device)
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=run_device, weights_only=False)
    config_data = dict(checkpoint["config"])
    config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
    cfg = NECConfig(**config_data)
    model = NECEmbeddingNetwork(
        int(checkpoint["obs_dim"]),
        cfg.embedding_dim,
        hidden_sizes=cfg.hidden_sizes,
    ).to(run_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dnd = DifferentiableNeuralDictionary.from_state(checkpoint["dnd_state"])
    return {
        "model": model,
        "dnd": dnd,
        "config": cfg,
        "task": checkpoint["task"],
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "device": run_device,
    }
