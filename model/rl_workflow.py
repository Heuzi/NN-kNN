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

from datasets.rl_tasks import RLTaskSpec, get_rl_task_spec, list_supported_rl_tasks
from model.device_utils import resolve_runtime_device


class DQNNetwork(nn.Module):
    """CleanRL-style MLP Q-network for flat observations and discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, int] = (120, 84)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(frozen=True)
class DQNConfig:
    """Training configuration for the repo-native DQN baseline."""

    profile: str = "fast"
    seed: int = 0
    total_timesteps: int = 150_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 10_000
    gamma: float = 0.99
    target_network_frequency: int = 500
    max_grad_norm: float = 0.5
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10_000
    train_frequency: int = 10
    eval_frequency: int = 10_000
    eval_episodes: int = 20
    eval_seed: int = 10_000
    success_threshold: float | None = 475.0
    hidden_sizes: tuple[int, int] = (120, 84)
    source_reference: str = "CleanRL dqn.py"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


@dataclass
class ReplayBuffer:
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    pos: int = 0
    size: int = 0

    @classmethod
    def create(cls, buffer_size: int, obs_dim: int) -> "ReplayBuffer":
        return cls(
            observations=np.zeros((buffer_size, obs_dim), dtype=np.float32),
            next_observations=np.zeros((buffer_size, obs_dim), dtype=np.float32),
            actions=np.zeros(buffer_size, dtype=np.int64),
            rewards=np.zeros(buffer_size, dtype=np.float32),
            dones=np.zeros(buffer_size, dtype=np.float32),
        )

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.observations.shape[0]
        self.size = min(self.size + 1, self.observations.shape[0])

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device).long(),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(self.next_observations[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }


def make_dqn_config(profile: str = "fast", **overrides: Any) -> DQNConfig:
    profiles: dict[str, dict[str, Any]] = {
        "smoke": {
            "profile": "smoke",
            "total_timesteps": 256,
            "buffer_size": 1_000,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "target_network_frequency": 100,
            "eval_frequency": 128,
            "eval_episodes": 2,
            "success_threshold": None,
        },
        "fast": {
            "profile": "fast",
            "total_timesteps": 150_000,
            "learning_rate": 1e-3,
            "learning_starts": 1_000,
            "train_frequency": 1,
            "target_network_frequency": 250,
            "eval_frequency": 5_000,
            "eval_episodes": 20,
            "success_threshold": 475.0,
        },
        "gold": {
            "profile": "gold",
            "total_timesteps": 500_000,
            "learning_rate": 1e-3,
            "learning_starts": 1_000,
            "train_frequency": 1,
            "target_network_frequency": 250,
            "eval_frequency": 10_000,
            "eval_episodes": 20,
            "success_threshold": 475.0,
        },
    }
    normalized = profile.strip().lower()
    if normalized not in profiles:
        raise ValueError(f"Unknown DQN profile '{profile}'. Choose one of: {', '.join(sorted(profiles))}")
    data = {**profiles[normalized], **overrides}
    if "hidden_sizes" in data and not isinstance(data["hidden_sizes"], tuple):
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
    return DQNConfig(**data)


def make_dqn_output_dir(
    task_name: str,
    *,
    parent: str | Path = "results/rl",
    suffix: str | None = None,
) -> Path:
    created_at = datetime.now(timezone.utc)
    stem = f"dqn_{task_name}_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
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
    raise FileExistsError(f"Could not create a unique DQN output directory under {parent_path}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device_arg(device: str | torch.device | None) -> torch.device:
    if device is None:
        return resolve_runtime_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    if duration <= 0:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def _require_gymnasium() -> Any:
    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gymnasium is required for RL workflows. Install dependencies with "
            "`bash codex/setup.sh` or `python -m pip install 'gymnasium[classic_control]'`."
        ) from exc
    return gym


def _make_env(spec: RLTaskSpec, seed: int | None = None) -> Any:
    gym = _require_gymnasium()
    env = gym.make(spec.env_id)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def _validate_env_spaces(env: Any, spec: RLTaskSpec) -> tuple[int, int]:
    gym = _require_gymnasium()
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError(f"{spec.env_id} must use a Box observation space for this DQN baseline.")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(f"{spec.env_id} must use a Discrete action space for this DQN baseline.")
    obs_shape = env.observation_space.shape
    if len(obs_shape) != 1:
        raise ValueError(f"{spec.env_id} observation shape must be flat, got {obs_shape}.")
    return int(np.prod(obs_shape)), int(env.action_space.n)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


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


def _first_threshold_step(
    eval_rows: list[dict[str, Any]],
    threshold: float | None,
) -> int | None:
    if threshold is None:
        return None
    for row in eval_rows:
        if float(row["mean_return"]) >= threshold:
            return int(row["global_step"])
    return None


def _build_training_efficiency(
    *,
    selected_eval: dict[str, Any],
    last_eval: dict[str, Any],
    selected_step: int,
    selected_source: str,
    total_timesteps: int,
    success_threshold: float | None,
    first_success_step: int | None,
) -> dict[str, Any]:
    selected_mean = float(selected_eval["mean_return"])
    last_mean = float(last_eval["mean_return"])
    best_minus_last = selected_mean - last_mean
    selected_fraction = selected_step / total_timesteps if total_timesteps > 0 else 0.0
    first_success_fraction = (
        first_success_step / total_timesteps
        if first_success_step is not None and total_timesteps > 0
        else None
    )
    if success_threshold is not None and first_success_step is None:
        budget_interpretation = "unsolved_or_underfit"
    elif selected_source == "best_eval" and best_minus_last > 0:
        budget_interpretation = "regressed_after_best"
    elif selected_source == "final":
        budget_interpretation = "final_is_best_check_for_underfit"
    else:
        budget_interpretation = "plateau_or_tied_best"
    return {
        "best_model_step": int(selected_step),
        "best_model_training_fraction": selected_fraction,
        "first_success_step": first_success_step,
        "first_success_training_fraction": first_success_fraction,
        "best_minus_last_return": best_minus_last,
        "budget_interpretation": budget_interpretation,
    }


def _select_action(
    model: DQNNetwork,
    obs: np.ndarray,
    *,
    action_dim: int,
    epsilon: float,
    env: Any,
    device: torch.device,
) -> int:
    if random.random() < epsilon:
        return int(env.action_space.sample())
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return int(model(obs_tensor).argmax(dim=1).item())


def evaluate_dqn(
    task_name: str,
    model: DQNNetwork,
    *,
    episodes: int = 20,
    seed: int = 10_000,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run greedy-policy evaluation and return per-episode and aggregate metrics."""

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
                action = _select_action(
                    model,
                    np.asarray(obs, dtype=np.float32),
                    action_dim=int(env.action_space.n),
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


def train_dqn(
    task_name: str = "cartpole",
    config: DQNConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train a DQN baseline and write reproducible run artifacts."""

    spec = get_rl_task_spec(task_name)
    cfg = config or make_dqn_config(spec.default_profile)
    seed_everything(cfg.seed)
    run_device = _resolve_device_arg(device)

    env = _make_env(spec, seed=cfg.seed)
    obs_dim, action_dim = _validate_env_spaces(env, spec)
    q_network = DQNNetwork(obs_dim, action_dim, hidden_sizes=cfg.hidden_sizes).to(run_device)
    target_network = DQNNetwork(obs_dim, action_dim, hidden_sizes=cfg.hidden_sizes).to(run_device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    replay_buffer = ReplayBuffer.create(cfg.buffer_size, obs_dim)

    run_dir = Path(output_dir) if output_dir is not None else make_dqn_output_dir(spec.name)
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
    best_target_state: dict[str, torch.Tensor] | None = None

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
            action = _select_action(
                q_network,
                np.asarray(obs, dtype=np.float32),
                action_dim=action_dim,
                epsilon=epsilon,
                env=env,
                device=run_device,
            )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            train_done = bool(terminated)
            episode_done = bool(terminated or truncated)
            replay_buffer.add(
                np.asarray(obs, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                train_done,
            )
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            if replay_buffer.size >= cfg.learning_starts and global_step % cfg.train_frequency == 0:
                batch = replay_buffer.sample(cfg.batch_size, run_device)
                with torch.no_grad():
                    target_max = target_network(batch["next_observations"]).max(dim=1).values
                    td_target = batch["rewards"] + cfg.gamma * target_max * (1.0 - batch["dones"])
                old_val = q_network(batch["observations"]).gather(1, batch["actions"].view(-1, 1)).squeeze(1)
                loss = F.smooth_l1_loss(old_val, td_target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), cfg.max_grad_norm)
                optimizer.step()
                latest_loss = float(loss.detach().cpu().item())
                latest_q_value = float(old_val.detach().mean().cpu().item())
                loss_rows.append(
                    {
                        "global_step": global_step,
                        "loss": latest_loss,
                        "mean_q_value": latest_q_value,
                        "epsilon": epsilon,
                    }
                )

            if global_step > 0 and global_step % cfg.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

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
                    }
                )
                if progress and (episode_index <= 5 or episode_index % 10 == 0):
                    print(
                        "[dqn] "
                        f"step={global_step + 1} episode={episode_index} "
                        f"return={episode_return:.1f} length={episode_length} "
                        f"epsilon={epsilon:.3f} loss={latest_loss}",
                        flush=True,
                    )
                obs, _ = env.reset(seed=cfg.seed + episode_index)
                episode_return = 0.0
                episode_length = 0

            if (
                cfg.eval_frequency > 0
                and global_step > 0
                and global_step % cfg.eval_frequency == 0
            ):
                eval_metrics = evaluate_dqn(
                    spec.name,
                    q_network,
                    episodes=cfg.eval_episodes,
                    seed=cfg.eval_seed,
                    device=run_device,
                )
                eval_row = {
                    "global_step": global_step,
                    "mean_return": eval_metrics["mean_return"],
                    "std_return": eval_metrics["std_return"],
                    "min_return": eval_metrics["min_return"],
                    "max_return": eval_metrics["max_return"],
                    "mean_length": eval_metrics["mean_length"],
                    "episodes": cfg.eval_episodes,
                }
                eval_rows.append(eval_row)
                if best_eval is None or eval_metrics["mean_return"] > best_eval["mean_return"]:
                    best_eval = eval_metrics
                    best_eval_step = global_step
                    best_model_state = _copy_state_dict_to_cpu(q_network)
                    best_target_state = _copy_state_dict_to_cpu(target_network)
                if progress:
                    print(
                        "[dqn][eval] "
                        f"step={global_step} mean_return={eval_row['mean_return']:.2f} "
                        f"max_return={eval_row['max_return']:.2f}",
                        flush=True,
                    )
    finally:
        env.close()

    last_eval = evaluate_dqn(
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
        selected_model_state = _copy_state_dict_to_cpu(q_network)
        selected_target_state = _copy_state_dict_to_cpu(target_network)
    else:
        selected_eval = best_eval
        selected_step = int(best_eval_step or 0)
        selected_source = "best_eval"
        selected_model_state = best_model_state or _copy_state_dict_to_cpu(q_network)
        selected_target_state = best_target_state or _copy_state_dict_to_cpu(target_network)
        q_network.load_state_dict(selected_model_state)
        target_network.load_state_dict(selected_target_state)

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
        "target_model_state_dict": selected_target_state,
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
            "[dqn] finished "
            f"profile={cfg.profile} mean_eval_return={selected_eval['mean_return']:.2f} "
            f"passed={passed} run_dir={run_dir}",
            flush=True,
        )
    return {
        "model": q_network,
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


def load_dqn_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    run_device = _resolve_device_arg(device)
    checkpoint = torch.load(checkpoint_path, map_location=run_device)
    config_data = dict(checkpoint["config"])
    config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
    cfg = DQNConfig(**config_data)
    model = DQNNetwork(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        hidden_sizes=cfg.hidden_sizes,
    ).to(run_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "config": cfg,
        "task": checkpoint["task"],
        "checkpoint": checkpoint,
        "device": run_device,
    }
