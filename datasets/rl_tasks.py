from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RLTaskSpec:
    """Metadata for a reinforcement-learning task supported by the repo."""

    name: str
    env_id: str
    family: str
    observation_kind: str
    action_kind: str
    max_episode_steps: int
    default_profile: str
    literature_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["literature_notes"] = list(self.literature_notes)
        return data


_RL_TASKS: dict[str, RLTaskSpec] = {
    "cartpole": RLTaskSpec(
        name="cartpole",
        env_id="CartPole-v1",
        family="classic_control",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=500,
        default_profile="fast",
        literature_notes=(
            "Fast DQN sanity task used by the official PyTorch DQN tutorial.",
            "Engineering gate before moving to Atari/ALE, the DQN and NEC paper-aligned benchmark family.",
        ),
    ),
    "pong": RLTaskSpec(
        name="pong",
        env_id="ALE/Pong-v5",
        family="atari",
        observation_kind="atari_frames",
        action_kind="discrete",
        max_episode_steps=108_000,
        default_profile="debug",
        literature_notes=(
            "Canonical Atari control task for DQN-style image-observation baselines.",
            "Use smoke/debug profiles for plumbing only; full Atari comparisons require much larger budgets.",
        ),
    ),
    "breakout": RLTaskSpec(
        name="breakout",
        env_id="ALE/Breakout-v5",
        family="atari",
        observation_kind="atari_frames",
        action_kind="discrete",
        max_episode_steps=108_000,
        default_profile="debug",
        literature_notes=(
            "Canonical Atari control task for DQN-style image-observation baselines.",
            "Use smoke/debug profiles for plumbing only; full Atari comparisons require much larger budgets.",
        ),
    ),
}


def normalize_rl_task_name(task_name: str) -> str:
    normalized = task_name.strip().lower().replace("-", "_")
    aliases = {
        "cart_pole": "cartpole",
        "cartpole_v1": "cartpole",
        "cartpole-v1": "cartpole",
        "cartpolev1": "cartpole",
        "ale_pong_v5": "pong",
        "ale/pong_v5": "pong",
        "ale/pong-v5": "pong",
        "pong_v5": "pong",
        "pong-v5": "pong",
        "ale_breakout_v5": "breakout",
        "ale/breakout_v5": "breakout",
        "ale/breakout-v5": "breakout",
        "breakout_v5": "breakout",
        "breakout-v5": "breakout",
    }
    return aliases.get(normalized, normalized)


def get_rl_task_spec(task_name: str) -> RLTaskSpec:
    normalized = normalize_rl_task_name(task_name)
    try:
        return _RL_TASKS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_RL_TASKS))
        raise ValueError(f"Unknown RL task '{task_name}'. Supported tasks: {supported}") from exc


def list_supported_rl_tasks() -> dict[str, list[str]]:
    by_family: dict[str, list[str]] = {}
    for task in _RL_TASKS.values():
        by_family.setdefault(task.family, []).append(task.name)
    return {family: sorted(names) for family, names in sorted(by_family.items())}
