from __future__ import annotations

import copy
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

from datasets.reg_data import DATATYPES, Reg_data, standardize_tensor
from model.device_utils import resolve_runtime_device
from model.nnknn_model import default_args, train_model


_TRUE_W_LINEAR = torch.tensor([2.5, -1.7, 0.0, 0.9, 3.2], dtype=torch.float32).view(-1, 1)
_REAL_DATASET_LOOKUP = {
    re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"): name for name in DATATYPES
}


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Purpose: make notebook and batch runs reproducible."""
    runtime_device = resolve_runtime_device()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        if runtime_device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False


def _normalize_name(name: str) -> str:
    """Purpose: normalize dataset and mode names for internal lookups."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def list_supported_regression_datasets() -> dict[str, list[str]]:
    """Purpose: list every synthetic and real regression dataset supported here."""
    return {
        "synthetic": [
            "linear_regression",
            "mixture_two_linear_models",
            "redundant_features",
        ],
        "real": sorted(DATATYPES.keys()),
    }


def make_regression_cfg(
    overrides: Mapping[str, Any] | None = None,
    *,
    base_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Purpose: build a regression config without repeating `default_args` in the notebook.

    Input:
        overrides: Key/value pairs that should overwrite the base config.
        base_cfg: Optional starting config. Defaults to `default_args`.

    Output:
        A deep-copied config dictionary ready for training.
    """
    cfg = copy.deepcopy(dict(base_cfg or default_args))
    if overrides:
        cfg.update(copy.deepcopy(dict(overrides)))
    return cfg


def make_linear_regression_dataset(
    n: int = 3000,
    d: int = 5,
    noise_scale: float = 1.0,
    seed: int = 42,
    true_w: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Purpose: create the synthetic linear-regression dataset used in the notebook."""
    if true_w is None:
        true_w = _TRUE_W_LINEAR.clone()
    true_w = true_w.to(torch.float32).view(-1, 1)
    if true_w.size(0) != d:
        raise ValueError(f"Expected true_w to have {d} rows, got {true_w.size(0)}.")

    generator = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=generator)
    y = X @ true_w + noise_scale * torch.randn(n, 1, generator=generator)

    feature_weights = true_w.squeeze().abs()
    denom = feature_weights.sum().clamp_min(1e-8)
    feature_weights = feature_weights / denom

    return {
        "dataset_name": "linear_regression",
        "display_name": "linear_regression",
        "dataset_kind": "synthetic",
        "X": X,
        "y": y,
        "regime_weights": False,
        "true_w": true_w,
        "true_feature_weights": feature_weights,
    }


def make_mixture_two_linear_models_dataset(
    n: int = 3000,
    d: int = 4,
    noise_scale: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Purpose: create the two-regime synthetic dataset used in the notebook."""
    if d != 4:
        raise ValueError("The default mixture generator currently expects d=4.")

    generator = torch.Generator().manual_seed(seed)

    w1 = torch.tensor([2.5, -1.5, 0.0, 0.0], dtype=torch.float32).view(d, 1)
    w2 = torch.tensor([0.0, 0.0, -2.0, 1.5], dtype=torch.float32).view(d, 1)

    w1_abs = w1.squeeze().abs()
    w2_abs = w2.squeeze().abs()
    true_feature_weights_glocal = torch.stack(
        [
            w1_abs / w1_abs.sum().clamp_min(1e-8),
            w2_abs / w2_abs.sum().clamp_min(1e-8),
        ],
        dim=0,
    )

    X = torch.randn(n, d, generator=generator)
    regime_labels = torch.zeros(n, dtype=torch.long)
    regime_labels[: n // 2] = 0
    regime_labels[n // 2 :] = 1

    perm = torch.randperm(n, generator=generator)
    X = X[perm]
    regime_labels = regime_labels[perm]

    y = torch.zeros(n, 1, dtype=torch.float32)
    idx1 = regime_labels == 0
    idx2 = regime_labels == 1

    X1 = X[idx1]
    X2 = X[idx2]
    y[idx1] = X1 @ w1 + noise_scale * torch.randn(X1.size(0), 1, generator=generator)
    y[idx2] = X2 @ w2 + noise_scale * torch.randn(X2.size(0), 1, generator=generator)

    return {
        "dataset_name": "mixture_two_linear_models",
        "display_name": "mixture_two_linear_models",
        "dataset_kind": "synthetic",
        "X": X,
        "y": y,
        "regime_weights": True,
        "regime_labels": regime_labels,
        "w1": w1,
        "w2": w2,
        "true_feature_weights_glocal": true_feature_weights_glocal,
    }


def make_redundant_features_dataset(
    n: int = 1500,
    seed: int = 42,
    noise_scale: float = 0.1,
) -> dict[str, Any]:
    """Purpose: create the redundant-feature synthetic dataset used in the notebook."""
    generator = torch.Generator().manual_seed(seed)

    s1 = torch.randn(n, 1, generator=generator)
    s2 = torch.randn(n, 1, generator=generator)

    x0 = s1 + 0.05 * torch.randn(n, 1, generator=generator)
    x1 = s2 + 0.05 * torch.randn(n, 1, generator=generator)
    x2 = s2 + 0.05 * torch.randn(n, 1, generator=generator)

    X = torch.cat([x0, x1, x2], dim=1)
    y = 1.0 * s1 + 1.0 * s2 + noise_scale * torch.randn(n, 1, generator=generator)

    true_w = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32).view(-1, 1)
    true_feature_weights = true_w.squeeze().abs()
    true_feature_weights = true_feature_weights / true_feature_weights.sum().clamp_min(1e-8)

    return {
        "dataset_name": "redundant_features",
        "display_name": "redundant_features",
        "dataset_kind": "synthetic",
        "X": X,
        "y": y,
        "regime_weights": False,
        "true_w": true_w,
        "true_feature_weights": true_feature_weights,
    }


def get_regression_dataset(dataset_name: str, **kwargs: Any) -> dict[str, Any]:
    """Purpose: load one supported regression dataset into a normalized bundle."""
    normalized = _normalize_name(dataset_name)

    if normalized == "linear_regression":
        return make_linear_regression_dataset(**kwargs)
    if normalized in {"mixture_two_linear_models", "mixture_of_two_linear_models"}:
        return make_mixture_two_linear_models_dataset(**kwargs)
    if normalized == "redundant_features":
        return make_redundant_features_dataset(**kwargs)

    real_name = _REAL_DATASET_LOOKUP.get(normalized)
    if real_name is None:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Supported datasets: {list_supported_regression_datasets()}"
        )

    X, y = Reg_data(real_name)
    return {
        "dataset_name": real_name,
        "display_name": real_name,
        "dataset_kind": "real",
        "X": X.to(torch.float32),
        "y": y.to(torch.float32),
        "regime_weights": False,
    }


def load_regression_dataset_state(
    dataset_name: str,
    *,
    dataset_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Purpose: create a notebook-friendly state dictionary for one dataset.

    Input:
        dataset_name: Synthetic or real dataset name.
        dataset_kwargs: Optional dataset builder/loader arguments.

    Output:
        A state dictionary exposing notebook variables like `X`, `y`, `Xs`, and `ys`.
    """
    bundle = get_regression_dataset(dataset_name, **dict(dataset_kwargs or {}))
    state = {key: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value)) for key, value in bundle.items()}
    state["dataset"] = state["display_name"]

    if state["dataset_kind"] == "real":
        state["Xs"] = state["X"]
        state["ys"] = state["y"]

    state["N"] = int(state["X"].shape[0])
    state["D"] = int(state["X"].shape[1])
    return state


def publish_workflow_state(
    state: Mapping[str, Any],
    namespace: MutableMapping[str, Any],
    *,
    keys: list[str] | None = None,
) -> dict[str, Any]:
    """Purpose: copy workflow-state values back into notebook globals.

    Input:
        state: Workflow state dictionary from this module.
        namespace: Usually `globals()` inside the notebook.
        keys: Optional subset of keys to publish.

    Output:
        The same state dictionary, returned unchanged for convenience.
    """
    selected_keys = keys or [key for key in state.keys() if not key.startswith("_")]
    for key in selected_keys:
        namespace[key] = state[key]
    return dict(state)


def print_dataset_summary(state: Mapping[str, Any], *, show_feature_weights: bool = False) -> None:
    """Purpose: print the same quick dataset summary the notebook previously assembled manually.

    Input:
        state: Dataset state from `load_regression_dataset_state`.
        show_feature_weights: When True, also prints true synthetic feature weights.

    Output:
        None. This helper prints notebook-friendly summary text.
    """
    X = state["X"]
    y = state["y"]
    print(X.shape, y.shape)

    if state.get("dataset_kind") == "real":
        print("y mean:", y.mean().item(), "y std:", y.std().item())
        print("Raw y min/max:", y.min().item(), y.max().item())

    if state.get("regime_weights") and state.get("regime_labels") is not None:
        regime_labels = state["regime_labels"]
        idx1 = int((regime_labels == 0).sum().item())
        idx2 = int((regime_labels == 1).sum().item())
        print("Regime 1 samples:", idx1, "Regime 2 samples:", idx2)

    if show_feature_weights and state.get("true_feature_weights_glocal") is not None:
        print("True glocal feature weights (regime 1, regime 2):")
        print(state["true_feature_weights_glocal"])
    elif show_feature_weights and state.get("true_feature_weights") is not None:
        print("True feature weights for similarity:", state["true_feature_weights"])


def _flatten_targets(y: torch.Tensor) -> torch.Tensor:
    """Purpose: convert `[N, 1]` regression targets to `[N]` when needed."""
    if y.dim() > 1 and y.size(-1) == 1:
        return y.view(-1)
    return y.clone()


def split_regression_data(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    regime_labels: torch.Tensor | None = None,
    flatten_targets: bool = True,
) -> dict[str, Any]:
    """Purpose: create one random train/validation split for regression data."""
    n_train = int(train_ratio * X.size(0))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(X.size(0), generator=generator)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return split_regression_data_from_indices(
        X,
        y,
        train_idx=train_idx,
        val_idx=val_idx,
        regime_labels=regime_labels,
        flatten_targets=flatten_targets,
    )


def split_regression_data_from_indices(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    train_idx: torch.Tensor | np.ndarray,
    val_idx: torch.Tensor | np.ndarray,
    regime_labels: torch.Tensor | None = None,
    flatten_targets: bool = True,
) -> dict[str, Any]:
    """Purpose: split regression data using explicit train/validation indices."""
    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    val_idx = torch.as_tensor(val_idx, dtype=torch.long)

    X_train = X[train_idx].clone()
    X_val = X[val_idx].clone()
    y_train = y[train_idx].clone()
    y_val = y[val_idx].clone()
    if flatten_targets:
        y_train = _flatten_targets(y_train)
        y_val = _flatten_targets(y_val)

    result = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }

    if regime_labels is not None:
        result["regime_labels_train"] = regime_labels[train_idx].clone()
        result["regime_labels_val"] = regime_labels[val_idx].clone()

    return result


def _build_unstandardized_aliases(split_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Purpose: attach raw/model-space aliases before any standardization happens."""
    X_train_raw = split_bundle["X_train"].clone()
    X_val_raw = split_bundle["X_val"].clone()
    y_train_raw = _flatten_targets(split_bundle["y_train"].clone())
    y_val_raw = _flatten_targets(split_bundle["y_val"].clone())

    return {
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_mean_raw": y_train_raw.mean().item(),
        "y_std_raw": y_train_raw.std().item(),
        "standardized_targets": False,
        "X_train_z": X_train_raw,
        "X_val_z": X_val_raw,
        "y_train_z": y_train_raw,
        "y_val_z": y_val_raw,
        "y_train_norm": y_train_raw,
        "y_val_norm": y_val_raw,
        "X_mean": None,
        "X_std": None,
        "y_mean": None,
        "y_std": None,
    }


def split_regression_state(
    state: Mapping[str, Any],
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    train_idx: torch.Tensor | np.ndarray | None = None,
    val_idx: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Purpose: add a train/validation split to a dataset workflow state.

    Input:
        state: Dataset state from `load_regression_dataset_state`.
        train_ratio: Fraction of rows placed in the training split.
        seed: RNG seed for random splits.
        train_idx: Optional explicit training indices, useful for k-fold runs.
        val_idx: Optional explicit validation indices, useful for k-fold runs.

    Output:
        A new workflow state containing split tensors plus notebook aliases.
    """
    flatten_targets = state.get("dataset_kind") != "real"
    if train_idx is None or val_idx is None:
        split_bundle = split_regression_data(
            state["X"],
            state["y"],
            train_ratio=train_ratio,
            seed=seed,
            regime_labels=state.get("regime_labels"),
            flatten_targets=flatten_targets,
        )
    else:
        split_bundle = split_regression_data_from_indices(
            state["X"],
            state["y"],
            train_idx=train_idx,
            val_idx=val_idx,
            regime_labels=state.get("regime_labels"),
            flatten_targets=flatten_targets,
        )

    new_state = dict(state)
    new_state.update(split_bundle)
    new_state.update(_build_unstandardized_aliases(split_bundle))
    new_state["train_ratio"] = train_ratio
    new_state["split_seed"] = seed
    return new_state


def print_split_summary(state: Mapping[str, Any]) -> None:
    """Purpose: print train/validation tensor shapes for the active split."""
    print(
        "Train:",
        state["X_train"].shape,
        state["y_train"].shape,
        "| Val:",
        state["X_val"].shape,
        state["y_val"].shape,
    )


def prepare_regression_split_for_training(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    standardize: bool = True,
) -> dict[str, Any]:
    """Purpose: convert a split into training-ready tensors and scaler metadata.

    Input:
        X_train, y_train: Training split tensors.
        X_val, y_val: Validation split tensors.
        standardize: When True, z-scores features and targets using train-split stats.

    Output:
        A dictionary with raw tensors, model-space tensors, and scaler statistics.
    """
    X_train_raw = X_train.clone()
    X_val_raw = X_val.clone()
    y_train_raw = _flatten_targets(y_train.clone())
    y_val_raw = _flatten_targets(y_val.clone())

    result: dict[str, Any] = {
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_mean_raw": y_train_raw.mean().item(),
        "y_std_raw": y_train_raw.std().item(),
        "standardized_targets": bool(standardize),
    }

    if standardize:
        X_train_used, X_mean, X_std = standardize_tensor(X_train_raw, dim=0, return_stats=True)
        X_val_used = standardize_tensor(X_val_raw, dim=0, mean=X_mean, std=X_std)

        y_train_used, y_mean, y_std = standardize_tensor(
            y_train_raw.view(-1, 1), dim=0, return_stats=True
        )
        y_val_used = standardize_tensor(
            y_val_raw.view(-1, 1), dim=0, mean=y_mean, std=y_std
        )

        y_train_used = y_train_used.view(-1)
        y_val_used = y_val_used.view(-1)

        result.update(
            {
                "X_train": X_train_used,
                "X_val": X_val_used,
                "y_train": y_train_used,
                "y_val": y_val_used,
                "X_mean": X_mean,
                "X_std": X_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "X_train_z": X_train_used,
                "X_val_z": X_val_used,
                "y_train_z": y_train_used,
                "y_val_z": y_val_used,
                "y_train_norm": y_train_used,
                "y_val_norm": y_val_used,
            }
        )
    else:
        result.update(
            {
                "X_train": X_train_raw,
                "X_val": X_val_raw,
                "y_train": y_train_raw,
                "y_val": y_val_raw,
                "X_train_z": X_train_raw,
                "X_val_z": X_val_raw,
                "y_train_z": y_train_raw,
                "y_val_z": y_val_raw,
                "y_train_norm": y_train_raw,
                "y_val_norm": y_val_raw,
                "X_mean": None,
                "X_std": None,
                "y_mean": None,
                "y_std": None,
            }
        )

    return result


def standardize_regression_state(
    state: Mapping[str, Any],
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    """Purpose: add standardized tensors to an existing split workflow state.

    Input:
        state: Split workflow state from `split_regression_state`.
        enabled: When True, z-scores the split. When False, refreshes raw aliases only.

    Output:
        A new workflow state with training-ready tensors and scaler statistics.
    """
    new_state = dict(state)
    if enabled:
        prepared = prepare_regression_split_for_training(
            state["X_train"],
            state["y_train"],
            state["X_val"],
            state["y_val"],
            standardize=True,
        )
        new_state.update(prepared)
    else:
        new_state.update(_build_unstandardized_aliases(state))
    return new_state


def print_standardization_summary(state: Mapping[str, Any]) -> None:
    """Purpose: print the same standardization sanity checks the notebook used before."""
    print("X_train mean abs (avg):", state["X_train"].mean(0).abs().mean().item())
    print(
        "X_train std  (min/mean/max):",
        state["X_train"].std(0).min().item(),
        state["X_train"].std(0).mean().item(),
        state["X_train"].std(0).max().item(),
    )
    print("y_train mean/std:", state["y_train"].mean().item(), state["y_train"].std().item())


def prepare_cfg_for_dataset(
    cfg: dict[str, Any],
    dataset_bundle: dict[str, Any],
    *,
    glocal_init_from_true: bool = False,
    glocal_init_alpha: float = 1.0,
) -> dict[str, Any]:
    """Purpose: inject dataset-specific similarity priors into a training config."""
    cfg_run = copy.deepcopy(cfg)

    for key in (
        "glocal_init_from_true",
        "glocal_init_alpha",
        "true_feature_weights_glocal",
        "true_feature_weights",
    ):
        cfg_run.pop(key, None)

    if dataset_bundle.get("regime_weights", False) and dataset_bundle.get("true_feature_weights_glocal") is not None:
        cfg_run["glocal_init_from_true"] = bool(glocal_init_from_true)
        cfg_run["glocal_init_alpha"] = float(glocal_init_alpha)
        cfg_run["true_feature_weights_glocal"] = dataset_bundle["true_feature_weights_glocal"].clone()
    elif dataset_bundle.get("true_feature_weights") is not None:
        cfg_run["true_feature_weights"] = dataset_bundle["true_feature_weights"].clone()

    return cfg_run


def configure_regression_cfg_for_state(
    cfg: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    glocal_init_from_true: bool = False,
    glocal_init_alpha: float = 1.0,
) -> dict[str, Any]:
    """Purpose: notebook-facing wrapper that prepares a config using the current workflow state.

    Input:
        cfg: Base training config.
        state: Workflow state containing true feature weights when available.
        glocal_init_from_true: Whether to initialize glocal weights from the true regime weights.
        glocal_init_alpha: Strength of that glocal-weight initialization blend.

    Output:
        A copied config dictionary ready for training.
    """
    return prepare_cfg_for_dataset(
        dict(cfg),
        dict(state),
        glocal_init_from_true=glocal_init_from_true,
        glocal_init_alpha=glocal_init_alpha,
    )


def _clone_feature_extractor(feature_extractor: Any) -> Any:
    """Purpose: keep repeated runs from mutating the same feature extractor instance."""
    if feature_extractor is None:
        return None
    return copy.deepcopy(feature_extractor)


def _resolve_standardize_flag(
    standardize: bool | str,
    dataset_bundle: dict[str, Any],
) -> bool:
    """Purpose: resolve `standardize=\"auto\"` based on dataset type."""
    if standardize == "auto":
        return dataset_bundle.get("dataset_kind") == "real"
    return bool(standardize)


def _checkpoint_path_for_run(
    checkpoint_path: str,
    *,
    dataset_name: str,
    run_label: str,
) -> str:
    """Purpose: create unique checkpoint filenames for repeated experiments."""
    base = Path(checkpoint_path)
    parent = base.parent if str(base.parent) not in {"", "."} else Path("checkpoints")
    parent.mkdir(parents=True, exist_ok=True)

    safe_dataset = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name)
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", run_label)
    suffix = base.suffix or ".pth"
    stem = base.stem if base.suffix else base.name
    return str(parent / f"{stem}_{safe_dataset}_{safe_label}{suffix}")


def _model_device(model: torch.nn.Module) -> torch.device:
    """Purpose: infer the device currently used by a trained model."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _raw_scale_tensors(
    *,
    y_mean_raw: float | None,
    y_std_raw: float | None,
    standardized_targets: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Purpose: build raw-space scaler tensors, or identity tensors when not standardized."""
    if standardized_targets and y_mean_raw is not None and y_std_raw is not None:
        y_mean_t = torch.as_tensor(y_mean_raw, dtype=torch.float32)
        y_std_t = torch.as_tensor(y_std_raw, dtype=torch.float32).clamp_min(1e-6)
        return y_mean_t, y_std_t

    return torch.tensor(0.0, dtype=torch.float32), torch.tensor(1.0, dtype=torch.float32)


def _plot_observed_vs_predicted(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    title: str,
    figure_size: tuple[int, int] = (5, 5),
) -> None:
    """Purpose: keep observed-vs-predicted plotting logic in one place."""
    plt.figure(figsize=figure_size)
    plt.scatter(y_true.numpy(), y_pred.numpy(), s=8, alpha=0.7)
    mn = float(min(y_true.min().item(), y_pred.min().item()))
    mx = float(max(y_true.max().item(), y_pred.max().item()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("True (raw)")
    plt.ylabel("Predicted (raw)")
    plt.title(title)
    plt.show()


def evaluate_regression_model(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    y_mean_raw: float | None = None,
    y_std_raw: float | None = None,
    standardized_targets: bool = True,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Purpose: compute post-adaptation predictions and RMSEs for one validation split."""
    model.eval()
    run_device = device or _model_device(model)

    preds = []
    with torch.no_grad():
        for i in range(0, X_val.size(0), batch_size):
            _, yhat, *_ = model(X_val[i : i + batch_size].to(run_device))
            preds.append(yhat.detach().cpu().view(-1))

    y_pred_model_space = torch.cat(preds, dim=0).to(torch.float32)
    y_true_model_space = _flatten_targets(y_val.detach().cpu()).to(torch.float32)
    rmse_model_space = torch.sqrt(F.mse_loss(y_pred_model_space, y_true_model_space)).item()

    y_mean_t, y_std_t = _raw_scale_tensors(
        y_mean_raw=y_mean_raw,
        y_std_raw=y_std_raw,
        standardized_targets=standardized_targets,
    )
    y_pred_raw = y_pred_model_space * y_std_t + y_mean_t
    y_true_raw = y_true_model_space * y_std_t + y_mean_t
    rmse_raw = torch.sqrt(F.mse_loss(y_pred_raw, y_true_raw)).item()
    rmse_z = rmse_model_space if standardized_targets else None

    return {
        "y_pred_model_space": y_pred_model_space,
        "y_true_model_space": y_true_model_space,
        "y_pred_raw": y_pred_raw,
        "y_true_raw": y_true_raw,
        "y_mean_t": y_mean_t,
        "y_std_t": y_std_t,
        "rmse_model_space": rmse_model_space,
        "rmse_raw": rmse_raw,
        "rmse_z": rmse_z,
    }


def train_regression_state(
    state: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    feature_extractor: Any = None,
    checkpoint_label: str | None = None,
    clone_feature_extractor: bool = True,
) -> dict[str, Any]:
    """Purpose: train NN-kNN from a workflow state instead of repeating notebook code.

    Input:
        state: Workflow state containing `X_train`, `X_val`, `y_train_norm`, and `y_val_norm`.
        cfg: Training config.
        feature_extractor: Optional feature extractor passed to `train_model`.
        checkpoint_label: Optional suffix used to create a unique checkpoint path.
        clone_feature_extractor: When True, deep-copies the feature extractor per run.

    Output:
        A new workflow state containing the trained model, best metric, and runtime config.
    """
    cfg_run = copy.deepcopy(dict(cfg))
    if checkpoint_label is not None:
        cfg_run["checkpoint_path"] = _checkpoint_path_for_run(
            cfg_run.get("checkpoint_path", "nnknn_regression_best.pth"),
            dataset_name=state.get("display_name", "dataset"),
            run_label=checkpoint_label,
        )

    run_feature_extractor = (
        _clone_feature_extractor(feature_extractor) if clone_feature_extractor else feature_extractor
    )
    best_metric, glocal_weightor, model = train_model(
        state["X_train"],
        state["y_train_norm"],
        state["X_val"],
        state["y_val_norm"],
        feature_extractor=run_feature_extractor,
        cfg=cfg_run,
    )

    new_state = dict(state)
    new_state.update(
        {
            "cfg": copy.deepcopy(dict(cfg)),
            "cfg_run": cfg_run,
            "feature_extractor": feature_extractor,
            "best_metric": best_metric,
            "best_acc": best_metric,
            "glocal_weightor": glocal_weightor,
            "model": model,
        }
    )
    return new_state


def evaluate_regression_state(
    state: Mapping[str, Any],
    *,
    batch_size: int = 512,
    show_plot: bool = True,
    print_metrics: bool = True,
    plot_title: str = "Observed vs Predicted (Validation)",
) -> dict[str, Any]:
    """Purpose: evaluate a trained workflow state and expose notebook-friendly metrics.

    Input:
        state: Workflow state containing a trained model and validation tensors.
        batch_size: Inference batch size.
        show_plot: When True, renders the observed-vs-predicted plot.
        print_metrics: When True, prints RMSE values.
        plot_title: Figure title for the observed-vs-predicted plot.

    Output:
        A new workflow state containing prediction tensors and RMSE summaries.
    """
    metrics = evaluate_regression_model(
        state["model"],
        state["X_val"],
        state["y_val"],
        y_mean_raw=state.get("y_mean_raw"),
        y_std_raw=state.get("y_std_raw"),
        standardized_targets=bool(state.get("standardized_targets", False)),
        batch_size=batch_size,
    )

    new_state = dict(state)
    new_state.update(metrics)
    new_state["y_pred_z"] = metrics["y_pred_model_space"]
    new_state["y_true_z"] = metrics["y_true_model_space"]

    if print_metrics:
        if new_state.get("standardized_targets", False):
            print(f"Validation RMSE (z): {new_state['rmse_z']:.4f}")
        print(f"Validation RMSE (raw): {new_state['rmse_raw']:.4f}")

    if show_plot:
        _plot_observed_vs_predicted(
            new_state["y_true_raw"],
            new_state["y_pred_raw"],
            title=plot_title,
        )

    return new_state


def _predict_pre_post_adaptation(
    state: Mapping[str, Any],
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Purpose: collect pre- and post-adaptation predictions in model space."""
    model = state["model"]
    run_device = device or _model_device(model)

    pre_preds, post_preds = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, state["X_val"].size(0), batch_size):
            _, yhat, pre_adapt_yhat, *_ = model(state["X_val"][i : i + batch_size].to(run_device))

            if pre_adapt_yhat.dim() == 3:
                pre_yhat = pre_adapt_yhat.sum(dim=1).squeeze(-1)
            elif pre_adapt_yhat.dim() == 2:
                pre_yhat = pre_adapt_yhat.squeeze(-1)
            else:
                pre_yhat = pre_adapt_yhat

            post_preds.append(yhat.detach().cpu().view(-1))
            pre_preds.append(pre_yhat.detach().cpu().view(-1))

    return {
        "y_pred_pre_z": torch.cat(pre_preds, dim=0).to(torch.float32),
        "y_pred_post_z": torch.cat(post_preds, dim=0).to(torch.float32),
        "y_true_z": _flatten_targets(state["y_val"].detach().cpu()).to(torch.float32),
    }


def evaluate_pre_post_adaptation_state(
    state: Mapping[str, Any],
    *,
    batch_size: int = 512,
    show_plots: bool = True,
    print_metrics: bool = True,
) -> dict[str, Any]:
    """Purpose: evaluate pre- vs post-adaptation predictions for one trained workflow state.

    Input:
        state: Workflow state containing a trained model and validation tensors.
        batch_size: Inference batch size.
        show_plots: When True, renders pre/post observed-vs-predicted plots.
        print_metrics: When True, prints pre/post RMSE summaries.

    Output:
        A new workflow state containing pre/post prediction tensors and RMSE summaries.
    """
    preds = _predict_pre_post_adaptation(state, batch_size=batch_size)
    standardized_targets = bool(state.get("standardized_targets", False))
    y_mean_t, y_std_t = _raw_scale_tensors(
        y_mean_raw=state.get("y_mean_raw"),
        y_std_raw=state.get("y_std_raw"),
        standardized_targets=standardized_targets,
    )

    rmse_pre_model = torch.sqrt(F.mse_loss(preds["y_pred_pre_z"], preds["y_true_z"])).item()
    rmse_post_model = torch.sqrt(F.mse_loss(preds["y_pred_post_z"], preds["y_true_z"])).item()

    y_pred_pre_raw = preds["y_pred_pre_z"] * y_std_t + y_mean_t
    y_pred_post_raw = preds["y_pred_post_z"] * y_std_t + y_mean_t
    y_true_raw = preds["y_true_z"] * y_std_t + y_mean_t
    rmse_pre_raw = torch.sqrt(F.mse_loss(y_pred_pre_raw, y_true_raw)).item()
    rmse_post_raw = torch.sqrt(F.mse_loss(y_pred_post_raw, y_true_raw)).item()

    new_state = dict(state)
    new_state.update(preds)
    new_state.update(
        {
            "y_mean_t": y_mean_t,
            "y_std_t": y_std_t,
            "y_pred_pre_raw": y_pred_pre_raw,
            "y_pred_post_raw": y_pred_post_raw,
            "y_true_raw": y_true_raw,
            "rmse_pre_z": rmse_pre_model if standardized_targets else None,
            "rmse_post_z": rmse_post_model if standardized_targets else None,
            "rmse_pre_model_space": rmse_pre_model,
            "rmse_post_model_space": rmse_post_model,
            "rmse_pre_raw": rmse_pre_raw,
            "rmse_post_raw": rmse_post_raw,
        }
    )

    if print_metrics:
        if standardized_targets:
            print(f"[Z] Pre-adaptation  RMSE: {new_state['rmse_pre_z']:.4f}")
            print(f"[Z] Post-adaptation RMSE: {new_state['rmse_post_z']:.4f}")
            print(f"[Z] ΔRMSE (pre - post):  {new_state['rmse_pre_z'] - new_state['rmse_post_z']:.4f}")
        print(f"[RAW] Pre-adaptation  RMSE: {new_state['rmse_pre_raw']:.4f}")
        print(f"[RAW] Post-adaptation RMSE: {new_state['rmse_post_raw']:.4f}")

    if show_plots:
        _plot_observed_vs_predicted(
            new_state["y_true_raw"],
            new_state["y_pred_pre_raw"],
            title="Observed vs Predicted (Validation, pre-adaptation)",
        )
        _plot_observed_vs_predicted(
            new_state["y_true_raw"],
            new_state["y_pred_post_raw"],
            title="Observed vs Predicted (Validation, post-adaptation)",
        )

    return new_state


def run_regression_workflow(
    dataset_name: str,
    cfg: Mapping[str, Any],
    *,
    feature_extractor: Any = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    run_seed: int = 42,
    dataset_seed: int | None = None,
    split_seed: int | None = None,
    train_ratio: float = 0.8,
    standardize: bool | str = "auto",
    glocal_init_from_true: bool = False,
    glocal_init_alpha: float = 1.0,
    checkpoint_label: str | None = None,
    include_eval: bool = True,
    include_pre_post: bool = False,
    show_plots: bool = False,
    print_metrics: bool = True,
) -> dict[str, Any]:
    """Purpose: execute the same dataset -> split -> train -> eval flow used by the notebook.

    Input:
        dataset_name: Synthetic or real dataset name.
        cfg: Base training config.
        feature_extractor: Optional feature extractor passed into `train_model`.
        dataset_kwargs: Optional dataset builder/loader arguments.
        run_seed: Seed used for training reproducibility.
        dataset_seed: Seed used for synthetic dataset creation.
        split_seed: Seed used when making the train/validation split.
        train_ratio: Fraction of rows assigned to training.
        standardize: `True`, `False`, or `"auto"` to standardize only real datasets.
        glocal_init_from_true: Whether to initialize glocal weights from true regime weights.
        glocal_init_alpha: Strength of that glocal-weight initialization blend.
        checkpoint_label: Optional suffix used to create a unique checkpoint file.
        include_eval: When True, computes post-adaptation metrics after training.
        include_pre_post: When True, also computes pre/post adaptation metrics.
        show_plots: When True, evaluation helpers render figures.
        print_metrics: When True, evaluation helpers print RMSE summaries.

    Output:
        A complete workflow state ready for notebook use or repeated-run aggregation.
    """
    dataset_kwargs = dict(dataset_kwargs or {})
    bundle_seed = dataset_seed if dataset_seed is not None else dataset_kwargs.pop("seed", 42)
    seed_everything(run_seed)

    state = load_regression_dataset_state(
        dataset_name,
        dataset_kwargs={**dataset_kwargs, "seed": bundle_seed},
    )
    state = split_regression_state(
        state,
        train_ratio=train_ratio,
        seed=split_seed if split_seed is not None else run_seed,
    )

    standardize_flag = _resolve_standardize_flag(standardize, state)
    if standardize_flag:
        state = standardize_regression_state(state, enabled=True)

    cfg_run = configure_regression_cfg_for_state(
        cfg,
        state,
        glocal_init_from_true=glocal_init_from_true,
        glocal_init_alpha=glocal_init_alpha,
    )
    state = train_regression_state(
        state,
        cfg_run,
        feature_extractor=feature_extractor,
        checkpoint_label=checkpoint_label,
    )
    state.update(
        {
            "dataset": state["display_name"],
            "run_seed": run_seed,
            "dataset_seed": bundle_seed,
            "split_seed": split_seed if split_seed is not None else run_seed,
        }
    )

    if include_eval:
        state = evaluate_regression_state(
            state,
            show_plot=show_plots,
            print_metrics=print_metrics,
        )

    if include_pre_post:
        state = evaluate_pre_post_adaptation_state(
            state,
            show_plots=show_plots,
            print_metrics=print_metrics,
        )

    return state


def run_single_regression_experiment(
    dataset_name: str,
    cfg: Mapping[str, Any],
    *,
    feature_extractor: Any = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    run_seed: int = 42,
    dataset_seed: int | None = None,
    split_seed: int | None = None,
    train_ratio: float = 0.8,
    standardize: bool | str = "auto",
    glocal_init_from_true: bool = False,
    glocal_init_alpha: float = 1.0,
    checkpoint_label: str | None = None,
) -> dict[str, Any]:
    """Purpose: backward-compatible single-run entry point built on the shared workflow helpers."""
    return run_regression_workflow(
        dataset_name,
        cfg,
        feature_extractor=feature_extractor,
        dataset_kwargs=dataset_kwargs,
        run_seed=run_seed,
        dataset_seed=dataset_seed,
        split_seed=split_seed,
        train_ratio=train_ratio,
        standardize=standardize,
        glocal_init_from_true=glocal_init_from_true,
        glocal_init_alpha=glocal_init_alpha,
        checkpoint_label=checkpoint_label,
        include_eval=True,
        include_pre_post=False,
        show_plots=False,
        print_metrics=False,
    )


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    """Purpose: format a metric as `mean +/- std` for paper tables."""
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def summarize_regression_runs(runs_df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Purpose: aggregate repeated-run results into table-ready mean/std summaries."""
    records: list[dict[str, Any]] = []
    grouped = runs_df.groupby(["dataset", "mode"], dropna=False, sort=False)

    for (dataset, mode), group in grouped:
        record: dict[str, Any] = {
            "dataset": dataset,
            "mode": mode,
            "num_runs": int(len(group)),
        }

        for metric_name in ("rmse_raw", "rmse_model_space", "rmse_z", "best_metric"):
            if metric_name not in group.columns:
                continue

            values = group[metric_name].dropna()
            if values.empty:
                continue

            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric_name}_mean"] = mean_value
            record[f"{metric_name}_std"] = std_value
            record[f"{metric_name}_table"] = format_mean_std(mean_value, std_value, digits=digits)

        records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def _build_run_record(state: Mapping[str, Any], *, mode: str, run_index: int, fold: int | None) -> dict[str, Any]:
    """Purpose: normalize one workflow state into a compact summary row."""
    return {
        "dataset": state["display_name"],
        "mode": mode,
        "run_index": run_index,
        "run_seed": state["run_seed"],
        "dataset_seed": state["dataset_seed"],
        "split_seed": state["split_seed"],
        "fold": fold,
        "best_metric": state["best_metric"],
        "standardized_targets": state["standardized_targets"],
        "rmse_model_space": state["rmse_model_space"],
        "rmse_raw": state["rmse_raw"],
        "rmse_z": state["rmse_z"],
    }


def run_repeated_regression_experiments(
    dataset_names: str | list[str],
    cfg: Mapping[str, Any],
    *,
    feature_extractor: Any = None,
    num_runs: int = 5,
    mode: str = "holdout",
    base_seed: int = 42,
    train_ratio: float = 0.8,
    standardize: bool | str = "auto",
    dataset_kwargs_map: Mapping[str, Mapping[str, Any]] | None = None,
    glocal_init_from_true: bool = False,
    glocal_init_alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    """Purpose: rerun one or more experiments and summarize RMSE mean/std for tables.

    Input:
        dataset_names: One dataset name or a list of datasets to evaluate.
        cfg: Base training config reused across runs.
        feature_extractor: Optional feature extractor passed into each training call.
        num_runs: Number of repeated holdout runs or number of CV folds.
        mode: Either `holdout`/`repeated_holdout` or `kfold`.
        base_seed: Base seed used to derive per-run randomness.
        train_ratio: Training split ratio used in holdout mode.
        standardize: `True`, `False`, or `"auto"` to standardize only real datasets.
        dataset_kwargs_map: Optional per-dataset loader/builder arguments.
        glocal_init_from_true: Whether to initialize glocal weights from true synthetic weights.
        glocal_init_alpha: Strength of that glocal-weight initialization blend.

    Output:
        A tuple of `(summary_df, run_df, artifacts)` containing paper-ready aggregates,
        per-run records, and the full workflow states for each run.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    dataset_kwargs_map = dataset_kwargs_map or {}
    mode_normalized = _normalize_name(mode)
    if mode_normalized not in {"holdout", "repeated_holdout", "kfold"}:
        raise ValueError("mode must be one of: 'holdout', 'repeated_holdout', 'kfold'.")

    run_records: list[dict[str, Any]] = []
    artifacts: dict[str, list[dict[str, Any]]] = {}

    for dataset_name in dataset_names:
        per_dataset_artifacts: list[dict[str, Any]] = []
        dataset_kwargs = dict(dataset_kwargs_map.get(dataset_name, {}))
        bundle_seed = dataset_kwargs.pop("seed", base_seed)

        if mode_normalized == "kfold":
            dataset_state = load_regression_dataset_state(
                dataset_name,
                dataset_kwargs={**dataset_kwargs, "seed": bundle_seed},
            )
            standardize_flag = _resolve_standardize_flag(standardize, dataset_state)
            kfold = KFold(n_splits=num_runs, shuffle=True, random_state=base_seed)

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset_state["X"])):
                run_seed = base_seed + fold_idx
                seed_everything(run_seed)

                run_state = split_regression_state(
                    dataset_state,
                    seed=base_seed,
                    train_idx=train_idx,
                    val_idx=val_idx,
                )
                if standardize_flag:
                    run_state = standardize_regression_state(run_state, enabled=True)

                cfg_run = configure_regression_cfg_for_state(
                    cfg,
                    run_state,
                    glocal_init_from_true=glocal_init_from_true,
                    glocal_init_alpha=glocal_init_alpha,
                )
                run_state = train_regression_state(
                    run_state,
                    cfg_run,
                    feature_extractor=feature_extractor,
                    checkpoint_label=f"fold_{fold_idx + 1}",
                )
                run_state.update(
                    {
                        "dataset": run_state["display_name"],
                        "run_seed": run_seed,
                        "dataset_seed": bundle_seed,
                        "split_seed": base_seed,
                        "mode": "kfold",
                        "run_index": fold_idx + 1,
                        "fold": fold_idx + 1,
                    }
                )
                run_state = evaluate_regression_state(
                    run_state,
                    show_plot=False,
                    print_metrics=False,
                )

                per_dataset_artifacts.append(run_state)
                run_records.append(
                    _build_run_record(
                        run_state,
                        mode="kfold",
                        run_index=fold_idx + 1,
                        fold=fold_idx + 1,
                    )
                )
        else:
            for run_idx in range(num_runs):
                run_seed = base_seed + run_idx
                artifact = run_regression_workflow(
                    dataset_name,
                    cfg,
                    feature_extractor=feature_extractor,
                    dataset_kwargs={**dataset_kwargs, "seed": bundle_seed},
                    run_seed=run_seed,
                    dataset_seed=bundle_seed,
                    split_seed=run_seed,
                    train_ratio=train_ratio,
                    standardize=standardize,
                    glocal_init_from_true=glocal_init_from_true,
                    glocal_init_alpha=glocal_init_alpha,
                    checkpoint_label=f"run_{run_idx + 1}",
                    include_eval=True,
                    include_pre_post=False,
                    show_plots=False,
                    print_metrics=False,
                )
                artifact["mode"] = "holdout"
                artifact["run_index"] = run_idx + 1
                artifact["fold"] = None
                per_dataset_artifacts.append(artifact)
                run_records.append(
                    _build_run_record(
                        artifact,
                        mode="holdout",
                        run_index=run_idx + 1,
                        fold=None,
                    )
                )

        artifacts[dataset_name] = per_dataset_artifacts

    runs_df = pd.DataFrame(run_records)
    summary_df = summarize_regression_runs(runs_df)
    return summary_df, runs_df, artifacts
