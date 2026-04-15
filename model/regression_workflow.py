from __future__ import annotations

import copy
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

from datasets.reg_data import DATATYPES, Reg_data, standardize_tensor
from model.nnknn_model import train_model


_TRUE_W_LINEAR = torch.tensor([2.5, -1.7, 0.0, 0.9, 3.2], dtype=torch.float32).view(-1, 1)
_REAL_DATASET_LOOKUP = {
    re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"): name for name in DATATYPES
}


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def list_supported_regression_datasets() -> dict[str, list[str]]:
    return {
        "synthetic": [
            "linear_regression",
            "mixture_two_linear_models",
            "redundant_features",
        ],
        "real": sorted(DATATYPES.keys()),
    }


def make_linear_regression_dataset(
    n: int = 3000,
    d: int = 5,
    noise_scale: float = 1.0,
    seed: int = 42,
    true_w: torch.Tensor | None = None,
) -> dict[str, Any]:
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


def _flatten_targets(y: torch.Tensor) -> torch.Tensor:
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
) -> dict[str, Any]:
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
    )


def split_regression_data_from_indices(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    train_idx: torch.Tensor | np.ndarray,
    val_idx: torch.Tensor | np.ndarray,
    regime_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    val_idx = torch.as_tensor(val_idx, dtype=torch.long)

    X_train = X[train_idx].clone()
    X_val = X[val_idx].clone()
    y_train = _flatten_targets(y[train_idx].clone())
    y_val = _flatten_targets(y[val_idx].clone())

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


def prepare_regression_split_for_training(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    standardize: bool = True,
) -> dict[str, Any]:
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


def prepare_cfg_for_dataset(
    cfg: dict[str, Any],
    dataset_bundle: dict[str, Any],
    *,
    glocal_init_from_true: bool = False,
    glocal_init_alpha: float = 1.0,
) -> dict[str, Any]:
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


def _clone_feature_extractor(feature_extractor: Any) -> Any:
    if feature_extractor is None:
        return None
    return copy.deepcopy(feature_extractor)


def _resolve_standardize_flag(
    standardize: bool | str,
    dataset_bundle: dict[str, Any],
) -> bool:
    if standardize == "auto":
        return dataset_bundle.get("dataset_kind") == "real"
    return bool(standardize)


def _checkpoint_path_for_run(
    checkpoint_path: str,
    *,
    dataset_name: str,
    run_label: str,
) -> str:
    base = Path(checkpoint_path)
    parent = base.parent if str(base.parent) not in {"", "."} else Path("checkpoints")
    parent.mkdir(parents=True, exist_ok=True)

    safe_dataset = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name)
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", run_label)
    suffix = base.suffix or ".pth"
    stem = base.stem if base.suffix else base.name
    return str(parent / f"{stem}_{safe_dataset}_{safe_label}{suffix}")


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


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

    if standardized_targets:
        y_mean_t = torch.as_tensor(y_mean_raw, dtype=torch.float32)
        y_std_t = torch.as_tensor(y_std_raw, dtype=torch.float32).clamp_min(1e-6)
        y_pred_raw = y_pred_model_space * y_std_t + y_mean_t
        y_true_raw = y_true_model_space * y_std_t + y_mean_t
        rmse_raw = torch.sqrt(F.mse_loss(y_pred_raw, y_true_raw)).item()
        rmse_z = rmse_model_space
    else:
        y_pred_raw = y_pred_model_space
        y_true_raw = y_true_model_space
        rmse_raw = rmse_model_space
        rmse_z = None

    return {
        "y_pred_model_space": y_pred_model_space,
        "y_true_model_space": y_true_model_space,
        "y_pred_raw": y_pred_raw,
        "y_true_raw": y_true_raw,
        "rmse_model_space": rmse_model_space,
        "rmse_raw": rmse_raw,
        "rmse_z": rmse_z,
    }


def run_single_regression_experiment(
    dataset_name: str,
    cfg: dict[str, Any],
    *,
    feature_extractor: Any = None,
    dataset_kwargs: dict[str, Any] | None = None,
    run_seed: int = 42,
    dataset_seed: int | None = None,
    split_seed: int | None = None,
    train_ratio: float = 0.8,
    standardize: bool | str = "auto",
    glocal_init_from_true: bool = False,
    glocal_init_alpha: float = 1.0,
    checkpoint_label: str | None = None,
) -> dict[str, Any]:
    dataset_kwargs = dict(dataset_kwargs or {})
    bundle_seed = dataset_seed if dataset_seed is not None else dataset_kwargs.pop("seed", 42)
    dataset_bundle = get_regression_dataset(dataset_name, seed=bundle_seed, **dataset_kwargs)

    split_bundle = split_regression_data(
        dataset_bundle["X"],
        dataset_bundle["y"],
        train_ratio=train_ratio,
        seed=split_seed if split_seed is not None else run_seed,
        regime_labels=dataset_bundle.get("regime_labels"),
    )

    standardize_flag = _resolve_standardize_flag(standardize, dataset_bundle)
    prepared_split = prepare_regression_split_for_training(
        split_bundle["X_train"],
        split_bundle["y_train"],
        split_bundle["X_val"],
        split_bundle["y_val"],
        standardize=standardize_flag,
    )

    cfg_run = prepare_cfg_for_dataset(
        cfg,
        dataset_bundle,
        glocal_init_from_true=glocal_init_from_true,
        glocal_init_alpha=glocal_init_alpha,
    )

    checkpoint_path = cfg_run.get("checkpoint_path", "nnknn_regression_best.pth")
    cfg_run["checkpoint_path"] = _checkpoint_path_for_run(
        checkpoint_path,
        dataset_name=dataset_bundle["display_name"],
        run_label=checkpoint_label or f"seed_{run_seed}",
    )

    seed_everything(run_seed)
    best_metric, glocal_weightor, model = train_model(
        prepared_split["X_train"],
        prepared_split["y_train_norm"],
        prepared_split["X_val"],
        prepared_split["y_val_norm"],
        feature_extractor=_clone_feature_extractor(feature_extractor),
        cfg=cfg_run,
    )

    metrics = evaluate_regression_model(
        model,
        prepared_split["X_val"],
        prepared_split["y_val"],
        y_mean_raw=prepared_split["y_mean_raw"],
        y_std_raw=prepared_split["y_std_raw"],
        standardized_targets=prepared_split["standardized_targets"],
    )

    return {
        "dataset": dataset_bundle["display_name"],
        "run_seed": run_seed,
        "dataset_seed": bundle_seed,
        "split_seed": split_seed if split_seed is not None else run_seed,
        "best_metric": best_metric,
        "cfg_run": cfg_run,
        "model": model,
        "glocal_weightor": glocal_weightor,
        **dataset_bundle,
        **split_bundle,
        **prepared_split,
        **metrics,
    }


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def summarize_regression_runs(runs_df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
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


def run_repeated_regression_experiments(
    dataset_names: str | list[str],
    cfg: dict[str, Any],
    *,
    feature_extractor: Any = None,
    num_runs: int = 5,
    mode: str = "holdout",
    base_seed: int = 42,
    train_ratio: float = 0.8,
    standardize: bool | str = "auto",
    dataset_kwargs_map: dict[str, dict[str, Any]] | None = None,
    glocal_init_from_true: bool = False,
    glocal_init_alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]]]:
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
            dataset_bundle = get_regression_dataset(dataset_name, seed=bundle_seed, **dataset_kwargs)
            standardize_flag = _resolve_standardize_flag(standardize, dataset_bundle)
            kfold = KFold(n_splits=num_runs, shuffle=True, random_state=base_seed)

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(dataset_bundle["X"])):
                run_seed = base_seed + fold_idx
                split_bundle = split_regression_data_from_indices(
                    dataset_bundle["X"],
                    dataset_bundle["y"],
                    train_idx=train_idx,
                    val_idx=val_idx,
                    regime_labels=dataset_bundle.get("regime_labels"),
                )
                prepared_split = prepare_regression_split_for_training(
                    split_bundle["X_train"],
                    split_bundle["y_train"],
                    split_bundle["X_val"],
                    split_bundle["y_val"],
                    standardize=standardize_flag,
                )
                cfg_run = prepare_cfg_for_dataset(
                    cfg,
                    dataset_bundle,
                    glocal_init_from_true=glocal_init_from_true,
                    glocal_init_alpha=glocal_init_alpha,
                )
                checkpoint_path = cfg_run.get("checkpoint_path", "nnknn_regression_best.pth")
                cfg_run["checkpoint_path"] = _checkpoint_path_for_run(
                    checkpoint_path,
                    dataset_name=dataset_bundle["display_name"],
                    run_label=f"fold_{fold_idx + 1}",
                )

                seed_everything(run_seed)
                best_metric, glocal_weightor, model = train_model(
                    prepared_split["X_train"],
                    prepared_split["y_train_norm"],
                    prepared_split["X_val"],
                    prepared_split["y_val_norm"],
                    feature_extractor=_clone_feature_extractor(feature_extractor),
                    cfg=cfg_run,
                )

                metrics = evaluate_regression_model(
                    model,
                    prepared_split["X_val"],
                    prepared_split["y_val"],
                    y_mean_raw=prepared_split["y_mean_raw"],
                    y_std_raw=prepared_split["y_std_raw"],
                    standardized_targets=prepared_split["standardized_targets"],
                )

                artifact = {
                    "dataset": dataset_bundle["display_name"],
                    "mode": "kfold",
                    "run_index": fold_idx + 1,
                    "run_seed": run_seed,
                    "fold": fold_idx + 1,
                    "best_metric": best_metric,
                    "model": model,
                    "glocal_weightor": glocal_weightor,
                    **dataset_bundle,
                    **split_bundle,
                    **prepared_split,
                    **metrics,
                }
                per_dataset_artifacts.append(artifact)
                run_records.append(
                    {
                        "dataset": dataset_bundle["display_name"],
                        "mode": "kfold",
                        "run_index": fold_idx + 1,
                        "run_seed": run_seed,
                        "dataset_seed": bundle_seed,
                        "split_seed": base_seed,
                        "fold": fold_idx + 1,
                        "best_metric": best_metric,
                        "standardized_targets": prepared_split["standardized_targets"],
                        "rmse_model_space": metrics["rmse_model_space"],
                        "rmse_raw": metrics["rmse_raw"],
                        "rmse_z": metrics["rmse_z"],
                    }
                )
        else:
            for run_idx in range(num_runs):
                run_seed = base_seed + run_idx
                artifact = run_single_regression_experiment(
                    dataset_name,
                    cfg,
                    feature_extractor=feature_extractor,
                    dataset_kwargs=dataset_kwargs,
                    run_seed=run_seed,
                    dataset_seed=bundle_seed,
                    split_seed=run_seed,
                    train_ratio=train_ratio,
                    standardize=standardize,
                    glocal_init_from_true=glocal_init_from_true,
                    glocal_init_alpha=glocal_init_alpha,
                    checkpoint_label=f"run_{run_idx + 1}",
                )
                artifact["mode"] = "holdout"
                artifact["run_index"] = run_idx + 1
                per_dataset_artifacts.append(artifact)
                run_records.append(
                    {
                        "dataset": artifact["dataset"],
                        "mode": "holdout",
                        "run_index": run_idx + 1,
                        "run_seed": artifact["run_seed"],
                        "dataset_seed": artifact["dataset_seed"],
                        "split_seed": artifact["split_seed"],
                        "fold": None,
                        "best_metric": artifact["best_metric"],
                        "standardized_targets": artifact["standardized_targets"],
                        "rmse_model_space": artifact["rmse_model_space"],
                        "rmse_raw": artifact["rmse_raw"],
                        "rmse_z": artifact["rmse_z"],
                    }
                )

        artifacts[dataset_name] = per_dataset_artifacts

    runs_df = pd.DataFrame(run_records)
    summary_df = summarize_regression_runs(runs_df)
    return summary_df, runs_df, artifacts
