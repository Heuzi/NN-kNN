from __future__ import annotations

import copy
import os
import random
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from datasets.classification_data import (
    list_image_classification_datasets,
    list_small_classification_datasets,
    load_image_classification_dataset,
    load_small_classification_dataset,
)
from model.device_utils import resolve_runtime_device
from model.feature_extractors import CIFAR10Classifier, MNISTClassifier, get_feature_extractor_from
from model.nnknn_model import default_args, train_model


_SMALL_METHODS = ["nnknn", "knn", "mlp"]
_IMAGE_METHODS = [
    "nnknn_conv_trainable",
    "nnknn_conv_frozen",
    "convnet",
    "knn_pixels",
    "knn_conv_frozen",
]
_METHOD_ALIASES = {
    "nnknn": "nnknn",
    "nn_knn": "nnknn",
    "knn": "knn",
    "mlp": "mlp",
    "nnknn_conv_trainable": "nnknn_conv_trainable",
    "nnknn_trainable": "nnknn_conv_trainable",
    "nnknn_conv_frozen": "nnknn_conv_frozen",
    "nnknn_frozen": "nnknn_conv_frozen",
    "convnet": "convnet",
    "cnn": "convnet",
    "knn_pixels": "knn_pixels",
    "knn_conv_frozen": "knn_conv_frozen",
    "knn_features": "knn_conv_frozen",
}


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def list_supported_classification_datasets() -> dict[str, list[str]]:
    return {
        "small": list_small_classification_datasets(),
        "image": list_image_classification_datasets(),
    }


def list_supported_classification_benchmark_methods() -> dict[str, list[str]]:
    return {"small": list(_SMALL_METHODS), "image": list(_IMAGE_METHODS)}


def _validate_classification_cfg(cfg: Mapping[str, Any]) -> None:
    if cfg.get("task_type") != "classification":
        raise ValueError("Classification configs must set task_type='classification'.")
    if not cfg.get("normalize_over_cases", False):
        raise ValueError("Classification configs must set normalize_over_cases=True.")
    if cfg.get("case_normalizer") not in {"softmax", "sparsemax"}:
        raise ValueError("Maintained classification supports case_normalizer='softmax' or 'sparsemax'.")
    if cfg.get("classification_loss") != "nll_class_mass":
        raise ValueError("Maintained classification uses classification_loss='nll_class_mass'.")
    if cfg.get("regression_locality", False) or cfg.get("use_nn_cdh", False):
        raise ValueError("Regression locality and NN-CDH are not classification components.")
    if cfg.get("neg_weight_flag", False) or cfg.get("case_score_mode") == "neg_distance_logw":
        raise ValueError("Legacy classification case weights are unsupported by the current core.")


def make_classification_cfg(
    overrides: Mapping[str, Any] | None = None,
    *,
    base_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg or default_args))
    cfg.update(
        {
            "task_type": "classification",
            "case_score_mode": "bias_minus_distance",
            "normalize_over_cases": True,
            "case_normalizer": "softmax",
            "classification_loss": "nll_class_mass",
            "regression_locality": False,
            "use_nn_cdh": False,
            "nn_cdh_pretrain": False,
            "neg_weight_flag": False,
            "checkpoint_path": "checkpoints/nnknn_classification_best.pth",
            "batch_size": 32,
            "patience": 20,
        }
    )
    if overrides:
        cfg.update(copy.deepcopy(dict(overrides)))
    _validate_classification_cfg(cfg)
    return cfg


def load_classification_dataset_state(
    dataset_name: str,
    *,
    dataset_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = _normalize_name(dataset_name)
    kwargs = dict(dataset_kwargs or {})
    if normalized in {"mnist", "cifar10", "cifar_10", "svhn"}:
        state = load_image_classification_dataset(dataset_name, **kwargs)
    else:
        state = load_small_classification_dataset(dataset_name, **kwargs)
    state = {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state.items()
    }
    state["dataset"] = state["display_name"]
    if "X" in state:
        state["N"] = int(state["X"].shape[0])
        state["D"] = int(state["X"].shape[1])
    else:
        state["N_train"] = int(state["X_train"].shape[0])
        state["N_val"] = int(state["X_val"].shape[0])
    return state


def split_classification_state(
    state: Mapping[str, Any],
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    train_idx: np.ndarray | torch.Tensor | None = None,
    val_idx: np.ndarray | torch.Tensor | None = None,
    standardize: bool = True,
    image_validation_ratio: float = 0.1,
) -> dict[str, Any]:
    if state.get("dataset_kind") == "image":
        full_X_train = state["X_train"]
        full_y_train = state["y_train"]
        num_classes = int(state["num_classes"])
        validation_size = max(num_classes, int(round(len(full_y_train) * image_validation_ratio)))
        validation_size = min(validation_size, len(full_y_train) - num_classes)
        if validation_size < num_classes:
            raise ValueError("Image training subset is too small for stratified inner validation.")
        all_indices = np.arange(len(full_y_train))
        image_train_idx, image_val_idx = train_test_split(
            all_indices,
            test_size=validation_size,
            stratify=full_y_train.numpy(),
            random_state=seed,
        )
        new_state = dict(state)
        new_state.update(
            {
                "X_train_full": full_X_train,
                "y_train_full": full_y_train,
                "X_test": state["X_val"],
                "y_test": state["y_val"],
                "X_train": full_X_train[image_train_idx],
                "y_train": full_y_train[image_train_idx],
                "X_val": full_X_train[image_val_idx],
                "y_val": full_y_train[image_val_idx],
                "train_idx": torch.tensor(image_train_idx, dtype=torch.long),
                "val_idx": torch.tensor(image_val_idx, dtype=torch.long),
                "split_seed": seed,
                "standardized_features": True,
                "evaluation_split": "official_test",
            }
        )
        return new_state

    X = state["X"].detach().cpu().numpy()
    y = state["y"].detach().cpu().numpy()
    if train_idx is None or val_idx is None:
        indices = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=y,
            random_state=seed,
        )
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    X_train_raw = X[train_idx]
    X_val_raw = X[val_idx]
    scaler = None
    if standardize:
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
    else:
        X_train, X_val = X_train_raw, X_val_raw

    new_state = dict(state)
    new_state.update(
        {
            "X_train_raw": torch.tensor(X_train_raw, dtype=torch.float32),
            "X_val_raw": torch.tensor(X_val_raw, dtype=torch.float32),
            "X_train": torch.tensor(X_train, dtype=torch.float32),
            "X_val": torch.tensor(X_val, dtype=torch.float32),
            "y_train": torch.tensor(y[train_idx], dtype=torch.long),
            "y_val": torch.tensor(y[val_idx], dtype=torch.long),
            "train_idx": torch.tensor(train_idx, dtype=torch.long),
            "val_idx": torch.tensor(val_idx, dtype=torch.long),
            "scaler": scaler,
            "standardized_features": bool(standardize),
            "split_seed": seed,
            "train_ratio": train_ratio,
        }
    )
    return new_state


def _checkpoint_path_for_run(path: str, dataset_name: str, label: str) -> str:
    checkpoint = Path(path)
    suffix = checkpoint.suffix or ".pth"
    stem = checkpoint.stem if checkpoint.suffix else checkpoint.name
    safe_dataset = _normalize_name(dataset_name)
    safe_label = _normalize_name(label)
    return str(checkpoint.with_name(f"{stem}_{safe_dataset}_{safe_label}{suffix}"))


def build_image_classifier(dataset_name: str) -> nn.Module:
    normalized = _normalize_name(dataset_name)
    if normalized == "mnist":
        return MNISTClassifier()
    if normalized in {"cifar10", "cifar_10", "svhn"}:
        return CIFAR10Classifier()
    raise KeyError(f"No image classifier architecture for '{dataset_name}'.")


def build_nnknn_image_feature_extractor(dataset_name: str) -> nn.Module:
    return get_feature_extractor_from(build_image_classifier(dataset_name))


def train_nnknn_classification_state(
    state: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    feature_extractor: nn.Module | None = None,
    checkpoint_label: str | None = None,
    clone_feature_extractor: bool = True,
) -> dict[str, Any]:
    _validate_classification_cfg(cfg)
    cfg_run = copy.deepcopy(dict(cfg))
    if checkpoint_label is not None:
        cfg_run["checkpoint_path"] = _checkpoint_path_for_run(
            str(cfg_run.get("checkpoint_path", "checkpoints/nnknn_classification_best.pth")),
            str(state["display_name"]),
            checkpoint_label,
        )
    checkpoint = Path(str(cfg_run["checkpoint_path"]))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    run_feature_extractor = (
        copy.deepcopy(feature_extractor) if feature_extractor is not None and clone_feature_extractor else feature_extractor
    )
    best_metric, glocal_weightor, model = train_model(
        state["X_train"],
        state["y_train"],
        state["X_val"],
        state["y_val"],
        feature_extractor=run_feature_extractor,
        cfg=cfg_run,
    )
    result = dict(state)
    result.update(
        {
            "cfg": copy.deepcopy(dict(cfg)),
            "cfg_run": cfg_run,
            "feature_extractor": feature_extractor,
            "best_metric": float(best_metric),
            "best_acc": float(best_metric),
            "glocal_weightor": glocal_weightor,
            "model": model,
        }
    )
    return result


def evaluate_classification_model(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    batch_size: int = 512,
) -> dict[str, Any]:
    run_device = next(model.parameters()).device
    model.eval()
    masses: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    top_cases: list[torch.Tensor] = []
    top_labels: list[torch.Tensor] = []
    top_activations: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, X_val.size(0), batch_size):
            class_mass, predicted, _, cases, labels, activations = model(
                X_val[start : start + batch_size].to(run_device)
            )
            masses.append(class_mass.detach().cpu())
            predictions.append(predicted.detach().cpu())
            if cases is not None:
                top_cases.append(cases.detach().cpu())
                top_labels.append(labels.detach().cpu())
                top_activations.append(activations.detach().cpu())
    predicted = torch.cat(predictions)
    metrics: dict[str, Any] = {
        "class_probabilities": torch.cat(masses),
        "predictions": predicted,
        "accuracy": float(accuracy_score(y_val.detach().cpu().numpy(), predicted.numpy())),
    }
    if top_cases:
        activated_labels = torch.cat(top_labels)
        metrics.update(
            {
                "most_activated_cases": torch.cat(top_cases),
                "most_activated_case_labels": activated_labels,
                "most_activated_class_ids": activated_labels.argmax(dim=-1),
                "most_activated_activations": torch.cat(top_activations),
            }
        )
    return metrics


def evaluate_nnknn_classification_state(
    state: Mapping[str, Any],
    *,
    batch_size: int = 512,
    print_metrics: bool = True,
) -> dict[str, Any]:
    X_eval = state.get("X_test", state["X_val"])
    y_eval = state.get("y_test", state["y_val"])
    metrics = evaluate_classification_model(
        state["model"], X_eval, y_eval, batch_size=batch_size
    )
    result = dict(state)
    result.update(metrics)
    if print_metrics:
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    return result


def run_single_nnknn_classification_experiment(
    dataset_name: str,
    cfg: Mapping[str, Any],
    *,
    feature_extractor: nn.Module | None = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    run_seed: int = 42,
    split_seed: int | None = None,
    train_ratio: float = 0.8,
    standardize: bool = True,
    checkpoint_label: str | None = None,
) -> dict[str, Any]:
    seed_everything(run_seed)
    state = load_classification_dataset_state(dataset_name, dataset_kwargs=dataset_kwargs)
    state = split_classification_state(
        state,
        train_ratio=train_ratio,
        seed=run_seed if split_seed is None else split_seed,
        standardize=standardize,
    )
    if feature_extractor is None and state["dataset_kind"] == "image":
        feature_extractor = build_nnknn_image_feature_extractor(str(state["dataset_name"]))
    state["run_seed"] = run_seed
    trained = train_nnknn_classification_state(
        state,
        cfg,
        feature_extractor=feature_extractor,
        checkpoint_label=checkpoint_label,
    )
    return evaluate_nnknn_classification_state(trained, print_metrics=False)


class _MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in (128, 64, 32, 16):
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(start_dim=1))


def _train_torch_classifier(
    model: nn.Module,
    state: Mapping[str, Any],
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
) -> tuple[nn.Module, float]:
    run_device = resolve_runtime_device()
    model = model.to(run_device)
    loader = DataLoader(
        TensorDataset(state["X_train"], state["y_train"]),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    stale = 0
    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X.to(run_device)), batch_y.to(run_device))
            loss.backward()
            optimizer.step()
        predicted = _predict_torch_classifier(model, state["X_val"], batch_size=batch_size)
        accuracy = float(accuracy_score(state["y_val"].numpy(), predicted.numpy()))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale > patience:
                break
    model.load_state_dict(best_state)
    return model, best_accuracy


@torch.no_grad()
def _predict_torch_classifier(
    model: nn.Module,
    X: torch.Tensor,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    run_device = next(model.parameters()).device
    model.eval()
    predictions = []
    for start in range(0, X.size(0), batch_size):
        predictions.append(model(X[start : start + batch_size].to(run_device)).argmax(dim=1).cpu())
    return torch.cat(predictions)


def _evaluate_knn(
    state: Mapping[str, Any],
    *,
    features_train: torch.Tensor | None = None,
    features_val: torch.Tensor | None = None,
    n_neighbors: int = 5,
) -> dict[str, Any]:
    X_train = features_train if features_train is not None else state["X_train"]
    X_eval = features_val if features_val is not None else state.get("X_test", state["X_val"])
    y_eval = state.get("y_test", state["y_val"])
    classifier = KNeighborsClassifier(n_neighbors=min(n_neighbors, len(state["y_train"])))
    classifier.fit(X_train.flatten(start_dim=1).numpy(), state["y_train"].numpy())
    predicted = classifier.predict(X_eval.flatten(start_dim=1).numpy())
    result = dict(state)
    result.update(
        {
            "baseline_model": classifier,
            "predictions": torch.tensor(predicted, dtype=torch.long),
            "accuracy": float(accuracy_score(y_eval.numpy(), predicted)),
        }
    )
    return result


@torch.no_grad()
def _extract_conv_features(
    classifier: nn.Module,
    X: torch.Tensor,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    extractor = get_feature_extractor_from(copy.deepcopy(classifier)).to(resolve_runtime_device())
    extractor.eval()
    features = []
    for start in range(0, X.size(0), batch_size):
        features.append(extractor(X[start : start + batch_size].to(resolve_runtime_device())).cpu())
    return torch.cat(features)


def run_classification_benchmark_methods_on_state(
    state: Mapping[str, Any],
    nnknn_cfg: Mapping[str, Any] | None = None,
    *,
    methods: list[str] | None = None,
    method_cfgs: Mapping[str, Mapping[str, Any]] | None = None,
    run_seed: int = 42,
    checkpoint_label_prefix: str | None = None,
) -> dict[str, dict[str, Any]]:
    is_image = state.get("dataset_kind") == "image"
    requested = methods or (_IMAGE_METHODS if is_image else _SMALL_METHODS)
    canonical = [_METHOD_ALIASES.get(_normalize_name(method), _normalize_name(method)) for method in requested]
    if is_image:
        canonical = [
            "nnknn_conv_trainable" if method == "nnknn" else method for method in canonical
        ]
    cfgs = {
        _METHOD_ALIASES.get(_normalize_name(name), _normalize_name(name)): dict(value)
        for name, value in (method_cfgs or {}).items()
    }
    results: dict[str, dict[str, Any]] = {}
    pretrained_classifier: nn.Module | None = None

    def get_pretrained_classifier() -> nn.Module:
        nonlocal pretrained_classifier
        if pretrained_classifier is None:
            kwargs = dict(cfgs.get("convnet", {}))
            pretrained_classifier, _ = _train_torch_classifier(
                build_image_classifier(str(state["dataset_name"])), state, **kwargs
            )
        return pretrained_classifier

    for method in canonical:
        seed_everything(run_seed)
        print(f"[classification benchmark] {state['display_name']} method={method}", flush=True)
        method_cfg = dict(cfgs.get(method, {}))
        if method in {"nnknn", "nnknn_conv_trainable", "nnknn_conv_frozen"}:
            if nnknn_cfg is None:
                raise ValueError("nnknn_cfg is required for NN-kNN benchmark methods.")
            if not is_image and method != "nnknn":
                raise ValueError(f"Method '{method}' is for image datasets.")
            extractor = None
            cfg_run = copy.deepcopy(dict(nnknn_cfg))
            if is_image:
                if method == "nnknn_conv_frozen":
                    extractor = get_feature_extractor_from(copy.deepcopy(get_pretrained_classifier()))
                    cfg_run["freeze_feature_extractor"] = True
                else:
                    extractor = build_nnknn_image_feature_extractor(str(state["dataset_name"]))
            label = f"{checkpoint_label_prefix}_{method}" if checkpoint_label_prefix else method
            artifact = train_nnknn_classification_state(
                state, cfg_run, feature_extractor=extractor, checkpoint_label=label
            )
            artifact = evaluate_nnknn_classification_state(artifact, print_metrics=False)
            artifact["baseline_name"] = "NN-kNN" if not is_image else method
            results[method] = artifact
        elif method in {"knn", "knn_pixels"}:
            artifact = _evaluate_knn(state, **method_cfg)
            artifact["baseline_name"] = method
            results[method] = artifact
        elif method == "mlp":
            model = _MLPClassifier(int(state["X_train"][0].numel()), int(state["num_classes"]))
            model, accuracy = _train_torch_classifier(model, state, **method_cfg)
            predicted = _predict_torch_classifier(model, state["X_val"])
            artifact = dict(state)
            artifact.update(
                {
                    "baseline_model": model,
                    "predictions": predicted,
                    "accuracy": accuracy,
                    "baseline_name": "mlp",
                }
            )
            results[method] = artifact
        elif method == "convnet":
            model = get_pretrained_classifier()
            X_eval = state.get("X_test", state["X_val"])
            y_eval = state.get("y_test", state["y_val"])
            predicted = _predict_torch_classifier(model, X_eval)
            artifact = dict(state)
            artifact.update(
                {
                    "baseline_model": model,
                    "predictions": predicted,
                    "accuracy": float(accuracy_score(y_eval.numpy(), predicted.numpy())),
                    "baseline_name": "convnet",
                }
            )
            results[method] = artifact
        elif method == "knn_conv_frozen":
            classifier = get_pretrained_classifier()
            artifact = _evaluate_knn(
                state,
                features_train=_extract_conv_features(classifier, state["X_train"]),
                features_val=_extract_conv_features(
                    classifier, state.get("X_test", state["X_val"])
                ),
                **method_cfg,
            )
            artifact["baseline_name"] = "knn_conv_frozen"
            results[method] = artifact
        else:
            raise ValueError(f"Unsupported classification benchmark method '{method}'.")
    return results


def summarize_classification_benchmark_runs(runs_df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (dataset, mode, method), group in runs_df.groupby(["dataset", "mode", "method"], sort=False):
        values = group["accuracy"]
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        records.append(
            {
                "dataset": dataset,
                "mode": mode,
                "method": method,
                "num_runs": len(values),
                "accuracy_mean": mean,
                "accuracy_std": std,
                "accuracy_table": f"{mean:.{digits}f} +/- {std:.{digits}f}",
            }
        )
    return pd.DataFrame(records)


def run_repeated_classification_model_benchmarks(
    dataset_names: str | list[str],
    nnknn_cfg: Mapping[str, Any] | None = None,
    *,
    methods: list[str] | None = None,
    method_cfgs: Mapping[str, Mapping[str, Any]] | None = None,
    num_runs: int = 5,
    mode: str = "holdout",
    base_seed: int = 42,
    train_ratio: float = 0.8,
    standardize: bool = True,
    dataset_kwargs_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, list[dict[str, Any]]]]]:
    names = [dataset_names] if isinstance(dataset_names, str) else list(dataset_names)
    normalized_mode = _normalize_name(mode)
    kwargs_map = dataset_kwargs_map or {}
    records: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset_name in names:
        initial = load_classification_dataset_state(
            dataset_name, dataset_kwargs=kwargs_map.get(dataset_name)
        )
        is_image = initial["dataset_kind"] == "image"
        if is_image and normalized_mode not in {"preset", "official", "holdout"}:
            raise ValueError("Image benchmarks use their official/preset test split.")
        requested = methods or (_IMAGE_METHODS if is_image else _SMALL_METHODS)
        method_names = [_METHOD_ALIASES.get(_normalize_name(method), _normalize_name(method)) for method in requested]
        if is_image:
            method_names = [
                "nnknn_conv_trainable" if method == "nnknn" else method
                for method in method_names
            ]
        artifacts[dataset_name] = {method: [] for method in method_names}
        if is_image:
            splits = [(None, None)] * num_runs
            run_mode = "official"
        elif normalized_mode == "kfold":
            splitter = StratifiedKFold(n_splits=num_runs, shuffle=True, random_state=base_seed)
            splits = list(splitter.split(initial["X"], initial["y"]))
            run_mode = "kfold"
        else:
            splits = [(None, None)] * num_runs
            run_mode = "holdout"

        for index, (train_idx, val_idx) in enumerate(splits, start=1):
            run_seed = base_seed + index - 1
            state = split_classification_state(
                initial,
                train_ratio=train_ratio,
                seed=run_seed,
                train_idx=train_idx,
                val_idx=val_idx,
                standardize=standardize,
            )
            state["run_index"] = index
            state["run_seed"] = run_seed
            results = run_classification_benchmark_methods_on_state(
                state,
                nnknn_cfg,
                methods=method_names,
                method_cfgs=method_cfgs,
                run_seed=run_seed,
                checkpoint_label_prefix=f"{run_mode}_{index}",
            )
            for method, artifact in results.items():
                artifacts[dataset_name].setdefault(method, []).append(artifact)
                records.append(
                    {
                        "dataset": str(state["display_name"]),
                        "mode": run_mode,
                        "method": method,
                        "run_index": index,
                        "run_seed": run_seed,
                        "accuracy": float(artifact["accuracy"]),
                    }
                )
    runs_df = pd.DataFrame(records)
    return summarize_classification_benchmark_runs(runs_df), runs_df, artifacts
