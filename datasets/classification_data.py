from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


DATA_DIR = Path(__file__).resolve().parent
_SMALL_ALIASES = {
    "iris": "iris",
    "zebra": "zebra",
    "zebra_special": "zebra_special",
    "wine": "wine",
    "breast_cancer": "breast_cancer",
    "breastcancer": "breast_cancer",
    "balance": "balance",
    "bal": "balance",
    "digits": "digits",
}
_IMAGE_ALIASES = {
    "mnist": "mnist",
    "cifar10": "cifar10",
    "cifar_10": "cifar10",
    "svhn": "svhn",
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def list_small_classification_datasets() -> list[str]:
    return ["iris", "zebra", "zebra_special", "wine", "breast_cancer", "balance", "digits"]


def list_image_classification_datasets() -> list[str]:
    return ["mnist", "cifar10", "svhn"]


def _zebra(seed: int, special: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    num_points = 110
    x1 = np.repeat(np.linspace(0, 100, num_points // 10), 10)
    y1 = rng.random(num_points) * 100
    labels1 = (x1 % 20 != 0).astype(np.int64)
    if not special:
        return torch.tensor(np.column_stack((x1, y1)), dtype=torch.float32), torch.tensor(labels1)

    y2 = np.repeat(np.linspace(0, 100, num_points // 10), 10)
    x2 = 100 + rng.random(num_points) * 100
    labels2 = (y2 % 20 != 0).astype(np.int64)
    X = np.column_stack((np.concatenate((x1, x2)), np.concatenate((y1, y2))))
    y = np.concatenate((labels1, labels2))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def _balance_scale() -> tuple[torch.Tensor, torch.Tensor]:
    label_map = {"L": 0, "B": 1, "R": 2}
    features: list[list[float]] = []
    labels: list[int] = []
    with (DATA_DIR / "balance-scale.data").open(newline="") as handle:
        for row in csv.reader(handle):
            labels.append(label_map[row[0]])
            features.append([float(value) for value in row[1:]])
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def load_small_classification_dataset(name: str, *, seed: int = 42) -> dict[str, Any]:
    """Load a paper small-dataset classification task without data leakage."""
    canonical = _SMALL_ALIASES.get(_normalize_name(name))
    if canonical is None:
        raise KeyError(f"Unknown small classification dataset '{name}'.")

    if canonical == "zebra":
        X, y = _zebra(seed, special=False)
    elif canonical == "zebra_special":
        X, y = _zebra(seed, special=True)
    elif canonical == "balance":
        X, y = _balance_scale()
    else:
        from sklearn import datasets as sk_datasets

        loaders = {
            "iris": sk_datasets.load_iris,
            "wine": sk_datasets.load_wine,
            "breast_cancer": sk_datasets.load_breast_cancer,
            "digits": sk_datasets.load_digits,
        }
        bunch = loaders[canonical]()
        X = torch.tensor(bunch.data, dtype=torch.float32)
        y = torch.tensor(bunch.target, dtype=torch.long)

    return {
        "dataset_name": canonical,
        "display_name": canonical,
        "dataset_kind": "small",
        "split_kind": "unsplit",
        "X": X,
        "y": y,
        "num_classes": int(torch.unique(y).numel()),
        "feature_dim": int(X.shape[1]),
    }


def _limited_indices(labels: np.ndarray, maximum: int | None, seed: int) -> np.ndarray:
    indices = np.arange(len(labels))
    if maximum is None or maximum >= len(indices):
        return indices
    if maximum < len(np.unique(labels)):
        raise ValueError("A stratified image subset must contain at least one item per class.")
    from sklearn.model_selection import train_test_split

    selected, _ = train_test_split(
        indices,
        train_size=maximum,
        stratify=labels,
        random_state=seed,
    )
    return np.sort(selected)


def _stack_images(dataset: Any, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    images = [dataset[int(index)][0] for index in indices]
    labels = [int(dataset[int(index)][1]) for index in indices]
    return torch.stack(images).to(torch.float32), torch.tensor(labels, dtype=torch.long)


def _image_labels(dataset: Any) -> np.ndarray:
    labels = getattr(dataset, "targets", None)
    if labels is None:
        labels = getattr(dataset, "labels")
    return np.asarray(labels)


def load_image_classification_dataset(
    name: str,
    *,
    data_root: str | Path | None = None,
    download: bool = True,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    seed: int = 42,
    retain_raw: bool = False,
) -> dict[str, Any]:
    """Load official image train/test splits and normalize using training images only."""
    canonical = _IMAGE_ALIASES.get(_normalize_name(name))
    if canonical is None:
        raise KeyError(f"Unknown image classification dataset '{name}'.")

    from torchvision import datasets as tv_datasets
    from torchvision import transforms

    root = Path(data_root) if data_root is not None else DATA_DIR
    to_tensor = transforms.ToTensor()
    if canonical == "mnist":
        train_set = tv_datasets.MNIST(root=str(root), train=True, download=download, transform=to_tensor)
        eval_set = tv_datasets.MNIST(root=str(root), train=False, download=download, transform=to_tensor)
    elif canonical == "cifar10":
        train_set = tv_datasets.CIFAR10(root=str(root), train=True, download=download, transform=to_tensor)
        eval_set = tv_datasets.CIFAR10(root=str(root), train=False, download=download, transform=to_tensor)
    else:
        svhn_root = root / "svhn"
        train_set = tv_datasets.SVHN(root=str(svhn_root), split="train", download=download, transform=to_tensor)
        eval_set = tv_datasets.SVHN(root=str(svhn_root), split="test", download=download, transform=to_tensor)

    train_labels = _image_labels(train_set)
    eval_labels = _image_labels(eval_set)
    train_indices = _limited_indices(train_labels, max_train_samples, seed)
    eval_indices = _limited_indices(eval_labels, max_eval_samples, seed + 1)
    X_train_raw, y_train = _stack_images(train_set, train_indices)
    X_val_raw, y_val = _stack_images(eval_set, eval_indices)

    channel_mean = X_train_raw.mean(dim=(0, 2, 3), keepdim=True)
    channel_std = X_train_raw.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    state = {
        "dataset_name": canonical,
        "display_name": canonical,
        "dataset_kind": "image",
        "split_kind": "official",
        "X_train": (X_train_raw - channel_mean) / channel_std,
        "y_train": y_train,
        "X_val": (X_val_raw - channel_mean) / channel_std,
        "y_val": y_val,
        "num_classes": int(torch.unique(y_train).numel()),
        "channel_mean": channel_mean,
        "channel_std": channel_std,
    }
    if retain_raw:
        state["X_train_raw"] = X_train_raw
        state["X_val_raw"] = X_val_raw
    return state
