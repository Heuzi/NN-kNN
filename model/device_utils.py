import os
import warnings

import torch


def _sm_tag(capability: tuple[int, int]) -> str:
    """Purpose: normalize CUDA capability tuples into PyTorch arch tags."""
    major, minor = capability
    return f"sm_{major}{minor}"


def cuda_build_supports_current_gpu() -> bool:
    """Purpose: report whether the installed CUDA build supports GPU0's SM version."""
    if not torch.cuda.is_available():
        return False
    try:
        current_sm = _sm_tag(torch.cuda.get_device_capability(0))
        return current_sm in set(torch.cuda.get_arch_list())
    except Exception:
        return False


def resolve_runtime_device(env_var: str = "NNKNN_DEVICE") -> torch.device:
    """Purpose: choose a safe runtime device, with optional env override.

    `NNKNN_DEVICE=cpu` forces CPU.
    `NNKNN_DEVICE=cuda` forces CUDA and skips compatibility fallback checks.
    """
    requested = os.getenv(env_var)
    if requested:
        return torch.device(requested)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        current_sm = _sm_tag(torch.cuda.get_device_capability(0))
        supported_sms = set(torch.cuda.get_arch_list())
    except Exception as exc:
        warnings.warn(
            f"Falling back to CPU because CUDA capability detection failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    if current_sm not in supported_sms:
        warnings.warn(
            "Falling back to CPU because the installed PyTorch CUDA build does not "
            f"support GPU0 ({current_sm}). Supported SMs: {sorted(supported_sms)}. "
            "Install a matching PyTorch CUDA wheel or set NNKNN_DEVICE=cuda to force it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    return torch.device("cuda")
