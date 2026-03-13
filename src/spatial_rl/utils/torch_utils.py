from __future__ import annotations

from typing import Any


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install with: pip install -e .[ml]"
        ) from exc
    return torch


def resolve_device(requested: str) -> str:
    torch = require_torch()
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS requested but not available")

    return requested


def resolve_dtype(torch: Any, dtype_name: str):
    normalized = dtype_name.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype_name}")
