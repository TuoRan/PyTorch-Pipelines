# src/utils/device.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import torch

def get_device(prefer: Optional[str] = None) -> Tuple[torch.device, Dict[str, Any]]:
    """
    Select the best available torch device.

    Args:
        prefer: Optional override: "cuda", "mps", or "cpu".
                If not available, falls back to the best available device.

    Returns:
        (device, info) where device is torch.device, info is a small metadata dict.
    """
    prefer_norm = prefer.lower() if isinstance(prefer, str) else None

    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    # Decide backend
    if prefer_norm == "cpu":
        backend = "cpu"
    elif prefer_norm == "cuda":
        backend = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    elif prefer_norm == "mps":
        backend = "mps" if mps_ok else ("cuda" if cuda_ok else "cpu")
    else:
        backend = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")

    device = torch.device("cuda:0" if backend == "cuda" else backend)

    # Metadata
    if backend == "cuda":
        name = torch.cuda.get_device_name(0)
        supports_amp = True
    elif backend == "mps":
        name = "Apple MPS"
        supports_amp = False  # MPS autocast exists but is limited; keep it conservative.
    else:
        name = "CPU"
        supports_amp = False

    info: Dict[str, Any] = {
        "backend": backend,
        "device": str(device),
        "name": name,
        "supports_amp": supports_amp,
        "preferred": prefer_norm,
        "cuda_available": cuda_ok,
        "mps_available": mps_ok,
    }
    return device, info
