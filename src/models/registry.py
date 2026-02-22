# src/models/registry.py
from __future__ import annotations
from typing import Dict, List, Optional, Type, Union
import torch
import torch.nn as nn

from src.models.vision.simple_cnn import SimpleCNN
from src.models.vision.resnet.resnet18 import ResNet18
from src.models.text.text_mlp import TextMLP

# Models registry data
_REGISTRY: Dict[str, Type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "resnet18": ResNet18,
    "text_mlp": TextMLP
}

# Optional aliases so users can type shorter names without guessing.
_ALIASES: Dict[str, str] = {
    "resnet": "resnet18",
    "cnn": "simple_cnn",
    "text": "text_mlp"
}


def available_models() -> List[str]:
    """Return sorted list of canonical model names."""
    return sorted(_REGISTRY.keys())


def _normalize_name(name: str) -> str:
    key = name.strip().lower()
    return _ALIASES.get(key, key)


def get_model(
    name: str,
    *,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
) -> nn.Module:
    """
    Construct a model from the registry.

    Args:
        name: Model key (e.g., 'simple_cnn', 'resnet18'). Aliases supported ('resnet').
        device: Optional device to move the model to.
        **kwargs: Forwarded to the model constructor.

    Returns:
        nn.Module on the requested device (if provided).
    """
    key = _normalize_name(name)
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {available_models()} (aliases: {sorted(_ALIASES.keys())})"
        )

    model_cls = _REGISTRY[key]
    model = model_cls(**kwargs)

    if device is not None:
        model = model.to(torch.device(device))

    return model
