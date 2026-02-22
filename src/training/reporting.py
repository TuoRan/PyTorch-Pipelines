# src/training/reporting.py
from __future__ import annotations
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import torch


def make_run_dir(root: str, run_name: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(root, f"{run_name}_{ts}")
    os.makedirs(path, exist_ok=False)
    return path


def _to_jsonable(x: Any) -> Any:
    # Minimal conversion for torch device / tensors etc.
    if hasattr(x, "type") and hasattr(x, "index"):  # torch.device-ish
        return str(x)
    if is_dataclass(x) and not isinstance(x, type):
        return asdict(x)
    return x


def save_json(path: str, data: Dict[str, Any]) -> None:
    clean = {k: _to_jsonable(v) for k, v in data.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, sort_keys=True)


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
