# src/utils/seed.py
from __future__ import annotations
import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, enables deterministic PyTorch behavior when possible.
                       This can reduce performance and may raise errors for some ops.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Deterministic settings (best-effort)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # This enforces determinism for ops that support it (may error otherwise).
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions or unsupported builds can throw here.
            pass
