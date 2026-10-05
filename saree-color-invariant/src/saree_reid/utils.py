"""Utility helpers: seeding, config loading, I/O."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions (may slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log.info(f"Seeded everything with {seed}")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config with env-var interpolation for ${VAR:default}."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # Simple env-var interpolation: ${VAR:default}
    import re
    def _replace(m: re.Match) -> str:
        var = m.group(1)
        default = m.group(2) if m.group(2) else ""
        return os.environ.get(var, default)

    text = re.sub(r"\$\{(\w+):([^}]*)\}", _replace, text)
    return yaml.safe_load(text)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_device() -> torch.device:
    """Return best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
