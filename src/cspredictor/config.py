"""Loads config.yaml once and exposes it as a plain dict."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a config-declared path relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
