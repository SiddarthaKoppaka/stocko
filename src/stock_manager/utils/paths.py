from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/Stock_manager")


def artifact_root(kind: str) -> Path:
    """Return configured artifact root for data, models, or reports."""
    env_names = {
        "data": "STOCK_MANAGER_DATA_DIR",
        "models": "STOCK_MANAGER_MODEL_DIR",
        "reports": "STOCK_MANAGER_REPORT_DIR",
    }
    if kind not in env_names:
        raise ValueError(f"Unknown artifact root kind: {kind}")
    default = DEFAULT_DRIVE_ROOT / kind
    return Path(os.environ.get(env_names[kind], str(default)))


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result

