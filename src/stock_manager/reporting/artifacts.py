from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stock_manager.utils.paths import ensure_dir


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact with stable indentation."""
    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return output


def write_predictions(path: str | Path, predictions: pd.DataFrame) -> Path:
    """Write predictions as parquet."""
    output = Path(path)
    ensure_dir(output.parent)
    predictions.to_parquet(output, index=False)
    return output


def run_metadata(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Build basic run metadata for reproducibility."""
    return {
        "model": model_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }

