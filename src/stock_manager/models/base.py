from __future__ import annotations

from pathlib import Path
from typing import Any

from stock_manager.config import require_keys
from stock_manager.models.lightgbm_alpha158 import train_lightgbm_alpha158
from stock_manager.models.torch_adapters import train_master, train_stockmixer


def train_from_config(config: dict[str, Any]) -> dict[str, Path]:
    """Dispatch to the configured model trainer."""
    require_keys(config, ["model.type"], context="model config")
    model_type = config["model"]["type"]
    if model_type == "lightgbm_alpha158":
        return train_lightgbm_alpha158(config)
    if model_type == "master":
        return train_master(config)
    if model_type == "stockmixer":
        return train_stockmixer(config)
    raise ValueError(f"Unsupported model type: {model_type}")

