from __future__ import annotations

from pathlib import Path


def train_master(config: dict) -> dict[str, Path]:
    """Train the upstream MASTER architecture with the local sequence data adapter."""
    try:
        from stock_manager.models.master_impl import train_master_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MASTER requires PyTorch. Install the project with the torch extra before training."
        ) from exc
    return train_master_model(config)


def train_stockmixer(config: dict) -> dict[str, Path]:
    """Train the upstream StockMixer architecture with the local rolling EOD data adapter."""
    try:
        from stock_manager.models.stockmixer_impl import train_stockmixer_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "StockMixer requires PyTorch. Install the project with the torch extra before training."
        ) from exc
    return train_stockmixer_model(config)

