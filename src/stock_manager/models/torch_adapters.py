from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from stock_manager.models.lightgbm_alpha158 import (
    _chronological_split,
    _feature_columns,
    _load_training_frame,
    _prediction_frame,
    _prediction_metrics,
    _write_training_outputs,
)


class AdapterBackedRegressor(Ridge):
    """Small deterministic fallback model for adapter smoke tests and artifact contracts."""

    def __init__(self, adapter_name: str, alpha: float = 1.0):
        super().__init__(alpha=alpha)
        self.adapter_name = adapter_name


def train_master(config: dict) -> dict[str, Path]:
    """Train the MASTER adapter with the shared artifact contract.

    TODO: replace the fallback regressor with the official SJTU-DMTai/MASTER architecture while
    preserving this function's inputs and outputs.
    """
    return _train_torch_adapter(config, "master")


def train_stockmixer(config: dict) -> dict[str, Path]:
    """Train the StockMixer adapter with the shared artifact contract.

    TODO: replace the fallback regressor with the official SJTU-DMTai/StockMixer architecture while
    preserving this function's inputs and outputs.
    """
    return _train_torch_adapter(config, "stockmixer")


def _train_torch_adapter(config: dict, model_name: str) -> dict[str, Path]:
    frame = _load_training_frame(config)
    label = config.get("data", {}).get("label_column", "label_5d")
    feature_columns = _feature_columns(frame, label)
    train, _valid, test = _chronological_split(frame, config["splits"])
    alpha = float(config.get("model", {}).get("params", {}).get("alpha", 1.0))
    model = AdapterBackedRegressor(adapter_name=model_name, alpha=alpha)
    model.fit(train[feature_columns], train[label])
    prediction = np.asarray(model.predict(test[feature_columns]))
    predictions = _prediction_frame(test, prediction, label)
    metrics = _prediction_metrics(predictions)
    return _write_training_outputs(model_name, config, model, predictions, metrics)

