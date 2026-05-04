from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from stock_manager.config import require_keys
from stock_manager.reporting.artifacts import run_metadata, write_json, write_predictions
from stock_manager.utils.paths import ensure_dir


def train_lightgbm_alpha158(config: dict) -> dict[str, Path]:
    """Train a LightGBM baseline from processed labeled data.

    This is the local baseline path. A full Qlib Alpha158 workflow can be wired through this
    adapter while preserving the same artifact contract.
    """
    require_keys(
        config,
        ["data.processed_path", "splits.train_end", "splits.valid_end", "paths.model_dir"],
        context="lightgbm config",
    )
    try:
        import lightgbm as lgb
    except OSError as exc:
        raise RuntimeError(
            "LightGBM could not load its native library. On macOS, install OpenMP with "
            "`brew install libomp`, then rerun training."
        ) from exc

    frame = _load_training_frame(config)
    label = config.get("data", {}).get("label_column", "label_5d")
    feature_columns = _feature_columns(frame, label)
    train, valid, test = _chronological_split(frame, config["splits"])

    model = lgb.LGBMRegressor(**config.get("model", {}).get("params", {}))
    model.fit(
        train[feature_columns],
        train[label],
        eval_set=[(valid[feature_columns], valid[label])],
        eval_metric="l2",
    )
    predictions = _prediction_frame(test, model.predict(test[feature_columns]), label)
    metrics = _prediction_metrics(predictions)
    return _write_training_outputs("lightgbm_alpha158", config, model, predictions, metrics)


def _load_training_frame(config: dict) -> pd.DataFrame:
    frame = pd.read_parquet(Path(config["data"]["processed_path"]))
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def _feature_columns(frame: pd.DataFrame, label: str) -> list[str]:
    excluded = {"date", "ticker", label}
    features = [column for column in frame.columns if column not in excluded]
    numeric_features = [
        column for column in features if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_features:
        raise ValueError("No numeric feature columns available for training")
    return numeric_features


def _chronological_split(
    frame: pd.DataFrame,
    splits: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(splits["train_end"])
    valid_end = pd.Timestamp(splits["valid_end"])
    test_end = pd.Timestamp(splits.get("test_end", frame["date"].max()))
    train = frame[frame["date"] <= train_end]
    valid = frame[(frame["date"] > train_end) & (frame["date"] <= valid_end)]
    test = frame[(frame["date"] > valid_end) & (frame["date"] <= test_end)]
    if train.empty or valid.empty or test.empty:
        raise ValueError("Chronological split produced an empty train, valid, or test set")
    return train, valid, test


def _prediction_frame(test: pd.DataFrame, prediction: np.ndarray, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": test["date"].values,
            "ticker": test["ticker"].values,
            "prediction": prediction,
            "label": test[label].values,
        }
    )


def _prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    corr = predictions[["prediction", "label"]].corr(method="spearman").iloc[0, 1]
    mse = float(np.mean((predictions["prediction"] - predictions["label"]) ** 2))
    return {"rank_ic": float(corr) if not np.isnan(corr) else 0.0, "mse": mse}


def _write_training_outputs(
    model_name: str,
    config: dict,
    model: object,
    predictions: pd.DataFrame,
    metrics: dict[str, float],
) -> dict[str, Path]:
    model_dir = ensure_dir(Path(config["paths"]["model_dir"]) / model_name)
    report_dir = ensure_dir(Path(config["paths"].get("report_dir", model_dir)))
    model_path = model_dir / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    prediction_path = write_predictions(
        report_dir / f"{model_name}_predictions.parquet",
        predictions,
    )
    metrics_path = write_json(report_dir / f"{model_name}_metrics.json", metrics)
    config_path = write_json(report_dir / f"{model_name}_config_snapshot.json", config)
    metadata_path = write_json(
        report_dir / f"{model_name}_run_metadata.json",
        run_metadata(model_name, config),
    )
    return {
        "model": model_path,
        "predictions": prediction_path,
        "metrics": metrics_path,
        "config": config_path,
        "metadata": metadata_path,
    }
