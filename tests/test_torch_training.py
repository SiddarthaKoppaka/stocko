from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from stock_manager.models.master_impl import train_master_model
from stock_manager.models.stockmixer_impl import _build_stockmixer_arrays, train_stockmixer_model


def _make_synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=36, freq="B")
    tickers = ["AAA", "BBB", "CCC"]
    raw_rows = []
    feature_rows = []
    for ticker_index, ticker in enumerate(tickers):
        close = 100 + ticker_index * 5 + np.cumsum(rng.normal(0.2, 0.5, len(dates)))
        open_ = close + rng.normal(0, 0.2, len(dates))
        high = np.maximum(open_, close) + 0.3
        low = np.minimum(open_, close) - 0.3
        volume = 1_000_000 + rng.integers(0, 50_000, len(dates))
        returns = pd.Series(close).shift(-1) / pd.Series(close) - 1
        for index, date in enumerate(dates):
            raw_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": float(open_[index]),
                    "high": float(high[index]),
                    "low": float(low[index]),
                    "close": float(close[index]),
                    "volume": float(volume[index]),
                }
            )
            row = {
                "date": date,
                "ticker": ticker,
                "LABEL0": float(returns.iloc[index]) if pd.notna(returns.iloc[index]) else np.nan,
            }
            base = close[index] / close[max(index - 1, 0)] - 1
            for feature_index in range(158):
                row[f"alpha_{feature_index:03d}"] = float(base + 0.001 * feature_index + rng.normal(0, 0.01))
            feature_rows.append(row)
    return pd.DataFrame(raw_rows), pd.DataFrame(feature_rows), dates


def test_master_training_smoke(tmp_path: Path) -> None:
    raw_frame, feature_frame, dates = _make_synthetic_frames()
    raw_path = tmp_path / "raw.parquet"
    alpha_path = tmp_path / "alpha.parquet"
    raw_frame.to_parquet(raw_path, index=False)
    feature_frame.to_parquet(alpha_path, index=False)

    outputs = train_master_model(
        {
            "data": {
                "processed_path": str(alpha_path),
                "market_source_path": str(raw_path),
                "label_column": "LABEL0",
            },
            "splits": {
                "train_end": str(dates[18].date()),
                "valid_end": str(dates[26].date()),
                "test_end": str(dates[-2].date()),
            },
            "paths": {
                "model_dir": str(tmp_path / "models"),
                "report_dir": str(tmp_path / "reports"),
            },
            "model": {
                "params": {
                    "lookback_window": 4,
                    "d_model": 32,
                    "t_nhead": 2,
                    "s_nhead": 2,
                    "t_dropout_rate": 0.1,
                    "s_dropout_rate": 0.1,
                    "beta": 2.0,
                    "n_epochs": 1,
                    "lr": 1e-3,
                    "train_stop_loss_thred": 0.0,
                    "seed": 7,
                }
            },
        }
    )

    assert set(outputs) == {"model", "predictions", "metrics", "config", "metadata"}
    predictions = pd.read_parquet(outputs["predictions"])
    assert not predictions.empty
    assert set(predictions.columns) == {"date", "ticker", "prediction", "label"}


def test_master_training_requires_label_column(tmp_path: Path) -> None:
    raw_frame, feature_frame, dates = _make_synthetic_frames()
    raw_path = tmp_path / "raw.parquet"
    alpha_path = tmp_path / "alpha_missing_label.parquet"
    raw_frame.to_parquet(raw_path, index=False)
    feature_frame.drop(columns=["LABEL0"]).to_parquet(alpha_path, index=False)

    with pytest.raises(ValueError, match="includes the configured label column 'LABEL0'"):
        train_master_model(
            {
                "data": {
                    "processed_path": str(alpha_path),
                    "market_source_path": str(raw_path),
                    "label_column": "LABEL0",
                },
                "splits": {
                    "train_end": str(dates[18].date()),
                    "valid_end": str(dates[26].date()),
                    "test_end": str(dates[-2].date()),
                },
                "paths": {
                    "model_dir": str(tmp_path / "models"),
                    "report_dir": str(tmp_path / "reports"),
                },
                "model": {
                    "params": {
                        "lookback_window": 4,
                        "d_model": 32,
                        "t_nhead": 2,
                        "s_nhead": 2,
                        "t_dropout_rate": 0.1,
                        "s_dropout_rate": 0.1,
                        "beta": 2.0,
                        "n_epochs": 1,
                        "lr": 1e-3,
                        "train_stop_loss_thred": 0.0,
                        "seed": 7,
                    }
                },
            }
        )


def test_stockmixer_training_smoke(tmp_path: Path) -> None:
    raw_frame, _, dates = _make_synthetic_frames()
    raw_path = tmp_path / "raw.parquet"
    raw_frame.to_parquet(raw_path, index=False)

    outputs = train_stockmixer_model(
        {
            "data": {
                "processed_path": str(raw_path),
            },
            "splits": {
                "train_end": str(dates[18].date()),
                "valid_end": str(dates[26].date()),
                "test_end": str(dates[-2].date()),
            },
            "paths": {
                "model_dir": str(tmp_path / "models"),
                "report_dir": str(tmp_path / "reports"),
            },
            "model": {
                "params": {
                    "lookback_length": 4,
                    "steps": 1,
                    "epochs": 1,
                    "learning_rate": 1e-3,
                    "alpha": 0.1,
                    "market_hidden_dim": 4,
                    "scale_factor": 3,
                    "seed": 7,
                }
            },
        }
    )

    assert set(outputs) == {"model", "predictions", "metrics", "config", "metadata"}
    predictions = pd.read_parquet(outputs["predictions"])
    assert not predictions.empty
    assert set(predictions.columns) == {"date", "ticker", "prediction", "label"}


def test_stockmixer_array_builder_normalizes_inputs() -> None:
    raw_frame, _, dates = _make_synthetic_frames()
    arrays = _build_stockmixer_arrays(
        raw_frame,
        {
            "splits": {
                "train_end": str(dates[18].date()),
                "valid_end": str(dates[26].date()),
                "test_end": str(dates[-2].date()),
            },
            "model": {
                "params": {
                    "lookback_length": 4,
                    "steps": 1,
                    "normalization_window": 5,
                }
            },
        },
    )

    assert np.isfinite(arrays["eod_data"]).all()
    assert float(np.nanmax(np.abs(arrays["eod_data"]))) < 5.0