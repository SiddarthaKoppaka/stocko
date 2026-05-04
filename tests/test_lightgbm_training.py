from __future__ import annotations

import pandas as pd
import pytest

from stock_manager.models import train_from_config


def test_lightgbm_tiny_training_smoke(tmp_path):
    try:
        import lightgbm  # noqa: F401
    except OSError as exc:
        pytest.skip(f"LightGBM native library is unavailable: {exc}")

    processed = tmp_path / "processed.parquet"
    rows = []
    for date, base in [
        ("2020-01-01", 1.0),
        ("2020-01-02", 1.1),
        ("2021-01-01", 1.2),
        ("2022-01-01", 1.3),
    ]:
        for ticker_offset, ticker in enumerate(["AAPL", "MSFT", "NVDA"]):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": base + ticker_offset,
                    "high": base + ticker_offset + 0.1,
                    "low": base + ticker_offset - 0.1,
                    "close": base + ticker_offset + 0.05,
                    "volume": 1000 + ticker_offset,
                    "label_5d": 0.01 * (ticker_offset + 1),
                }
            )
    pd.DataFrame(rows).to_parquet(processed, index=False)

    outputs = train_from_config(
        {
            "model": {
                "type": "lightgbm_alpha158",
                "params": {
                    "n_estimators": 2,
                    "min_data_in_leaf": 1,
                    "min_data_in_bin": 1,
                    "verbose": -1,
                    "random_state": 42,
                },
            },
            "data": {"processed_path": str(processed), "label_column": "label_5d"},
            "splits": {
                "train_end": "2020-12-31",
                "valid_end": "2021-12-31",
                "test_end": "2022-12-31",
            },
            "paths": {
                "model_dir": str(tmp_path / "models"),
                "report_dir": str(tmp_path / "reports"),
            },
        }
    )

    assert outputs["model"].exists()
    assert outputs["predictions"].exists()
    assert outputs["metrics"].exists()
