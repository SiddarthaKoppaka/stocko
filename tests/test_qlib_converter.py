from __future__ import annotations

import pandas as pd
import pytest

from stock_manager.qlib import build_qlib_dataset


def test_build_qlib_dataset_writes_expected_layout(tmp_path):
    processed = tmp_path / "processed.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [100, 120],
            "label_5d": [0.1, 0.2],
        }
    ).to_parquet(processed, index=False)

    outputs = build_qlib_dataset(
        {
            "data": {"processed_path": str(processed)},
            "qlib": {"provider_uri": str(tmp_path / "qlib")},
        }
    )

    assert outputs["calendar"].exists()
    assert outputs["instruments"].exists()
    assert outputs["alpha158"].exists()
    assert (tmp_path / "qlib" / "features" / "aapl" / "factor.day.bin").exists()
    assert (tmp_path / "qlib" / "features" / "aapl" / "close.day.bin").stat().st_size > 0

    alpha158 = pd.read_parquet(outputs["alpha158"])
    assert {"date", "ticker", "LABEL0", "KMID", "ROC5", "VSUMD60"}.issubset(alpha158.columns)
    feature_columns = [col for col in alpha158.columns if col not in {"date", "ticker", "LABEL0"}]
    assert len(feature_columns) == 158


def test_build_qlib_dataset_uses_five_day_label_and_daily_zscore(tmp_path):
    processed = tmp_path / "processed_long.parquet"
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    rows = []
    close_map = {
        "AAPL": [100, 101, 102, 103, 104, 110, 111, 112],
        "MSFT": [100, 100, 100, 100, 100, 90, 89, 88],
    }
    for ticker, closes in close_map.items():
        for index, date in enumerate(dates):
            close = float(closes[index])
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + index * 1000,
                    "label_5d": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(processed, index=False)

    outputs = build_qlib_dataset(
        {
            "data": {"processed_path": str(processed), "label_horizon": 5},
            "qlib": {
                "provider_uri": str(tmp_path / "qlib"),
                "alpha158": {"clip_quantiles": None},
            },
        }
    )

    alpha158 = pd.read_parquet(outputs["alpha158"]).dropna(subset=["LABEL0"])
    first_date = alpha158["date"].min()
    first_cross_section = alpha158[alpha158["date"] == first_date].sort_values("ticker")

    assert list(first_cross_section["ticker"]) == ["AAPL", "MSFT"]
    assert first_cross_section["LABEL0"].mean() == pytest.approx(0.0, abs=1e-9)
    assert first_cross_section["LABEL0"].std(ddof=0) == pytest.approx(1.0, rel=1e-6)
    assert first_cross_section.loc[first_cross_section["ticker"] == "AAPL", "LABEL0"].iloc[0] == pytest.approx(1.0)
    assert first_cross_section.loc[first_cross_section["ticker"] == "MSFT", "LABEL0"].iloc[0] == pytest.approx(-1.0)
