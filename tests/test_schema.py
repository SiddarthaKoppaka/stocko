from __future__ import annotations

import pandas as pd
import pytest

from stock_manager.data.schema import missing_data_report, validate_ohlcv


def test_validate_ohlcv_sorts_and_normalizes_tickers():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01"],
            "ticker": ["aapl", "AAPL"],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [2, 1],
            "volume": [100, 100],
        }
    )

    result = validate_ohlcv(frame)

    assert list(result["date"].dt.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]
    assert result["ticker"].tolist() == ["AAPL", "AAPL"]


def test_validate_ohlcv_requires_close():
    with pytest.raises(ValueError, match="close"):
        validate_ohlcv(pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAPL"]}))


def test_missing_data_report_counts_duplicates():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "ticker": ["AAPL", "AAPL"],
            "close": [1.0, None],
        }
    )

    report = missing_data_report(frame)

    assert report.duplicate_rows == 1
    assert report.missing_by_column["close"] == 1

