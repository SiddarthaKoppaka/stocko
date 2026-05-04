from __future__ import annotations

import pandas as pd
import pytest

from stock_manager.features.labels import add_forward_return_label


def test_forward_return_label_is_grouped_sorted_and_drops_future_nulls():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT", "MSFT"],
            "close": [121.0, 100.0, 110.0, 50.0, 55.0, 66.0],
        }
    )

    result = add_forward_return_label(frame, horizon=2)

    assert list(result["ticker"]) == ["AAPL", "MSFT"]
    assert result.loc[result["ticker"] == "AAPL", "label_2d"].iloc[0] == pytest.approx(0.21)
    assert result.loc[result["ticker"] == "MSFT", "label_2d"].iloc[0] == pytest.approx(0.32)
