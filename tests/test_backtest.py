from __future__ import annotations

import pandas as pd
import pytest

from stock_manager.backtest.metrics import max_drawdown, summarize_predictions


def test_summarize_predictions_returns_expected_keys():
    predictions = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "ticker": ["A", "B", "A", "B"],
            "prediction": [0.2, 0.1, 0.1, 0.2],
            "label": [0.03, -0.01, -0.02, 0.04],
        }
    )

    metrics = summarize_predictions(predictions, cost_bps=5)

    assert "rank_ic" in metrics
    assert metrics["cost_bps"] == 5


def test_max_drawdown():
    assert max_drawdown(pd.Series([0.2, -0.15])) == pytest.approx(-0.15)
