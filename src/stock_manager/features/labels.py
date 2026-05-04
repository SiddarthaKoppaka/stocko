from __future__ import annotations

import pandas as pd


def add_forward_return_label(
    frame: pd.DataFrame,
    *,
    horizon: int = 5,
    price_column: str = "close",
    label_column: str | None = None,
) -> pd.DataFrame:
    """Add leakage-safe forward return labels grouped by ticker.

    Features at time t must only use information available at or before t. This function uses
    future prices only in the target column and drops rows whose future price is unavailable.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {"date", "ticker", price_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required label columns: {', '.join(sorted(missing))}")
    label_name = label_column or f"label_{horizon}d"
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    future_price = result.groupby("ticker", group_keys=False)[price_column].shift(-horizon)
    result[label_name] = future_price / result[price_column] - 1
    return result.dropna(subset=[label_name]).reset_index(drop=True)

