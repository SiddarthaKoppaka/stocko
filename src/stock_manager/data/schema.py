from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

OHLCV_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class MissingDataReport:
    row_count: int
    ticker_count: int
    missing_by_column: dict[str, int]
    duplicate_rows: int


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an OHLCV frame with date/ticker keys."""
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")
    result = frame.loc[:, OHLCV_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    result["ticker"] = result["ticker"].astype(str).str.upper()
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["date", "ticker", "close"]].isna().any().any():
        raise ValueError("OHLCV data contains null date, ticker, or close values")
    if (result["volume"] < 0).any():
        raise ValueError("OHLCV data contains negative volume values")
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result


def missing_data_report(frame: pd.DataFrame) -> MissingDataReport:
    """Summarize null and duplicate rows in a market data frame."""
    duplicate_rows = int(frame.duplicated(subset=["date", "ticker"]).sum())
    return MissingDataReport(
        row_count=len(frame),
        ticker_count=int(frame["ticker"].nunique()) if "ticker" in frame.columns else 0,
        missing_by_column={column: int(frame[column].isna().sum()) for column in frame.columns},
        duplicate_rows=duplicate_rows,
    )

