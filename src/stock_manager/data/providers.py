from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import date

import pandas as pd

from stock_manager.data.schema import validate_ohlcv


def download_yfinance_bars(
    tickers: Iterable[str],
    start: str | date,
    end: str | date,
) -> pd.DataFrame:
    """Download daily OHLCV bars from yfinance.

    Compatible with yfinance >=0.2 which returns MultiIndex columns when
    downloading a single ticker with group_by='ticker'.  We download one
    ticker at a time and squeeze any 2-D single-column Series to 1-D.
    """
    import yfinance as yf

    def _squeeze(series_or_frame) -> pd.Series:
        """Flatten a 1-column DataFrame or MultiIndex Series to a plain Series."""
        if isinstance(series_or_frame, pd.DataFrame):
            return series_or_frame.iloc[:, 0]
        # MultiIndex Series — drop the ticker level
        if isinstance(series_or_frame.index, pd.MultiIndex):
            return series_or_frame.droplevel(0)
        return series_or_frame

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        data = yf.download(
            ticker,
            start=str(start),
            end=str(end),
            auto_adjust=False,
            progress=False,
            multi_level_index=False,  # yfinance >=0.2.48: flat columns for single ticker
        )
        if data.empty:
            continue
        # Flatten MultiIndex columns if present (yfinance < 0.2.48 or group_by default)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[1] == ticker else "_".join(col).strip("_")
                            for col in data.columns]
        data = data.reset_index()
        # Column name normalisation — yfinance uses "Adj Close" or "Close"
        close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
        frames.append(
            pd.DataFrame(
                {
                    "date":   _squeeze(data["Date"]),
                    "ticker": ticker,
                    "open":   _squeeze(data["Open"]),
                    "high":   _squeeze(data["High"]),
                    "low":    _squeeze(data["Low"]),
                    "close":  _squeeze(data[close_col]),
                    "volume": _squeeze(data["Volume"]),
                }
            )
        )
    if not frames:
        raise RuntimeError("yfinance returned no data")
    return validate_ohlcv(pd.concat(frames, ignore_index=True))


def download_alpaca_bars(
    tickers: Iterable[str],
    start: str | date,
    end: str | date,
) -> pd.DataFrame:
    """Download daily OHLCV bars from Alpaca using environment credentials."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required for Alpaca data")

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=list(tickers),
        timeframe=TimeFrame.Day,
        start=pd.Timestamp(start).to_pydatetime(),
        end=pd.Timestamp(end).to_pydatetime(),
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise RuntimeError("Alpaca returned no data")
    bars = bars.reset_index()
    bars = bars.rename(columns={"symbol": "ticker", "timestamp": "date"})
    return validate_ohlcv(bars[["date", "ticker", "open", "high", "low", "close", "volume"]])


def download_fred_series(
    series_ids: Iterable[str],
    start: str | date,
    end: str | date,
) -> pd.DataFrame:
    """Download optional FRED macro series; requires FRED_API_KEY."""
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for FRED macro data")
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    frames = []
    for series_id in series_ids:
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        frames.append(series.rename(series_id))
    if not frames:
        return pd.DataFrame(columns=["date"])
    return pd.concat(frames, axis=1).reset_index(names="date")
