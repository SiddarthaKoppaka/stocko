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
    """Download daily OHLCV bars from yfinance."""
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        data = yf.download(
            ticker,
            start=str(start),
            end=str(end),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if data.empty:
            continue
        data = data.reset_index()
        frames.append(
            pd.DataFrame(
                {
                    "date": data["Date"],
                    "ticker": ticker,
                    "open": data["Open"],
                    "high": data["High"],
                    "low": data["Low"],
                    "close": data.get("Adj Close", data["Close"]),
                    "volume": data["Volume"],
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
