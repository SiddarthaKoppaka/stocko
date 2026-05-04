from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_SP500_URL = "https://www.slickcharts.com/sp500"


def load_universe(config: dict) -> list[str]:
    """Load configured trade universe tickers."""
    source = config.get("source", "inline")
    if source == "inline":
        tickers = config.get("tickers", [])
    elif source == "csv":
        tickers = pd.read_csv(Path(config["path"]))[config.get("ticker_column", "ticker")].tolist()
    elif source == "slickcharts_sp500":
        tickers = pd.read_html(DEFAULT_SP500_URL)[0]["Symbol"].tolist()
    else:
        raise ValueError(f"Unsupported universe source: {source}")
    return normalize_tickers(tickers)


def normalize_tickers(tickers: list[str]) -> list[str]:
    """Normalize tickers for data providers while preserving stable order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for ticker in tickers:
        value = str(ticker).strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized

