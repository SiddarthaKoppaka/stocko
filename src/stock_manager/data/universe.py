from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DEFAULT_SP500_URL = "https://www.slickcharts.com/sp500"
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
GITHUB_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)


def load_universe(config: dict) -> list[str]:
    """Load configured trade universe tickers."""
    source = config.get("source", "inline")
    if source == "inline":
        tickers = config.get("tickers", [])
    elif source == "csv":
        tickers = pd.read_csv(Path(config["path"]))[config.get("ticker_column", "ticker")].tolist()
    elif source == "wikipedia_sp500":
        tickers = _load_sp500_from_wikipedia()
    elif source == "slickcharts_sp500":
        tickers = _load_sp500_from_slickcharts()
    elif source == "sp500_auto":
        tickers = _load_sp500_auto()
    else:
        raise ValueError(f"Unsupported universe source: {source}")
    return normalize_tickers(tickers)


def _load_sp500_auto() -> list[str]:
    """Load S&P 500 tickers with resilient fallbacks for notebook/cloud environments."""
    errors: list[str] = []
    for loader in (
        _load_sp500_from_wikipedia,
        _load_sp500_from_slickcharts,
        _load_sp500_from_github_csv,
    ):
        try:
            tickers = loader()
            if tickers:
                return tickers
            errors.append(f"{loader.__name__}: empty ticker list")
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError("Unable to fetch S&P 500 universe. " + " | ".join(errors))


def _load_sp500_from_wikipedia() -> list[str]:
    """Load S&P 500 symbols from Wikipedia constituents table."""
    response = requests.get(
        WIKIPEDIA_SP500_URL,
        timeout=20,
        headers=_browser_headers(),
    )
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    return table["Symbol"].tolist()


def _load_sp500_from_slickcharts() -> list[str]:
    """Load S&P 500 symbols from Slickcharts with browser-like headers."""
    response = requests.get(
        DEFAULT_SP500_URL,
        timeout=20,
        headers=_browser_headers(),
    )
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text))[0]
    return table["Symbol"].tolist()


def _load_sp500_from_github_csv() -> list[str]:
    """Load S&P 500 symbols from a public GitHub dataset mirror."""
    frame = pd.read_csv(GITHUB_SP500_CSV_URL)
    if "Symbol" not in frame.columns:
        raise ValueError("GitHub constituents CSV does not contain 'Symbol' column")
    return frame["Symbol"].tolist()


def _browser_headers() -> dict[str, str]:
    """Headers that reduce 403 responses from anti-bot edge filters."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }


def normalize_tickers(tickers: list[str]) -> list[str]:
    """Normalize tickers for data providers while preserving stable order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for ticker in tickers:
        value = str(ticker).strip().upper().replace(".", "-")
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized

