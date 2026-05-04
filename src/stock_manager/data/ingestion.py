from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_manager.config import require_keys
from stock_manager.data.providers import (
    download_alpaca_bars,
    download_fred_series,
    download_yfinance_bars,
)
from stock_manager.data.schema import missing_data_report, validate_ohlcv
from stock_manager.data.universe import load_universe
from stock_manager.features.labels import add_forward_return_label
from stock_manager.utils.paths import ensure_dir


def ingest_market_data(config: dict) -> dict[str, Path]:
    """Run configured market data ingestion and write raw/processed artifacts."""
    require_keys(
        config,
        ["universe", "data.start", "data.end", "paths.raw_dir", "paths.processed_dir"],
        context="ingestion config",
    )
    tickers = load_universe(config["universe"])
    start = config["data"]["start"]
    end = config["data"]["end"]
    providers = config["data"].get("providers", ["alpaca", "yfinance"])

    last_error: Exception | None = None
    frame: pd.DataFrame | None = None
    selected_provider = ""
    for provider in providers:
        try:
            if provider == "alpaca":
                frame = download_alpaca_bars(tickers, start, end)
            elif provider == "yfinance":
                frame = download_yfinance_bars(tickers, start, end)
            else:
                raise ValueError(f"Unsupported data provider: {provider}")
            selected_provider = provider
            break
        except Exception as exc:  # pragma: no cover - exercised with real providers.
            last_error = exc
    if frame is None:
        raise RuntimeError(f"All market data providers failed. Last error: {last_error}")

    frame = validate_ohlcv(frame)
    raw_dir = ensure_dir(config["paths"]["raw_dir"])
    processed_dir = ensure_dir(config["paths"]["processed_dir"])
    raw_path = raw_dir / "ohlcv.parquet"
    processed_path = processed_dir / "ohlcv_labeled.parquet"
    report_path = processed_dir / "missing_data_report.json"

    frame.to_parquet(raw_path, index=False)
    labeled = add_forward_return_label(frame, horizon=int(config["data"].get("label_horizon", 5)))
    labeled.to_parquet(processed_path, index=False)
    report = missing_data_report(frame).__dict__
    report["provider"] = selected_provider
    report["tickers_requested"] = len(tickers)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    outputs = {"raw": raw_path, "processed": processed_path, "report": report_path}
    fred_config = config.get("fred", {})
    if fred_config.get("enabled"):
        macro = download_fred_series(fred_config.get("series", []), start, end)
        macro_path = raw_dir / "fred_macro.parquet"
        macro.to_parquet(macro_path, index=False)
        outputs["fred"] = macro_path
    return outputs

