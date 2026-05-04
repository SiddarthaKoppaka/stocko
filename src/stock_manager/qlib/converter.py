from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_manager.config import require_keys
from stock_manager.utils.paths import ensure_dir

QLIB_FIELDS = ["open", "high", "low", "close", "volume", "factor"]


def build_qlib_dataset(config: dict) -> dict[str, Path]:
    """Convert processed OHLCV data into a simple Qlib-compatible directory layout.

    Qlib's binary format is specialized; this v1 converter writes calendar, instrument metadata,
    and per-instrument field files with the expected field names so the layout is explicit and
    easy to replace with Qlib's official dump_bin utilities later.
    """
    require_keys(
        config,
        ["data.processed_path", "qlib.provider_uri"],
        context="qlib config",
    )
    processed_path = Path(config["data"]["processed_path"])
    provider_uri = ensure_dir(config["qlib"]["provider_uri"])
    frame = pd.read_parquet(processed_path)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed data missing Qlib fields: {', '.join(sorted(missing))}")

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame["factor"] = 1.0
    calendars_dir = ensure_dir(provider_uri / "calendars")
    instruments_dir = ensure_dir(provider_uri / "instruments")
    features_dir = ensure_dir(provider_uri / "features")

    dates = sorted(frame["date"].unique())
    (calendars_dir / "day.txt").write_text("\n".join(dates) + "\n", encoding="utf-8")

    instruments = []
    for ticker, ticker_frame in frame.groupby("ticker", sort=True):
        ticker_dates = sorted(ticker_frame["date"].unique())
        instruments.append(f"{ticker}\t{ticker_dates[0]}\t{ticker_dates[-1]}")
        ticker_dir = ensure_dir(features_dir / ticker.lower())
        ticker_frame = ticker_frame.sort_values("date")
        for field in QLIB_FIELDS:
            output = ticker_dir / f"{field}.csv"
            ticker_frame[["date", field]].to_csv(output, index=False, header=False)
    (instruments_dir / "all.txt").write_text("\n".join(instruments) + "\n", encoding="utf-8")
    return {
        "provider_uri": provider_uri,
        "calendar": calendars_dir / "day.txt",
        "instruments": instruments_dir / "all.txt",
    }

