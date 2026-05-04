from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from stock_manager.config import require_keys
from stock_manager.features.alpha158 import DEFAULT_LABEL_COLUMN, write_alpha158_frame
from stock_manager.utils.paths import ensure_dir

QLIB_FIELDS = ["open", "high", "low", "close", "volume", "factor"]
INSTRUMENTS_FILE_NAME = "all.txt"
INSTRUMENTS_SEP = "\t"
START_FIELD = "start_datetime"
END_FIELD = "end_datetime"


def build_qlib_dataset(config: dict) -> dict[str, Path]:
    """Convert processed OHLCV data into Qlib's daily `.bin` provider format.

    The output layout mirrors the pieces consumed by `qlib.init`: calendars/day.txt,
    instruments/all.txt, and per-symbol field files under features/<symbol>/*.day.bin.
    """
    require_keys(
        config,
        ["data.processed_path", "qlib.provider_uri"],
        context="qlib config",
    )
    processed_path = Path(config["data"]["processed_path"])
    provider_uri = ensure_dir(config["qlib"]["provider_uri"])
    mode = config.get("qlib", {}).get("mode", "all")
    frame = pd.read_parquet(processed_path)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Processed data missing Qlib fields: {', '.join(sorted(missing))}")

    frame = _normalize_source_frame(frame)
    if mode == "all":
        _reset_provider_uri(provider_uri)
        calendar_list = sorted(frame["date"].unique())
        instrument_rows = _dump_all(frame, provider_uri, calendar_list)
    elif mode == "update":
        instrument_rows, calendar_list = _dump_update(frame, provider_uri)
    else:
        raise ValueError(f"Unsupported qlib dump mode: {mode}")

    alpha158_path = _build_alpha158_artifact(config, provider_uri, frame)
    _save_calendars(provider_uri / "calendars" / "day.txt", calendar_list)
    _save_instruments(provider_uri / "instruments" / INSTRUMENTS_FILE_NAME, instrument_rows)
    return {
        "provider_uri": provider_uri,
        "calendar": provider_uri / "calendars" / "day.txt",
        "instruments": provider_uri / "instruments" / INSTRUMENTS_FILE_NAME,
        "alpha158": alpha158_path,
    }


def _normalize_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
    result["symbol"] = result["ticker"].astype(str).str.upper()
    result["factor"] = 1.0
    return result.loc[:, ["symbol", "date", *QLIB_FIELDS]].sort_values(["symbol", "date"])


def _reset_provider_uri(provider_uri: Path) -> None:
    for name in ("calendars", "instruments", "features"):
        path = provider_uri / name
        if path.exists():
            shutil.rmtree(path)


def _dump_all(frame: pd.DataFrame, provider_uri: Path, calendar_list: list[pd.Timestamp]) -> list[dict[str, str]]:
    instrument_rows: list[dict[str, str]] = []
    features_dir = ensure_dir(provider_uri / "features")
    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        symbol_frame = symbol_frame.drop_duplicates(subset=["date"]).sort_values("date")
        _write_symbol_bins(symbol_frame, features_dir / symbol.lower(), calendar_list, append=False)
        instrument_rows.append(
            {
                "symbol": symbol,
                START_FIELD: _format_datetime(symbol_frame["date"].min()),
                END_FIELD: _format_datetime(symbol_frame["date"].max()),
            }
        )
    return instrument_rows


def _dump_update(
    frame: pd.DataFrame, provider_uri: Path
) -> tuple[list[dict[str, str]], list[pd.Timestamp]]:
    calendar_path = provider_uri / "calendars" / "day.txt"
    instruments_path = provider_uri / "instruments" / INSTRUMENTS_FILE_NAME
    if not calendar_path.exists() or not instruments_path.exists():
        calendar_list = sorted(frame["date"].unique())
        return _dump_all(frame, provider_uri, calendar_list), calendar_list

    old_calendar_list = _read_calendars(calendar_path)
    new_calendar_tail = sorted(date for date in frame["date"].unique() if date > old_calendar_list[-1])
    calendar_list = old_calendar_list + new_calendar_tail

    instrument_rows = _read_instruments(instruments_path)
    instrument_map = {row["symbol"]: row for row in instrument_rows}
    features_dir = ensure_dir(provider_uri / "features")

    for symbol, symbol_frame in frame.groupby("symbol", sort=True):
        symbol_frame = symbol_frame.drop_duplicates(subset=["date"]).sort_values("date")
        if symbol in instrument_map:
            existing_end = pd.Timestamp(instrument_map[symbol][END_FIELD])
            new_rows = symbol_frame[symbol_frame["date"] > existing_end]
            if new_rows.empty:
                continue
            _write_symbol_bins(new_rows, features_dir / symbol.lower(), calendar_list, append=True)
            instrument_map[symbol][END_FIELD] = _format_datetime(new_rows["date"].max())
        else:
            _write_symbol_bins(symbol_frame, features_dir / symbol.lower(), calendar_list, append=False)
            instrument_map[symbol] = {
                "symbol": symbol,
                START_FIELD: _format_datetime(symbol_frame["date"].min()),
                END_FIELD: _format_datetime(symbol_frame["date"].max()),
            }
    return sorted(instrument_map.values(), key=lambda row: row["symbol"]), calendar_list


def _write_symbol_bins(
    frame: pd.DataFrame,
    features_dir: Path,
    calendar_list: list[pd.Timestamp],
    *,
    append: bool,
) -> None:
    if frame.empty:
        return
    features_dir.mkdir(parents=True, exist_ok=True)
    aligned = _align_to_calendar(frame, calendar_list)
    if aligned.empty:
        return

    date_index = float(calendar_list.index(aligned.index.min()))
    for field in QLIB_FIELDS:
        bin_path = features_dir / f"{field}.day.bin"
        values = np.asarray(aligned[field], dtype="<f4")
        if append and bin_path.exists():
            with bin_path.open("ab") as handle:
                values.tofile(handle)
        else:
            payload = np.hstack([np.array([date_index], dtype="<f4"), values])
            payload.astype("<f4").tofile(bin_path)


def _align_to_calendar(frame: pd.DataFrame, calendar_list: list[pd.Timestamp]) -> pd.DataFrame:
    calendars = pd.DataFrame({"date": calendar_list})
    calendars = calendars[
        (calendars["date"] >= frame["date"].min()) & (calendars["date"] <= frame["date"].max())
    ]
    if calendars.empty:
        return pd.DataFrame()
    base = frame.set_index("date").sort_index()
    return base.reindex(calendars["date"])


def _save_calendars(path: Path, calendar_list: list[pd.Timestamp]) -> None:
    ensure_dir(path.parent)
    values = [_format_datetime(date) for date in calendar_list]
    np.savetxt(path, values, fmt="%s", encoding="utf-8")


def _save_instruments(path: Path, instrument_rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    lines = [
        INSTRUMENTS_SEP.join([row["symbol"], row[START_FIELD], row[END_FIELD]])
        for row in sorted(instrument_rows, key=lambda row: row["symbol"])
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_calendars(path: Path) -> list[pd.Timestamp]:
    return sorted(map(pd.Timestamp, pd.read_csv(path, header=None).iloc[:, 0].tolist()))


def _read_instruments(path: Path) -> list[dict[str, str]]:
    frame = pd.read_csv(
        path,
        sep=INSTRUMENTS_SEP,
        names=["symbol", START_FIELD, END_FIELD],
        dtype=str,
    )
    return frame.to_dict(orient="records")


def _format_datetime(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _build_alpha158_artifact(config: dict, provider_uri: Path, frame: pd.DataFrame) -> Path:
    alpha_cfg = config.get("qlib", {}).get("alpha158", {})
    output_path = Path(
        alpha_cfg.get(
            "output_path",
            str(provider_uri / "alpha158.parquet"),
        )
    )
    label_column = alpha_cfg.get("label_column", DEFAULT_LABEL_COLUMN)
    vwap_mode = alpha_cfg.get("vwap_mode", "typical_price")
    source_frame = frame.rename(columns={"symbol": "ticker"}).copy()
    return write_alpha158_frame(
        source_frame,
        output_path,
        label_column=label_column,
        vwap_mode=vwap_mode,
    )

