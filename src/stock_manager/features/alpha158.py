from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-12
DEFAULT_ROLLING_WINDOWS = (5, 10, 20, 30, 60)
DEFAULT_LABEL_COLUMN = "LABEL0"


def build_alpha158_frame(
    frame: pd.DataFrame,
    *,
    label_column: str = DEFAULT_LABEL_COLUMN,
    vwap_mode: str = "typical_price",
) -> pd.DataFrame:
    """Build the default Qlib Alpha158 factor set from daily bars.

    This mirrors the default Alpha158 handler surface closely: 9 kbar factors,
    4 price-ratio factors (OPEN0/HIGH0/LOW0/VWAP0), and 29 rolling operator
    families across windows [5, 10, 20, 30, 60] for 158 factors in total.

    The upstream default expects a VWAP field. Public daily providers in this
    repository do not currently supply a true daily VWAP, so the default
    runnable path uses the explicit `typical_price` proxy. Set
    `vwap_mode="require"` to fail instead.
    """
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required Alpha158 columns: {', '.join(sorted(missing))}")

    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None)
    normalized = normalized.sort_values(["ticker", "date"]).reset_index(drop=True)
    normalized["vwap"] = _resolve_vwap(normalized, vwap_mode=vwap_mode)

    per_ticker = []
    for ticker, ticker_frame in normalized.groupby("ticker", sort=True):
        features = _build_alpha158_for_ticker(ticker_frame, label_column=label_column)
        features.insert(0, "ticker", ticker)
        features.insert(0, "date", ticker_frame["date"].to_numpy())
        per_ticker.append(features)

    return pd.concat(per_ticker, ignore_index=True)


def write_alpha158_frame(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    label_column: str = DEFAULT_LABEL_COLUMN,
    vwap_mode: str = "typical_price",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    alpha158 = build_alpha158_frame(frame, label_column=label_column, vwap_mode=vwap_mode)
    alpha158.to_parquet(output, index=False)
    return output


def _resolve_vwap(frame: pd.DataFrame, *, vwap_mode: str) -> pd.Series:
    if "vwap" in frame.columns and frame["vwap"].notna().any():
        return pd.to_numeric(frame["vwap"], errors="coerce")
    if vwap_mode == "typical_price":
        return (frame["open"] + frame["high"] + frame["low"] + frame["close"]) / 4.0
    if vwap_mode == "close":
        return frame["close"]
    if vwap_mode == "require":
        raise ValueError(
            "Alpha158 requires a VWAP field, but the input frame only contains OHLCV bars. "
            "Use a provider with VWAP or set qlib.alpha158.vwap_mode to an explicit proxy."
        )
    raise ValueError(f"Unsupported Alpha158 VWAP mode: {vwap_mode}")


def _build_alpha158_for_ticker(frame: pd.DataFrame, *, label_column: str) -> pd.DataFrame:
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    vwap = frame["vwap"].astype(float)

    features: dict[str, pd.Series] = {}

    bar_range = high - low
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low

    features["KMID"] = (close - open_) / (open_ + EPSILON)
    features["KLEN"] = bar_range / (open_ + EPSILON)
    features["KMID2"] = (close - open_) / (bar_range + EPSILON)
    features["KUP"] = upper_shadow / (open_ + EPSILON)
    features["KUP2"] = upper_shadow / (bar_range + EPSILON)
    features["KLOW"] = lower_shadow / (open_ + EPSILON)
    features["KLOW2"] = lower_shadow / (bar_range + EPSILON)
    features["KSFT"] = (2 * close - high - low) / (open_ + EPSILON)
    features["KSFT2"] = (2 * close - high - low) / (bar_range + EPSILON)

    features["OPEN0"] = open_ / (close + EPSILON)
    features["HIGH0"] = high / (close + EPSILON)
    features["LOW0"] = low / (close + EPSILON)
    features["VWAP0"] = vwap / (close + EPSILON)

    close_delta = close - close.shift(1)
    close_ratio = close / (close.shift(1) + EPSILON)
    volume_delta = volume - volume.shift(1)
    weighted_return = close_ratio.sub(1.0).abs() * volume

    for window in DEFAULT_ROLLING_WINDOWS:
        features[f"ROC{window}"] = close.shift(window) / (close + EPSILON)
        features[f"MA{window}"] = close.rolling(window).mean() / (close + EPSILON)
        features[f"STD{window}"] = close.rolling(window).std() / (close + EPSILON)
        features[f"BETA{window}"] = _rolling_regression(close, window, "slope") / (close + EPSILON)
        features[f"RSQR{window}"] = _rolling_regression(close, window, "rsquare")
        features[f"RESI{window}"] = _rolling_regression(close, window, "residual") / (close + EPSILON)
        features[f"MAX{window}"] = high.rolling(window).max() / (close + EPSILON)
        features[f"MIN{window}"] = low.rolling(window).min() / (close + EPSILON)
        features[f"QTLU{window}"] = close.rolling(window).quantile(0.8) / (close + EPSILON)
        features[f"QTLD{window}"] = close.rolling(window).quantile(0.2) / (close + EPSILON)
        features[f"RANK{window}"] = close.rolling(window).apply(_current_percentile_rank, raw=False)
        features[f"RSV{window}"] = (
            (close - low.rolling(window).min())
            / (high.rolling(window).max() - low.rolling(window).min() + EPSILON)
        )
        features[f"IMAX{window}"] = high.rolling(window).apply(_days_since_recent_max, raw=True) / window
        features[f"IMIN{window}"] = low.rolling(window).apply(_days_since_recent_min, raw=True) / window
        features[f"IMXD{window}"] = high.rolling(window).apply(_days_since_recent_max, raw=True) / window - \
            low.rolling(window).apply(_days_since_recent_min, raw=True) / window
        features[f"CORR{window}"] = close.rolling(window).corr(np.log(volume + 1.0))
        features[f"CORD{window}"] = close_ratio.rolling(window).corr(
            np.log(volume / (volume.shift(1) + EPSILON) + 1.0)
        )
        features[f"CNTP{window}"] = close_delta.gt(0).astype(float).rolling(window).mean()
        features[f"CNTN{window}"] = close_delta.lt(0).astype(float).rolling(window).mean()
        features[f"CNTD{window}"] = features[f"CNTP{window}"] - features[f"CNTN{window}"]

        abs_close_delta_sum = close_delta.abs().rolling(window).sum()
        features[f"SUMP{window}"] = close_delta.clip(lower=0).rolling(window).sum() / (abs_close_delta_sum + EPSILON)
        features[f"SUMN{window}"] = (-close_delta.clip(upper=0)).rolling(window).sum() / (abs_close_delta_sum + EPSILON)
        features[f"SUMD{window}"] = (
            close_delta.clip(lower=0).rolling(window).sum()
            - (-close_delta.clip(upper=0)).rolling(window).sum()
        ) / (abs_close_delta_sum + EPSILON)

        features[f"VMA{window}"] = volume.rolling(window).mean() / (volume + EPSILON)
        features[f"VSTD{window}"] = volume.rolling(window).std() / (volume + EPSILON)
        features[f"WVMA{window}"] = weighted_return.rolling(window).std() / (
            weighted_return.rolling(window).mean() + EPSILON
        )

        abs_volume_delta_sum = volume_delta.abs().rolling(window).sum()
        features[f"VSUMP{window}"] = volume_delta.clip(lower=0).rolling(window).sum() / (
            abs_volume_delta_sum + EPSILON
        )
        features[f"VSUMN{window}"] = (-volume_delta.clip(upper=0)).rolling(window).sum() / (
            abs_volume_delta_sum + EPSILON
        )
        features[f"VSUMD{window}"] = (
            volume_delta.clip(lower=0).rolling(window).sum()
            - (-volume_delta.clip(upper=0)).rolling(window).sum()
        ) / (abs_volume_delta_sum + EPSILON)

    features[label_column] = close.shift(-2) / (close.shift(-1) + EPSILON) - 1.0
    return pd.DataFrame(features)


def _rolling_regression(series: pd.Series, window: int, kind: str) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denominator = np.sum((x - x_mean) ** 2)

    def _apply(values: np.ndarray) -> float:
        y = values.astype(float)
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
        intercept = y_mean - slope * x_mean
        fitted = intercept + slope * x
        residual = y[-1] - fitted[-1]
        total = np.sum((y - y_mean) ** 2)
        explained = np.sum((fitted - y_mean) ** 2)
        rsquare = explained / total if total > EPSILON else 0.0
        if kind == "slope":
            return float(slope)
        if kind == "rsquare":
            return float(rsquare)
        if kind == "residual":
            return float(residual)
        raise ValueError(f"Unsupported regression output: {kind}")

    return series.rolling(window).apply(_apply, raw=True)


def _current_percentile_rank(values: pd.Series) -> float:
    return float(values.rank(pct=True).iloc[-1])


def _days_since_recent_max(values: np.ndarray) -> float:
    return float(np.argmax(values[::-1]))


def _days_since_recent_min(values: np.ndarray) -> float:
    return float(np.argmin(values[::-1]))