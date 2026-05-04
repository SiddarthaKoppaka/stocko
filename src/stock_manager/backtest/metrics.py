from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(predictions: pd.DataFrame) -> float:
    """Mean daily Spearman correlation between predictions and forward returns."""
    values = []
    for _, day in predictions.groupby("date"):
        if day["prediction"].nunique() < 2 or day["label"].nunique() < 2:
            continue
        values.append(day[["prediction", "label"]].corr(method="spearman").iloc[0, 1])
    return float(np.nanmean(values)) if values else 0.0


def icir(daily_ic: pd.Series) -> float:
    """Information coefficient information ratio."""
    if daily_ic.empty or daily_ic.std(ddof=0) == 0:
        return 0.0
    return float(daily_ic.mean() / daily_ic.std(ddof=0))


def long_short_returns(predictions: pd.DataFrame, *, top_quantile: float = 0.2) -> pd.Series:
    """Compute equal-weight long-short returns from prediction ranks."""
    returns = {}
    for date, day in predictions.groupby("date"):
        day = day.sort_values("prediction")
        bucket_size = max(1, int(len(day) * top_quantile))
        short = day.head(bucket_size)["label"].mean()
        long = day.tail(bucket_size)["label"].mean()
        returns[pd.Timestamp(date)] = float(long - short)
    return pd.Series(returns).sort_index()


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline of cumulative strategy value."""
    if returns.empty:
        return 0.0
    equity = (1 + returns.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def summarize_predictions(predictions: pd.DataFrame, *, cost_bps: float = 0.0) -> dict[str, float]:
    """Summarize prediction quality and simple long-short backtest metrics."""
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    daily_ls = long_short_returns(predictions)
    cost = cost_bps / 10_000
    cost_adjusted = daily_ls - cost
    daily_ic = predictions.groupby("date")[["prediction", "label"]].apply(
        lambda day: day.corr(method="spearman").iloc[0, 1]
    )
    annualized_return = float(cost_adjusted.mean() * 252 / 5) if not cost_adjusted.empty else 0.0
    return {
        "rank_ic": rank_ic(predictions),
        "ic": float(predictions[["prediction", "label"]].corr().iloc[0, 1]),
        "icir": icir(daily_ic.dropna()),
        "sharpe": _annualized_sharpe(cost_adjusted),
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown(cost_adjusted),
        "turnover": 1.0,
        "hit_rate": float((daily_ls > 0).mean()) if not daily_ls.empty else 0.0,
        "long_short_spread": float(daily_ls.mean()) if not daily_ls.empty else 0.0,
        "cost_bps": float(cost_bps),
    }


def _annualized_sharpe(returns: pd.Series) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=0) * np.sqrt(252 / 5))
