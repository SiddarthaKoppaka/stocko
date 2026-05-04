from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_manager.backtest.metrics import summarize_predictions
from stock_manager.config import require_keys
from stock_manager.reporting.artifacts import write_json
from stock_manager.utils.paths import ensure_dir


def compare_models(config: dict) -> dict[str, Path]:
    """Compare model prediction artifacts under shared cost assumptions."""
    require_keys(config, ["models", "report_dir"], context="comparison config")
    report_dir = ensure_dir(config["report_dir"])
    cost_bps_values = config.get("cost_bps", [0, 5, 10])
    rows = []
    for model in config["models"]:
        name = model["name"]
        predictions = pd.read_parquet(model["predictions_path"])
        for cost_bps in cost_bps_values:
            metrics = summarize_predictions(predictions, cost_bps=cost_bps)
            rows.append({"model": name, **metrics})
    metrics_frame = pd.DataFrame(rows)
    csv_path = report_dir / "model_comparison.csv"
    metrics_frame.to_csv(csv_path, index=False)
    json_path = write_json(report_dir / "model_comparison.json", {"rows": rows})
    summary_path = report_dir / "summary.md"
    summary_path.write_text(_summary_markdown(metrics_frame), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "summary": summary_path}


def _summary_markdown(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "# Model Comparison\n\nNo metrics available.\n"
    best = metrics.sort_values(["rank_ic", "sharpe"], ascending=False).iloc[0]
    table = _markdown_table(metrics)
    return (
        "# Model Comparison\n\n"
        "Research-only output. Backtest performance does not guarantee future performance.\n\n"
        f"Best row by Rank IC: `{best['model']}` at `{best['cost_bps']}` bps.\n\n"
        + table
        + "\n"
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
