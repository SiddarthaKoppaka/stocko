from __future__ import annotations

import pandas as pd

from stock_manager.qlib import build_qlib_dataset


def test_build_qlib_dataset_writes_expected_layout(tmp_path):
    processed = tmp_path / "processed.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["AAPL", "AAPL"],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [100, 120],
            "label_5d": [0.1, 0.2],
        }
    ).to_parquet(processed, index=False)

    outputs = build_qlib_dataset(
        {
            "data": {"processed_path": str(processed)},
            "qlib": {"provider_uri": str(tmp_path / "qlib")},
        }
    )

    assert outputs["calendar"].exists()
    assert outputs["instruments"].exists()
    assert outputs["alpha158"].exists()
    assert (tmp_path / "qlib" / "features" / "aapl" / "factor.day.bin").exists()
    assert (tmp_path / "qlib" / "features" / "aapl" / "close.day.bin").stat().st_size > 0

    alpha158 = pd.read_parquet(outputs["alpha158"])
    assert {"date", "ticker", "LABEL0", "KMID", "ROC5", "VSUMD60"}.issubset(alpha158.columns)
    feature_columns = [col for col in alpha158.columns if col not in {"date", "ticker", "LABEL0"}]
    assert len(feature_columns) == 158
