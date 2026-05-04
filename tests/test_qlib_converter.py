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
    assert (tmp_path / "qlib" / "features" / "aapl" / "factor.csv").exists()
