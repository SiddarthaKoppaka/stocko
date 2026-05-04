from __future__ import annotations

import pandas as pd

from stock_manager.models.lightgbm_alpha158 import _chronological_split


def test_chronological_split_boundaries():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            "ticker": ["A", "A", "A"],
            "close": [1.0, 2.0, 3.0],
            "label_5d": [0.1, 0.2, 0.3],
        }
    )

    train, valid, test = _chronological_split(
        frame,
        {"train_end": "2020-12-31", "valid_end": "2021-12-31", "test_end": "2022-12-31"},
    )

    assert train["date"].dt.year.tolist() == [2020]
    assert valid["date"].dt.year.tolist() == [2021]
    assert test["date"].dt.year.tolist() == [2022]

