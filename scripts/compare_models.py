from __future__ import annotations

import argparse

from stock_manager.backtest import compare_models
from stock_manager.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model prediction artifacts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    outputs = compare_models(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

