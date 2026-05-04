from __future__ import annotations

import argparse

from stock_manager.config import load_config
from stock_manager.data.ingestion import ingest_market_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OHLCV and optional macro data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    outputs = ingest_market_data(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

