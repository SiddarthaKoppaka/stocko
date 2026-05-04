from __future__ import annotations

import argparse

from stock_manager.config import load_config
from stock_manager.models import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configured model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    outputs = train_from_config(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

