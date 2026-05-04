from __future__ import annotations

import argparse

from stock_manager.config import load_config
from stock_manager.qlib import build_qlib_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qlib-compatible dataset layout.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    outputs = build_qlib_dataset(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

