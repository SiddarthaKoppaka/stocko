# Stock Manager / DiffSTOCK

Research-only US equities training and backtesting pipeline. v1 trains models, compares predictions, and reports backtest metrics. It does not place trades, connect to broker execution endpoints, or provide financial advice.

## Scope

- Universe: current S&P 500 constituents for v1.
- Target: 5-trading-day forward return, `close[t + 5] / close[t] - 1`.
- Models: LightGBM + Alpha158 baseline, MASTER adapter, StockMixer adapter.
- Backtesting: research comparison only, with 0, 5, and 10 bps cost cases.

Using current S&P 500 constituents over historical periods introduces survivorship bias. Treat v1 results as exploratory until historical index membership is implemented.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional model/data extras:

```bash
pip install -e ".[all]"
```

On macOS, LightGBM may require OpenMP:

```bash
brew install libomp
```

Copy `.env.example` to `.env` locally and set API keys only in your environment. Do not commit secrets or generated artifacts.

## Colab Flow

1. Open `notebooks/00_colab_setup.ipynb`.
2. Mount Google Drive.
3. Clone or pull the GitHub repo into `/content/stock-manager`.
4. Install dependencies.
5. Set `STOCK_MANAGER_DATA_DIR`, `STOCK_MANAGER_MODEL_DIR`, and `STOCK_MANAGER_REPORT_DIR`.
6. Run ingestion, Qlib conversion, model training, then comparison notebooks.

Default Colab artifact root:

```text
/content/drive/MyDrive/Stock_manager
```

## Data Sources

The pipeline tries Alpaca daily bars first when credentials exist, then yfinance fallback. FRED macro features are optional and config-controlled. Raw, processed, Qlib binary, model, prediction, and report files are generated artifacts and should stay outside GitHub.

## Commands

```bash
python scripts/ingest_data.py --config configs/data_alpaca_yfinance_fred.yaml
python scripts/build_qlib_dataset.py --config configs/qlib_lightgbm_alpha158.yaml
python scripts/train_model.py --config configs/qlib_lightgbm_alpha158.yaml
python scripts/train_model.py --config configs/qlib_master.yaml
python scripts/train_model.py --config configs/qlib_stockmixer.yaml
python scripts/compare_models.py --config configs/backtest_compare.yaml
pytest
```

MASTER and StockMixer are implemented as local adapter entry points. They provide a common training/prediction artifact contract now and are isolated so official architecture integrations can be tightened without changing notebooks.

## Outputs

Each training run writes:

- model artifact
- predictions
- metrics
- config snapshot
- run metadata

Comparison writes machine-readable metrics and a human-readable summary under the configured reports directory.
