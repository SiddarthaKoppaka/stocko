# Training Pipeline Plan

## Summary
Build the v1 training portion of `Stock Manager / DiffSTOCK` as a GitHub-ready Python repo that runs locally and in Colab, trains three model families, saves comparable artifacts, and stops at research/backtesting.

Primary implementation order:
1. Repo skeleton, config system, README, `.gitignore`, `.env.example`
2. Data ingestion and validation
3. 5-day forward-return labeling
4. Qlib dataset conversion
5. LightGBM + Alpha158 training
6. MASTER adapter
7. StockMixer adapter
8. Shared prediction, metrics, and backtest comparison outputs

## Source Links
Use these as implementation references:

- Qlib repo and workflow: https://github.com/microsoft/qlib
- Qlib LightGBM Alpha158 config: https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
- Qlib data format docs: https://qlib.readthedocs.io/en/latest/component/data.html
- LightGBM repo/docs: https://github.com/microsoft/LightGBM and https://lightgbm.readthedocs.io/
- MASTER official repo: https://github.com/SJTU-DMTai/MASTER
- MASTER paper: https://huggingface.co/papers/2312.15235
- StockMixer official repo: https://github.com/SJTU-DMTai/StockMixer
- StockMixer paper/DOI: https://mlanthology.org/aaai/2024/fan2024aaai-stockmixer/
- Alpaca historical bars API: https://docs.alpaca.markets/reference/stockbars
- Alpaca Python SDK market data: https://alpaca.markets/sdks/python/market_data.html
- yfinance package: https://pypi.org/project/yfinance/
- fredapi repo/package: https://github.com/mortada/fredapi and https://pypi.org/project/fredapi/
- S&P 500 current constituents fallback source: https://www.slickcharts.com/sp500

## Key Changes
- Create the repo structure from `guide.md`, keeping notebooks thin and putting reusable logic under `src/stock_manager`.
- Add `pyproject.toml` with grouped dependencies:
  - core/data: `pandas`, `numpy`, `pyarrow`, `pyyaml`, `yfinance`, `alpaca-py`, `fredapi`
  - modeling: `scikit-learn`, `lightgbm`, `torch`
  - qlib: `pyqlib`
  - reporting/dev: `matplotlib`, `pytest`, `ruff`
- Add YAML configs for universe, data, Qlib provider URI, model params, train/valid/test windows, artifact paths, and cost assumptions.
- Use current S&P 500 constituents for v1, with explicit survivorship-bias documentation.
- Implement data ingestion with Alpaca first, yfinance fallback, FRED macro support, schema validation, and missing-data reports.
- Store raw, processed, Qlib binary, model, prediction, and report artifacts outside GitHub, defaulting to `/content/drive/MyDrive/Stock_manager/...` in Colab.
- Implement leakage-safe `label_5d = close.shift(-5) / close - 1` grouped by ticker and sorted by date.
- Build a Qlib conversion path that writes OHLCV plus required `factor` field into Qlib-compatible `.bin` layout.
- Implement one common training contract for all models:
  - load config
  - load dataset
  - train
  - save model artifact
  - save predictions
  - save metrics
  - save config snapshot
  - save run metadata
- For LightGBM, start from Qlib’s Alpha158 workflow and adapt market, dates, provider URI, and artifact paths to US equities.
- For MASTER and StockMixer, wrap their official architectures behind local adapters rather than copying notebook logic into notebooks; isolate version-sensitive Torch requirements if needed.
- Implement model comparison using identical universe, labels, splits, and cost assumptions across all three models.

## Training Workflow
- `00_colab_setup.ipynb`: mount Drive, clone/pull repo, install dependencies, set env vars, verify GPU/runtime/imports.
- `01_data_ingestion.ipynb`: run configured S&P 500 OHLCV and macro ingestion.
- `02_qlib_dataset_build.ipynb`: convert processed data into Qlib format and validate provider URI.
- `03_train_lightgbm_alpha158.ipynb`: call the LightGBM training entry point.
- `04_train_master.ipynb`: call the MASTER training entry point.
- `05_train_stockmixer.ipynb`: call the StockMixer training entry point.
- `06_backtest_compare_models.ipynb`: load predictions and generate comparable metrics/reports.

## Tests
- Unit tests:
  - config loading and required-field validation
  - ticker/date sorting
  - 5-day label correctness
  - no future label rows retained
  - schema validation
  - missing-data detection
  - chronological split boundaries
- Integration tests:
  - tiny universe: `AAPL`, `MSFT`, `NVDA`
  - mocked Alpaca/yfinance/FRED calls
  - small processed dataset to Qlib conversion smoke test
  - LightGBM tiny training smoke test only
- Acceptance criteria:
  - repo installs locally
  - `pytest` passes without API keys
  - notebooks import package code rather than holding business logic
  - no secrets, datasets, model artifacts, reports, `.parquet`, `.csv`, or `.pkl` outputs are committed
  - README documents Colab flow, data limitations, survivorship bias, no-live-trading scope, and research-only disclaimer

## Assumptions
- v1 uses current S&P 500 constituents only.
- Nasdaq 100 is config-ready but not implemented as a required first training universe.
- Alpaca daily bars are attempted first; yfinance is the fallback/cross-check.
- FRED macro features are optional in the first pass and can be enabled by config once the equity pipeline works.
- Backtests use next-open execution where feasible and compare 0, 5, and 10 bps cost cases.
- MASTER and StockMixer are required, but their integration can begin as adapter-backed training entry points with clear compatibility notes if full Qlib-native integration takes longer.
