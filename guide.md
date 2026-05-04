<project_handoff_prompt>

  <project_identity>
    <project_name>Stock Manager / DiffSTOCK</project_name>
    <project_type>US equities quantitative research and backtesting system</project_type>
    <primary_goal>
      Build a GitHub-based quant research codebase that can be pulled into Google Colab for model training and backtesting.
      The system should start with US equities, train multiple models, compare their performance, and stop at backtesting for v1.
    </primary_goal>
    <current_phase>Project kickoff / repository skeleton / training pipeline setup</current_phase>
  </project_identity>

  <confirmed_architectural_decisions>
    <market>
      US equities only for now.
    </market>

    <universe>
      Use S&P 500 as the first trading universe.
      If feasible, design the system so Nasdaq 100 or other liquid universes can be added later.
      For v1, current S&P 500 constituents are acceptable, but document survivorship-bias limitations.
    </universe>

    <target_label>
      Use 5-trading-day forward return as the primary prediction target.

      Formula:
      forward_return_5d = close[t + 5] / close[t] - 1

      Meaning:
      The model should learn to rank which stocks are likely to outperform over the next 5 trading days.
      It does not need to predict exact future prices.
    </target_label>

    <v1_scope>
      v1 ends at model training, backtesting, and model comparison.
      No live trading in v1.
      No broker order placement in v1.
      No full LLM agent execution in v1.
    </v1_scope>

    <models>
      Implement support for:
      1. LightGBM + Alpha158 baseline
      2. MASTER
      3. StockMixer

      Do not skip MASTER or StockMixer. The user has tried simpler approaches before and wants all three model families supported.
    </models>

    <llm_role>
      LLM agents are a later feature.
      For now, LLM behavior should be advisory-only.
      LLMs should never directly place trades.
      LLMs may later explain signals, flag contrary evidence, and log reasoning.
    </llm_role>

    <future_execution_plan>
      Fully automated paper trading is a future phase after v1 backtest validation.
      Alpaca is the likely paper-trading broker/data API for that future phase.
    </future_execution_plan>

    <data_plan>
      Start with free or low-friction data sources.
      Use Alpaca historical data if available and practical.
      Use yfinance as a fallback/cross-check.
      Use FRED for macro features.
      Consider Polygon later only if data quality, corporate actions, intraday data, or production-grade backtesting become blockers.
    </data_plan>

    <development_workflow>
      Code should live in GitHub.
      Colab notebooks should clone or pull the GitHub repo when running.
      Google Drive should store large datasets, Qlib binary data, model artifacts, and generated reports.
      Do not commit large data files, trained model artifacts, API keys, or secrets to GitHub.
    </development_workflow>
  </confirmed_architectural_decisions>

  <alpaca_data_decision>
    <summary>
      Alpaca free/basic historical data appears sufficient for v1 research/backtesting on daily S&P 500 OHLCV,
      especially for 5-day forward-return modeling.
    </summary>

    <limitations>
      Free Alpaca market data may use IEX rather than full SIP coverage.
      This is acceptable for initial research, but not institutional-grade execution simulation.
      Alpaca does not solve historical S&P 500 membership / survivorship bias by itself.
      Recent data access and rate limits may depend on account/data plan.
    </limitations>

    <recommended_usage>
      For v1:
      - Try Alpaca daily historical bars for S&P 500 OHLCV.
      - Cross-check a small sample against yfinance.
      - Use yfinance fallback when Alpaca limits or auth issues block progress.
      - Use FRED for macro features.

      For later:
      - Use Alpaca for paper trading.
      - Evaluate Polygon or paid Alpaca/SIP if data quality becomes a blocker.
    </recommended_usage>
  </alpaca_data_decision>

  <risk_management_context>
    <drawdown_definition>
      Max drawdown is the largest peak-to-trough decline in portfolio value.

      Example:
      Portfolio reaches $120,000 and later falls to $102,000.
      Drawdown = 102000 / 120000 - 1 = -15%.
    </drawdown_definition>

    <v1_decision>
      In v1, only measure and report drawdown.
      Do not stop training/backtests based on drawdown.
    </v1_decision>

    <future_paper_trading_recommendation>
      For future paper trading:
      - 10% portfolio drawdown alert
      - 15% hard pause
      - Evaluate ATR-based position stops, but do not assume they improve performance until tested.
    </future_paper_trading_recommendation>
  </risk_management_context>

  <remaining_open_decisions>
    <universe_construction>
      Decide whether v1 uses:
      - current S&P 500 constituents only, or
      - approximated historical S&P 500 membership.

      Recommendation:
      Use current S&P 500 constituents for v1 simplicity.
      Clearly document survivorship bias.
    </universe_construction>

    <additional_universes>
      Decide whether to add Nasdaq 100 from day one.

      Recommendation:
      Make the config extensible, but start with S&P 500 only.
    </additional_universes>

    <context_instruments>
      Decide whether to include ETFs/macros as context features.

      Recommendation:
      Add context-only instruments later:
      - SPY
      - QQQ
      - IWM
      - TLT
      - GLD
      - VIX proxy

      These should not be trade candidates in v1 unless explicitly configured.
    </context_instruments>

    <backtest_execution_assumption>
      Decide how trades are simulated.

      Recommendation:
      Use next-open execution where feasible:
      signal generated after market close -> trade at next market open.
    </backtest_execution_assumption>

    <transaction_costs>
      Recommendation:
      Compare three cases:
      - 0 bps
      - 5 bps per trade
      - 10 bps per trade

      If a strategy only works at 0 bps and fails at 5-10 bps, mark it fragile.
    </transaction_costs>

    <model_winner_selection>
      Recommendation:
      All models must use the same:
      - universe
      - label horizon
      - date ranges
      - train/validation/test splits
      - walk-forward windows
      - cost assumptions

      Primary metric:
      Rank IC stability

      Secondary metrics:
      - IC
      - ICIR
      - Sharpe
      - max drawdown
      - turnover
      - hit rate
      - long-short spread
    </model_winner_selection>
  </remaining_open_decisions>

  <target_repository_structure>
    <tree>
stock-manager/
├── notebooks/
│   ├── 00_colab_setup.ipynb
│   ├── 01_data_ingestion.ipynb
│   ├── 02_qlib_dataset_build.ipynb
│   ├── 03_train_lightgbm_alpha158.ipynb
│   ├── 04_train_master.ipynb
│   ├── 05_train_stockmixer.ipynb
│   └── 06_backtest_compare_models.ipynb
├── src/
│   └── stock_manager/
│       ├── __init__.py
│       ├── config/
│       ├── data/
│       ├── features/
│       ├── qlib/
│       ├── models/
│       ├── backtest/
│       ├── reporting/
│       └── utils/
├── configs/
│   ├── universe_sp500.yaml
│   ├── universe_nasdaq100.yaml
│   ├── data_alpaca_yfinance_fred.yaml
│   ├── qlib_lightgbm_alpha158.yaml
│   ├── qlib_master.yaml
│   └── qlib_stockmixer.yaml
├── scripts/
├── reports/
├── tests/
├── pyproject.toml
├── README.md
├── .env.example
└── .gitignore
    </tree>
  </target_repository_structure>

  <implementation_requirements>
    <notebook_requirements>
      Keep notebooks simple.
      Notebooks should orchestrate steps, not contain large business logic.
      Reusable logic must live in src/stock_manager.

      00_colab_setup.ipynb should:
      - Mount Google Drive.
      - Clone or pull the GitHub repo.
      - Install dependencies.
      - Set environment variables such as STOCK_MANAGER_DATA_DIR.
      - Verify GPU/runtime.
      - Verify imports.

      Each training notebook should be config-driven.
    </notebook_requirements>

    <data_ingestion_requirements>
      Build modules for:
      - S&P 500 universe loading.
      - Alpaca historical OHLCV download.
      - yfinance fallback download.
      - FRED macro download.
      - basic data schema validation.
      - missing data reporting.
      - adjusted close handling if available.

      Store raw data separately from processed/model-ready data.
    </data_ingestion_requirements>

    <labeling_requirements>
      Implement 5-day forward return label generation.

      Requirements:
      - Group by ticker.
      - Sort by date.
      - label_5d = close.shift(-5) / close - 1
      - Avoid leakage.
      - Drop rows where the future label is unavailable.
      - Add tests for label correctness.
    </labeling_requirements>

    <qlib_requirements>
      Build Qlib dataset conversion utilities.
      Support Qlib-compatible data directory layout.
      Keep provider_uri configurable.
      Make Google Drive path configurable.
      Support Alpha158 feature setup for LightGBM.
      Add config files for MASTER and StockMixer.
    </qlib_requirements>

    <model_training_requirements>
      Implement model training entry points for:
      - LightGBM + Alpha158
      - MASTER
      - StockMixer

      Each model should:
      - load config
      - train on the configured dataset
      - write artifacts to Google Drive output path
      - save metrics
      - save predictions
      - save model config snapshot
    </model_training_requirements>

    <backtesting_requirements>
      Implement backtest comparison utilities.

      Include metrics:
      - IC
      - Rank IC
      - ICIR
      - Sharpe
      - annualized return
      - max drawdown
      - turnover
      - hit rate
      - long-short spread
      - cost-adjusted returns

      Support cost assumptions:
      - 0 bps
      - 5 bps
      - 10 bps
    </backtesting_requirements>

    <artifact_requirements>
      Save the following artifacts:
      - raw data snapshots
      - processed datasets
      - Qlib binary data
      - trained model artifacts
      - predictions
      - backtest reports
      - plots
      - config snapshots
      - run metadata

      Do not commit artifacts to GitHub.
    </artifact_requirements>
  </implementation_requirements>

  <coding_style>
    <language>Python</language>
    <python_version>Use Python 3.10 or 3.11 unless Qlib compatibility requires otherwise.</python_version>

    <style_rules>
      Use typed functions where practical.
      Prefer small modules and pure functions.
      Use dataclasses or Pydantic models for structured config when useful.
      Avoid notebook-only logic.
      Avoid hardcoded local paths.
      Avoid hardcoded secrets.
      Prefer pathlib.Path over raw string file paths.
      Prefer logging over print in src modules.
      Keep scripts thin and call src package functions.
      Write clear error messages for missing API keys, missing data, and invalid schemas.
    </style_rules>

    <dependency_management>
      Use pyproject.toml.
      Keep dependencies grouped by purpose if possible:
      - core
      - data
      - qlib
      - dev
      - notebook

      Do not add heavy dependencies unless needed.
      Avoid dependency sprawl.
    </dependency_management>

    <config_style>
      Use YAML config files under configs/.
      No important training parameter should be hardcoded in notebooks.
      Configs should include:
      - universe
      - date range
      - data provider
      - storage paths
      - model type
      - feature set
      - label horizon
      - train/validation/test windows
      - backtest assumptions
    </config_style>
  </coding_style>

  <documentation_style>
    <readme_requirements>
      README.md should explain:
      - project purpose
      - v1 scope
      - what is intentionally out of scope
      - setup instructions
      - Colab workflow
      - data sources
      - artifact storage strategy
      - model training workflow
      - backtest comparison workflow
      - known limitations
      - survivorship bias warning
      - no-live-trading warning
    </readme_requirements>

    <docstring_style>
      Public functions should include concise docstrings.
      Docstrings should explain:
      - what the function does
      - expected input schema
      - output schema
      - leakage/safety assumptions where relevant
    </docstring_style>

    <comments_style>
      Use comments to explain non-obvious quant logic.
      Do not over-comment simple Python.
      Always comment leakage-sensitive sections, such as label generation and train/test splitting.
    </comments_style>

    <reports_style>
      Reports should be saved in reports/.
      Prefer both machine-readable and human-readable outputs:
      - metrics.json
      - predictions.parquet
      - summary.md
      - plots as png/html
    </reports_style>
  </documentation_style>

  <testing_guidelines>
    <unit_tests>
      Add unit tests for:
      - 5-day forward return label generation
      - ticker/date sorting
      - schema validation
      - missing data detection
      - config loading
      - train/validation/test split boundaries
    </unit_tests>

    <integration_tests>
      Add lightweight integration tests using a tiny sample universe such as:
      - AAPL
      - MSFT
      - NVDA

      Do not require full S&P 500 downloads in CI.
    </integration_tests>

    <test_constraints>
      Tests should not require real API keys by default.
      Mock external APIs where possible.
      Never hit paid APIs in tests.
      Avoid long-running model training in tests.
    </test_constraints>
  </testing_guidelines>

  <guardrails>
    <trading_safety>
      v1 must not place trades.
      v1 must not connect to broker execution endpoints.
      v1 must not include live-trading automation.
      Any paper-trading code must be explicitly separated into a future module or feature flag.
    </trading_safety>

    <data_safety>
      Use only public or user-authorized data.
      Do not use non-public insider information.
      Do not scrape sources that prohibit scraping without checking terms.
      Do not commit downloaded datasets to GitHub.
    </data_safety>

    <secret_management>
      Never commit API keys.
      Use environment variables.
      Provide .env.example only.
      Add .env to .gitignore.
      Add Google Drive artifact directories to .gitignore where applicable.
    </secret_management>

    <financial_disclaimer>
      This project is for research and backtesting.
      Outputs are not financial advice.
      Backtest performance does not guarantee future performance.
      Mark all v1 outputs as research-only.
    </financial_disclaimer>

    <leakage_prevention>
      Avoid lookahead bias.
      Do not use future data in features.
      Label generation must shift future returns only into the target column.
      Feature windows must only use information available at or before time t.
      Train/validation/test splits must be chronological, not random.
    </leakage_prevention>

    <survivorship_bias>
      If using current S&P 500 constituents historically, document survivorship bias.
      Do not claim institutional-grade backtest quality until historical membership is handled.
    </survivorship_bias>
  </guardrails>

  <colab_workflow>
    <expected_flow>
      1. Open 00_colab_setup.ipynb.
      2. Mount Google Drive.
      3. Clone or pull GitHub repo.
      4. Install dependencies.
      5. Set paths.
      6. Run data ingestion.
      7. Build Qlib dataset.
      8. Train models.
      9. Run backtest comparison.
      10. Save reports to Google Drive.
    </expected_flow>

    <path_expectations>
      Code path:
      /content/stock-manager

      Google Drive project root:
      /content/drive/MyDrive/Stock_manager

      Suggested artifact paths:
      /content/drive/MyDrive/Stock_manager/data/raw
      /content/drive/MyDrive/Stock_manager/data/processed
      /content/drive/MyDrive/Stock_manager/data/qlib
      /content/drive/MyDrive/Stock_manager/models
      /content/drive/MyDrive/Stock_manager/reports
    </path_expectations>
  </colab_workflow>

  <initial_tasks_for_codex>
    <task_1>
      Create the repository skeleton exactly as specified.
    </task_1>

    <task_2>
      Add pyproject.toml with initial dependencies for:
      - pandas
      - numpy
      - pyarrow
      - pyyaml
      - yfinance
      - alpaca-py
      - fredapi
      - scikit-learn
      - lightgbm
      - matplotlib
      - pytest
      - ruff
    </task_2>

    <task_3>
      Add README.md with setup, scope, and warnings.
    </task_3>

    <task_4>
      Add .gitignore for:
      - .env
      - __pycache__
      - .pytest_cache
      - .ruff_cache
      - data/
      - models/
      - reports/
      - *.pkl
      - *.parquet
      - *.csv generated artifacts
      - notebooks checkpoints
    </task_4>

    <task_5>
      Implement config loader.
    </task_5>

    <task_6>
      Implement initial S&P 500 universe loader.
      For v1, current S&P 500 is acceptable.
      Document survivorship bias.
    </task_6>

    <task_7>
      Implement Alpaca/yfinance OHLCV ingestion with clean fallback behavior.
    </task_7>

    <task_8>
      Implement 5-day forward return label generator and tests.
    </task_8>

    <task_9>
      Create placeholder Qlib conversion module with clear TODOs if full Qlib implementation is not completed immediately.
    </task_9>

    <task_10>
      Create simple Colab setup notebook or notebook-generating script.
    </task_10>
  </initial_tasks_for_codex>

  <success_criteria_for_first_commit>
    <criteria>
      The repo installs locally.
      Tests run successfully.
      README explains the project.
      Configs are present.
      Basic data ingestion modules exist.
      Label generation is implemented and tested.
      No secrets or large artifacts are committed.
      Colab workflow is documented.
    </criteria>
  </success_criteria_for_first_commit>

</project_handoff_prompt>