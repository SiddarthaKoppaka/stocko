from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from stock_manager.config import require_keys
from stock_manager.models.lightgbm_alpha158 import _prediction_frame, _prediction_metrics, _write_training_outputs
from stock_manager.utils.logging import get_logger, progress_iter

EPSILON = 1e-12
MASTER_FACTOR_COUNT = 158
MASTER_MARKET_FEATURE_COUNT = 63
MASTER_TOTAL_FEATURE_COUNT = MASTER_FACTOR_COUNT + MASTER_MARKET_FEATURE_COUNT
LOGGER = get_logger(__name__)


def train_master_model(config: dict) -> dict[str, Path]:
    require_keys(
        config,
        [
            "data.processed_path",
            "data.market_source_path",
            "data.label_column",
            "splits.train_end",
            "splits.valid_end",
            "paths.model_dir",
        ],
        context="master config",
    )

    feature_path = Path(config["data"]["processed_path"])
    feature_frame = pd.read_parquet(feature_path)
    market_source = pd.read_parquet(Path(config["data"]["market_source_path"]))
    label_column = config["data"]["label_column"]
    params = config.get("model", {}).get("params", {})
    show_progress = config.get("runtime", {}).get("show_progress", True)

    LOGGER.info("Preparing MASTER training data from %s", config["data"]["processed_path"])

    _validate_master_feature_frame(feature_frame, label_column, feature_path)

    merged, feature_columns = _prepare_master_frame(
        feature_frame,
        market_source,
        label_column,
        config,
        show_progress=show_progress,
    )
    datasets = _build_master_datasets(
        merged,
        feature_columns,
        label_column,
        config,
        show_progress=show_progress,
    )
    LOGGER.info(
        "MASTER datasets ready: train=%s valid=%s test=%s samples",
        len(datasets.train),
        len(datasets.valid),
        len(datasets.test),
    )
    model_bundle = _fit_master(datasets, feature_columns, config, show_progress=show_progress)
    predictions = _predict_master(model_bundle, datasets.test, label_column, show_progress=show_progress)
    metrics = _prediction_metrics(predictions)
    model_bundle.model.to(torch.device("cpu"))
    config.setdefault("model", {}).setdefault("params", {}).update(
        {
            "resolved_feature_count": len(feature_columns),
            "resolved_market_feature_count": MASTER_MARKET_FEATURE_COUNT,
            "resolved_factor_count": MASTER_FACTOR_COUNT,
            **params,
        }
    )
    LOGGER.info("MASTER training complete: rank_ic=%.4f mse=%.6f", metrics["rank_ic"], metrics["mse"])
    return _write_training_outputs("master", config, model_bundle.model, predictions, metrics)


def _validate_master_feature_frame(
    feature_frame: pd.DataFrame,
    label_column: str,
    feature_path: Path,
) -> None:
    if label_column in feature_frame.columns:
        return

    available = ", ".join(sorted(feature_frame.columns[:10]))
    raise ValueError(
        "MASTER expects a materialized Alpha158 parquet that includes the configured label column "
        f"'{label_column}', but it was not found in {feature_path}. Rebuild the Alpha158 artifact "
        "from the notebook's Qlib Dataset Build step or rerun the dataset builder before training. "
        f"Sample columns: {available}"
    )


@dataclass
class MasterDatasets:
    train: "MasterSequenceDataset"
    valid: "MasterSequenceDataset"
    test: "MasterSequenceDataset"


@dataclass
class MasterBundle:
    model: nn.Module
    device: torch.device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1], :]


class SAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)
        dim = int(self.d_model / self.nhead)
        outputs = []
        for index in range(self.nhead):
            if index == self.nhead - 1:
                qh = q[:, :, index * dim :]
                kh = k[:, :, index * dim :]
                vh = v[:, :, index * dim :]
            else:
                qh = q[:, :, index * dim : (index + 1) * dim]
                kh = k[:, :, index * dim : (index + 1) * dim]
                vh = v[:, :, index * dim : (index + 1) * dim]
            attention = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            attention = self.attn_dropout[index](attention)
            outputs.append(torch.matmul(attention, vh).transpose(0, 1))
        attended = torch.cat(outputs, dim=-1)
        residual = x + attended
        residual = self.norm2(residual)
        return residual + self.ffn(residual)


class TAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)
        dim = int(self.d_model / self.nhead)
        outputs = []
        for index in range(self.nhead):
            if index == self.nhead - 1:
                qh = q[:, :, index * dim :]
                kh = k[:, :, index * dim :]
                vh = v[:, :, index * dim :]
            else:
                qh = q[:, :, index * dim : (index + 1) * dim]
                kh = k[:, :, index * dim : (index + 1) * dim]
                vh = v[:, :, index * dim : (index + 1) * dim]
            attention = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            attention = self.attn_dropout[index](attention)
            outputs.append(torch.matmul(attention, vh))
        attended = torch.cat(outputs, dim=-1)
        residual = x + attended
        residual = self.norm2(residual)
        return residual + self.ffn(residual)


class Gate(nn.Module):
    def __init__(self, d_input: int, d_output: int, beta: float = 1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.temperature = beta

    def forward(self, gate_input: torch.Tensor) -> torch.Tensor:
        output = self.trans(gate_input)
        output = torch.softmax(output / self.temperature, dim=-1)
        return self.d_output * output


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.trans(z)
        query = hidden[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(hidden, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        return torch.matmul(lam, z).squeeze(1)


class MASTER(nn.Module):
    def __init__(
        self,
        d_feat: int = MASTER_FACTOR_COUNT,
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        t_dropout_rate: float = 0.5,
        s_dropout_rate: float = 0.5,
        gate_input_start_index: int = MASTER_FACTOR_COUNT,
        gate_input_end_index: int = MASTER_TOTAL_FEATURE_COUNT,
        beta: float = 2.0,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.feature_gate = Gate(gate_input_end_index - gate_input_start_index, d_feat, beta=beta)
        self.x2y = nn.Linear(d_feat, d_model)
        self.pe = PositionalEncoding(d_model)
        self.tatten = TAttention(d_model=d_model, nhead=t_nhead, dropout=t_dropout_rate)
        self.satten = SAttention(d_model=d_model, nhead=s_nhead, dropout=s_dropout_rate)
        self.temporalatten = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        src = x[:, :, : self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index : self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        hidden = self.x2y(src)
        hidden = self.pe(hidden)
        hidden = self.tatten(hidden)
        hidden = self.satten(hidden)
        hidden = self.temporalatten(hidden)
        return self.decoder(hidden).squeeze(-1)


class MasterSequenceDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        label_column: str,
        *,
        lookback_window: int,
        split_name: str,
        train_end: pd.Timestamp,
        valid_end: pd.Timestamp,
        test_end: pd.Timestamp,
        show_progress: bool,
    ):
        records = []
        self.series: dict[str, np.ndarray] = {}
        self.feature_columns = feature_columns
        self.label_column = label_column

        grouped = frame.groupby("ticker", sort=True)
        for ticker, ticker_frame in progress_iter(
            grouped,
            total=frame["ticker"].nunique(),
            desc=f"MASTER {split_name} windows",
            enabled=show_progress,
        ):
            ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
            values = ticker_frame.loc[:, [*feature_columns, label_column]].to_numpy(dtype=np.float32)
            dates = pd.to_datetime(ticker_frame["date"]).to_numpy()
            self.series[ticker] = values
            for end_pos in range(lookback_window - 1, len(ticker_frame)):
                date = pd.Timestamp(dates[end_pos])
                if pd.isna(values[end_pos, -1]):
                    continue
                if split_name == "train" and date <= train_end:
                    records.append((date, ticker, end_pos))
                elif split_name == "valid" and train_end < date <= valid_end:
                    records.append((date, ticker, end_pos))
                elif split_name == "test" and valid_end < date <= test_end:
                    records.append((date, ticker, end_pos))

        records.sort(key=lambda item: (item[0], item[1]))
        self.records = records
        self.lookback_window = lookback_window
        self.index = pd.MultiIndex.from_tuples(
            [(date, ticker) for date, ticker, _ in records],
            names=["datetime", "instrument"],
        )
        self.day_slices = _daily_slices(self.index)

    def __len__(self) -> int:
        return len(self.records)

    def get_index(self) -> pd.MultiIndex:
        return self.index

    def iter_batches(self, *, shuffle: bool, drop_last: bool) -> tuple[np.ndarray, pd.MultiIndex]:
        order = np.arange(len(self.day_slices))
        if shuffle:
            np.random.shuffle(order)
        for day_index in order:
            start, stop = self.day_slices[day_index]
            if drop_last and start == stop:
                continue
            batch = np.stack([self._window_at(pos) for pos in range(start, stop + 1)], axis=0)
            yield batch, self.index[start : stop + 1]

    def _window_at(self, position: int) -> np.ndarray:
        _, ticker, end_pos = self.records[position]
        values = self.series[ticker]
        start = end_pos - self.lookback_window + 1
        return values[start : end_pos + 1]


def _prepare_master_frame(
    feature_frame: pd.DataFrame,
    market_source: pd.DataFrame,
    label_column: str,
    config: dict,
    *,
    show_progress: bool,
) -> tuple[pd.DataFrame, list[str]]:
    frame = feature_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    market_source = market_source.copy()
    market_source["date"] = pd.to_datetime(market_source["date"]).dt.tz_localize(None)

    excluded = {"date", "ticker", label_column}
    factor_columns = [column for column in frame.columns if column not in excluded]
    if len(factor_columns) != MASTER_FACTOR_COUNT:
        raise ValueError(
            f"MASTER expects {MASTER_FACTOR_COUNT} Alpha158 factors, found {len(factor_columns)}."
        )

    LOGGER.info("Building MASTER market-information features")
    market_features = _build_market_information(market_source)
    merged = frame.merge(market_features, on="date", how="left")

    train_end = pd.Timestamp(config["splits"]["train_end"])
    feature_columns = [*factor_columns, *[col for col in market_features.columns if col != "date"]]
    stats = _fit_robust_stats(merged.loc[merged["date"] <= train_end, feature_columns], feature_columns)
    merged.loc[:, feature_columns] = _apply_robust_stats(merged.loc[:, feature_columns], feature_columns, stats)
    merged.loc[:, feature_columns] = merged.loc[:, feature_columns].astype(np.float32)
    if show_progress:
        LOGGER.info("MASTER feature matrix ready: %s rows, %s columns", len(merged), len(feature_columns))
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True), feature_columns


def _build_market_information(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"MASTER market information requires columns: {', '.join(sorted(missing))}")

    source = frame.loc[:, ["date", "ticker", "close", "volume"]].copy()
    source = source.sort_values(["ticker", "date"]).reset_index(drop=True)
    source["ret_1d"] = source.groupby("ticker", group_keys=False)["close"].pct_change()
    source["amount"] = source["close"] * source["volume"]

    by_date = source.groupby("date")
    equal_weight_ret = by_date["ret_1d"].mean()
    median_ret = by_date["ret_1d"].median()
    equal_weight_amount = by_date["amount"].mean()
    median_amount = by_date["amount"].median()

    valid = source["ret_1d"].notna() & source["amount"].notna()
    weighted_numerator = (source.loc[valid, "ret_1d"] * source.loc[valid, "amount"]).groupby(source.loc[valid, "date"]).sum()
    weighted_denominator = source.loc[valid].groupby("date")["amount"].sum()
    volume_weighted_ret = weighted_numerator / (weighted_denominator + EPSILON)
    volume_weighted_amount = by_date["amount"].sum()

    market_frame = pd.DataFrame({"date": sorted(source["date"].unique())})
    market_frame = market_frame.set_index("date")
    specs = {
        "EQ": (equal_weight_ret, equal_weight_amount),
        "VW": (volume_weighted_ret, volume_weighted_amount),
        "MD": (median_ret, median_amount),
    }
    for prefix, (return_series, amount_series) in specs.items():
        return_series = return_series.reindex(market_frame.index)
        amount_series = amount_series.reindex(market_frame.index)
        market_frame[f"{prefix}_RET1"] = return_series
        for window in (5, 10, 20, 30, 60):
            market_frame[f"{prefix}_RET_MEAN_{window}"] = return_series.rolling(window).mean()
            market_frame[f"{prefix}_RET_STD_{window}"] = return_series.rolling(window).std()
            market_frame[f"{prefix}_AMOUNT_MEAN_{window}"] = amount_series.rolling(window).mean() / (amount_series + EPSILON)
            market_frame[f"{prefix}_AMOUNT_STD_{window}"] = amount_series.rolling(window).std() / (amount_series + EPSILON)
    market_frame = market_frame.reset_index()
    feature_columns = [column for column in market_frame.columns if column != "date"]
    if len(feature_columns) != MASTER_MARKET_FEATURE_COUNT:
        raise AssertionError(f"Expected {MASTER_MARKET_FEATURE_COUNT} market features, found {len(feature_columns)}")
    return market_frame


def _fit_robust_stats(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.Series, pd.Series]:
    median = frame[feature_columns].median()
    mad = (frame[feature_columns] - median).abs().median()
    scale = (1.4826 * mad).replace(0, np.nan)
    return median, scale


def _apply_robust_stats(
    frame: pd.DataFrame,
    feature_columns: list[str],
    stats: tuple[pd.Series, pd.Series],
) -> pd.DataFrame:
    median, scale = stats
    transformed = (frame[feature_columns] - median) / (scale + EPSILON)
    return transformed.clip(-3, 3).fillna(0.0)


def _build_master_datasets(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    config: dict,
    *,
    show_progress: bool,
) -> MasterDatasets:
    lookback_window = int(config.get("model", {}).get("params", {}).get("lookback_window", 8))
    train_end = pd.Timestamp(config["splits"]["train_end"])
    valid_end = pd.Timestamp(config["splits"]["valid_end"])
    test_end = pd.Timestamp(config["splits"].get("test_end", frame["date"].max()))
    return MasterDatasets(
        train=MasterSequenceDataset(
            frame,
            feature_columns,
            label_column,
            lookback_window=lookback_window,
            split_name="train",
            train_end=train_end,
            valid_end=valid_end,
            test_end=test_end,
            show_progress=show_progress,
        ),
        valid=MasterSequenceDataset(
            frame,
            feature_columns,
            label_column,
            lookback_window=lookback_window,
            split_name="valid",
            train_end=train_end,
            valid_end=valid_end,
            test_end=test_end,
            show_progress=show_progress,
        ),
        test=MasterSequenceDataset(
            frame,
            feature_columns,
            label_column,
            lookback_window=lookback_window,
            split_name="test",
            train_end=train_end,
            valid_end=valid_end,
            test_end=test_end,
            show_progress=show_progress,
        ),
    )


def _fit_master(
    datasets: MasterDatasets,
    feature_columns: list[str],
    config: dict,
    *,
    show_progress: bool,
) -> MasterBundle:
    params = config.get("model", {}).get("params", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(params.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = MASTER(
        d_feat=MASTER_FACTOR_COUNT,
        d_model=int(params.get("d_model", 256)),
        t_nhead=int(params.get("t_nhead", 4)),
        s_nhead=int(params.get("s_nhead", 2)),
        t_dropout_rate=float(params.get("t_dropout_rate", 0.5)),
        s_dropout_rate=float(params.get("s_dropout_rate", 0.5)),
        gate_input_start_index=MASTER_FACTOR_COUNT,
        gate_input_end_index=len(feature_columns),
        beta=float(params.get("beta", 2.0)),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(params.get("lr", 1e-4)))
    train_stop_loss_thred = float(params.get("train_stop_loss_thred", 0.95))
    n_epochs = int(params.get("n_epochs", 40))
    LOGGER.info("MASTER training started: epochs=%s, train batches=%s", n_epochs, len(datasets.train.day_slices))

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    for epoch_index in progress_iter(
        range(n_epochs),
        total=n_epochs,
        desc="MASTER epochs",
        enabled=show_progress,
    ):
        train_loss = _train_master_epoch(
            model,
            optimizer,
            datasets.train,
            device,
            show_progress=show_progress,
            epoch_label=f"MASTER epoch {epoch_index + 1}/{n_epochs}",
        )
        LOGGER.info("MASTER epoch %s/%s train_loss=%.6f", epoch_index + 1, n_epochs, train_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = copy.deepcopy(model.state_dict())
        if train_loss <= train_stop_loss_thred:
            LOGGER.info("MASTER early stop threshold reached at epoch %s", epoch_index + 1)
            break
    model.load_state_dict(best_state)
    return MasterBundle(model=model, device=device)


def _train_master_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    dataset: MasterSequenceDataset,
    device: torch.device,
    *,
    show_progress: bool,
    epoch_label: str,
) -> float:
    model.train()
    losses: list[float] = []
    batch_iterator = dataset.iter_batches(shuffle=True, drop_last=True)
    for batch, _ in progress_iter(
        batch_iterator,
        total=len(dataset.day_slices),
        desc=epoch_label,
        enabled=show_progress,
    ):
        tensor = torch.tensor(batch, dtype=torch.float32, device=device)
        features = tensor[:, :, :-1]
        labels = tensor[:, -1, -1]
        mask = _drop_extreme(labels)
        if mask.sum() == 0:
            continue
        labels = _zscore(labels[mask])
        features = features[mask]
        prediction = model(features)
        loss = torch.mean((prediction - labels) ** 2)
        losses.append(float(loss.item()))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
        optimizer.step()
    if not losses:
        raise ValueError("MASTER training produced no valid batches")
    return float(np.mean(losses))


def _predict_master(
    model_bundle: MasterBundle,
    dataset: MasterSequenceDataset,
    label_column: str,
    *,
    show_progress: bool,
) -> pd.DataFrame:
    model_bundle.model.eval()
    rows = []
    batch_iterator = dataset.iter_batches(shuffle=False, drop_last=False)
    for batch, batch_index in progress_iter(
        batch_iterator,
        total=len(dataset.day_slices),
        desc="MASTER inference",
        enabled=show_progress,
    ):
        tensor = torch.tensor(batch, dtype=torch.float32, device=model_bundle.device)
        features = tensor[:, :, :-1]
        labels = tensor[:, -1, -1].detach().cpu().numpy()
        with torch.no_grad():
            prediction = model_bundle.model(features).detach().cpu().numpy()
        dates = batch_index.get_level_values("datetime")
        tickers = batch_index.get_level_values("instrument")
        frame = _prediction_frame(
            pd.DataFrame({"date": dates, "ticker": tickers, label_column: labels}),
            prediction,
            label_column,
        )
        rows.append(frame)
    if not rows:
        raise ValueError("MASTER test split is empty")
    return pd.concat(rows, ignore_index=True)


def _drop_extreme(labels: torch.Tensor) -> torch.Tensor:
    sorted_values, indices = labels.sort()
    size = labels.shape[0]
    cutoff = int(0.025 * size)
    if cutoff == 0 or size - cutoff <= cutoff:
        return ~torch.isnan(labels)
    keep = indices[cutoff : size - cutoff]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[keep] = True
    mask &= ~torch.isnan(labels)
    return mask


def _zscore(labels: torch.Tensor) -> torch.Tensor:
    std = labels.std(unbiased=False)
    if torch.isnan(std) or std <= 0:
        return labels - labels.mean()
    return (labels - labels.mean()) / std


def _daily_slices(index: pd.MultiIndex) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    if len(index) == 0:
        return slices
    dates = index.get_level_values("datetime")
    start = 0
    current = dates[0]
    for position, value in enumerate(dates[1:], start=1):
        if value != current:
            slices.append((start, position - 1))
            start = position
            current = value
    slices.append((start, len(index) - 1))
    return slices