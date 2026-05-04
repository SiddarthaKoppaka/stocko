from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from stock_manager.config import require_keys
from stock_manager.models.lightgbm_alpha158 import _prediction_metrics, _write_training_outputs
from stock_manager.utils.logging import get_logger, progress_iter

EPSILON = 1e-12
LOGGER = get_logger(__name__)


def train_stockmixer_model(config: dict) -> dict[str, Path]:
    require_keys(
        config,
        ["data.processed_path", "splits.train_end", "splits.valid_end", "paths.model_dir"],
        context="stockmixer config",
    )
    frame = pd.read_parquet(Path(config["data"]["processed_path"]))
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    show_progress = config.get("runtime", {}).get("show_progress", True)

    LOGGER.info("Preparing StockMixer arrays from %s", config["data"]["processed_path"])
    arrays = _build_stockmixer_arrays(frame, config)
    LOGGER.info(
        "StockMixer arrays ready: stocks=%s dates=%s lookback=%s",
        len(arrays["tickers"]),
        len(arrays["dates"]),
        arrays["lookback_length"],
    )
    bundle = _fit_stockmixer(arrays, config, show_progress=show_progress)
    predictions = _predict_stockmixer(bundle, arrays, show_progress=show_progress)
    metrics = _prediction_metrics(predictions)
    bundle["model"].to(torch.device("cpu"))
    LOGGER.info("StockMixer training complete: rank_ic=%.4f mse=%.6f", metrics["rank_ic"], metrics["mse"])
    return _write_training_outputs("stockmixer", config, bundle["model"], predictions, metrics)


acv = nn.GELU()


def get_loss(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    base_price: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32, device=device)
    return_ratio = (prediction - base_price) / (base_price + EPSILON)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = return_ratio @ all_one.t() - all_one @ return_ratio.t()
    gt_pw_dif = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.ln = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.ln(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout)
        return x


class TriU(nn.Module):
    def __init__(self, time_step: int):
        super().__init__()
        self.time_step = time_step
        self.tri_u = nn.ParameterList([nn.Linear(index + 1, 1) for index in range(time_step)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.tri_u[0](inputs[:, :, 0].unsqueeze(-1))
        for index in range(1, self.time_step):
            x = torch.cat([x, self.tri_u[index](inputs[:, :, : index + 1])], dim=-1)
        return x


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps: int, channels: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm([time_steps, channels])
        self.ln_2 = nn.LayerNorm([time_steps, channels])
        self.time_mixer = TriU(time_steps)
        self.channel_mixer = MixerBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.time_mixer(x)
        x = x.permute(0, 2, 1)
        x = self.ln_2(x + inputs)
        y = self.channel_mixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step: int, channel: int, scale_dim: int = 8):
        super().__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    def __init__(self, stocks: int, hidden_dim: int = 20):
        super().__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.permute(1, 0)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x.permute(1, 0)


class StockMixer(nn.Module):
    def __init__(self, stocks: int, time_steps: int, channels: int, market: int, scale: int):
        super().__init__()
        scale_dim = max(1, time_steps // 2)
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        self.stock_mixer = NoGraphMixer(stocks, market)
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        y = self.mixer(inputs, x)
        y = self.channel_fc(y).squeeze(-1)
        z = self.stock_mixer(y)
        y = self.time_fc(y)
        z = self.time_fc_(z)
        return y + z


def _build_stockmixer_arrays(frame: pd.DataFrame, config: dict) -> dict:
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"StockMixer requires OHLCV columns: {', '.join(sorted(missing))}")

    params = config.get("model", {}).get("params", {})
    normalized = _normalize_stockmixer_frame(
        frame,
        normalization_window=int(params.get("normalization_window", 20)),
    )
    dates = pd.Index(sorted(normalized["date"].unique()))
    tickers = sorted(normalized["ticker"].astype(str).unique())
    feature_columns = ["open_input", "high_input", "low_input", "close_input", "volume_input"]
    pivots = [
        normalized.pivot(index="date", columns="ticker", values=column).reindex(index=dates, columns=tickers)
        for column in feature_columns
    ]
    eod_data = np.stack([pivot.T.to_numpy(dtype=np.float32) for pivot in pivots], axis=-1)
    mask_data = (~np.isnan(eod_data).any(axis=2)).astype(np.float32)
    price_data = normalized.pivot(index="date", columns="ticker", values="close").reindex(index=dates, columns=tickers)
    price_data = price_data.T.to_numpy(dtype=np.float32)
    gt_data = np.full_like(price_data, np.nan, dtype=np.float32)
    steps = int(params.get("steps", 1))
    for row in range(steps, price_data.shape[1]):
        previous = price_data[:, row - steps]
        current = price_data[:, row]
        valid = (mask_data[:, row] > 0.5) & (mask_data[:, row - steps] > 0.5)
        gt_data[valid, row] = (current[valid] - previous[valid]) / (previous[valid] + EPSILON)
    eod_data = np.nan_to_num(eod_data, nan=0.0, posinf=0.0, neginf=0.0)

    train_end = pd.Timestamp(config["splits"]["train_end"])
    valid_end = pd.Timestamp(config["splits"]["valid_end"])
    test_end = pd.Timestamp(config["splits"].get("test_end", dates.max()))
    train_index = int(dates.searchsorted(train_end, side="right"))
    valid_index = int(dates.searchsorted(valid_end, side="right"))
    test_index = int(dates.searchsorted(test_end, side="right"))
    if test_index <= 0:
        test_index = len(dates)

    return {
        "eod_data": eod_data,
        "mask_data": mask_data,
        "price_data": price_data,
        "gt_data": gt_data,
        "dates": dates,
        "tickers": tickers,
        "train_index": train_index,
        "valid_index": valid_index,
        "test_index": test_index,
        "lookback_length": int(config.get("model", {}).get("params", {}).get("lookback_length", 16)),
        "steps": steps,
    }


def _normalize_stockmixer_frame(frame: pd.DataFrame, *, normalization_window: int) -> pd.DataFrame:
    if normalization_window <= 0:
        raise ValueError("StockMixer normalization_window must be positive")

    result = frame.copy()
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)

    close_anchor = result.groupby("ticker", group_keys=False)["close"].transform(
        lambda series: series.rolling(normalization_window, min_periods=1).mean().shift(1)
    )
    close_anchor = close_anchor.fillna(result["close"])
    volume_anchor = result.groupby("ticker", group_keys=False)["volume"].transform(
        lambda series: series.rolling(normalization_window, min_periods=1).mean().shift(1)
    )
    volume_anchor = volume_anchor.fillna(result["volume"])

    for column in ("open", "high", "low", "close"):
        result[f"{column}_input"] = ((result[column] / (close_anchor + EPSILON)) - 1.0).clip(-5.0, 5.0)
    result["volume_input"] = np.log1p(result["volume"] / (volume_anchor + EPSILON)).clip(-5.0, 5.0)
    return result


def _fit_stockmixer(arrays: dict, config: dict, *, show_progress: bool) -> dict:
    params = config.get("model", {}).get("params", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(params.get("seed", 12345678))
    np.random.seed(seed)
    torch.manual_seed(seed)

    stock_num = len(arrays["tickers"])
    model = StockMixer(
        stocks=stock_num,
        time_steps=int(params.get("lookback_length", arrays["lookback_length"])),
        channels=arrays["eod_data"].shape[2],
        market=int(params.get("market_hidden_dim", 20)),
        scale=int(params.get("scale_factor", 3)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("learning_rate", 0.001)))
    epochs = int(params.get("epochs", 100))
    alpha = float(params.get("alpha", 0.1))
    batch_offsets = np.arange(start=0, stop=arrays["train_index"], dtype=int)
    best_state = model.state_dict()
    best_valid_loss = float("inf")
    train_steps = max(0, arrays["train_index"] - arrays["lookback_length"] - arrays["steps"] + 1)
    LOGGER.info("StockMixer training started: epochs=%s, train_steps=%s", epochs, train_steps)

    for epoch_index in progress_iter(
        range(epochs),
        total=epochs,
        desc="StockMixer epochs",
        enabled=show_progress,
    ):
        np.random.shuffle(batch_offsets)
        upper = train_steps
        train_losses = []
        for j in progress_iter(
            range(max(0, upper)),
            total=max(0, upper),
            desc=f"StockMixer epoch {epoch_index + 1}/{epochs}",
            enabled=show_progress,
        ):
            offset = int(batch_offsets[j])
            data_batch, mask_batch, price_batch, gt_batch = [
                torch.tensor(item, dtype=torch.float32, device=device)
                for item in _get_batch(arrays, offset)
            ]
            optimizer.zero_grad()
            prediction = model(data_batch)
            loss, _, _, _ = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, alpha)
            if not torch.isfinite(loss):
                LOGGER.warning("Skipping non-finite StockMixer batch loss at offset=%s", offset)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        valid_loss, _ = _validate_stockmixer(model, arrays, arrays["train_index"], arrays["valid_index"], device, alpha)
        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        LOGGER.info(
            "StockMixer epoch %s/%s train_loss=%.6f valid_loss=%.6f",
            epoch_index + 1,
            epochs,
            mean_train_loss,
            valid_loss,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return {"model": model, "device": device, "alpha": alpha}


def _validate_stockmixer(
    model: nn.Module,
    arrays: dict,
    start_index: int,
    end_index: int,
    device: torch.device,
    alpha: float,
) -> tuple[float, dict]:
    if end_index <= start_index:
        return float("inf"), {}
    stock_num = len(arrays["tickers"])
    current_pred = np.zeros((stock_num, end_index - start_index), dtype=float)
    current_gt = np.zeros((stock_num, end_index - start_index), dtype=float)
    current_mask = np.zeros((stock_num, end_index - start_index), dtype=float)
    losses = []
    for current_offset in range(start_index - arrays["lookback_length"] - arrays["steps"] + 1, end_index - arrays["lookback_length"] - arrays["steps"] + 1):
        batch_items = [torch.tensor(item, dtype=torch.float32, device=device) for item in _get_batch(arrays, current_offset)]
        data_batch, mask_batch, price_batch, gt_batch = batch_items
        with torch.no_grad():
            prediction = model(data_batch)
            loss, _, _, return_ratio = get_loss(prediction, gt_batch, price_batch, mask_batch, stock_num, alpha)
        if not torch.isfinite(loss):
            continue
        losses.append(float(loss.item()))
        position = current_offset - (start_index - arrays["lookback_length"] - arrays["steps"] + 1)
        current_pred[:, position] = return_ratio[:, 0].detach().cpu().numpy()
        current_gt[:, position] = gt_batch[:, 0].detach().cpu().numpy()
        current_mask[:, position] = mask_batch[:, 0].detach().cpu().numpy()
    if not losses:
        return float("inf"), {}
    return float(np.mean(losses)), _evaluate_stockmixer(current_pred, current_gt, current_mask)


def _predict_stockmixer(bundle: dict, arrays: dict, *, show_progress: bool) -> pd.DataFrame:
    model = bundle["model"]
    device = bundle["device"]
    alpha = bundle["alpha"]
    start_index = arrays["valid_index"]
    end_index = arrays["test_index"]
    rows = []
    offsets = range(
        start_index - arrays["lookback_length"] - arrays["steps"] + 1,
        end_index - arrays["lookback_length"] - arrays["steps"] + 1,
    )
    total = max(0, end_index - start_index)
    for current_offset in progress_iter(
        offsets,
        total=total,
        desc="StockMixer inference",
        enabled=show_progress,
    ):
        batch_items = [torch.tensor(item, dtype=torch.float32, device=device) for item in _get_batch(arrays, current_offset)]
        data_batch, mask_batch, price_batch, gt_batch = batch_items
        with torch.no_grad():
            prediction = model(data_batch)
            _, _, _, return_ratio = get_loss(prediction, gt_batch, price_batch, mask_batch, len(arrays["tickers"]), alpha)
        target_date = arrays["dates"][current_offset + arrays["lookback_length"] + arrays["steps"] - 1]
        pred_np = return_ratio[:, 0].detach().cpu().numpy()
        gt_np = gt_batch[:, 0].detach().cpu().numpy()
        mask_np = mask_batch[:, 0].detach().cpu().numpy()
        for ticker, pred_value, gt_value, is_valid in zip(arrays["tickers"], pred_np, gt_np, mask_np):
            if is_valid < 0.5 or np.isnan(gt_value):
                continue
            rows.append(
                {
                    "date": pd.Timestamp(target_date),
                    "ticker": ticker,
                    "prediction": float(pred_value),
                    "label": float(gt_value),
                }
            )
    if not rows:
        raise ValueError("StockMixer test split is empty")
    return pd.DataFrame(rows)


def _get_batch(arrays: dict, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_len = arrays["lookback_length"]
    steps = arrays["steps"]
    mask_batch = arrays["mask_data"][:, offset : offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1, keepdims=True)
    return (
        arrays["eod_data"][:, offset : offset + seq_len, :],
        mask_batch,
        np.expand_dims(arrays["price_data"][:, offset + seq_len - 1], axis=1),
        np.expand_dims(arrays["gt_data"][:, offset + seq_len + steps - 1], axis=1),
    )


def _evaluate_stockmixer(prediction: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    df_pred = pd.DataFrame(prediction * mask)
    df_gt = pd.DataFrame(ground_truth * mask)
    ic = []
    sharpe_li5 = []
    prec_10 = []
    for column in range(prediction.shape[1]):
        ic.append(df_pred[column].corr(df_gt[column]))
        rank_pre = np.argsort(prediction[:, column])
        pre_top10 = []
        for rank in range(1, prediction.shape[0] + 1):
            current = rank_pre[-rank]
            if mask[current][column] < 0.5:
                continue
            if len(pre_top10) < 10:
                pre_top10.append(current)
        if not pre_top10:
            continue
        real_ret_rat_top10 = np.mean([ground_truth[index][column] for index in pre_top10])
        sharpe_li5.append(real_ret_rat_top10)
        prec_10.append(np.mean([ground_truth[index][column] >= 0 for index in pre_top10]))
    sharpe_arr = np.array(sharpe_li5) if sharpe_li5 else np.array([0.0])
    return {
        "IC": float(np.nanmean(ic)) if ic else 0.0,
        "RIC": float(np.nanmean(ic) / (np.nanstd(ic) + EPSILON)) if ic else 0.0,
        "sharpe5": float((np.mean(sharpe_arr) / (np.std(sharpe_arr) + EPSILON)) * 15.87),
        "prec_10": float(np.mean(prec_10)) if prec_10 else 0.0,
    }