"""Indicator helpers with optional torch backend."""

from __future__ import annotations

import numpy as np
import talib


def _to_numpy(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def compute_sma(values, period: int, backend: str = "cpu") -> np.ndarray:
    """Compute SMA using the selected backend."""
    series = _to_numpy(values)
    if backend == "torch":
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.as_tensor(series, dtype=torch.float64, device=device)
            if tensor.numel() < period:
                return np.full(series.shape, np.nan)
            kernel = torch.ones(period, dtype=torch.float64, device=device) / float(period)
            sma = torch.nn.functional.conv1d(
                tensor.view(1, 1, -1),
                kernel.view(1, 1, -1),
                padding=0,
            ).view(-1)
            out = torch.full((tensor.numel(),), torch.nan, dtype=torch.float64, device=device)
            out[period - 1 :] = sma
            return out.cpu().numpy()
        except Exception:
            return talib.SMA(series, timeperiod=period)
    return talib.SMA(series, timeperiod=period)


def compute_rsi(values, period: int, backend: str = "cpu") -> np.ndarray:
    """Compute RSI using the selected backend."""
    series = _to_numpy(values)
    if backend == "torch":
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.as_tensor(series, dtype=torch.float64, device=device)
            if tensor.numel() < period + 1:
                return np.full(series.shape, np.nan)

            delta = tensor[1:] - tensor[:-1]
            gain = torch.clamp(delta, min=0.0)
            loss = torch.clamp(-delta, min=0.0)

            avg_gain = torch.zeros_like(tensor)
            avg_loss = torch.zeros_like(tensor)
            avg_gain[period] = gain[:period].mean()
            avg_loss[period] = loss[:period].mean()

            alpha = 1.0 / float(period)
            for idx in range(period + 1, tensor.numel()):
                avg_gain[idx] = (1 - alpha) * avg_gain[idx - 1] + alpha * gain[idx - 1]
                avg_loss[idx] = (1 - alpha) * avg_loss[idx - 1] + alpha * loss[idx - 1]

            rs = torch.where(avg_loss == 0, torch.full_like(avg_loss, np.inf), avg_gain / avg_loss)
            rsi = 100 - (100 / (1 + rs))
            rsi[:period] = torch.nan
            return rsi.cpu().numpy()
        except Exception:
            return talib.RSI(series, timeperiod=period)
    return talib.RSI(series, timeperiod=period)
