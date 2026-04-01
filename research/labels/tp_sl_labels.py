# flake8: noqa
"""
TP/SL labeling for maker-like scenarios.

We evaluate TP/SL on the mid-price excursion relative to entry price:
- Long entry: current best_bid_px_01
- Short entry: current best_ask_px_01

For each observation, over a finite holding window [t, t + holding_ms]:
- if TP is hit before SL -> +1
- if SL is hit before TP -> -1
- otherwise -> 0 (timeout / neither)

All labels are future-looking by construction, but computed only from mid values
strictly after observation time on the reconstruction grid.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _first_hit_step_from_bool(cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    cond: boolean matrix of shape (n, K), where cond[i, k] means "hit at step k+1".
    Returns (first_step, any_hit):
    - first_step: float array (n,) with NaN if never hit, else 1..K
    - any_hit: boolean array (n,)
    """
    any_hit = cond.any(axis=1)
    first_step = np.full(cond.shape[0], np.nan, dtype=np.float64)
    if np.any(any_hit):
        # argmax gives first True index for boolean arrays
        first_step[any_hit] = np.argmax(cond[any_hit], axis=1) + 1
    return first_step, any_hit


def build_tp_sl_labels(
    df: pd.DataFrame,
    *,
    tick_size: float,
    grid_ms: int,
    take_profit_ticks: tuple[int, ...] = (1, 2, 3),
    stop_loss_ticks: tuple[int, ...] = (1, 2),
    holding_ms: tuple[int, ...] = (100, 250, 500, 1000),
    alignment_method: Literal["shift"] = "shift",
) -> pd.DataFrame:
    if alignment_method != "shift":
        raise NotImplementedError("tp_sl_labels currently implemented only for shift-based alignment on regular grids")

    _require_cols(df, ["mid_px", "bid_px_01", "ask_px_01"])

    df = df.copy().reset_index(drop=True)

    mid = df["mid_px"].to_numpy(dtype=np.float64)
    best_bid = df["bid_px_01"].to_numpy(dtype=np.float64)
    best_ask = df["ask_px_01"].to_numpy(dtype=np.float64)

    # Convert holding windows to steps (ceil because exact timestamp might not exist).
    holding_steps = {h: int(math.ceil(h / grid_ms)) for h in holding_ms}
    max_step = int(max(holding_steps.values()))

    # Precompute future mid matrix for steps 1..max_step:
    # future_mid_mat[i, k-1] = mid[i+k] if available else NaN
    n = mid.shape[0]
    future_mid_mat = np.full((n, max_step), np.nan, dtype=np.float32)
    for k in range(1, max_step + 1):
        if k >= n:
            break
        future_mid_mat[: n - k, k - 1] = mid[k:].astype(np.float32)

    out = pd.DataFrame(index=df.index)

    # Long side (entry at bid): TP when future_mid >= entry + tp*tick, SL when <= entry - sl*tick.
    long_entry = best_bid
    for tp in take_profit_ticks:
        tp_prices = long_entry + float(tp) * tick_size
        for sl in stop_loss_ticks:
            sl_prices = long_entry - float(sl) * tick_size

            # Boolean matrices: shape (n, max_step)
            cond_tp = np.isfinite(future_mid_mat) & (future_mid_mat >= tp_prices[:, None].astype(np.float32))
            cond_sl = np.isfinite(future_mid_mat) & (future_mid_mat <= sl_prices[:, None].astype(np.float32))

            first_tp_step, _ = _first_hit_step_from_bool(cond_tp)
            first_sl_step, _ = _first_hit_step_from_bool(cond_sl)

            for hold_ms_val, hold_step in holding_steps.items():
                tp_within = np.isfinite(first_tp_step) & (first_tp_step <= hold_step)
                sl_within = np.isfinite(first_sl_step) & (first_sl_step <= hold_step)

                label = np.zeros(n, dtype=np.int8)
                both = tp_within & sl_within
                only_tp = tp_within & ~sl_within
                only_sl = sl_within & ~tp_within

                label[only_tp] = 1
                label[only_sl] = -1
                if np.any(both):
                    label[both] = np.where(first_tp_step[both] < first_sl_step[both], 1, -1).astype(np.int8)

                col = f"label_tp{tp}_sl{sl}_hold{hold_ms_val}_long"
                out[col] = label

    # Short side (entry at ask): TP when future_mid <= entry - tp*tick, SL when >= entry + sl*tick.
    short_entry = best_ask
    for tp in take_profit_ticks:
        tp_prices = short_entry - float(tp) * tick_size
        for sl in stop_loss_ticks:
            sl_prices = short_entry + float(sl) * tick_size

            cond_tp = np.isfinite(future_mid_mat) & (future_mid_mat <= tp_prices[:, None].astype(np.float32))
            cond_sl = np.isfinite(future_mid_mat) & (future_mid_mat >= sl_prices[:, None].astype(np.float32))

            first_tp_step, _ = _first_hit_step_from_bool(cond_tp)
            first_sl_step, _ = _first_hit_step_from_bool(cond_sl)

            for hold_ms_val, hold_step in holding_steps.items():
                tp_within = np.isfinite(first_tp_step) & (first_tp_step <= hold_step)
                sl_within = np.isfinite(first_sl_step) & (first_sl_step <= hold_step)

                label = np.zeros(n, dtype=np.int8)
                both = tp_within & sl_within
                only_tp = tp_within & ~sl_within
                only_sl = sl_within & ~tp_within

                label[only_tp] = 1
                label[only_sl] = -1
                if np.any(both):
                    label[both] = np.where(first_tp_step[both] < first_sl_step[both], 1, -1).astype(np.int8)

                col = f"label_tp{tp}_sl{sl}_hold{hold_ms_val}_short"
                out[col] = label

    return out

