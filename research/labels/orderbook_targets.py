# flake8: noqa
"""
Horizon-based labeling for order book modeling.

Creates:
- future move in ticks for mid / best bid / best ask
- raw directional target in {-1, 0, +1}
- tick-threshold directional targets (binary)

All labels are computed using future states only (shift(-k) or nearest-forward).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonAlignment:
    method: Literal["shift", "nearest_forward"]
    # mapping from requested horizon_ms -> aligned horizon_ms (for naming / reporting)
    aligned_horizon_ms: dict[int, int]
    # mapping from requested horizon_ms -> aligned step in grid points
    aligned_steps: dict[int, int]
    # counts where nearest-forward used a strictly later timestamp than requested
    non_exact_alignment_counts: dict[int, int]


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _infer_alignment_shift_if_grid(
    df: pd.DataFrame,
    *,
    ts_col: str,
    grid_ms: int,
    sample_n: int = 200_000,
) -> bool:
    if ts_col not in df.columns:
        return False
    if len(df) < 2:
        return False

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    ts = ts.dropna()
    if len(ts) < 2:
        return False

    # Work on a sample to keep it fast.
    if len(ts) > sample_n:
        ts = ts.iloc[:: max(1, len(ts) // sample_n)]

    ts_ms = ts.values.astype("datetime64[ms]").astype("int64")
    diffs = ts_ms[1:] - ts_ms[:-1]
    if diffs.size == 0:
        return False

    ratio_exact = float((diffs == int(grid_ms)).mean())
    # Must be almost always exact to trust shift-based labels.
    return ratio_exact >= 0.999


def _build_nearest_forward_future_values(
    ts: np.ndarray,
    values: np.ndarray,
    horizon_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each t_i find first index j such that ts_j >= ts_i + horizon_ms.
    Returns (future_values, exact_mask) where exact_mask means ts_j == ts_i + horizon_ms.
    """
    target = ts + int(horizon_ms)
    idx = np.searchsorted(ts, target, side="left")
    n = ts.shape[0]
    future = np.full(n, np.nan, dtype=np.float64)
    exact = np.zeros(n, dtype=bool)

    valid = idx < n
    future[valid] = values[idx[valid]]
    exact[valid] = ts[idx[valid]] == target[valid]
    return future, exact


def build_horizon_tick_direction_targets(
    df: pd.DataFrame,
    *,
    tick_size: float,
    grid_ms: int,
    horizons_ms: list[int],
    tick_thresholds: tuple[int, ...] = (1, 2, 3),
    ts_col: str = "ts_event",
    alignment_method: Literal["auto", "shift", "nearest_forward"] = "auto",
) -> tuple[pd.DataFrame, HorizonAlignment]:
    """
    Build horizon targets for tick-directionality.
    """
    required = ["mid_px", "bid_px_01", "ask_px_01", ts_col]
    _require_cols(df, required)

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col, "mid_px", "bid_px_01", "ask_px_01"]).sort_values(ts_col).reset_index(drop=True)

    ts_ms = df[ts_col].values.astype("datetime64[ms]").astype("int64")
    mid = df["mid_px"].to_numpy(dtype=np.float64)
    bid = df["bid_px_01"].to_numpy(dtype=np.float64)
    ask = df["ask_px_01"].to_numpy(dtype=np.float64)

    if alignment_method == "auto":
        use_shift = _infer_alignment_shift_if_grid(df, ts_col=ts_col, grid_ms=grid_ms)
        method = "shift" if use_shift else "nearest_forward"
    else:
        method = alignment_method

    aligned_steps: dict[int, int] = {}
    aligned_horizon_ms: dict[int, int] = {}
    non_exact_alignment_counts: dict[int, int] = {}

    max_step = 0
    for h in horizons_ms:
        step = int(math.ceil(h / grid_ms))
        aligned_steps[h] = step
        aligned_horizon_ms[h] = step * grid_ms
        max_step = max(max_step, step)
        non_exact_alignment_counts[h] = 0

    out = pd.DataFrame(index=df.index)

    # Common helper: compute future arrays either via shift or nearest-forward.
    for h in horizons_ms:
        step = aligned_steps[h]
        horizon_tag = f"h{h}"
        aligned_ms = aligned_horizon_ms[h]

        if method == "shift":
            future_mid = pd.Series(mid).shift(-step).to_numpy(dtype=np.float64)
            future_bid = pd.Series(bid).shift(-step).to_numpy(dtype=np.float64)
            future_ask = pd.Series(ask).shift(-step).to_numpy(dtype=np.float64)
            # Exactness is guaranteed for shift-based alignment on a perfect grid.
            non_exact_alignment_counts[h] = 0
        elif method == "nearest_forward":
            future_mid, exact_mid = _build_nearest_forward_future_values(ts_ms, mid, horizon_ms=h)
            future_bid, exact_bid = _build_nearest_forward_future_values(ts_ms, bid, horizon_ms=h)
            future_ask, exact_ask = _build_nearest_forward_future_values(ts_ms, ask, horizon_ms=h)
            # Count non-exact alignments where future timestamp > requested.
            # If any of mid/bid/ask is non-exact, we count it (same idx, so it's consistent).
            non_exact = ~(exact_mid)
            non_exact_alignment_counts[h] = int(np.sum(non_exact & np.isfinite(future_mid)))
        else:
            raise ValueError(f"Unknown alignment method: {method}")

        # tick moves (required)
        mid_move_ticks = (future_mid - mid) / tick_size
        bid_move_ticks = (future_bid - bid) / tick_size
        ask_move_ticks = (future_ask - ask) / tick_size

        out[f"mid_move_ticks_{horizon_tag}"] = mid_move_ticks
        out[f"best_bid_move_ticks_{horizon_tag}"] = bid_move_ticks
        out[f"best_ask_move_ticks_{horizon_tag}"] = ask_move_ticks

        # raw directional: {-1,0,1} using 0.5 tick as "flat" threshold
        flat_thr_ticks = 0.5
        out[f"target_mid_dir_{horizon_tag}"] = np.where(
            mid_move_ticks > flat_thr_ticks,
            1,
            np.where(mid_move_ticks < -flat_thr_ticks, -1, 0),
        ).astype(np.int8)

        # tick-threshold binary targets
        for n in tick_thresholds:
            out[f"target_mid_ge_{n}tick_{horizon_tag}"] = (mid_move_ticks >= float(n)).astype(np.int8)
            out[f"target_mid_le_minus_{n}tick_{horizon_tag}"] = (mid_move_ticks <= -float(n)).astype(np.int8)

    alignment = HorizonAlignment(
        method=method, aligned_horizon_ms=aligned_horizon_ms, aligned_steps=aligned_steps, non_exact_alignment_counts=non_exact_alignment_counts
    )

    return out, alignment

