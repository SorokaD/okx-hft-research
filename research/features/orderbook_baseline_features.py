# flake8: noqa
"""
Baseline feature engineering for order book modeling.

This module creates a "clean" and "honest" feature set from reconstructed order book
snapshots on a fixed grid.

Leakage note:
- All features must use only current and past information.
- This module uses only `shift(+k)` and rolling windows on past data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from research.features.imbalance import compute_l1_imbalance, compute_l10_imbalance
from research.features.microprice import compute_microprice
from research.features.spread import compute_relative_spread


@dataclass(frozen=True)
class LookbackSpec:
    # lookback windows in milliseconds (e.g., 100, 200, 500, 1000)
    lookback_ms: tuple[int, ...]


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _top_volume(df: pd.DataFrame, side: str, k: int) -> pd.Series:
    cols = [f"{side}_sz_{i:02d}" for i in range(1, k + 1)]
    _require_cols(df, cols)
    return df[cols].sum(axis=1)


def _top_imbalance(df: pd.DataFrame, k: int) -> pd.Series:
    bid = _top_volume(df, "bid", k)
    ask = _top_volume(df, "ask", k)
    denom = (bid + ask).replace(0, np.nan)
    return (bid - ask) / denom


def _cumulative_depth(df: pd.DataFrame, side: str, k: int) -> pd.Series:
    cols = [f"{side}_sz_{i:02d}" for i in range(1, k + 1)]
    _require_cols(df, cols)
    return df[cols].cumsum(axis=1).iloc[:, -1]


def build_orderbook_baseline_features(
    df: pd.DataFrame,
    *,
    grid_ms: int = 100,
    tick_size: float,
    lookback_ms: tuple[int, ...] = (100, 200, 500, 1000),
    depth_k: int = 10,
) -> pd.DataFrame:
    """
    Build baseline features at time t.

    Required columns (minimum):
    - ts_event, mid_px, spread_px
    - bid_px_01..bid_px_{depth_k}, bid_sz_01..bid_sz_{depth_k}
    - ask_px_01..ask_px_{depth_k}, ask_sz_01..ask_sz_{depth_k}

    Returns:
    - DataFrame with feature columns only (ts_event preserved if present in input).
    """
    required = [
        "mid_px",
        "spread_px",
    ]
    for side in ("bid", "ask"):
        for i in range(1, depth_k + 1):
            required.append(f"{side}_px_{i:02d}")
            required.append(f"{side}_sz_{i:02d}")

    _require_cols(df, required)

    out = pd.DataFrame(index=df.index)
    if "ts_event" in df.columns:
        out["ts_event"] = df["ts_event"]
    if "inst_id" in df.columns:
        out["inst_id"] = df["inst_id"]
    if "anchor_snapshot_ts" in df.columns:
        out["anchor_snapshot_ts"] = df["anchor_snapshot_ts"]
    if "reconstruction_mode" in df.columns:
        out["reconstruction_mode"] = df["reconstruction_mode"]

    out["tick_size"] = float(tick_size)

    best_bid_px = df["bid_px_01"]
    best_ask_px = df["ask_px_01"]
    best_bid_sz = df["bid_sz_01"]
    best_ask_sz = df["ask_sz_01"]

    out["best_bid_px"] = best_bid_px
    out["best_ask_px"] = best_ask_px
    out["best_bid_sz"] = best_bid_sz
    out["best_ask_sz"] = best_ask_sz

    # Core book state
    out["mid_px"] = df["mid_px"]
    out["spread_px"] = df["spread_px"]
    out["mid_to_best_bid_ticks"] = (out["mid_px"] - best_bid_px) / tick_size
    out["best_ask_to_mid_ticks"] = (best_ask_px - out["mid_px"]) / tick_size
    out["relative_spread"] = compute_relative_spread(df)
    out["spread_bps"] = out["relative_spread"] * 10_000.0

    # Volumes top-k
    for k in (1, 3, 5, 10):
        out[f"bid_vol_top{k}"] = _top_volume(df, "bid", k)
        out[f"ask_vol_top{k}"] = _top_volume(df, "ask", k)

    # Imbalance top-k
    for k in (1, 3, 5, 10):
        out[f"imbalance_top{k}"] = _top_imbalance(df, k)

    # Microprice
    out["microprice"] = compute_microprice(df)
    out["microprice_minus_mid_ticks"] = (out["microprice"] - out["mid_px"]) / tick_size

    # Shape features: distances of each level from mid (in ticks) and cumulative depth
    for i in range(1, depth_k + 1):
        out[f"bid_level_{i:02d}_dist_ticks"] = (df[f"bid_px_{i:02d}"] - out["mid_px"]) / tick_size
        out[f"ask_level_{i:02d}_dist_ticks"] = (out["mid_px"] - df[f"ask_px_{i:02d}"]) / tick_size

    # cumulative depth at each i (last cumulative value at i)
    for i in range(1, depth_k + 1):
        out[f"cum_bid_sz_{i:02d}"] = df[[f"bid_sz_{j:02d}" for j in range(1, i + 1)]].sum(axis=1)
        out[f"cum_ask_sz_{i:02d}"] = df[[f"ask_sz_{j:02d}" for j in range(1, i + 1)]].sum(axis=1)

    # Asymmetry between bid and ask depth at full depth_k
    bid_depth = out[f"cum_bid_sz_{depth_k:02d}"]
    ask_depth = out[f"cum_ask_sz_{depth_k:02d}"]
    denom = (bid_depth + ask_depth).replace(0, np.nan)
    out["depth_asymmetry_top10"] = (bid_depth - ask_depth) / denom

    # Short-history features (past only)
    # Convert ms lookbacks into steps. Since the data is on a fixed 100ms grid, we map:
    # lookback_steps = ceil(lookback_ms / grid_ms)
    for lb_ms in lookback_ms:
        steps = int(math.ceil(lb_ms / grid_ms))
        tag = f"h{lb_ms}"

        # returns
        out[f"mid_ret_{tag}"] = out["mid_px"] / out["mid_px"].shift(steps) - 1.0
        out[f"spread_change_{tag}"] = out["spread_px"] - out["spread_px"].shift(steps)
        out[f"spread_change_ticks_{tag}"] = out[f"spread_change_{tag}"] / tick_size

        # imbalance changes
        out[f"imbalance_top1_change_{tag}"] = out["imbalance_top1"].shift(0) - out["imbalance_top1"].shift(steps)
        out[f"imbalance_top10_change_{tag}"] = out["imbalance_top10"].shift(0) - out["imbalance_top10"].shift(steps)

        # top sizes changes
        out[f"best_bid_sz_change_{tag}"] = out["best_bid_sz"] - out["best_bid_sz"].shift(steps)
        out[f"best_ask_sz_change_{tag}"] = out["best_ask_sz"] - out["best_ask_sz"].shift(steps)

        # rolling volatility & realized range over the same number of steps
        # volatility: std of mid_ret over last window (past only)
        # realized range: (max-min)/tick_size over last window
        roll_mid = out["mid_px"]
        out[f"mid_vol_{tag}"] = roll_mid.pct_change().rolling(window=steps, min_periods=steps).std()
        out[f"mid_realized_range_ticks_{tag}"] = (roll_mid.rolling(window=steps, min_periods=steps).max()
                                                     - roll_mid.rolling(window=steps, min_periods=steps).min()) / tick_size

        # count of mid changes in the last window
        mid_changed = (roll_mid.diff() != 0).astype(np.float64)
        out[f"mid_change_count_{tag}"] = mid_changed.rolling(window=steps, min_periods=steps).sum()

    return out

