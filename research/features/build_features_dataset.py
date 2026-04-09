# flake8: noqa
"""Build leakage-safe feature datasets from reconstructed order book data.

All dynamic features use configurable wall-clock lookback windows (``windows_sec``),
defaulting to ``[1, 3, 5, 10, 30]`` seconds. Pass ``--windows-sec 1,3,...,3600`` on the
CLI to extend horizons (e.g. up to one hour) without changing targets.

Leakage-safe rule (unchanged): for row ``i`` at ``ts_event[i]``, every feature uses only
rows ``j`` with ``ts_event[j] <= ts_event[i]``. Past boundaries are resolved via
``searchsorted`` on sorted event times; rolling sums use prefix arrays (O(n) per window).

**Note on long windows:** ``realized_vol_{w}s`` is the std of *per-update* mid changes
(tick steps) inside the window, not calendar-time sampled returns. For very large ``w``
you aggregate many micro-updates — the estimate is still causal but can be dominated
by microstructure noise; ``volatility_regime`` uses qcut on the chosen vol column, which
may collapse buckets if variation is low (handled in ``_bucketize_quantiles``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.utils.tick_size import infer_tick_size

LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOWS_SEC: list[int] = [1, 3, 5, 10, 30]

BASE_COLUMNS = [
    "ts_event",
    "inst_id",
    "anchor_snapshot_ts",
    "reconstruction_mode",
    "mid_px",
    "spread_px",
]

BOOK_COLUMNS = [
    *[f"bid_px_{i:02d}" for i in range(1, 11)],
    *[f"bid_sz_{i:02d}" for i in range(1, 11)],
    *[f"ask_px_{i:02d}" for i in range(1, 11)],
    *[f"ask_sz_{i:02d}" for i in range(1, 11)],
]

_DEPTH_FEATURE_NAMES: list[str] = [
    *[f"bid_dist_{i}_ticks" for i in range(1, 11)],
    *[f"ask_dist_{i}_ticks" for i in range(1, 11)],
    *[f"bid_sz_{i}_norm" for i in range(1, 11)],
    *[f"ask_sz_{i}_norm" for i in range(1, 11)],
    "cum_bid_sz_1",
    "cum_bid_sz_3",
    "cum_bid_sz_5",
    "cum_bid_sz_10",
    "cum_ask_sz_1",
    "cum_ask_sz_3",
    "cum_ask_sz_5",
    "cum_ask_sz_10",
    "cum_bid_sz_1_over_10",
    "cum_bid_sz_3_over_10",
    "cum_bid_sz_5_over_10",
    "cum_ask_sz_1_over_10",
    "cum_ask_sz_3_over_10",
    "cum_ask_sz_5_over_10",
    "book_slope_bid",
    "book_slope_ask",
    "book_convexity_bid",
    "book_convexity_ask",
]

_REGIME_FEATURE_NAMES: list[str] = [
    "spread_regime",
    "volatility_regime",
    "update_regime",
    "seconds_since_large_move",
]


def feature_groups_for_windows(windows_sec: list[int]) -> dict[str, list[str]]:
    """Column names per group for summarization (matches columns emitted by builders)."""

    price: list[str] = [
        "spread_ticks",
        "relative_spread",
        *[f"mid_return_{w}s" for w in windows_sec],
        *[f"mid_velocity_{w}s" for w in windows_sec],
        "mid_acceleration_1s",
        *[f"realized_vol_{w}s" for w in windows_sec],
    ]
    imbalance: list[str] = [
        "imbalance_1",
        "imbalance_3",
        "imbalance_5",
        "imbalance_10",
        *[
            f"imbalance_{k}_delta_{w}s"
            for w in windows_sec
            for k in (1, 3, 5, 10)
        ],
        "imbalance_5_acceleration",
    ]
    order_flow: list[str] = []
    for w in windows_sec:
        order_flow.extend(
            [
                f"bid_size_added_{w}s",
                f"ask_size_added_{w}s",
                f"bid_size_removed_{w}s",
                f"ask_size_removed_{w}s",
                f"net_bid_flow_{w}s",
                f"net_ask_flow_{w}s",
                f"order_flow_imbalance_{w}s",
            ]
        )
    order_flow.extend([f"update_intensity_{w}s" for w in windows_sec])

    return {
        "price": price,
        "depth": list(_DEPTH_FEATURE_NAMES),
        "imbalance": imbalance,
        "order_flow": order_flow,
        "regime": list(_REGIME_FEATURE_NAMES),
    }


FEATURE_GROUPS: dict[str, list[str]] = feature_groups_for_windows(DEFAULT_WINDOWS_SEC)


def parse_windows_sec(value: str | None) -> list[int]:
    """Parse comma-separated positive integers (e.g. ``'1,3,5,10,30'``)."""

    if value is None or not str(value).strip():
        return list(DEFAULT_WINDOWS_SEC)
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        return list(DEFAULT_WINDOWS_SEC)
    try:
        out = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"--windows-sec must be comma-separated integers, got {value!r}"
        ) from exc
    return validate_windows_sec(out)


def validate_windows_sec(windows: list[int]) -> list[int]:
    """Require unique positive integers; return sorted ascending."""

    if not windows:
        raise ValueError("windows_sec must be non-empty")
    for w in windows:
        if not isinstance(w, (int, np.integer)):
            raise TypeError(f"window must be int, got {type(w).__name__}: {w!r}")
        if int(w) <= 0:
            raise ValueError(f"all windows_sec must be positive, got {w!r}")
    w_int = sorted({int(x) for x in windows})
    if len(w_int) != len(windows):
        raise ValueError(f"windows_sec must be unique, got {windows!r}")
    return w_int


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num_arr = np.asarray(num, dtype=np.float64)
    den_arr = np.asarray(den, dtype=np.float64)
    num_b, den_b = np.broadcast_arrays(num_arr, den_arr)
    out = np.full(num_b.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(num_b) & np.isfinite(den_b) & (den_b != 0)
    np.divide(num_b, den_b, out=out, where=valid)
    return out


def _prepare_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort and validate the input feature dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = ["ts_event", "mid_px", "spread_px", *BOOK_COLUMNS]
    _require_columns(df, required)

    work = df.copy()
    work["ts_event"] = pd.to_datetime(work["ts_event"], utc=True, errors="coerce")
    if "anchor_snapshot_ts" in work.columns:
        work["anchor_snapshot_ts"] = pd.to_datetime(
            work["anchor_snapshot_ts"], utc=True, errors="coerce"
        )

    numeric_cols = ["mid_px", "spread_px", *BOOK_COLUMNS]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["ts_event", "mid_px", "spread_px"]).sort_values(
        "ts_event"
    ).reset_index(drop=True)

    if work.empty:
        raise ValueError("Input DataFrame is empty after cleaning")
    if not work["ts_event"].is_monotonic_increasing:
        raise ValueError("ts_event must be monotonic increasing after sorting")

    return work


def _compute_past_indices(
    ts_ns: np.ndarray,
    windows_sec: list[int],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return start indices, validity masks, and elapsed seconds for each window."""

    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for seconds in windows_sec:
        target = ts_ns - np.int64(seconds) * np.int64(1_000_000_000)
        start_idx = np.searchsorted(ts_ns, target, side="left").astype(np.int64)
        valid = target >= ts_ns[0]
        elapsed_sec = np.full(ts_ns.shape[0], np.nan, dtype=np.float64)
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size:
            elapsed_sec[valid_idx] = (
                ts_ns[valid_idx] - ts_ns[start_idx[valid_idx]]
            ) / 1e9
        out[seconds] = (start_idx, valid, elapsed_sec)
    return out


def _log_window_validity_stats(n_rows: int, past_index: dict[int, Any], windows_sec: list[int]) -> None:
    """Rows where clock lookback crosses start of series (features legitimately NaN)."""

    for w in windows_sec:
        _, valid, _ = past_index[w]
        n_ok = int(np.sum(valid))
        n_bad = n_rows - n_ok
        pct = 100.0 * n_bad / n_rows if n_rows else 0.0
        LOGGER.info(
            "Window %ds: rows without full clock-history boundary: %d / %d (%.2f%%)",
            w,
            n_bad,
            n_rows,
            pct,
        )


def _values_at_indices(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return values[indices]


def _window_sum_from_prefix(
    prefix: np.ndarray,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    out = np.full(start_idx.shape[0], np.nan, dtype=np.float64)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return out
    left = start_idx[valid_idx]
    right = end_idx[valid_idx]
    out[valid_idx] = prefix[right + 1] - prefix[left]
    return out


def _window_step_std(
    values: np.ndarray,
    start_idx: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Std of one-step values over the past window [start_idx + 1, i]."""

    n = values.shape[0]
    prefix = np.zeros(n + 1, dtype=np.float64)
    prefix_sq = np.zeros(n + 1, dtype=np.float64)
    prefix[1:] = np.cumsum(values)
    prefix_sq[1:] = np.cumsum(values * values)

    out = np.full(n, np.nan, dtype=np.float64)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return out

    left = start_idx[valid_idx] + 1
    right = valid_idx
    count = right - start_idx[valid_idx]
    enough = count >= 2
    if not enough.any():
        return out

    idx2 = valid_idx[enough]
    left2 = left[enough]
    right2 = right[enough]
    count2 = count[enough].astype(np.float64)

    sums = prefix[right2 + 1] - prefix[left2]
    sums_sq = prefix_sq[right2 + 1] - prefix_sq[left2]
    mean = sums / count2
    var = (sums_sq - count2 * mean * mean) / (count2 - 1.0)
    out[idx2] = np.sqrt(np.maximum(var, 0.0))
    return out


def _bucketize_quantiles(values: pd.Series, q: int = 3) -> pd.Series:
    """Quantile buckets as integer codes; missing values stay missing."""

    bucket = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.notna()
    if valid.sum() == 0:
        return bucket

    labels = list(range(q))
    try:
        bucket.loc[valid] = pd.qcut(
            values.loc[valid],
            q=q,
            labels=labels,
            duplicates="drop",
        ).astype("Int64")
    except ValueError:
        bucket.loc[valid] = 0
    return bucket


def _compute_book_shape_features(
    dist_ticks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate slope and convexity of one side of the book."""

    x = np.arange(1, dist_ticks.shape[1] + 1, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = np.sum(x_centered * x_centered)

    y_mean = np.nanmean(dist_ticks, axis=1)
    slope = np.nansum((dist_ticks - y_mean[:, None]) * x_centered[None, :], axis=1) / x_var
    convexity = np.nanmean(np.diff(dist_ticks, n=2, axis=1), axis=1)
    return slope.astype(np.float64), convexity.astype(np.float64)


def build_price_features(
    df: pd.DataFrame,
    *,
    tick_size: float,
    past_index: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    windows_sec: list[int],
) -> pd.DataFrame:
    """Build price/spread/volatility features without future leakage."""

    out = pd.DataFrame(index=df.index)
    mid = df["mid_px"].to_numpy(dtype=np.float64)
    spread = df["spread_px"].to_numpy(dtype=np.float64)

    out["spread_ticks"] = spread / float(tick_size)
    out["relative_spread"] = _safe_divide(spread, mid)

    for seconds in windows_sec:
        start_idx, valid, elapsed_sec = past_index[seconds]
        past_mid = _values_at_indices(mid, start_idx)
        ret_ticks = np.full(len(df), np.nan, dtype=np.float64)
        ret_ticks[valid] = (mid[valid] - past_mid[valid]) / float(tick_size)
        out[f"mid_return_{seconds}s"] = ret_ticks
        out[f"mid_velocity_{seconds}s"] = _safe_divide(ret_ticks, elapsed_sec)

    out["mid_acceleration_1s"] = np.nan
    if 1 in past_index:
        velocity_1s = out["mid_velocity_1s"].to_numpy(dtype=np.float64)
        start_idx_1s, valid_1s, elapsed_1s = past_index[1]
        prev_velocity = _values_at_indices(velocity_1s, start_idx_1s)
        accel = np.full(len(df), np.nan, dtype=np.float64)
        valid_acc = valid_1s & np.isfinite(prev_velocity) & np.isfinite(velocity_1s)
        accel[valid_acc] = (velocity_1s[valid_acc] - prev_velocity[valid_acc]) / elapsed_1s[
            valid_acc
        ]
        out["mid_acceleration_1s"] = accel

    mid_step_returns = np.zeros(len(df), dtype=np.float64)
    mid_step_returns[1:] = np.diff(mid) / float(tick_size)
    for seconds in windows_sec:
        start_idx, valid, _ = past_index[seconds]
        out[f"realized_vol_{seconds}s"] = _window_step_std(mid_step_returns, start_idx, valid)

    return out


def build_depth_features(
    df: pd.DataFrame,
    *,
    tick_size: float,
) -> pd.DataFrame:
    """Build normalized depth, distance, and shape features from the book."""

    out = pd.DataFrame(index=df.index)
    mid = df["mid_px"].to_numpy(dtype=np.float64)

    bid_px = np.column_stack([df[f"bid_px_{i:02d}"].to_numpy(dtype=np.float64) for i in range(1, 11)])
    ask_px = np.column_stack([df[f"ask_px_{i:02d}"].to_numpy(dtype=np.float64) for i in range(1, 11)])
    bid_sz = np.column_stack([df[f"bid_sz_{i:02d}"].to_numpy(dtype=np.float64) for i in range(1, 11)])
    ask_sz = np.column_stack([df[f"ask_sz_{i:02d}"].to_numpy(dtype=np.float64) for i in range(1, 11)])

    bid_dist = (mid[:, None] - bid_px) / float(tick_size)
    ask_dist = (ask_px - mid[:, None]) / float(tick_size)
    total_bid = np.sum(bid_sz, axis=1)
    total_ask = np.sum(ask_sz, axis=1)
    bid_norm = _safe_divide(bid_sz, total_bid[:, None])
    ask_norm = _safe_divide(ask_sz, total_ask[:, None])

    for i in range(10):
        level = i + 1
        out[f"bid_dist_{level}_ticks"] = bid_dist[:, i]
        out[f"ask_dist_{level}_ticks"] = ask_dist[:, i]
        out[f"bid_sz_{level}_norm"] = bid_norm[:, i]
        out[f"ask_sz_{level}_norm"] = ask_norm[:, i]

    cum_bid = np.cumsum(bid_sz, axis=1)
    cum_ask = np.cumsum(ask_sz, axis=1)
    for k in [1, 3, 5, 10]:
        out[f"cum_bid_sz_{k}"] = cum_bid[:, k - 1]
        out[f"cum_ask_sz_{k}"] = cum_ask[:, k - 1]

    out["cum_bid_sz_1_over_10"] = _safe_divide(
        out["cum_bid_sz_1"].to_numpy(dtype=np.float64),
        out["cum_bid_sz_10"].to_numpy(dtype=np.float64),
    )
    out["cum_bid_sz_3_over_10"] = _safe_divide(
        out["cum_bid_sz_3"].to_numpy(dtype=np.float64),
        out["cum_bid_sz_10"].to_numpy(dtype=np.float64),
    )
    out["cum_bid_sz_5_over_10"] = _safe_divide(
        out["cum_bid_sz_5"].to_numpy(dtype=np.float64),
        out["cum_bid_sz_10"].to_numpy(dtype=np.float64),
    )
    out["cum_ask_sz_1_over_10"] = _safe_divide(
        out["cum_ask_sz_1"].to_numpy(dtype=np.float64),
        out["cum_ask_sz_10"].to_numpy(dtype=np.float64),
    )
    out["cum_ask_sz_3_over_10"] = _safe_divide(
        out["cum_ask_sz_3"].to_numpy(dtype=np.float64),
        out["cum_ask_sz_10"].to_numpy(dtype=np.float64),
    )
    out["cum_ask_sz_5_over_10"] = _safe_divide(
        out["cum_ask_sz_5"].to_numpy(dtype=np.float64),
        out["cum_ask_sz_10"].to_numpy(dtype=np.float64),
    )

    slope_bid, convex_bid = _compute_book_shape_features(bid_dist)
    slope_ask, convex_ask = _compute_book_shape_features(ask_dist)
    out["book_slope_bid"] = slope_bid
    out["book_slope_ask"] = slope_ask
    out["book_convexity_bid"] = convex_bid
    out["book_convexity_ask"] = convex_ask
    return out


def build_imbalance_features(
    df: pd.DataFrame,
    *,
    past_index: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    windows_sec: list[int],
) -> pd.DataFrame:
    """Build imbalance levels, deltas, and acceleration features."""

    out = pd.DataFrame(index=df.index)
    imbalance_now: dict[int, np.ndarray] = {}

    for k in [1, 3, 5, 10]:
        bid_sum = sum(df[f"bid_sz_{i:02d}"].to_numpy(dtype=np.float64) for i in range(1, k + 1))
        ask_sum = sum(df[f"ask_sz_{i:02d}"].to_numpy(dtype=np.float64) for i in range(1, k + 1))
        imbalance = _safe_divide(bid_sum - ask_sum, bid_sum + ask_sum)
        imbalance_now[k] = imbalance
        out[f"imbalance_{k}"] = imbalance

    for seconds in windows_sec:
        start_idx, valid, _ = past_index[seconds]
        for k in [1, 3, 5, 10]:
            prev = _values_at_indices(imbalance_now[k], start_idx)
            delta = np.full(len(df), np.nan, dtype=np.float64)
            mask = valid & np.isfinite(prev) & np.isfinite(imbalance_now[k])
            delta[mask] = imbalance_now[k][mask] - prev[mask]
            out[f"imbalance_{k}_delta_{seconds}s"] = delta

    out["imbalance_5_acceleration"] = np.nan
    if 1 in past_index and "imbalance_5_delta_1s" in out.columns:
        imbalance_5_delta_1s = out["imbalance_5_delta_1s"].to_numpy(dtype=np.float64)
        start_idx_1s, valid_1s, elapsed_1s = past_index[1]
        prev_delta = _values_at_indices(imbalance_5_delta_1s, start_idx_1s)
        accel = np.full(len(df), np.nan, dtype=np.float64)
        mask = valid_1s & np.isfinite(prev_delta) & np.isfinite(imbalance_5_delta_1s)
        accel[mask] = (imbalance_5_delta_1s[mask] - prev_delta[mask]) / elapsed_1s[mask]
        out["imbalance_5_acceleration"] = accel
    return out


def build_order_flow_features(
    df: pd.DataFrame,
    *,
    past_index: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    windows_sec: list[int],
) -> pd.DataFrame:
    """Build order-flow and update-intensity features from snapshot-to-snapshot deltas."""

    out = pd.DataFrame(index=df.index)

    bid_sz = np.column_stack([df[f"bid_sz_{i:02d}"].to_numpy(dtype=np.float64) for i in range(1, 11)])
    ask_sz = np.column_stack([df[f"ask_sz_{i:02d}"].to_numpy(dtype=np.float64) for i in range(1, 11)])

    bid_diff = np.zeros_like(bid_sz)
    ask_diff = np.zeros_like(ask_sz)
    bid_diff[1:] = bid_sz[1:] - bid_sz[:-1]
    ask_diff[1:] = ask_sz[1:] - ask_sz[:-1]

    bid_added_step = np.sum(np.clip(bid_diff, 0.0, None), axis=1)
    ask_added_step = np.sum(np.clip(ask_diff, 0.0, None), axis=1)
    bid_removed_step = np.sum(np.clip(-bid_diff, 0.0, None), axis=1)
    ask_removed_step = np.sum(np.clip(-ask_diff, 0.0, None), axis=1)
    bid_net_step = np.sum(bid_diff, axis=1)
    ask_net_step = np.sum(ask_diff, axis=1)

    prefix_bid_added = np.zeros(len(df) + 1, dtype=np.float64)
    prefix_ask_added = np.zeros(len(df) + 1, dtype=np.float64)
    prefix_bid_removed = np.zeros(len(df) + 1, dtype=np.float64)
    prefix_ask_removed = np.zeros(len(df) + 1, dtype=np.float64)
    prefix_bid_net = np.zeros(len(df) + 1, dtype=np.float64)
    prefix_ask_net = np.zeros(len(df) + 1, dtype=np.float64)

    prefix_bid_added[1:] = np.cumsum(bid_added_step)
    prefix_ask_added[1:] = np.cumsum(ask_added_step)
    prefix_bid_removed[1:] = np.cumsum(bid_removed_step)
    prefix_ask_removed[1:] = np.cumsum(ask_removed_step)
    prefix_bid_net[1:] = np.cumsum(bid_net_step)
    prefix_ask_net[1:] = np.cumsum(ask_net_step)

    end_idx = np.arange(len(df), dtype=np.int64)
    for seconds in windows_sec:
        start_idx, valid, _ = past_index[seconds]
        bid_added = _window_sum_from_prefix(prefix_bid_added, start_idx, end_idx, valid)
        ask_added = _window_sum_from_prefix(prefix_ask_added, start_idx, end_idx, valid)
        bid_removed = _window_sum_from_prefix(prefix_bid_removed, start_idx, end_idx, valid)
        ask_removed = _window_sum_from_prefix(prefix_ask_removed, start_idx, end_idx, valid)
        bid_net = _window_sum_from_prefix(prefix_bid_net, start_idx, end_idx, valid)
        ask_net = _window_sum_from_prefix(prefix_ask_net, start_idx, end_idx, valid)

        out[f"bid_size_added_{seconds}s"] = bid_added
        out[f"ask_size_added_{seconds}s"] = ask_added
        out[f"bid_size_removed_{seconds}s"] = bid_removed
        out[f"ask_size_removed_{seconds}s"] = ask_removed
        out[f"net_bid_flow_{seconds}s"] = bid_net
        out[f"net_ask_flow_{seconds}s"] = ask_net
        out[f"order_flow_imbalance_{seconds}s"] = _safe_divide(
            bid_net - ask_net,
            np.abs(bid_net) + np.abs(ask_net),
        )

    for seconds in windows_sec:
        start_idx, valid, _ = past_index[seconds]
        out[f"update_intensity_{seconds}s"] = np.where(
            valid,
            np.arange(len(df)) - start_idx + 1,
            np.nan,
        )
    return out


def _regime_vol_series(price_features: pd.DataFrame, windows_sec: list[int]) -> pd.Series:
    """Prefer 30s vol (legacy regime); else longest window (stabler qcut than very short)."""

    if 30 in windows_sec:
        return price_features["realized_vol_30s"]
    wmax = max(windows_sec)
    return price_features[f"realized_vol_{wmax}s"]


def _regime_update_series(order_flow_features: pd.DataFrame, windows_sec: list[int]) -> pd.Series:
    """Prefer 5s update intensity; else smallest window >= 5, else minimum window."""

    if 5 in windows_sec:
        return order_flow_features["update_intensity_5s"]
    ge5 = [w for w in windows_sec if w >= 5]
    if ge5:
        w = min(ge5)
    else:
        w = min(windows_sec)
    return order_flow_features[f"update_intensity_{w}s"]


def _large_move_mask(price_features: pd.DataFrame, windows_sec: list[int]) -> pd.Series:
    """Same 20-tick threshold on shortest available return horizon (prefer 1s)."""

    if 1 in windows_sec:
        s = price_features["mid_return_1s"]
    else:
        w = min(windows_sec)
        s = price_features[f"mid_return_{w}s"]
    return s.abs() > 20.0


def build_regime_features(
    df: pd.DataFrame,
    *,
    price_features: pd.DataFrame,
    order_flow_features: pd.DataFrame,
    windows_sec: list[int],
) -> pd.DataFrame:
    """Build cross-sectional regime and persistence features."""

    out = pd.DataFrame(index=df.index)
    out["spread_regime"] = _bucketize_quantiles(price_features["spread_ticks"], q=3)
    out["volatility_regime"] = _bucketize_quantiles(_regime_vol_series(price_features, windows_sec), q=3)
    out["update_regime"] = _bucketize_quantiles(
        _regime_update_series(order_flow_features, windows_sec),
        q=3,
    )

    ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    large_move_mask = _large_move_mask(price_features, windows_sec)
    last_large_move_ts = ts.where(large_move_mask).ffill()
    seconds_since = (ts - last_large_move_ts).dt.total_seconds()
    out["seconds_since_large_move"] = seconds_since.to_numpy(dtype=np.float64)
    return out


def build_features_dataset(
    df: pd.DataFrame,
    tick_size: float,
    *,
    windows_sec: list[int] | None = None,
    lookback_sec: int | None = None,
    keep_raw_book: bool = False,
) -> pd.DataFrame:
    """Build a feature dataset while preserving existing target columns.

    Parameters
    ----------
    df:
        Input DataFrame with order book columns and already built target columns.
    tick_size:
        Tick size used to normalize price distances and returns.
    windows_sec:
        Sorted unique positive wall-clock windows (seconds) for all dynamic features.
        Default: ``[1, 3, 5, 10, 30]``.
    lookback_sec:
        Must be ``>= max(windows_sec)``. If omitted, set to ``max(windows_sec)``.
        (Validates alignment; computation uses ``windows_sec`` only.)
    keep_raw_book:
        If False, raw ``bid_px_*``, ``bid_sz_*``, ``ask_px_*``, ``ask_sz_*`` columns are
        dropped from the output after features are built.
    """

    if not np.isfinite(tick_size) or float(tick_size) <= 0:
        raise ValueError(f"tick_size must be positive finite, got {tick_size!r}")

    ws = validate_windows_sec(list(windows_sec) if windows_sec is not None else list(DEFAULT_WINDOWS_SEC))
    max_w = max(ws)

    if lookback_sec is None:
        lb = max_w
    else:
        lb = int(lookback_sec)
        if lb < max_w:
            raise ValueError(
                f"lookback_sec ({lb}) must be >= max(windows_sec) ({max_w}); "
                f"windows_sec={ws!r}"
            )

    work = _prepare_input_frame(df)
    ts_ns = work["ts_event"].to_numpy(dtype="datetime64[ns]").astype("int64")
    past_index = _compute_past_indices(ts_ns, ws)

    LOGGER.info(
        "Building features: n_rows=%d windows_sec=%s max_lookback_sec=%d (validated lookback_sec=%d) "
        "keep_raw_book=%s",
        len(work),
        ws,
        max_w,
        lb,
        keep_raw_book,
    )
    _log_window_validity_stats(len(work), past_index, ws)

    price_features = build_price_features(
        work,
        tick_size=float(tick_size),
        past_index=past_index,
        windows_sec=ws,
    )
    depth_features = build_depth_features(
        work,
        tick_size=float(tick_size),
    )
    imbalance_features = build_imbalance_features(
        work,
        past_index=past_index,
        windows_sec=ws,
    )
    order_flow_features = build_order_flow_features(
        work,
        past_index=past_index,
        windows_sec=ws,
    )
    regime_features = build_regime_features(
        work,
        price_features=price_features,
        order_flow_features=order_flow_features,
        windows_sec=ws,
    )

    feature_frames = [
        price_features,
        depth_features,
        imbalance_features,
        order_flow_features,
        regime_features,
    ]
    features = pd.concat(feature_frames, axis=1)

    duplicate_cols = features.columns[features.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(f"Duplicate feature columns generated: {duplicate_cols}")

    out = pd.concat([work, features], axis=1)

    if not keep_raw_book:
        raw_present = [col for col in BOOK_COLUMNS if col in out.columns]
        out = out.drop(columns=raw_present)

    LOGGER.info("Feature dataset shape=%s", out.shape)
    return out


def summarize_features(
    df: pd.DataFrame,
    *,
    windows_sec: list[int] | None = None,
) -> dict[str, Any]:
    """Return and log a compact summary of generated feature columns.

    Parameters
    ----------
    df:
        DataFrame after ``build_features_dataset``.
    windows_sec:
        Same windows used when building; default ``DEFAULT_WINDOWS_SEC`` for group labels.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    ws = validate_windows_sec(list(windows_sec) if windows_sec is not None else list(DEFAULT_WINDOWS_SEC))
    groups = feature_groups_for_windows(ws)

    feature_columns_by_group: dict[str, list[str]] = {}
    all_feature_columns: list[str] = []
    for group_name, group_cols in groups.items():
        present = [col for col in group_cols if col in df.columns]
        feature_columns_by_group[group_name] = present
        all_feature_columns.extend(present)

    all_feature_columns = sorted(set(all_feature_columns))
    nan_pct = (
        df[all_feature_columns].isna().mean().mul(100.0).sort_values(ascending=False)
        if all_feature_columns
        else pd.Series(dtype=np.float64)
    )

    numeric_feature_cols = [
        col for col in all_feature_columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    basic_stats = (
        df[numeric_feature_cols]
        .describe()
        .T.loc[:, ["mean", "std", "min", "50%", "max"]]
        .rename(columns={"50%": "median"})
        if numeric_feature_cols
        else pd.DataFrame(columns=["mean", "std", "min", "median", "max"])
    )

    summary = {
        "n_rows": int(len(df)),
        "n_feature_columns": int(len(all_feature_columns)),
        "feature_columns_by_group": feature_columns_by_group,
        "nan_pct": nan_pct,
        "basic_stats": basic_stats,
    }

    LOGGER.info("Feature summary: rows=%d feature_columns=%d", summary["n_rows"], summary["n_feature_columns"])
    for group_name, cols in feature_columns_by_group.items():
        LOGGER.info("Feature group %-12s -> %d columns", group_name, len(cols))
    if not nan_pct.empty:
        LOGGER.info("Top NaN %% columns:\n%s", nan_pct.head(20).to_string())
    if not basic_stats.empty:
        LOGGER.info("Basic stats sample:\n%s", basic_stats.head(20).to_string())

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe feature parquet from a target dataset parquet. "
            "Dynamic features use wall-clock windows from --windows-sec (default 1,3,5,10,30). "
            "Only past data at or before each ts_event is used."
        ),
    )
    parser.add_argument(
        "--parquet-path",
        "--input-path",
        dest="parquet_path",
        required=True,
        help="Input parquet with targets and raw order book columns.",
    )
    parser.add_argument("--output-path", required=True, help="Output parquet path.")
    parser.add_argument("--tick-size", type=float, default=None, help="Tick size; if omitted, inferred from mid_px.")
    parser.add_argument(
        "--windows-sec",
        type=str,
        default=None,
        help=(
            "Comma-separated wall-clock windows in seconds, e.g. "
            "'1,3,5,10,30,60,300,900,1800,3600'. Default: 1,3,5,10,30. "
            "Must be unique positive integers."
        ),
    )
    parser.add_argument(
        "--lookback-sec",
        type=int,
        default=None,
        help="Must be >= max(windows). Default: max(windows). Used for validation/logging alignment.",
    )
    parser.add_argument("--keep-raw-book", action="store_true", help="Keep raw bid/ask px/sz columns.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI: target parquet -> feature parquet."""

    args = _parse_args()
    _configure_logging(args.log_level)

    parquet_path = Path(args.parquet_path)
    output_path = Path(args.output_path)
    windows_sec = parse_windows_sec(args.windows_sec)
    max_w = max(windows_sec)
    lookback = int(args.lookback_sec) if args.lookback_sec is not None else max_w
    if lookback < max_w:
        raise SystemExit(f"--lookback-sec ({lookback}) must be >= max(--windows-sec) ({max_w})")

    LOGGER.info("Reading parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    tick_size = args.tick_size
    if tick_size is None:
        tick_size = infer_tick_size(df, price_col="mid_px").tick_size
        LOGGER.info("Inferred tick_size=%s", tick_size)
    else:
        LOGGER.info("Using provided tick_size=%s", tick_size)

    df_features = build_features_dataset(
        df,
        tick_size=float(tick_size),
        windows_sec=windows_sec,
        lookback_sec=lookback,
        keep_raw_book=bool(args.keep_raw_book),
    )
    summarize_features(df_features, windows_sec=windows_sec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    LOGGER.info("Saved feature dataset to %s", output_path)


if __name__ == "__main__":
    main()
