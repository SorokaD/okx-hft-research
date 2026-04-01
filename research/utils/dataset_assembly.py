# flake8: noqa
"""
Dataset assembly for baseline short-horizon order book modeling.

This is a thin orchestration layer that:
- computes baseline features from reconstructed order book snapshots
- computes multi-horizon labels (directional, tick-based, profitability-aware, TP/SL)
- performs strict leakage-safe construction:
  - features use only current/past values
  - labels use only future values aligned to the horizon
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from research.features.orderbook_baseline_features import build_orderbook_baseline_features
from research.labels.orderbook_targets import build_horizon_tick_direction_targets
from research.labels.profitability_targets import (
    MakerTakerFeeModel,
    build_gross_profitability_targets,
    build_net_profitability_targets,
)
from research.labels.tp_sl_labels import build_tp_sl_labels
from research.utils.tick_size import infer_tick_size, validate_tick_size


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def _check_grid_regular(
    df: pd.DataFrame,
    *,
    ts_col: str,
    grid_ms: int,
    sample_n: int = 200_000,
    min_ratio: float = 0.999,
) -> tuple[bool, float]:
    if ts_col not in df.columns:
        raise ValueError(f"Missing ts column: {ts_col}")
    if len(df) < 2:
        return

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return

    # If there are duplicate timestamps, consecutive diffs include 0 which would
    # artificially lower the exact-match ratio. For grid-alignment validation we
    # use unique timestamps.
    ts = ts.drop_duplicates()
    if len(ts) < 2:
        return

    if len(ts) > sample_n:
        ts = ts.iloc[:: max(1, len(ts) // sample_n)]
    ts_ms = ts.values.astype("datetime64[ms]").astype("int64")
    diffs = ts_ms[1:] - ts_ms[:-1]
    if diffs.size == 0:
        return True, 1.0
    ratio = float((diffs == int(grid_ms)).mean())
    is_regular = ratio >= min_ratio
    return is_regular, ratio


@dataclass(frozen=True)
class ModelingDatasetConfig:
    grid_ms: int = 100
    depth_k: int = 10
    horizons_ms: tuple[int, ...] = (100, 150, 200, 250, 500, 1000)
    tick_thresholds: tuple[int, ...] = (1, 2, 3)
    lookback_ms: tuple[int, ...] = (100, 200, 500, 1000)

    # Profitability proxy config
    gross_profit_min_ticks: float = 1.0
    fee_model: MakerTakerFeeModel = MakerTakerFeeModel()

    # TP/SL config
    take_profit_ticks: tuple[int, ...] = (1, 2, 3)
    stop_loss_ticks: tuple[int, ...] = (1, 2)
    holding_ms: tuple[int, ...] = (100, 250, 500, 1000)

    # Tick size: if None -> infer from df
    tick_size: float | None = None
    tick_infer_sample_n: int = 200_000


def build_modeling_dataset(
    df: pd.DataFrame,
    *,
    config: ModelingDatasetConfig,
    ts_col: str = "ts_event",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build modeling dataset from reconstructed parquet DataFrame.
    Returns (dataset_df, metadata_dict).
    """
    _require_cols(df, ["mid_px", "spread_px", ts_col])
    _require_cols(df, ["bid_px_01", "ask_px_01", "bid_sz_01", "ask_sz_01"])

    depth_k = int(config.depth_k)

    for side in ("bid", "ask"):
        for i in range(1, depth_k + 1):
            _require_cols(df, [f"{side}_px_{i:02d}", f"{side}_sz_{i:02d}"])

    df_work = df.copy()
    df_work[ts_col] = pd.to_datetime(df_work[ts_col], utc=True, errors="coerce")
    df_work = df_work.dropna(subset=[ts_col, "mid_px", "spread_px"]).sort_values(ts_col).reset_index(drop=True)

    # Grid regularity: shift-based alignment is correct only on a strictly regular grid.
    # If it is not regular, we still build a dataset using nearest-forward alignment
    # (first observation at or after t+H) via `alignment_method="auto"` in label builder.
    is_regular, regular_ratio = _check_grid_regular(
        df_work, ts_col=ts_col, grid_ms=config.grid_ms
    )

    if config.tick_size is None:
        tick_res = infer_tick_size(df_work, price_col="mid_px", sample_n=config.tick_infer_sample_n)
        tick_size = validate_tick_size(tick_res.tick_size)
        tick_inferred_from = tick_res
    else:
        tick_size = validate_tick_size(config.tick_size)
        tick_inferred_from = None

    # Ensure tick size is >0
    if tick_size <= 0:
        raise ValueError(f"Invalid tick_size inferred: {tick_size}")

    # Features
    features_df = build_orderbook_baseline_features(
        df_work,
        grid_ms=config.grid_ms,
        tick_size=tick_size,
        lookback_ms=config.lookback_ms,
        depth_k=depth_k,
    )

    # Targets: horizon mid/bid/ask move and directional/tick-threshold
    horizons_ms = list(config.horizons_ms)
    targets_df, alignment = build_horizon_tick_direction_targets(
        df_work,
        tick_size=tick_size,
        grid_ms=config.grid_ms,
        horizons_ms=horizons_ms,
        tick_thresholds=config.tick_thresholds,
        ts_col=ts_col,
        alignment_method="auto",
    )

    # Profitability targets require aligned future best bid/ask at each horizon.
    # Under shift alignment with ceil-to-next-grid-step, future states are at -step.
    for h in horizons_ms:
        step = alignment.aligned_steps[h]
        horizon_tag = f"h{h}"
        future_bid = df_work["bid_px_01"].shift(-step)
        future_ask = df_work["ask_px_01"].shift(-step)
        gross_df = build_gross_profitability_targets(
            df_work,
            tick_size=tick_size,
            future_best_bid_px=future_bid,
            future_best_ask_px=future_ask,
            horizon_tag=horizon_tag,
            gross_profit_min_ticks=config.gross_profit_min_ticks,
        )
        net_df = build_net_profitability_targets(
            df_work,
            tick_size=tick_size,
            future_best_bid_px=future_bid,
            future_best_ask_px=future_ask,
            horizon_tag=horizon_tag,
            fee_model=config.fee_model,
        )
        targets_df = pd.concat([targets_df, gross_df, net_df], axis=1)

    # TP/SL labels
    tp_sl_df = build_tp_sl_labels(
        df_work,
        tick_size=tick_size,
        grid_ms=config.grid_ms,
        take_profit_ticks=config.take_profit_ticks,
        stop_loss_ticks=config.stop_loss_ticks,
        holding_ms=config.holding_ms,
        alignment_method="shift",
    )

    # Assemble dataset: features + targets + identifiers.
    # We keep only columns that are useful for training + traceability.
    id_cols = [c for c in ["ts_event", "inst_id", "anchor_snapshot_ts", "reconstruction_mode"] if c in df_work.columns]
    dataset = pd.concat([features_df, targets_df, tp_sl_df], axis=1)
    # drop rows with NaNs in any target column
    target_cols = list(targets_df.columns) + list(tp_sl_df.columns)
    dataset = pd.concat([dataset, df_work[id_cols]], axis=1) if id_cols and "ts_event" not in dataset.columns else dataset
    dataset = dataset.dropna(subset=target_cols).reset_index(drop=True)

    metadata: dict[str, Any] = {
        "grid_ms": config.grid_ms,
        "depth_k": depth_k,
        "horizons_ms": horizons_ms,
        "tick_thresholds": list(config.tick_thresholds),
        "lookback_ms": list(config.lookback_ms),
        "tick_size": float(tick_size),
        "tick_inferred": tick_inferred_from is not None,
        "tick_inference_notes": getattr(tick_inferred_from, "notes", None) if tick_inferred_from is not None else None,
        "aligned_steps": alignment.aligned_steps,
        "aligned_horizon_ms": alignment.aligned_horizon_ms,
        "alignment_method": alignment.method,
        "grid_regular": is_regular,
        "grid_exact_diff_ratio": regular_ratio,
        "num_rows_before_dropna": int(len(df_work)),
        "num_rows_after_dropna": int(len(dataset)),
        "targets": {
            "directional": [c for c in targets_df.columns if c.startswith("target_mid_dir_")],
            "tick_direction": [c for c in targets_df.columns if c.startswith("target_mid_ge_") or c.startswith("target_mid_le_")],
            "profitability": [
                c
                for c in targets_df.columns
                if c.startswith("target_gross_profitable_") or c.startswith("target_net_profitable_")
            ],
            "tp_sl": list(tp_sl_df.columns),
        },
    }

    return dataset, metadata

