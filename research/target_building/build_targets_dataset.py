# flake8: noqa
"""Build time-based ML target datasets from reconstructed order book data.

This module focuses on target construction only:
- time-based horizon windows using real `ts_event`,
- symmetric and asymmetric first-hit labels,
- MFE / MAE-style excursion targets in ticks,
- compact summary helpers,
- CLI/demo entrypoint for parquet -> parquet workflows.

The implementation deliberately avoids feature engineering so it can later be
composed with a separate `build_features(...)` stage.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.utils.tick_size import infer_tick_size

LOGGER = logging.getLogger(__name__)

BASE_ID_COLUMNS = [
    "ts_event",
    "inst_id",
    "anchor_snapshot_ts",
    "reconstruction_mode",
    "mid_px",
    "spread_px",
]


@dataclass(frozen=True)
class PreparedInputs:
    """Prepared arrays and metadata for target construction."""

    df: pd.DataFrame
    ts_ns: np.ndarray
    mid_px: np.ndarray
    base_columns: list[str]


def _validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _validate_positive_number(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


def _validate_int_list(name: str, values: list[int]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(int(v) <= 0 for v in values):
        raise ValueError(f"{name} must contain only positive integers, got {values!r}")


def _prepare_input_frame(df: pd.DataFrame) -> PreparedInputs:
    """Validate and normalize the input frame used for target construction."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    _validate_required_columns(df, ["ts_event", "mid_px", "spread_px"])

    work = df.copy()
    work["ts_event"] = pd.to_datetime(work["ts_event"], utc=True, errors="coerce")

    for col in ["anchor_snapshot_ts"]:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], utc=True, errors="coerce")

    for col in ["mid_px", "spread_px"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["ts_event", "mid_px", "spread_px"]).sort_values("ts_event").reset_index(drop=True)
    if work.empty:
        raise ValueError("Input DataFrame is empty after cleaning")

    if not work["ts_event"].is_monotonic_increasing:
        raise ValueError("ts_event must be monotonic increasing after sorting")

    ts_ns = work["ts_event"].to_numpy(dtype="datetime64[ns]").astype("int64")
    if np.any(np.diff(ts_ns) < 0):
        raise ValueError("ts_event must be monotonic increasing in nanoseconds representation")

    mid_px = work["mid_px"].to_numpy(dtype=np.float64)
    base_columns = [col for col in BASE_ID_COLUMNS if col in work.columns]

    return PreparedInputs(df=work, ts_ns=ts_ns, mid_px=mid_px, base_columns=base_columns)


def _compute_window_end_indices(ts_ns: np.ndarray, horizon_sec: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exclusive right bounds and complete-window mask for a time horizon."""

    horizon_ns = np.int64(horizon_sec) * np.int64(1_000_000_000)
    target_end_ns = ts_ns + horizon_ns
    end_idx = np.searchsorted(ts_ns, target_end_ns, side="right")
    complete_mask = target_end_ns <= ts_ns[-1]
    return end_idx.astype(np.int64), complete_mask


def _scan_relative_ticks(
    rel_ticks: np.ndarray,
    *,
    sym_thresholds: list[int],
    asym_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    """Compute all target values for one future path in tick space."""

    up_hits = rel_ticks > 0
    down_hits = rel_ticks < 0

    if up_hits.any():
        long_mfe = float(np.max(rel_ticks[up_hits]))
    else:
        long_mfe = 0.0

    if down_hits.any():
        long_mae = float(np.max(-rel_ticks[down_hits]))
    else:
        long_mae = 0.0

    short_mfe = long_mae
    short_mae = long_mfe

    result: dict[str, Any] = {
        "long_mfe": long_mfe,
        "long_mae": long_mae,
        "short_mfe": short_mfe,
        "short_mae": short_mae,
    }

    for threshold in sym_thresholds:
        up_idx = np.flatnonzero(rel_ticks >= float(threshold))
        down_idx = np.flatnonzero(rel_ticks <= -float(threshold))

        up_hit = up_idx.size > 0
        down_hit = down_idx.size > 0

        result[f"long_hit_up_{threshold}"] = int(up_hit)
        result[f"short_hit_down_{threshold}"] = int(down_hit)

        if up_hit and not down_hit:
            result[f"long_before_down_{threshold}"] = 1
        elif not up_hit:
            result[f"long_before_down_{threshold}"] = 0
        elif not down_hit:
            result[f"long_before_down_{threshold}"] = 1
        else:
            result[f"long_before_down_{threshold}"] = int(up_idx[0] < down_idx[0])

        if down_hit and not up_hit:
            result[f"short_before_up_{threshold}"] = 1
        elif not down_hit:
            result[f"short_before_up_{threshold}"] = 0
        elif not up_hit:
            result[f"short_before_up_{threshold}"] = 1
        else:
            result[f"short_before_up_{threshold}"] = int(down_idx[0] < up_idx[0])

    for target_ticks, adverse_ticks in asym_pairs:
        up_idx = np.flatnonzero(rel_ticks >= float(target_ticks))
        down_idx = np.flatnonzero(rel_ticks <= -float(adverse_ticks))

        if up_idx.size == 0:
            result[f"long_asym_{target_ticks}_{adverse_ticks}"] = 0
        elif down_idx.size == 0:
            result[f"long_asym_{target_ticks}_{adverse_ticks}"] = 1
        else:
            result[f"long_asym_{target_ticks}_{adverse_ticks}"] = int(up_idx[0] < down_idx[0])

        down_target_idx = np.flatnonzero(rel_ticks <= -float(target_ticks))
        up_adverse_idx = np.flatnonzero(rel_ticks >= float(adverse_ticks))

        if down_target_idx.size == 0:
            result[f"short_asym_{target_ticks}_{adverse_ticks}"] = 0
        elif up_adverse_idx.size == 0:
            result[f"short_asym_{target_ticks}_{adverse_ticks}"] = 1
        else:
            result[f"short_asym_{target_ticks}_{adverse_ticks}"] = int(down_target_idx[0] < up_adverse_idx[0])

    return result


def _empty_float_array(n: int) -> np.ndarray:
    return np.full(n, np.nan, dtype=np.float32)


def _allocate_target_arrays(
    n: int,
    horizons_sec: list[int],
    thresholds_ticks: list[int],
    asym_pairs: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Allocate result arrays once to avoid repeated DataFrame appends."""

    arrays: dict[str, np.ndarray] = {}
    for horizon_sec in horizons_sec:
        suffix = f"within_{horizon_sec}s"
        arrays[f"long_max_favorable_excursion_ticks_{suffix}"] = _empty_float_array(n)
        arrays[f"long_max_adverse_excursion_ticks_{suffix}"] = _empty_float_array(n)
        arrays[f"short_max_favorable_excursion_ticks_{suffix}"] = _empty_float_array(n)
        arrays[f"short_max_adverse_excursion_ticks_{suffix}"] = _empty_float_array(n)

        for threshold in thresholds_ticks:
            arrays[f"long_hit_up_{threshold}t_before_down_within_{horizon_sec}s"] = _empty_float_array(n)
            arrays[f"long_hit_up_{threshold}t_within_{horizon_sec}s"] = _empty_float_array(n)
            arrays[f"short_hit_down_{threshold}t_before_up_within_{horizon_sec}s"] = _empty_float_array(n)
            arrays[f"short_hit_down_{threshold}t_within_{horizon_sec}s"] = _empty_float_array(n)

        for target_ticks, adverse_ticks in asym_pairs:
            arrays[f"long_hit_up_{target_ticks}t_before_down_{adverse_ticks}t_within_{horizon_sec}s"] = _empty_float_array(n)
            arrays[f"short_hit_down_{target_ticks}t_before_up_{adverse_ticks}t_within_{horizon_sec}s"] = _empty_float_array(n)

    return arrays


def build_targets_dataset(
    df: pd.DataFrame,
    *,
    tick_size: float,
    horizons_sec: list[int],
    thresholds_ticks: list[int],
    adverse_ticks: list[int] | None = None,
    sampling_step_rows: int = 1,
    drop_last_incomplete: bool = True,
) -> pd.DataFrame:
    """Build a pure-target dataset from reconstructed order book snapshots.

    Parameters
    ----------
    df:
        Source DataFrame containing at least `ts_event`, `mid_px`, `spread_px`.
    tick_size:
        Instrument tick size used to express all excursions in ticks.
    horizons_sec:
        Forward-looking real-time horizons in seconds.
    thresholds_ticks:
        Symmetric target thresholds in ticks.
    adverse_ticks:
        Optional adverse thresholds for asymmetric TP/SL-like labels. When
        provided, it must have the same length as `thresholds_ticks`, and pairs
        are matched positionally.
    sampling_step_rows:
        Keep every N-th observation as an anchor row in the output.
    drop_last_incomplete:
        If True, rows whose full horizon does not fit into the future are
        removed. Otherwise, they are kept and target columns remain NaN.
    """

    _validate_positive_number("tick_size", tick_size)
    _validate_int_list("horizons_sec", horizons_sec)
    _validate_int_list("thresholds_ticks", thresholds_ticks)

    if int(sampling_step_rows) <= 0:
        raise ValueError(f"sampling_step_rows must be positive, got {sampling_step_rows!r}")

    asym_pairs: list[tuple[int, int]] = []
    if adverse_ticks is not None:
        _validate_int_list("adverse_ticks", adverse_ticks)
        if len(adverse_ticks) != len(thresholds_ticks):
            raise ValueError(
                "adverse_ticks must have the same length as thresholds_ticks "
                "and is matched positionally"
            )
        asym_pairs = list(zip(thresholds_ticks, adverse_ticks))

    prepared = _prepare_input_frame(df)
    work = prepared.df
    ts_ns = prepared.ts_ns
    mid_px = prepared.mid_px
    n = len(work)

    anchor_idx = np.arange(0, n, int(sampling_step_rows), dtype=np.int64)
    LOGGER.info(
        "Building targets for %d rows (%d anchor rows), horizons=%s thresholds=%s adverse=%s",
        n,
        len(anchor_idx),
        horizons_sec,
        thresholds_ticks,
        adverse_ticks,
    )

    target_arrays = _allocate_target_arrays(n, horizons_sec, thresholds_ticks, asym_pairs)
    complete_mask_by_horizon: dict[int, np.ndarray] = {}

    for horizon_sec in horizons_sec:
        end_idx, complete_mask = _compute_window_end_indices(ts_ns, horizon_sec)
        complete_mask_by_horizon[horizon_sec] = complete_mask
        LOGGER.info(
            "Processing horizon=%ss complete_rows=%d incomplete_rows=%d",
            horizon_sec,
            int(complete_mask.sum()),
            int((~complete_mask).sum()),
        )

        for i in anchor_idx:
            if not complete_mask[i]:
                continue

            start = int(i + 1)
            stop = int(end_idx[i])
            if start >= stop:
                rel_ticks = np.empty(0, dtype=np.float64)
            else:
                rel_ticks = (mid_px[start:stop] - mid_px[i]) / float(tick_size)

            stats = _scan_relative_ticks(rel_ticks, sym_thresholds=thresholds_ticks, asym_pairs=asym_pairs)
            suffix = f"within_{horizon_sec}s"
            target_arrays[f"long_max_favorable_excursion_ticks_{suffix}"][i] = stats["long_mfe"]
            target_arrays[f"long_max_adverse_excursion_ticks_{suffix}"][i] = stats["long_mae"]
            target_arrays[f"short_max_favorable_excursion_ticks_{suffix}"][i] = stats["short_mfe"]
            target_arrays[f"short_max_adverse_excursion_ticks_{suffix}"][i] = stats["short_mae"]

            for threshold in thresholds_ticks:
                target_arrays[f"long_hit_up_{threshold}t_before_down_within_{horizon_sec}s"][i] = stats[
                    f"long_before_down_{threshold}"
                ]
                target_arrays[f"long_hit_up_{threshold}t_within_{horizon_sec}s"][i] = stats[f"long_hit_up_{threshold}"]
                target_arrays[f"short_hit_down_{threshold}t_before_up_within_{horizon_sec}s"][i] = stats[
                    f"short_before_up_{threshold}"
                ]
                target_arrays[f"short_hit_down_{threshold}t_within_{horizon_sec}s"][i] = stats[
                    f"short_hit_down_{threshold}"
                ]

            for target_ticks, adverse_tick in asym_pairs:
                target_arrays[f"long_hit_up_{target_ticks}t_before_down_{adverse_tick}t_within_{horizon_sec}s"][i] = stats[
                    f"long_asym_{target_ticks}_{adverse_tick}"
                ]
                target_arrays[
                    f"short_hit_down_{target_ticks}t_before_up_{adverse_tick}t_within_{horizon_sec}s"
                ][i] = stats[f"short_asym_{target_ticks}_{adverse_tick}"]

    sampled_original_idx = anchor_idx
    if drop_last_incomplete:
        keep_mask = np.ones(len(sampled_original_idx), dtype=bool)
        for horizon_sec in horizons_sec:
            keep_mask &= complete_mask_by_horizon[horizon_sec][sampled_original_idx]
        sampled_original_idx = sampled_original_idx[keep_mask]

    # Build output row set first, then append target columns one-by-one.
    # This avoids a large contiguous copy during pd.concat on very wide frames.
    out = work.iloc[sampled_original_idx].loc[:, prepared.base_columns].reset_index(drop=True).copy()
    for col_name, values in target_arrays.items():
        out[col_name] = values[sampled_original_idx]

    LOGGER.info("Targets dataset shape=%s", out.shape)
    return out


def summarize_targets(df_targets: pd.DataFrame) -> dict[str, Any]:
    """Return a compact summary of generated target columns and their distributions."""

    if not isinstance(df_targets, pd.DataFrame):
        raise TypeError("df_targets must be a pandas DataFrame")

    target_columns = [col for col in df_targets.columns if col not in BASE_ID_COLUMNS]
    binary_columns = [
        col
        for col in target_columns
        if ("_hit_" in col or "_before_" in col) and df_targets[col].dropna().isin([0, 1]).all()
    ]
    excursion_columns = [col for col in target_columns if "excursion_ticks" in col]

    binary_share = (
        df_targets[binary_columns]
        .mean(numeric_only=True)
        .sort_index()
        .rename("share_of_ones")
        .reset_index()
        .rename(columns={"index": "target"})
        if binary_columns
        else pd.DataFrame(columns=["target", "share_of_ones"])
    )

    excursion_stats = (
        df_targets[excursion_columns]
        .describe()
        .T.loc[:, ["mean", "std", "min", "50%", "max"]]
        .rename(columns={"50%": "median"})
        .reset_index()
        .rename(columns={"index": "target"})
        if excursion_columns
        else pd.DataFrame(columns=["target", "mean", "std", "min", "median", "max"])
    )

    summary: dict[str, Any] = {
        "n_rows": int(len(df_targets)),
        "n_target_columns": int(len(target_columns)),
        "target_columns": target_columns,
        "binary_target_share": binary_share,
        "excursion_stats": excursion_stats,
    }
    return summary


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build time-based target parquet from reconstructed order book parquet.")
    parser.add_argument("--parquet-path", required=True, help="Input parquet with reconstructed order book snapshots.")
    parser.add_argument("--output-path", required=True, help="Output parquet path for the generated targets dataset.")
    parser.add_argument("--tick-size", type=float, default=None, help="Tick size. If omitted, inferred from mid_px.")
    parser.add_argument("--horizons-sec", default="30,60,120,180", help="Comma-separated horizons in seconds.")
    parser.add_argument(
        "--thresholds-ticks",
        default="50,100,150,200,260,300",
        help="Comma-separated symmetric target thresholds in ticks.",
    )
    parser.add_argument(
        "--adverse-ticks",
        default="",
        help="Optional comma-separated adverse thresholds matched positionally to thresholds-ticks.",
    )
    parser.add_argument("--sampling-step-rows", type=int, default=1, help="Keep every N-th row as an anchor timestamp.")
    parser.add_argument(
        "--keep-last-incomplete",
        action="store_true",
        help="Keep rows whose full horizon does not fit into the future and leave targets as NaN.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser


def _log_summary(summary: dict[str, Any]) -> None:
    LOGGER.info("Targets summary: rows=%d target_columns=%d", summary["n_rows"], summary["n_target_columns"])

    binary_share = summary["binary_target_share"]
    if not binary_share.empty:
        LOGGER.info("Binary targets share of ones:\n%s", binary_share.to_string(index=False))

    excursion_stats = summary["excursion_stats"]
    if not excursion_stats.empty:
        LOGGER.info("Excursion target stats:\n%s", excursion_stats.to_string(index=False))


def main() -> None:
    """Command-line demo: parquet -> targets parquet with compact summary."""

    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    parquet_path = Path(args.parquet_path)
    output_path = Path(args.output_path)

    LOGGER.info("Reading parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    tick_size = args.tick_size
    if tick_size is None:
        tick_size = infer_tick_size(df, price_col="mid_px").tick_size
        LOGGER.info("Inferred tick_size=%s", tick_size)
    else:
        LOGGER.info("Using provided tick_size=%s", tick_size)

    thresholds_ticks = _parse_int_list(args.thresholds_ticks)
    adverse_ticks = _parse_int_list(args.adverse_ticks) if args.adverse_ticks else None
    horizons_sec = _parse_int_list(args.horizons_sec)

    df_targets = build_targets_dataset(
        df,
        tick_size=float(tick_size),
        horizons_sec=horizons_sec,
        thresholds_ticks=thresholds_ticks,
        adverse_ticks=adverse_ticks,
        sampling_step_rows=int(args.sampling_step_rows),
        drop_last_incomplete=not args.keep_last_incomplete,
    )

    summary = summarize_targets(df_targets)
    _log_summary(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_targets.to_parquet(output_path, index=False)
    LOGGER.info("Saved targets dataset to %s", output_path)


if __name__ == "__main__":
    main()
