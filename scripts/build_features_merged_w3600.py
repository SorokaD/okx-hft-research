"""Merge reconstructed book + targets parquet, then build features (long windows incl. 3600s).

Run from repo root:
  python scripts/build_features_merged_w3600.py

Processes **hourly** reconstructed shards one at a time (plus the previous shard for a full
3600s clock lookback), loads only the matching slice of the targets parquet via row-group
filters, and appends rows to the output. This avoids holding the full wide targets frame in RAM
(~10+ GiB for multi-hundred target columns).

Optional:
  python scripts/build_features_merged_w3600.py --targets-path outputs/targets/....parquet
"""
from __future__ import annotations

import argparse
import gc
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.features.build_features_dataset import (  # noqa: E402
    build_features_dataset,
    summarize_features,
    validate_windows_sec,
)
from research.utils.tick_size import infer_tick_size  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("build_features_merged_w3600")

WINDOWS_SEC = validate_windows_sec([1, 3, 5, 10, 30, 60, 300, 900, 1800, 3600])

DEFAULT_RECONSTRUCTED_DIR = ROOT / "data" / "reconstructed" / "BTC-USDT-SWAP" / "grid100ms"
DEFAULT_TARGETS_PATH = ROOT / "outputs" / "targets" / "btc_usdt_swap_targets_dataset_260_550_h30_300.parquet"
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "features" / "btc_usdt_swap_features_w3600.parquet"

BOOK_REQUIRED = [
    *[f"bid_px_{i:02d}" for i in range(1, 11)],
    *[f"bid_sz_{i:02d}" for i in range(1, 11)],
    *[f"ask_px_{i:02d}" for i in range(1, 11)],
    *[f"ask_sz_{i:02d}" for i in range(1, 11)],
]
BASE_JOIN = [
    "ts_event",
    "inst_id",
    "anchor_snapshot_ts",
    "reconstruction_mode",
    "mid_px",
    "spread_px",
]


def downcast_float64_inplace(df: pd.DataFrame) -> int:
    """Return count of columns converted float64 -> float32."""

    n = 0
    for c in df.columns:
        if str(df[c].dtype) == "float64":
            df[c] = df[c].astype(np.float32)
            n += 1
    return n


def _emit_ts_bounds(shard_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    ts = pd.read_parquet(shard_path, columns=["ts_event"])["ts_event"]
    return ts.min(), ts.max()


def _emit_ts_half_open(shard_path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return (inclusive_lo, exclusive_hi) from shard filename if parseable.

    Hourly book shards often share the boundary timestamp across consecutive files
    (e.g. both contain 01:00:00). Emitting with ``ts < exclusive_hi`` avoids duplicate rows.
    """

    parsed = _shard_ts_range_from_name(shard_path)
    return parsed if parsed is not None else None


# e.g. ..._2026-03-26_00-00-00__2026-03-26_01-00-00.parquet
_RE_SHARD_TS = re.compile(
    r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})__(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.parquet$"
)


def _shard_ts_range_from_name(path: Path) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    m = _RE_SHARD_TS.search(path.name)
    if not m:
        return None
    lo = pd.Timestamp(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}+00:00")
    hi = pd.Timestamp(f"{m.group(5)}T{m.group(6)}:{m.group(7)}:{m.group(8)}+00:00")
    return lo, hi


def _shard_overlaps_target(
    shard_path: Path, tgt_lo: pd.Timestamp, tgt_hi: pd.Timestamp
) -> bool:
    bounds = _shard_ts_range_from_name(shard_path)
    if bounds is None:
        lo, hi = _emit_ts_bounds(shard_path)
    else:
        lo, hi = bounds
    return lo <= tgt_hi and hi >= tgt_lo


def _targets_ts_bounds(targets_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    col = pq.read_table(targets_path, columns=["ts_event"]).column(0)
    return pd.Timestamp(pc.min(col).as_py()), pd.Timestamp(pc.max(col).as_py())


def _infer_tick_from_book(raw_files: list[Path]) -> float:
    for p in raw_files:
        try:
            df = pd.read_parquet(p, columns=["mid_px"])
        except (OSError, ValueError):
            continue
        if len(df) < 2:
            continue
        try:
            return float(infer_tick_size(df, price_col="mid_px").tick_size)
        except ValueError:
            continue
    raise SystemExit("Could not infer tick_size from any reconstructed shard (mid_px).")


def _merge_book_targets(
    df_raw: pd.DataFrame,
    df_targets: pd.DataFrame,
    join_keys: list[str],
    target_only: list[str],
) -> pd.DataFrame | None:
    if df_raw.empty or df_targets.empty:
        return None

    df_raw = df_raw.sort_values(join_keys).reset_index(drop=True)
    df_targets = df_targets.sort_values(join_keys).reset_index(drop=True)

    rk = df_raw[join_keys].copy()
    rk["_raw_row_id"] = np.arange(len(rk), dtype=np.int64)
    rk["_merge_occurrence"] = rk.groupby(join_keys, dropna=False).cumcount()

    tk = df_targets[join_keys].copy()
    tk["_merge_occurrence"] = tk.groupby(join_keys, dropna=False).cumcount()

    mkeys = join_keys + ["_merge_occurrence"]
    matched = rk.merge(
        pd.concat([tk[mkeys], df_targets[target_only]], axis=1),
        on=mkeys,
        how="inner",
        validate="one_to_one",
    )

    if matched.empty:
        return None

    return pd.concat(
        [
            df_raw.iloc[matched["_raw_row_id"].to_numpy()].reset_index(drop=True),
            matched[target_only].reset_index(drop=True),
        ],
        axis=1,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reconstructed-dir",
        type=Path,
        default=DEFAULT_RECONSTRUCTED_DIR,
        help="Root directory of reconstructed book parquet shards",
    )
    p.add_argument(
        "--targets-path",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
        help="Targets parquet to merge (wide column set is OK; read in time slices)",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output features parquet",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    reconstructed_dir = args.reconstructed_dir
    targets_path = args.targets_path
    output_path = args.output_path

    raw_files = sorted(reconstructed_dir.rglob("*.parquet"))
    if not raw_files:
        raise SystemExit(f"No parquet under {reconstructed_dir}")

    schema_cols = pq.read_schema(raw_files[0]).names
    missing = [c for c in BOOK_REQUIRED if c not in schema_cols]
    if missing:
        raise SystemExit(f"Reconstructed schema missing book cols: {missing[:5]}...")

    tgt_schema = pq.read_schema(targets_path).names
    raw_read = [c for c in BASE_JOIN + BOOK_REQUIRED if c in schema_cols]
    join_keys = [c for c in BASE_JOIN if c in raw_read and c in tgt_schema]
    target_only = [c for c in tgt_schema if c not in set(raw_read)]
    tgt_cols = join_keys + target_only

    tgt_lo, tgt_hi = _targets_ts_bounds(targets_path)
    log.info(
        "Targets ts_event range (UTC): %s .. %s (%d rows)",
        tgt_lo,
        tgt_hi,
        pq.read_metadata(targets_path).num_rows,
    )

    n_shards_total = len(raw_files)
    raw_files_to_emit = [p for p in raw_files if _shard_overlaps_target(p, tgt_lo, tgt_hi)]
    log.info(
        "%d reconstructed shards under %s; %d overlap target period",
        n_shards_total,
        reconstructed_dir,
        len(raw_files_to_emit),
    )
    if not raw_files_to_emit:
        raise SystemExit("No reconstructed shards overlap the targets time range.")

    tick_size = _infer_tick_from_book(raw_files)
    log.info("tick_size=%s; windows_sec=%s", tick_size, WINDOWS_SEC)

    writer: pq.ParquetWriter | None = None
    total_out = 0
    last_chunk_for_summary: pd.DataFrame | None = None

    targets_dset = ds.dataset(str(targets_path), format="parquet")
    raw_files_to_emit_set = set(raw_files_to_emit)
    emit_shard_idx = 0

    for i, shard_path in enumerate(raw_files):
        if shard_path not in raw_files_to_emit_set:
            continue
        paths_win = ([raw_files[i - 1]] if i > 0 else []) + [shard_path]
        df_raw = pd.concat(
            [pd.read_parquet(p, columns=raw_read) for p in paths_win],
            ignore_index=True,
        )
        if df_raw.empty:
            continue

        t_min = df_raw["ts_event"].min()
        t_max = df_raw["ts_event"].max()
        ts_lo = pa.scalar(t_min, type=pa.timestamp("ns", tz="UTC"))
        ts_hi = pa.scalar(t_max, type=pa.timestamp("ns", tz="UTC"))
        flt = (pc.field("ts_event") >= ts_lo) & (pc.field("ts_event") <= ts_hi)
        table_tgt = targets_dset.to_table(columns=tgt_cols, filter=flt)
        df_targets = table_tgt.to_pandas()
        del table_tgt

        df_for = _merge_book_targets(df_raw, df_targets, join_keys, target_only)
        del df_raw, df_targets
        gc.collect()

        if df_for is None:
            log.warning("No matched rows for shard %s", shard_path.name)
            continue

        emit_bounds = _emit_ts_half_open(shard_path)
        n_dc = downcast_float64_inplace(df_for)

        df_feat = build_features_dataset(
            df_for,
            tick_size=tick_size,
            windows_sec=WINDOWS_SEC,
            lookback_sec=None,
            keep_raw_book=False,
        )
        del df_for
        if emit_bounds is not None:
            emit_lo, emit_hi_excl = emit_bounds
            mask_emit = (df_feat["ts_event"] >= emit_lo) & (df_feat["ts_event"] < emit_hi_excl)
        else:
            emit_lo, emit_hi_incl = _emit_ts_bounds(shard_path)
            mask_emit = (df_feat["ts_event"] >= emit_lo) & (df_feat["ts_event"] <= emit_hi_incl)
        mask = mask_emit & (df_feat["ts_event"] >= tgt_lo) & (df_feat["ts_event"] <= tgt_hi)
        df_out = df_feat.loc[mask].reset_index(drop=True)
        del df_feat

        if df_out.empty:
            log.warning("Zero rows after emit filter for %s", shard_path.name)
            gc.collect()
            continue

        table = pa.Table.from_pandas(df_out, preserve_index=False)
        if writer is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(output_path), table.schema, compression="snappy")
        writer.write_table(table)
        total_out += len(df_out)
        last_chunk_for_summary = df_out
        emit_shard_idx += 1
        log.info(
            "Emit shard %d/%d (file %d/%d) %s -> wrote %d rows (total %d); float64->float32: %d cols",
            emit_shard_idx,
            len(raw_files_to_emit),
            i + 1,
            len(raw_files),
            shard_path.name,
            len(df_out),
            total_out,
            n_dc,
        )
        gc.collect()

    if writer is not None:
        writer.close()

    if total_out == 0:
        raise SystemExit("No rows written; check paths and join keys.")

    log.info("Done: %d rows -> %s", total_out, output_path)

    if last_chunk_for_summary is not None:
        summarize_features(last_chunk_for_summary, windows_sec=WINDOWS_SEC)


if __name__ == "__main__":
    main()
