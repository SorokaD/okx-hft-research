#!/usr/bin/env python3
"""
Reconstruct order book from snapshot + updates and save to Parquet.

Supports two data sources:
- core: okx_core tables (fact_orderbook_l10_snapshot, fact_orderbook_update_level)
- raw: okx_raw tables (orderbook_snapshots, orderbook_updates) — OKX native format

Output layout: {output_dir}/{inst_id}/{mode}/YYYY-MM-DD/filename.parquet

Example (raw, grid 100ms):
    python scripts/reconstruct_orderbook_to_parquet.py \
      --source raw \
      --schema okx_raw \
      --inst-id BTC-USDT-SWAP \
      --start "2026-03-03 00:00:00" \
      --end "2026-03-04 00:00:00" \
      --chunk-hours 1 \
      --mode grid \
      --grid-ms 100 \
      --output-dir data/reconstructed \
      --filename-prefix book

Example (core, event mode):
    python scripts/reconstruct_orderbook_to_parquet.py \
      --source core \
      --snapshot-schema okx_core \
      --snapshot-table fact_orderbook_l10_snapshot \
      --updates-schema okx_core \
      --updates-table fact_orderbook_update_level \
      --time-column ts_event \
      --price-column price_px \
      --size-column size_qty \
      --inst-id BTC-USDT-SWAP \
      --start "2026-03-01 00:00:00" \
      --end "2026-03-01 06:00:00" \
      --chunk-hours 1 \
      --mode event \
      --output-dir data/reconstructed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

# Add project root so "research" package is importable
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from research.db.connection import get_engine  # noqa: E402
from research.reconstruction.reconstructor import (  # noqa: E402
    find_anchor_snapshot,
    find_anchor_snapshot_raw,
    load_updates,
    load_updates_raw,
    reconstruct_event_states,
    reconstruct_grid_states,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_ts(s: str) -> pd.Timestamp:
    """Parse timestamp string to timezone-aware Timestamp (UTC)."""
    ts = pd.to_datetime(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts


def _chunk_ranges(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_hours: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield (chunk_start, chunk_end) pairs."""
    chunk_delta = timedelta(hours=chunk_hours)
    chunks = []
    t = start
    while t < end:
        t_end = min(t + chunk_delta, end)
        chunks.append((t, t_end))
        t = t_end
    return chunks


def _run_check(engine, args) -> None:
    """Check DB connection and first chunk data."""
    from research.reconstruction.reconstructor import (
        find_anchor_snapshot_raw,
        load_updates_raw,
    )

    chunk_start, chunk_end = _chunk_ranges(
        _parse_ts(args.start), _parse_ts(args.end), args.chunk_hours
    )[0]
    chunk_start_ms = int(chunk_start.timestamp() * 1000)
    chunk_end_ms = int(chunk_end.timestamp() * 1000)
    log.info("Check: chunk [%s .. %s] -> ms [%d .. %d]", chunk_start, chunk_end, chunk_start_ms, chunk_end_ms)
    try:
        anchor = find_anchor_snapshot_raw(
            engine, args.schema, "orderbook_snapshots", args.inst_id, chunk_start
        )
        if anchor is None:
            log.warning("No anchor snapshot found")
            return
        anchor_ms = int(anchor["ts_event"].timestamp() * 1000)
        log.info("Anchor: ts_event_ms=%d (%s)", anchor_ms, anchor["ts_event"])
        updates_df = load_updates_raw(
            engine, args.schema, "orderbook_updates", args.inst_id, anchor["ts_event"], chunk_end
        )
        log.info("Updates: %d rows", len(updates_df))
        if updates_df.empty:
            log.warning("No updates in chunk - check ts_event_ms range")
    except Exception as e:
        log.exception("Check failed: %s", e)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct order book from snapshot + updates to Parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["core", "raw"],
        default="raw",
        help="Data source: raw (okx_raw.orderbook_*) or core (okx_core.fact_*)",
    )
    parser.add_argument(
        "--schema",
        default="okx_raw",
        help="Schema for raw source (used for both snapshot and updates)",
    )
    parser.add_argument("--snapshot-schema", help="Snapshot schema (for core source)")
    parser.add_argument("--snapshot-table", help="Snapshot table (for core: fact_orderbook_l10_snapshot)")
    parser.add_argument("--updates-schema", help="Updates schema (for core source)")
    parser.add_argument("--updates-table", help="Updates table (for core: fact_orderbook_update_level)")
    parser.add_argument(
        "--time-column",
        default="ts_event",
        help="Timestamp column for filtering (core source only)",
    )
    parser.add_argument(
        "--price-column",
        default="price_px",
        help="Price column for core source (fact_orderbook_update_level)",
    )
    parser.add_argument(
        "--size-column",
        default="size_qty",
        help="Size column for core source (fact_orderbook_update_level)",
    )
    parser.add_argument("--inst-id", required=True, help="Instrument ID (e.g. BTC-USDT-SWAP)")
    parser.add_argument(
        "--start",
        required=True,
        help="Start time (e.g. '2026-03-01 00:00:00')",
    )
    parser.add_argument("--end", required=True, help="End time")
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=1,
        help="Process period in chunks of N hours",
    )
    parser.add_argument(
        "--mode",
        choices=["event", "grid"],
        default="event",
        help="Reconstruction mode",
    )
    parser.add_argument(
        "--grid-ms",
        type=int,
        default=50,
        help="Grid step in ms (for mode=grid)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Book depth (L10)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        help="Parquet compression (snappy, gzip, etc.)",
    )
    parser.add_argument(
        "--filename-prefix",
        default="book",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging (print chunk_start_ms, row counts)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check DB connection and first chunk, then exit",
    )
    args = parser.parse_args()

    if args.source == "core":
        for name in ("snapshot_schema", "snapshot_table", "updates_schema", "updates_table"):
            if not getattr(args, name, None):
                parser.error(f"--source core requires --{name.replace('_', '-')}")

    start_ts = _parse_ts(args.start)
    end_ts = _parse_ts(args.end)
    engine = get_engine()

    if args.check:
        _run_check(engine, args)
        return 0

    mode_suffix = f"grid{args.grid_ms}ms" if args.mode == "grid" else "event"
    chunks = _chunk_ranges(start_ts, end_ts, args.chunk_hours)
    files_written = 0
    rows_total = 0
    chunks_skipped = 0

    t0 = time.perf_counter()
    for chunk_start, chunk_end in chunks:
        log.info("Chunk [%s .. %s]", chunk_start, chunk_end)
        chunk_t0 = time.perf_counter()

        if args.source == "raw":
            if args.verbose:
                chunk_ms = int(chunk_start.timestamp() * 1000)
                log.info("Chunk start ms: %d", chunk_ms)
            anchor = find_anchor_snapshot_raw(
                engine,
                args.schema,
                "orderbook_snapshots",
                args.inst_id,
                chunk_start,
            )
            if anchor is None:
                log.warning(
                    "No anchor snapshot before %s, skipping chunk", chunk_start
                )
                chunks_skipped += 1
                continue
            anchor_ts = anchor["ts_event"]
            updates_df = load_updates_raw(
                engine,
                args.schema,
                "orderbook_updates",
                args.inst_id,
                anchor_ts,
                chunk_end,
            )
        else:
            anchor = find_anchor_snapshot(
                engine,
                args.snapshot_schema,
                args.snapshot_table,
                args.inst_id,
                args.time_column,
                chunk_start,
            )
            if anchor is None:
                log.warning(
                    "No anchor snapshot before %s, skipping chunk", chunk_start
                )
                chunks_skipped += 1
                continue
            anchor_ts = anchor["ts_event"]
            updates_df = load_updates(
                engine,
                args.updates_schema,
                args.updates_table,
                args.inst_id,
                args.time_column,
                anchor_ts,
                chunk_end,
                price_column=args.price_column,
                size_column=args.size_column,
            )

        if updates_df.empty:
            log.warning("No updates in chunk, skipping")
            chunks_skipped += 1
            continue

        log.info("Loaded %d update rows", len(updates_df))

        if args.mode == "event":
            df = reconstruct_event_states(
                anchor,
                updates_df,
                chunk_start,
                chunk_end,
                depth=args.depth,
                price_col="price",
                size_col="size",
            )
        else:
            df = reconstruct_grid_states(
                anchor,
                updates_df,
                chunk_start,
                chunk_end,
                grid_ms=args.grid_ms,
                depth=args.depth,
                price_col="price",
                size_col="size",
            )

        if df.empty:
            log.warning("Reconstructed DataFrame empty, skipping file")
            chunks_skipped += 1
            continue

        date_str = chunk_start.strftime("%Y-%m-%d")
        time_start_str = chunk_start.strftime("%H-%M-%S")
        time_end_str = chunk_end.strftime("%H-%M-%S")
        fname = (
            f"{args.filename_prefix}_{mode_suffix}_{args.inst_id}_"
            f"{date_str}_{time_start_str}__"
            f"{chunk_end.strftime('%Y-%m-%d')}_{time_end_str}.parquet"
        )
        out_dir = args.output_dir / args.inst_id / mode_suffix / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / fname

        df.to_parquet(out_path, index=False, compression=args.compression)
        files_written += 1
        rows_total += len(df)
        chunk_elapsed = time.perf_counter() - chunk_t0
        log.info(
            "Wrote %s (%d rows) in %.1fs", out_path, len(df), chunk_elapsed
        )

    total_elapsed = time.perf_counter() - t0
    log.info(
        "Summary: chunks=%d, files=%d, rows=%d, skipped=%d, total=%.1fs",
        len(chunks),
        files_written,
        rows_total,
        chunks_skipped,
        total_elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
