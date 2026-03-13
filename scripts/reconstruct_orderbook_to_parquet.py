#!/usr/bin/env python3
"""
Reconstruct order book from snapshot + updates and save to Parquet.

Chunks the requested period by --chunk-hours, finds anchor snapshot per chunk,
loads updates, reconstructs states (event or grid), and writes Parquet files.

Output layout: {output_dir}/{inst_id}/{mode}/YYYY-MM-DD/filename.parquet

Example (event mode):
    python scripts/reconstruct_orderbook_to_parquet.py \
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
      --depth 10 \
      --output-dir data/reconstructed \
      --filename-prefix book

Example (grid 50ms):
    python scripts/reconstruct_orderbook_to_parquet.py \
      --snapshot-schema okx_core \
      --snapshot-table fact_orderbook_l10_snapshot \
      --updates-schema okx_core \
      --updates-table fact_orderbook_update_level \
      --time-column ts_event \
      --price-column price_px \
      --size-column size_qty \
      --inst-id BTC-USDT-SWAP \
      --start "2026-03-01 00:00:00" \
      --end "2026-03-01 01:00:00" \
      --chunk-hours 1 \
      --mode grid \
      --grid-ms 50 \
      --output-dir data/reconstructed \
      --filename-prefix book
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
    load_updates,
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
    """Parse timestamp string to timezone-aware Timestamp."""
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct order book from snapshot + updates to Parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot-schema", required=True, help="Snapshot table schema")
    parser.add_argument("--snapshot-table", required=True, help="Snapshot table name")
    parser.add_argument("--updates-schema", required=True, help="Updates table schema")
    parser.add_argument("--updates-table", required=True, help="Updates table name")
    parser.add_argument(
        "--time-column",
        default="ts_event",
        help="Timestamp column for filtering",
    )
    parser.add_argument(
        "--price-column",
        default="price",
        help="Price column (use price_px for fact_orderbook_update_level)",
    )
    parser.add_argument(
        "--size-column",
        default="size",
        help="Size column (use size_qty for fact_orderbook_update_level)",
    )
    parser.add_argument("--inst-id", required=True, help="Instrument ID")
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
    args = parser.parse_args()

    start_ts = _parse_ts(args.start)
    end_ts = _parse_ts(args.end)
    engine = get_engine()

    mode_suffix = f"grid{args.grid_ms}ms" if args.mode == "grid" else "event"
    chunks = _chunk_ranges(start_ts, end_ts, args.chunk_hours)
    files_written = 0
    rows_total = 0
    chunks_skipped = 0

    t0 = time.perf_counter()
    for chunk_start, chunk_end in chunks:
        log.info("Chunk [%s .. %s]", chunk_start, chunk_end)
        chunk_t0 = time.perf_counter()

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
