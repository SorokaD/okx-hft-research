#!/usr/bin/env python3
# flake8: noqa
"""
Snapshot-aligned validation pipeline for order book reconstruction.

Why:
- Grid-based comparisons are misleading because reconstructed rows land on a grid (e.g. 100ms)
  while real exchange snapshots have independent timestamps.
- This tool reconstructs book state by replaying updates strictly up to each snapshot_ts.

Modes:
- snapshot_aligned: rebuild from raw snapshots + raw updates and compare EXACTLY at snapshot_ts
- grid_nearest: compare to reconstructed parquet row using a "floor" policy (last rec ts <= snapshot_ts)
  to demonstrate how metrics degrade for grid-based validation

Artifacts:
- validation_summary.json
- worst_points.csv
- forensic_cases/case_snapshot_<snapshot_ms>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

import sys

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from research.db.connection import get_engine  # noqa: E402
from research.reconstruction.book_state import OrderBookState  # noqa: E402
from research.reconstruction.reconstructor import load_snapshot_raw_by_ts  # noqa: E402


def _parse_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, utc=True)


def _validate_identifier(ident: str, kind: str) -> None:
    # Minimal safety for f-string SQL identifiers (schema/table names).
    if not ident.replace("_", "").isalnum():
        raise ValueError(f"Invalid {kind}: {ident!r}")


def _normalize_side(v: Any) -> str:
    s = str(v).strip().lower()
    if s in {"1", "bid", "bids", "buy"}:
        return "bid"
    if s in {"2", "ask", "asks", "sell"}:
        return "ask"
    return s


def _book_row_to_levels_from_record(rec_row: pd.Series, depth: int) -> dict[str, list[tuple[float | None, float | None]]]:
    bids: list[tuple[float | None, float | None]] = []
    asks: list[tuple[float | None, float | None]] = []
    for i in range(1, depth + 1):
        bpx = rec_row.get(f"bid_px_{i:02d}")
        bsz = rec_row.get(f"bid_sz_{i:02d}")
        apx = rec_row.get(f"ask_px_{i:02d}")
        asz = rec_row.get(f"ask_sz_{i:02d}")
        bids.append((float(bpx) if pd.notna(bpx) else None, float(bsz) if pd.notna(bsz) else None))
        asks.append((float(apx) if pd.notna(apx) else None, float(asz) if pd.notna(asz) else None))
    return {"bids": bids, "asks": asks}


def _levels_from_book_state(book: OrderBookState, depth: int) -> dict[str, list[tuple[float | None, float | None]]]:
    levels = book.get_top_levels(depth)
    bids: list[tuple[float | None, float | None]] = []
    asks: list[tuple[float | None, float | None]] = []
    for i in range(depth):
        if i < len(levels["bids"]):
            bids.append((float(levels["bids"][i][0]), float(levels["bids"][i][1])))
        else:
            bids.append((None, None))
        if i < len(levels["asks"]):
            asks.append((float(levels["asks"][i][0]), float(levels["asks"][i][1])))
        else:
            asks.append((None, None))
    return {"bids": bids, "asks": asks}


@dataclass
class SnapshotDiff:
    snapshot_ms: int
    snapshot_ts: pd.Timestamp
    rec_ts: pd.Timestamp
    top1_bid_match: bool
    top1_ask_match: bool
    crossed_book_real: bool
    crossed_book_rec: bool
    max_px_diff: float
    max_sz_diff: float
    mean_px_diff: float
    mean_sz_diff: float

    # Full breakdown: [("bids"/"asks", level_index_1_based, real_px, rec_px, abs_px_diff, real_sz, rec_sz, abs_sz_diff)]
    breakdown: list[dict[str, Any]]

    def _to_jsonable(self) -> dict[str, Any]:
        return {
            "snapshot_ms": int(self.snapshot_ms),
            "snapshot_ts": self.snapshot_ts.isoformat(),
            "rec_ts": self.rec_ts.isoformat(),
            "top1_bid_match": bool(self.top1_bid_match),
            "top1_ask_match": bool(self.top1_ask_match),
            "crossed_book_real": bool(self.crossed_book_real),
            "crossed_book_rec": bool(self.crossed_book_rec),
            "max_px_diff": float(self.max_px_diff),
            "max_sz_diff": float(self.max_sz_diff),
            "mean_px_diff": float(self.mean_px_diff),
            "mean_sz_diff": float(self.mean_sz_diff),
            "breakdown": self.breakdown,
        }


def _compare_levels(
    real_levels: dict[str, list[tuple[float | None, float | None]]],
    rec_levels: dict[str, list[tuple[float | None, float | None]]],
    depth: int,
) -> tuple[bool, bool, bool, bool, float, float, float, float, list[dict[str, Any]]]:
    px_diffs: list[float] = []
    sz_diffs: list[float] = []
    breakdown: list[dict[str, Any]] = []

    for side in ("bids", "asks"):
        for lvl0 in range(depth):
            level_index = lvl0 + 1
            real_px, real_sz = real_levels[side][lvl0]
            rec_px, rec_sz = rec_levels[side][lvl0]

            if real_px is None or rec_px is None:
                abs_px_diff = math.nan
            else:
                abs_px_diff = abs(float(rec_px) - float(real_px))

            if real_sz is None or rec_sz is None:
                abs_sz_diff = math.nan
            else:
                abs_sz_diff = abs(float(rec_sz) - float(real_sz))

            # Collect distributions only when both sides are present.
            if not math.isnan(abs_px_diff):
                px_diffs.append(abs_px_diff)
            if not math.isnan(abs_sz_diff):
                sz_diffs.append(abs_sz_diff)

            breakdown.append(
                {
                    "side": "bid" if side == "bids" else "ask",
                    "level": level_index,
                    "real_px": real_px,
                    "rec_px": rec_px,
                    "abs_px_diff": None if math.isnan(abs_px_diff) else abs_px_diff,
                    "real_sz": real_sz,
                    "rec_sz": rec_sz,
                    "abs_sz_diff": None if math.isnan(abs_sz_diff) else abs_sz_diff,
                }
            )

    # Crossed book checks on top of book (best bid/ask).
    rec_bid0 = rec_levels["bids"][0][0]
    rec_ask0 = rec_levels["asks"][0][0]
    real_bid0 = real_levels["bids"][0][0]
    real_ask0 = real_levels["asks"][0][0]
    crossed_book_rec = rec_bid0 is not None and rec_ask0 is not None and float(rec_bid0) >= float(rec_ask0)
    crossed_book_real = real_bid0 is not None and real_ask0 is not None and float(real_bid0) >= float(real_ask0)

    tol = 1e-9
    top1_bid_match = real_bid0 is not None and rec_bid0 is not None and abs(float(real_bid0) - float(rec_bid0)) <= tol
    top1_ask_match = real_ask0 is not None and rec_ask0 is not None and abs(float(real_ask0) - float(rec_ask0)) <= tol

    max_px_diff = float(np.nanmax(px_diffs)) if px_diffs else math.nan
    max_sz_diff = float(np.nanmax(sz_diffs)) if sz_diffs else math.nan
    mean_px_diff = float(np.nanmean(px_diffs)) if px_diffs else math.nan
    mean_sz_diff = float(np.nanmean(sz_diffs)) if sz_diffs else math.nan

    return (
        top1_bid_match,
        top1_ask_match,
        crossed_book_real,
        crossed_book_rec,
        max_px_diff,
        max_sz_diff,
        mean_px_diff,
        mean_sz_diff,
        breakdown,
    )


def _load_real_snapshot_top_levels(
    engine,
    schema: str,
    snapshot_table: str,
    inst_id: str,
    snapshot_ms_list: list[int],
    depth: int,
) -> dict[int, dict[str, list[tuple[float | None, float | None]]]]:
    if not snapshot_ms_list:
        return {}

    _validate_identifier(schema, "schema")
    _validate_identifier(snapshot_table, "table")

    # We need only top `depth` levels for validation comparison.
    # Use BETWEEN to avoid huge IN lists.
    start_ms = min(snapshot_ms_list)
    end_ms = max(snapshot_ms_list)
    sql = text(
        f"""
        SELECT ts_event_ms, side, level, price, size
        FROM {schema}.{snapshot_table}
        WHERE instid = :inst_id
          AND ts_event_ms BETWEEN :start_ms AND :end_ms
          AND level <= :depth
        ORDER BY ts_event_ms, side, level
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"inst_id": inst_id, "start_ms": start_ms, "end_ms": end_ms, "depth": depth})

    if df.empty:
        raise ValueError("No real snapshot levels found in requested range (after level<=depth filter).")

    # Build: snapshot_ms -> {"bids":[(px,sz)*depth], "asks":[(px,sz)*depth]}
    out: dict[int, dict[str, list[tuple[float | None, float | None]]]] = {}
    for ms in snapshot_ms_list:
        out[int(ms)] = {"bids": [(None, None)] * depth, "asks": [(None, None)] * depth}

    df["side"] = df["side"].map(_normalize_side)
    df["level"] = pd.to_numeric(df["level"], errors="coerce").astype("Int64")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["side", "level", "price", "size"])

    for _, r in df.iterrows():
        ms = int(r["ts_event_ms"])
        side = str(r["side"])
        lvl = int(r["level"])
        if ms not in out:
            continue
        idx = lvl - 1
        if idx < 0 or idx >= depth:
            continue
        px = float(r["price"])
        sz = float(r["size"])
        if side == "bid":
            out[ms]["bids"][idx] = (px, sz)
        elif side == "ask":
            out[ms]["asks"][idx] = (px, sz)
    return out


def _load_snapshot_ms_candidates(engine, schema: str, snapshot_table: str, inst_id: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[int]:
    _validate_identifier(schema, "schema")
    _validate_identifier(snapshot_table, "table")
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    sql = text(
        f"""
        SELECT DISTINCT ts_event_ms
        FROM {schema}.{snapshot_table}
        WHERE instid = :inst_id
          AND ts_event_ms BETWEEN :start_ms AND :end_ms
        ORDER BY ts_event_ms
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"inst_id": inst_id, "start_ms": start_ms, "end_ms": end_ms})
    if df.empty:
        raise ValueError("No snapshots found in requested window for validation.")
    return [int(v) for v in df["ts_event_ms"].tolist()]


def _find_anchor_snapshot_ms_raw(engine, schema: str, snapshot_table: str, inst_id: str, first_snapshot_ms: int) -> int:
    _validate_identifier(schema, "schema")
    _validate_identifier(snapshot_table, "table")
    sql = text(
        f"""
        SELECT MAX(ts_event_ms) AS anchor_ms
        FROM {schema}.{snapshot_table}
        WHERE instid = :inst_id
          AND ts_event_ms <= :first_snapshot_ms
        """
    )
    with engine.connect() as conn:
        row = pd.read_sql(sql, conn, params={"inst_id": inst_id, "first_snapshot_ms": first_snapshot_ms}).iloc[0]
    anchor_ms = row["anchor_ms"]
    if pd.isna(anchor_ms):
        raise FileNotFoundError("No raw anchor snapshot found at or before the first snapshot in the window.")
    return int(anchor_ms)


def _parse_level_item(level: Any) -> tuple[float, float] | None:
    if isinstance(level, dict):
        p = level.get("price")
        s = level.get("size")
        if p is None or s is None:
            return None
        try:
            return float(p), float(s)
        except (TypeError, ValueError):
            return None
    if isinstance(level, (list, tuple)) and len(level) >= 2:
        try:
            return float(level[0]), float(level[1])
        except (TypeError, ValueError):
            return None
    return None


def _apply_updates_streamed_to_book(
    book: OrderBookState,
    updates_df: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    ptr: int,
) -> tuple[int, int]:
    """
    Apply all update rows with ts_event <= snapshot_ts.

    Returns:
    - new pointer
    - number of applied rows
    """
    applied_rows = 0
    target_ms = int(snapshot_ts.timestamp() * 1000)

    while ptr < len(updates_df):
        row_ms = int(updates_df["ts_event_ms"].iloc[ptr])
        if row_ms > target_ms:
            break

        bids_delta = updates_df["bids_delta"].iloc[ptr]
        asks_delta = updates_df["asks_delta"].iloc[ptr]

        # bids_delta and asks_delta are JSONB arrays of levels [[price,size], ...] (or collector dicts)
        for side_name, delta in (("bid", bids_delta), ("ask", asks_delta)):
            if delta is None:
                continue
            if isinstance(delta, str):
                delta = json.loads(delta) if delta.strip() else []
            if not isinstance(delta, list):
                continue
            for item in delta:
                parsed = _parse_level_item(item)
                if parsed is None:
                    continue
                price, size = parsed
                book.apply_update(side_name, price, float(size))

        ptr += 1
        applied_rows += 1

    return ptr, applied_rows


def _run_snapshot_aligned(
    engine,
    schema: str,
    inst_id: str,
    snapshot_table: str,
    updates_table: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    depth: int,
    mode_output_dir: Path,
    worst_k: int,
    snapshot_limit: int | None,
) -> dict[str, Any]:
    # 1) Get snapshot timestamps to validate.
    snapshot_ms_list = _load_snapshot_ms_candidates(engine, schema, snapshot_table, inst_id, start_ts, end_ts)
    if snapshot_limit and snapshot_limit > 0:
        snapshot_ms_list = snapshot_ms_list[:snapshot_limit]
    snapshot_ms_list_sorted = sorted(snapshot_ms_list)
    snapshot_ts_list = [pd.Timestamp(ms, unit="ms", tz="utc") for ms in snapshot_ms_list_sorted]

    first_snapshot_ms = snapshot_ms_list_sorted[0]
    last_snapshot_ms = snapshot_ms_list_sorted[-1]

    # 2) Build book state incrementally:
    anchor_ms = _find_anchor_snapshot_ms_raw(engine, schema, snapshot_table, inst_id, first_snapshot_ms)
    anchor_row = load_snapshot_raw_by_ts(
        engine=engine,
        schema=schema,
        snapshot_table=snapshot_table,
        inst_id=inst_id,
        ts_event_ms=anchor_ms,
        snapshot_max_depth=None,  # Critical for correctness: avoid hard depth truncation.
    )
    book = OrderBookState.from_snapshot_row(anchor_row)

    # 3) Pre-load real snapshot top levels (only depth<=N, so it's smaller).
    real_top = _load_real_snapshot_top_levels(
        engine=engine,
        schema=schema,
        snapshot_table=snapshot_table,
        inst_id=inst_id,
        snapshot_ms_list=snapshot_ms_list_sorted,
        depth=depth,
    )

    # 4) Load updates rows once for entire range and apply incrementally.
    _validate_identifier(schema, "schema")
    _validate_identifier(updates_table, "table")
    updates_start_ms = anchor_ms
    updates_end_ms = last_snapshot_ms
    updates_sql = text(
        f"""
        SELECT ts_event_ms, ts_ingest_ms, bids_delta, asks_delta
        FROM {schema}.{updates_table}
        WHERE instid = :inst_id
          AND ts_event_ms BETWEEN :start_ms AND :end_ms
        ORDER BY ts_event_ms, ts_ingest_ms
        """
    )
    with engine.connect() as conn:
        updates_df = pd.read_sql(
            updates_sql,
            conn,
            params={"inst_id": inst_id, "start_ms": updates_start_ms, "end_ms": updates_end_ms},
        )

    if updates_df.empty:
        raise ValueError("No updates found for snapshot-aligned validation range.")

    # 5) Iterate snapshots, apply updates strictly up to snapshot_ts, compare.
    ptr = 0
    px_diffs_all: list[float] = []
    sz_diffs_all: list[float] = []
    top1_bid_matches = 0
    top1_ask_matches = 0

    per_snapshot_scores: list[dict[str, Any]] = []
    # Keep per-snapshot forensic details in-memory so worst-K export doesn't require re-play.
    snapshot_details: dict[int, dict[str, Any]] = {}

    for snap_ts in snapshot_ts_list:
        snap_ms = int(snap_ts.timestamp() * 1000)
        ptr, _ = _apply_updates_streamed_to_book(book, updates_df, snap_ts, ptr)

        rec_levels = _levels_from_book_state(book, depth=depth)
        real_levels = real_top[snap_ms]

        (
            top1_bid_match,
            top1_ask_match,
            crossed_book_real,
            crossed_book_rec,
            max_px_diff,
            max_sz_diff,
            mean_px_diff,
            mean_sz_diff,
            breakdown,
        ) = _compare_levels(real_levels, rec_levels, depth=depth)

        if top1_bid_match:
            top1_bid_matches += 1
        if top1_ask_match:
            top1_ask_matches += 1

        # Collect px/sz diffs from breakdown.
        for side in ("bids", "asks"):
            pass
        # breakdown contains abs diffs or None; gather numeric diffs.
        for item in breakdown:
            if item["abs_px_diff"] is not None:
                px_diffs_all.append(float(item["abs_px_diff"]))
            if item["abs_sz_diff"] is not None:
                sz_diffs_all.append(float(item["abs_sz_diff"]))

        diff_obj = SnapshotDiff(
            snapshot_ms=snap_ms,
            snapshot_ts=snap_ts,
            rec_ts=snap_ts,
            top1_bid_match=top1_bid_match,
            top1_ask_match=top1_ask_match,
            crossed_book_real=crossed_book_real,
            crossed_book_rec=crossed_book_rec,
            max_px_diff=max_px_diff,
            max_sz_diff=max_sz_diff,
            mean_px_diff=mean_px_diff,
            mean_sz_diff=mean_sz_diff,
            breakdown=breakdown,
        )

        snapshot_details[snap_ms] = {
            "snapshot_ms": snap_ms,
            "snapshot_ts": snap_ts,
            "top1_bid_match": top1_bid_match,
            "top1_ask_match": top1_ask_match,
            "crossed_book_real": crossed_book_real,
            "crossed_book_rec": crossed_book_rec,
            "max_px_diff": max_px_diff,
            "max_sz_diff": max_sz_diff,
            "mean_px_diff": mean_px_diff,
            "mean_sz_diff": mean_sz_diff,
            "rec_levels": rec_levels,
            "real_levels": real_levels,
            "breakdown": breakdown,
        }

        per_snapshot_scores.append(
            {
                "snapshot_ms": snap_ms,
                "snapshot_ts": snap_ts.isoformat(),
                "top1_bid_match": top1_bid_match,
                "top1_ask_match": top1_ask_match,
                "crossed_book_real": crossed_book_real,
                "crossed_book_rec": crossed_book_rec,
                "max_px_diff": max_px_diff,
                "max_sz_diff": max_sz_diff,
                "mean_px_diff": mean_px_diff,
                "mean_sz_diff": mean_sz_diff,
            }
        )

    scores_df = pd.DataFrame(per_snapshot_scores)
    if scores_df.empty:
        raise ValueError("No validation points produced.")

    # Determine worst K by (max_px_diff desc, max_sz_diff desc).
    scores_df = scores_df.sort_values(["max_px_diff", "max_sz_diff"], ascending=False).reset_index(drop=True)
    worst_df = scores_df.head(worst_k).copy()

    # Write forensic cases for worst points.
    forensic_cases_dir = mode_output_dir / "forensic_cases"
    forensic_cases_dir.mkdir(parents=True, exist_ok=True)

    for _, row in worst_df.iterrows():
        snap_ms = int(row["snapshot_ms"])
        snap = snapshot_details[snap_ms]
        rec_levels = snap["rec_levels"]
        real_levels = snap["real_levels"]

        (
            top1_bid_match,
            top1_ask_match,
            crossed_book_real,
            crossed_book_rec,
            max_px_diff,
            max_sz_diff,
            mean_px_diff,
            mean_sz_diff,
            breakdown,
        ) = _compare_levels(real_levels, rec_levels, depth=depth)

        # Store reconstructed/real top levels for readability.
        def _levels_to_list(levels: dict[str, list[tuple[float | None, float | None]]]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for side in ("bids", "asks"):
                side_out = "bid" if side == "bids" else "ask"
                for i, (px, sz) in enumerate(levels[side], start=1):
                    rows.append(
                        {
                            "side": side_out,
                            "level": i,
                            "px": px,
                            "sz": sz,
                        }
                    )
            return rows

        case_obj = {
            "snapshot_ms": snap_ms,
            "snapshot_ts": snap["snapshot_ts"].isoformat(),
            "depth": depth,
            "top1_bid_match": bool(top1_bid_match),
            "top1_ask_match": bool(top1_ask_match),
            "crossed_book_real": bool(crossed_book_real),
            "crossed_book_rec": bool(crossed_book_rec),
            "max_px_diff": float(max_px_diff),
            "max_sz_diff": float(max_sz_diff),
            "mean_px_diff": float(mean_px_diff),
            "mean_sz_diff": float(mean_sz_diff),
            "reconstructed_top_levels": _levels_to_list(rec_levels),
            "real_snapshot_top_levels": _levels_to_list(real_levels),
            "diff_breakdown": breakdown,
        }

        case_path = forensic_cases_dir / f"case_snapshot_{snap_ms}.json"
        case_path.write_text(json.dumps(case_obj, indent=2), encoding="utf-8")

    # Metrics.
    px_diffs_arr = np.array(px_diffs_all, dtype=float) if px_diffs_all else np.array([], dtype=float)
    sz_diffs_arr = np.array(sz_diffs_all, dtype=float) if sz_diffs_all else np.array([], dtype=float)

    top1_bid_match_rate = float(top1_bid_matches / len(snapshot_ts_list)) if snapshot_ts_list else math.nan
    top1_ask_match_rate = float(top1_ask_matches / len(snapshot_ts_list)) if snapshot_ts_list else math.nan

    summary = {
        "mode": "snapshot_aligned",
        "window_start": start_ts.isoformat(),
        "window_end": end_ts.isoformat(),
        "inst_id": inst_id,
        "depth": depth,
        "validated_points": len(snapshot_ts_list),
        "worst_k": worst_k,
        "top1_bid_match_rate": top1_bid_match_rate,
        "top1_ask_match_rate": top1_ask_match_rate,
        "mean_px_diff": float(np.mean(px_diffs_arr)) if px_diffs_arr.size else math.nan,
        "p95_px_diff": float(np.percentile(px_diffs_arr, 95)) if px_diffs_arr.size else math.nan,
        "max_px_diff": float(np.max(px_diffs_arr)) if px_diffs_arr.size else math.nan,
        "mean_sz_diff": float(np.mean(sz_diffs_arr)) if sz_diffs_arr.size else math.nan,
        "p95_sz_diff": float(np.percentile(sz_diffs_arr, 95)) if sz_diffs_arr.size else math.nan,
        "max_sz_diff": float(np.max(sz_diffs_arr)) if sz_diffs_arr.size else math.nan,
        "worst_points": worst_df.to_dict(orient="records"),
        "forensic_cases_written": int(len(worst_df)),
    }

    (mode_output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    worst_df.to_csv(mode_output_dir / "worst_points.csv", index=False)

    return summary


def _run_grid_nearest(
    engine,
    reconstructed_parquet: str,
    schema: str,
    inst_id: str,
    snapshot_table: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    depth: int,
    mode_output_dir: Path,
    worst_k: int,
    snapshot_limit: int | None,
) -> dict[str, Any]:
    _validate_identifier(schema, "schema")
    _validate_identifier(snapshot_table, "table")

    rec_df = pd.read_parquet(reconstructed_parquet).copy()
    if rec_df.empty:
        raise ValueError("Reconstructed parquet is empty")
    rec_df["ts_event"] = pd.to_datetime(rec_df["ts_event"], utc=True, errors="coerce")
    rec_df = rec_df.sort_values("ts_event").reset_index(drop=True)

    window_start = start_ts
    window_end = end_ts
    rec_window = rec_df[(rec_df["ts_event"] >= window_start) & (rec_df["ts_event"] <= window_end)].copy()
    if rec_window.empty:
        raise ValueError("No reconstructed rows inside requested window.")

    snapshot_ms_list = _load_snapshot_ms_candidates(engine, schema, snapshot_table, inst_id, start_ts, end_ts)
    if snapshot_limit and snapshot_limit > 0:
        snapshot_ms_list = snapshot_ms_list[:snapshot_limit]
    snapshot_ms_list_sorted = sorted(snapshot_ms_list)
    snapshot_ts_list = [pd.Timestamp(ms, unit="ms", tz="utc") for ms in snapshot_ms_list_sorted]
    if not snapshot_ts_list:
        raise ValueError("No snapshot timestamps to validate.")

    # Load real top levels for snapshots.
    real_top = _load_real_snapshot_top_levels(
        engine=engine,
        schema=schema,
        snapshot_table=snapshot_table,
        inst_id=inst_id,
        snapshot_ms_list=snapshot_ms_list_sorted,
        depth=depth,
    )

    px_diffs_all: list[float] = []
    sz_diffs_all: list[float] = []
    top1_bid_matches = 0
    top1_ask_matches = 0
    per_snapshot_scores: list[dict[str, Any]] = []

    for snap_ts in snapshot_ts_list:
        snap_ms = int(snap_ts.timestamp() * 1000)

        c = rec_window.loc[rec_window["ts_event"] <= snap_ts]
        if c.empty:
            continue
        rec_row = c.iloc[-1]

        rec_levels = _book_row_to_levels_from_record(rec_row, depth=depth)
        real_levels = real_top[snap_ms]

        (
            top1_bid_match,
            top1_ask_match,
            crossed_book_real,
            crossed_book_rec,
            max_px_diff,
            max_sz_diff,
            mean_px_diff,
            mean_sz_diff,
            breakdown,
        ) = _compare_levels(real_levels, rec_levels, depth=depth)

        if top1_bid_match:
            top1_bid_matches += 1
        if top1_ask_match:
            top1_ask_matches += 1

        for item in breakdown:
            if item["abs_px_diff"] is not None:
                px_diffs_all.append(float(item["abs_px_diff"]))
            if item["abs_sz_diff"] is not None:
                sz_diffs_all.append(float(item["abs_sz_diff"]))

        per_snapshot_scores.append(
            {
                "snapshot_ms": snap_ms,
                "snapshot_ts": snap_ts.isoformat(),
                "top1_bid_match": bool(top1_bid_match),
                "top1_ask_match": bool(top1_ask_match),
                "crossed_book_real": bool(crossed_book_real),
                "crossed_book_rec": bool(crossed_book_rec),
                "max_px_diff": max_px_diff,
                "max_sz_diff": max_sz_diff,
                "mean_px_diff": mean_px_diff,
                "mean_sz_diff": mean_sz_diff,
            }
        )

    if not per_snapshot_scores:
        raise ValueError("Grid_nearest validation produced zero comparable points (maybe window mismatch).")

    scores_df = pd.DataFrame(per_snapshot_scores).sort_values(["max_px_diff", "max_sz_diff"], ascending=False).reset_index(drop=True)
    worst_df = scores_df.head(worst_k).copy()

    # Forensic cases: just store reconstructed and real levels from parquet row.
    forensic_cases_dir = mode_output_dir / "forensic_cases"
    forensic_cases_dir.mkdir(parents=True, exist_ok=True)

    # We need rec_row for the worst snapshots to store top levels.
    for _, row in worst_df.iterrows():
        snap_ms = int(row["snapshot_ms"])
        snap_ts = pd.Timestamp(snap_ms, unit="ms", tz="utc")

        c = rec_window.loc[rec_window["ts_event"] <= snap_ts]
        if c.empty:
            continue
        rec_row = c.iloc[-1]

        rec_levels = _book_row_to_levels_from_record(rec_row, depth=depth)
        real_levels = real_top[snap_ms]

        (
            top1_bid_match,
            top1_ask_match,
            crossed_book_real,
            crossed_book_rec,
            max_px_diff,
            max_sz_diff,
            mean_px_diff,
            mean_sz_diff,
            breakdown,
        ) = _compare_levels(real_levels, rec_levels, depth=depth)

        def _levels_to_list(levels: dict[str, list[tuple[float | None, float | None]]]) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for side in ("bids", "asks"):
                side_out = "bid" if side == "bids" else "ask"
                for i, (px, sz) in enumerate(levels[side], start=1):
                    rows.append({"side": side_out, "level": i, "px": px, "sz": sz})
            return rows

        case_obj = {
            "snapshot_ms": snap_ms,
            "snapshot_ts": snap_ts.isoformat(),
            "depth": depth,
            "top1_bid_match": bool(top1_bid_match),
            "top1_ask_match": bool(top1_ask_match),
            "crossed_book_real": bool(crossed_book_real),
            "crossed_book_rec": bool(crossed_book_rec),
            "max_px_diff": float(max_px_diff) if not math.isnan(max_px_diff) else math.nan,
            "max_sz_diff": float(max_sz_diff) if not math.isnan(max_sz_diff) else math.nan,
            "mean_px_diff": float(mean_px_diff) if not math.isnan(mean_px_diff) else math.nan,
            "mean_sz_diff": float(mean_sz_diff) if not math.isnan(mean_sz_diff) else math.nan,
            "reconstructed_top_levels": _levels_to_list(rec_levels),
            "real_snapshot_top_levels": _levels_to_list(real_levels),
            "diff_breakdown": breakdown,
        }
        (forensic_cases_dir / f"case_snapshot_{snap_ms}.json").write_text(
            json.dumps(case_obj, indent=2),
            encoding="utf-8",
        )

    px_diffs_arr = np.array(px_diffs_all, dtype=float) if px_diffs_all else np.array([], dtype=float)
    sz_diffs_arr = np.array(sz_diffs_all, dtype=float) if sz_diffs_all else np.array([], dtype=float)

    validated_points = len(per_snapshot_scores)
    top1_bid_match_rate = float(top1_bid_matches / validated_points) if validated_points else math.nan
    top1_ask_match_rate = float(top1_ask_matches / validated_points) if validated_points else math.nan

    summary = {
        "mode": "grid_nearest",
        "window_start": start_ts.isoformat(),
        "window_end": end_ts.isoformat(),
        "inst_id": inst_id,
        "depth": depth,
        "validated_points": validated_points,
        "worst_k": worst_k,
        "top1_bid_match_rate": top1_bid_match_rate,
        "top1_ask_match_rate": top1_ask_match_rate,
        "mean_px_diff": float(np.mean(px_diffs_arr)) if px_diffs_arr.size else math.nan,
        "p95_px_diff": float(np.percentile(px_diffs_arr, 95)) if px_diffs_arr.size else math.nan,
        "max_px_diff": float(np.max(px_diffs_arr)) if px_diffs_arr.size else math.nan,
        "mean_sz_diff": float(np.mean(sz_diffs_arr)) if sz_diffs_arr.size else math.nan,
        "p95_sz_diff": float(np.percentile(sz_diffs_arr, 95)) if sz_diffs_arr.size else math.nan,
        "max_sz_diff": float(np.max(sz_diffs_arr)) if sz_diffs_arr.size else math.nan,
        "worst_points": worst_df.to_dict(orient="records"),
        "forensic_cases_written": int(len(worst_df)),
    }

    (mode_output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    worst_df.to_csv(mode_output_dir / "worst_points.csv", index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate order book reconstruction (snapshot-aligned).")
    parser.add_argument("--mode", choices=["snapshot_aligned", "grid_nearest"], default="snapshot_aligned")
    parser.add_argument("--inst-id", required=True)
    parser.add_argument("--schema", default="okx_raw")
    parser.add_argument("--snapshot-table", default="orderbook_snapshots")
    parser.add_argument("--updates-table", default="orderbook_updates")
    parser.add_argument("--start", required=True, help="UTC timestamp start")
    parser.add_argument("--end", required=True, help="UTC timestamp end")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--worst-k", type=int, default=20)
    parser.add_argument("--snapshot-limit", type=int, default=0, help="0 = no limit; otherwise validate only first N snapshot points.")
    parser.add_argument("--reconstructed-parquet", help="Required for grid_nearest mode")
    parser.add_argument(
        "--output-dir",
        default="data/diagnostics/orderbook_validation",
        help="Directory for validation artifacts",
    )
    args = parser.parse_args()

    start_ts = _parse_ts(args.start)
    end_ts = _parse_ts(args.end)
    if end_ts < start_ts:
        raise ValueError("--end must be >= --start")

    snapshot_limit = args.snapshot_limit if args.snapshot_limit and args.snapshot_limit > 0 else None

    engine = get_engine()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize output directory by mode for easier comparison.
    mode_out_dir = out_dir / args.mode / f"{args.inst_id}_{start_ts.strftime('%Y-%m-%d_%H-%M-%S')}__{end_ts.strftime('%Y-%m-%d_%H-%M-%S')}"
    mode_out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "grid_nearest":
        if not args.reconstructed_parquet:
            raise ValueError("--reconstructed-parquet is required for grid_nearest mode")
        _run_grid_nearest(
            engine=engine,
            reconstructed_parquet=args.reconstructed_parquet,
            schema=args.schema,
            inst_id=args.inst_id,
            snapshot_table=args.snapshot_table,
            start_ts=start_ts,
            end_ts=end_ts,
            depth=args.depth,
            mode_output_dir=mode_out_dir,
            worst_k=args.worst_k,
            snapshot_limit=snapshot_limit,
        )
    else:
        _run_snapshot_aligned(
            engine=engine,
            schema=args.schema,
            inst_id=args.inst_id,
            snapshot_table=args.snapshot_table,
            updates_table=args.updates_table,
            start_ts=start_ts,
            end_ts=end_ts,
            depth=args.depth,
            mode_output_dir=mode_out_dir,
            worst_k=args.worst_k,
            snapshot_limit=snapshot_limit,
        )

    print(f"Validation artifacts written to: {mode_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

