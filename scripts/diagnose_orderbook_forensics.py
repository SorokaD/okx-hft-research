#!/usr/bin/env python3
# flake8: noqa
"""
Diagnostics-first forensic tool for order book reconstruction.

Scope:
- identify worst-case validation points
- audit ordering stability for updates with same ts_event
- add lightweight order book sanity counters in summary

This tool is intentionally observability-focused and does NOT change
reconstruction semantics.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

import sys

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from research.db.connection import get_engine  # noqa: E402
from research.reconstruction.book_state import OrderBookState  # noqa: E402
from research.reconstruction.reconstructor import (  # noqa: E402
    load_snapshot_raw_by_ts,
)


@dataclass
class ValidationPoint:
    snapshot_ts: pd.Timestamp
    snapshot_ms: int
    rec_ts: pd.Timestamp
    max_px_diff: float
    max_sz_diff: float
    top1_bid_match: bool
    top1_ask_match: bool
    crossed_book_real: bool
    crossed_book_rec: bool
    negative_or_zero_size_real: int
    negative_or_zero_size_rec: int


def _parse_ts(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, utc=True)
    return ts


def _normalize_side(v: Any) -> str:
    s = str(v).strip().lower()
    if s in {"1", "bid", "bids", "buy"}:
        return "bid"
    if s in {"2", "ask", "asks", "sell"}:
        return "ask"
    return s


def _real_snapshot_to_top_levels(real_df: pd.DataFrame, depth: int) -> dict[str, list[tuple[float, float]]]:
    df = real_df.copy()
    df["side"] = df["side"].map(_normalize_side)
    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["side", "level", "price", "size"])

    bids = (
        df[df["side"] == "bid"]
        .sort_values("level")
        .head(depth)[["price", "size"]]
        .itertuples(index=False, name=None)
    )
    asks = (
        df[df["side"] == "ask"]
        .sort_values("level")
        .head(depth)[["price", "size"]]
        .itertuples(index=False, name=None)
    )
    return {"bids": list(bids), "asks": list(asks)}


def _rec_row_to_top_levels(rec_row: pd.Series, depth: int) -> dict[str, list[tuple[float, float]]]:
    bids: list[tuple[float, float]] = []
    asks: list[tuple[float, float]] = []
    for i in range(1, depth + 1):
        bpx = rec_row.get(f"bid_px_{i:02d}")
        bsz = rec_row.get(f"bid_sz_{i:02d}")
        apx = rec_row.get(f"ask_px_{i:02d}")
        asz = rec_row.get(f"ask_sz_{i:02d}")
        if pd.notna(bpx) and pd.notna(bsz):
            bids.append((float(bpx), float(bsz)))
        if pd.notna(apx) and pd.notna(asz):
            asks.append((float(apx), float(asz)))
    return {"bids": bids, "asks": asks}


def _compare_levels(real_levels: dict[str, list[tuple[float, float]]], rec_levels: dict[str, list[tuple[float, float]]], depth: int) -> dict[str, Any]:
    max_px_diff = 0.0
    max_sz_diff = 0.0
    top1_bid_match = False
    top1_ask_match = False

    for side in ("bids", "asks"):
        real_side = real_levels[side]
        rec_side = rec_levels[side]
        for i in range(depth):
            if i >= len(real_side) or i >= len(rec_side):
                continue
            rp, rs = real_side[i]
            cp, cs = rec_side[i]
            max_px_diff = max(max_px_diff, abs(cp - rp))
            max_sz_diff = max(max_sz_diff, abs(cs - rs))
            if i == 0 and side == "bids":
                top1_bid_match = abs(cp - rp) <= 1e-9
            if i == 0 and side == "asks":
                top1_ask_match = abs(cp - rp) <= 1e-9

    crossed_book_real = False
    crossed_book_rec = False
    if real_levels["bids"] and real_levels["asks"]:
        crossed_book_real = real_levels["bids"][0][0] >= real_levels["asks"][0][0]
    if rec_levels["bids"] and rec_levels["asks"]:
        crossed_book_rec = rec_levels["bids"][0][0] >= rec_levels["asks"][0][0]

    neg_zero_real = sum(1 for _, s in real_levels["bids"] + real_levels["asks"] if s <= 0)
    neg_zero_rec = sum(1 for _, s in rec_levels["bids"] + rec_levels["asks"] if s <= 0)

    return {
        "max_px_diff": float(max_px_diff),
        "max_sz_diff": float(max_sz_diff),
        "top1_bid_match": bool(top1_bid_match),
        "top1_ask_match": bool(top1_ask_match),
        "crossed_book_real": bool(crossed_book_real),
        "crossed_book_rec": bool(crossed_book_rec),
        "negative_or_zero_size_real": int(neg_zero_real),
        "negative_or_zero_size_rec": int(neg_zero_rec),
    }


def _get_rec_row_at_or_before(rec_df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    c = rec_df.loc[rec_df["ts_event"] <= ts]
    if c.empty:
        raise ValueError("No reconstructed row at or before snapshot timestamp")
    return c.iloc[-1].copy()


def _parse_delta_item(level: Any) -> tuple[float, float] | None:
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


def _flatten_updates_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for db_idx, r in rows_df.reset_index(drop=True).iterrows():
        ts_event = pd.Timestamp(int(r["ts_event_ms"]), unit="ms", tz="utc")
        ts_ingest = pd.Timestamp(int(r["ts_ingest_ms"]), unit="ms", tz="utc")

        for side_col, side_name in (("bids_delta", "bid"), ("asks_delta", "ask")):
            delta = r.get(side_col)
            if isinstance(delta, str):
                delta = json.loads(delta) if delta.strip() else []
            if not isinstance(delta, list):
                continue
            for pos, item in enumerate(delta):
                parsed = _parse_delta_item(item)
                if parsed is None:
                    continue
                price, size = parsed
                out.append(
                    {
                        "db_row_pos": db_idx,
                        "delta_pos": pos,
                        "ts_event": ts_event,
                        "ts_ingest": ts_ingest,
                        "side": side_name,
                        "price": price,
                        "size": size,
                    }
                )
    if not out:
        return pd.DataFrame(columns=["ts_event", "side", "price", "size", "ts_ingest", "db_row_pos", "delta_pos"])
    return pd.DataFrame(out)


def _replay_to_snapshot(anchor_row: pd.Series, updates_flat_df: pd.DataFrame, snapshot_ts: pd.Timestamp, depth: int) -> pd.Series:
    book = OrderBookState.from_snapshot_row(anchor_row)
    if not updates_flat_df.empty:
        upd = updates_flat_df.loc[updates_flat_df["ts_event"] <= snapshot_ts]
        for _, u in upd.iterrows():
            book.apply_update(str(u["side"]), float(u["price"]), float(u["size"]))
    return pd.Series(
        book.to_record(
            ts_event=snapshot_ts,
            inst_id=str(anchor_row.get("inst_id", "")),
            anchor_snapshot_ts=anchor_row.get("ts_event"),
            reconstruction_mode="forensic_replay",
            depth=depth,
        )
    )


def _ordering_audit(engine, schema: str, inst_id: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> dict[str, Any]:
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    stats_sql = text(
        f"""
        WITH u AS (
          SELECT ts_event_ms, ts_ingest_ms
          FROM {schema}.orderbook_updates
          WHERE instid = :inst_id
            AND ts_event_ms BETWEEN :start_ms AND :end_ms
        ),
        g AS (
          SELECT
            ts_event_ms,
            COUNT(*) AS rows_per_ts,
            COUNT(DISTINCT ts_ingest_ms) AS distinct_ingest_per_ts
          FROM u
          GROUP BY ts_event_ms
        )
        SELECT
          (SELECT COUNT(*) FROM u) AS total_rows,
          (SELECT COUNT(*) FROM g) AS total_distinct_ts_event,
          (SELECT COUNT(*) FROM g WHERE rows_per_ts > 1) AS duplicated_ts_events,
          (SELECT COALESCE(SUM(rows_per_ts), 0) FROM g WHERE rows_per_ts > 1) AS rows_inside_duplicated_ts_events,
          (SELECT COALESCE(MAX(rows_per_ts), 0) FROM g) AS max_rows_per_single_ts_event
        """
    )
    sample_sql = text(
        f"""
        SELECT
          ts_event_ms,
          COUNT(*) AS rows_per_ts,
          COUNT(DISTINCT ts_ingest_ms) AS distinct_ingest_per_ts,
          MIN(ts_ingest_ms) AS min_ingest_ms,
          MAX(ts_ingest_ms) AS max_ingest_ms
        FROM {schema}.orderbook_updates
        WHERE instid = :inst_id
          AND ts_event_ms BETWEEN :start_ms AND :end_ms
        GROUP BY ts_event_ms
        HAVING COUNT(*) > 1
        ORDER BY rows_per_ts DESC, ts_event_ms DESC
        LIMIT 30
        """
    )
    with engine.connect() as conn:
        stats_row = pd.read_sql(stats_sql, conn, params={"inst_id": inst_id, "start_ms": start_ms, "end_ms": end_ms}).iloc[0]
        sample_df = pd.read_sql(sample_sql, conn, params={"inst_id": inst_id, "start_ms": start_ms, "end_ms": end_ms})

    report = {
        "total_rows": int(stats_row["total_rows"]),
        "total_distinct_ts_event": int(stats_row["total_distinct_ts_event"]),
        "duplicated_ts_events": int(stats_row["duplicated_ts_events"]),
        "rows_inside_duplicated_ts_events": int(stats_row["rows_inside_duplicated_ts_events"]),
        "max_rows_per_single_ts_event": int(stats_row["max_rows_per_single_ts_event"]),
        "has_intra_timestamp_ambiguity_risk": bool(int(stats_row["duplicated_ts_events"]) > 0),
        "note": "Current pipeline orders raw updates by ts_event only; equal ts_event ordering can be ambiguous unless additional tie-breakers are enforced.",
        "sample_duplicated_ts_events": sample_df.to_dict(orient="records"),
    }
    return report


def run(args: argparse.Namespace) -> int:
    rec_df = pd.read_parquet(args.reconstructed_parquet).copy()
    if rec_df.empty:
        raise ValueError("Reconstructed parquet is empty")
    rec_df["ts_event"] = pd.to_datetime(rec_df["ts_event"], utc=True, errors="coerce")
    if "anchor_snapshot_ts" in rec_df.columns:
        rec_df["anchor_snapshot_ts"] = pd.to_datetime(rec_df["anchor_snapshot_ts"], utc=True, errors="coerce")
    rec_df = rec_df.sort_values("ts_event").reset_index(drop=True)

    window_start = _parse_ts(args.start) if args.start else rec_df["ts_event"].min()
    window_end = _parse_ts(args.end) if args.end else rec_df["ts_event"].max()
    rec_window = rec_df[(rec_df["ts_event"] >= window_start) & (rec_df["ts_event"] <= window_end)].copy()
    if rec_window.empty:
        raise ValueError("No reconstructed rows inside requested window")

    engine = get_engine()
    start_ms = int(window_start.timestamp() * 1000)
    end_ms = int(window_end.timestamp() * 1000)

    snapshot_candidates_sql = text(
        f"""
        SELECT DISTINCT ts_event_ms
        FROM {args.schema}.orderbook_snapshots
        WHERE instid = :inst_id
          AND ts_event_ms BETWEEN :start_ms AND :end_ms
        ORDER BY ts_event_ms
        """
    )
    snapshot_levels_sql = text(
        f"""
        SELECT snapshot_id, instid, ts_event_ms, ts_ingest_ms, side, price, size, level
        FROM {args.schema}.orderbook_snapshots
        WHERE instid = :inst_id
          AND ts_event_ms = :snapshot_ms
        ORDER BY side, level
        """
    )
    updates_rows_sql = text(
        f"""
        SELECT ts_event_ms, ts_ingest_ms, bids_delta, asks_delta
        FROM {args.schema}.orderbook_updates
        WHERE instid = :inst_id
          AND ts_event_ms BETWEEN :start_ms AND :end_ms
        ORDER BY ts_event_ms
        """
    )

    with engine.connect() as conn:
        snap_df = pd.read_sql(
            snapshot_candidates_sql,
            conn,
            params={"inst_id": args.inst_id, "start_ms": start_ms, "end_ms": end_ms},
        )

    if snap_df.empty:
        raise ValueError("No snapshots found in selected window")

    all_snapshot_ms = [int(v) for v in snap_df["ts_event_ms"].tolist()]
    random.seed(args.seed)
    n = min(args.sample_points, len(all_snapshot_ms))
    sampled_ms = sorted(random.sample(all_snapshot_ms, n))

    points: list[ValidationPoint] = []
    for snapshot_ms in sampled_ms:
        snapshot_ts = pd.Timestamp(snapshot_ms, unit="ms", tz="utc")
        with engine.connect() as conn:
            real_df = pd.read_sql(
                snapshot_levels_sql,
                conn,
                params={"inst_id": args.inst_id, "snapshot_ms": snapshot_ms},
            )
        if real_df.empty:
            continue

        rec_row = _get_rec_row_at_or_before(rec_window, snapshot_ts)
        real_levels = _real_snapshot_to_top_levels(real_df, depth=args.depth)
        rec_levels = _rec_row_to_top_levels(rec_row, depth=args.depth)
        cmp = _compare_levels(real_levels, rec_levels, depth=args.depth)

        points.append(
            ValidationPoint(
                snapshot_ts=snapshot_ts,
                snapshot_ms=snapshot_ms,
                rec_ts=pd.Timestamp(rec_row["ts_event"]),
                max_px_diff=float(cmp["max_px_diff"]),
                max_sz_diff=float(cmp["max_sz_diff"]),
                top1_bid_match=bool(cmp["top1_bid_match"]),
                top1_ask_match=bool(cmp["top1_ask_match"]),
                crossed_book_real=bool(cmp["crossed_book_real"]),
                crossed_book_rec=bool(cmp["crossed_book_rec"]),
                negative_or_zero_size_real=int(cmp["negative_or_zero_size_real"]),
                negative_or_zero_size_rec=int(cmp["negative_or_zero_size_rec"]),
            )
        )

    if not points:
        raise ValueError("Validation produced zero points")

    points_df = pd.DataFrame([p.__dict__ for p in points]).sort_values(
        ["max_px_diff", "max_sz_diff"], ascending=False
    )
    worst_df = points_df.head(args.worst_k).copy()
    # Ensure JSON-serializable scalars (pandas Timestamp / numpy scalars).
    def _to_jsonable(v: Any) -> Any:
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        # numpy scalar types expose `.item()`, convert them to python scalars
        if hasattr(v, "item") and callable(getattr(v, "item")):
            try:
                return v.item()
            except Exception:
                pass
        return v

    worst_points = [{k: _to_jsonable(v) for k, v in rec.items()} for rec in worst_df.to_dict(orient="records")]

    ordering_report = _ordering_audit(engine, args.schema, args.inst_id, window_start, window_end)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "forensic_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    # Forensic replay for worst points
    forensic_cases: list[dict[str, Any]] = []
    for _, row in worst_df.iterrows():
        snapshot_ms = int(row["snapshot_ms"])
        snapshot_ts = pd.Timestamp(snapshot_ms, unit="ms", tz="utc")
        rec_row = _get_rec_row_at_or_before(rec_window, snapshot_ts)
        anchor_ts = rec_row.get("anchor_snapshot_ts")
        if pd.isna(anchor_ts):
            anchor_ts = snapshot_ts
        anchor_ts = pd.Timestamp(anchor_ts)
        if anchor_ts.tzinfo is None:
            anchor_ts = anchor_ts.tz_localize("UTC")

        anchor_row = load_snapshot_raw_by_ts(
            engine=engine,
            schema=args.schema,
            snapshot_table="orderbook_snapshots",
            inst_id=args.inst_id,
            ts_event_ms=int(anchor_ts.timestamp() * 1000),
            snapshot_max_depth=None,
        )

        with engine.connect() as conn:
            raw_updates_rows = pd.read_sql(
                updates_rows_sql,
                conn,
                params={
                    "inst_id": args.inst_id,
                    "start_ms": int(anchor_ts.timestamp() * 1000),
                    "end_ms": snapshot_ms,
                },
            )

            real_df = pd.read_sql(
                snapshot_levels_sql,
                conn,
                params={"inst_id": args.inst_id, "snapshot_ms": snapshot_ms},
            )

        flat_current = _flatten_updates_rows(raw_updates_rows)
        # deterministic audit order candidate (not pipeline semantics)
        flat_deterministic = flat_current.sort_values(
            ["ts_event", "ts_ingest", "db_row_pos", "delta_pos"]
        ).reset_index(drop=True)

        replay_current = _replay_to_snapshot(anchor_row, flat_current, snapshot_ts, args.depth)
        replay_det = _replay_to_snapshot(anchor_row, flat_deterministic, snapshot_ts, args.depth)

        real_levels = _real_snapshot_to_top_levels(real_df, depth=args.depth)
        cmp_current = _compare_levels(real_levels, _rec_row_to_top_levels(replay_current, args.depth), args.depth)
        cmp_det = _compare_levels(real_levels, _rec_row_to_top_levels(replay_det, args.depth), args.depth)

        max_px_diff_current = float(cmp_current["max_px_diff"])
        max_px_diff_det = float(cmp_det["max_px_diff"])
        max_sz_diff_current = float(cmp_current["max_sz_diff"])
        max_sz_diff_det = float(cmp_det["max_sz_diff"])

        case = {
            "snapshot_ms": snapshot_ms,
            "snapshot_ts": snapshot_ts.isoformat(),
            "anchor_snapshot_ts": anchor_ts.isoformat(),
            "rec_ts": pd.Timestamp(rec_row["ts_event"]).isoformat(),
            "original_rec_max_px_diff": float(row["max_px_diff"]),
            "original_rec_max_sz_diff": float(row["max_sz_diff"]),
            "replay_current_order_cmp": cmp_current,
            "replay_deterministic_order_cmp": cmp_det,
            "replay_current_max_px_diff": max_px_diff_current,
            "replay_deterministic_max_px_diff": max_px_diff_det,
            "replay_px_diff_delta_det_minus_current": max_px_diff_det - max_px_diff_current,
            "replay_current_max_sz_diff": max_sz_diff_current,
            "replay_deterministic_max_sz_diff": max_sz_diff_det,
            "replay_sz_diff_delta_det_minus_current": max_sz_diff_det - max_sz_diff_current,
            "deterministic_improved": bool(
                (max_px_diff_det < max_px_diff_current)
                or (max_sz_diff_det < max_sz_diff_current)
            ),
            "updates_rows_count": int(len(raw_updates_rows)),
            "updates_flat_count": int(len(flat_current)),
            "duplicated_ts_event_groups_in_case": int(
                raw_updates_rows["ts_event_ms"].value_counts().gt(1).sum()
            )
            if not raw_updates_rows.empty
            else 0,
        }
        forensic_cases.append(case)
        case_path = cases_dir / f"case_snapshot_{snapshot_ms}.json"
        case_path.write_text(json.dumps(case, indent=2), encoding="utf-8")

    sanity_summary = {
        "validated_points": int(len(points_df)),
        "crossed_book_real_count": int(points_df["crossed_book_real"].sum()),
        "crossed_book_rec_count": int(points_df["crossed_book_rec"].sum()),
        "negative_or_zero_size_real_total": int(points_df["negative_or_zero_size_real"].sum()),
        "negative_or_zero_size_rec_total": int(points_df["negative_or_zero_size_rec"].sum()),
        "top1_bid_match_rate": float(points_df["top1_bid_match"].mean()),
        "top1_ask_match_rate": float(points_df["top1_ask_match"].mean()),
        "max_px_diff_worst": float(points_df["max_px_diff"].max()),
        "max_sz_diff_worst": float(points_df["max_sz_diff"].max()),
    }

    ordering_conclusion = {
        "is_real_source_of_drift_risk": bool(ordering_report["has_intra_timestamp_ambiguity_risk"]),
        "reason": (
            "Multiple rows share the same ts_event and pipeline order is by ts_event only; "
            "intra-timestamp tie ordering is potentially unstable and can propagate to depth mismatches."
            if ordering_report["has_intra_timestamp_ambiguity_risk"]
            else "No duplicated ts_event groups detected in the selected window."
        ),
    }

    summary = {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "inst_id": args.inst_id,
        "depth": args.depth,
        "sample_points": int(n),
        "worst_k": int(args.worst_k),
        "sanity_summary": sanity_summary,
        "ordering_audit": ordering_report,
        "ordering_conclusion": ordering_conclusion,
        "worst_points": worst_points,
        "forensic_cases_written": len(forensic_cases),
    }

    (out_dir / "forensic_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    points_df.to_csv(out_dir / "validation_points.csv", index=False)
    worst_df.to_csv(out_dir / "worst_points.csv", index=False)
    (out_dir / "ordering_audit.json").write_text(
        json.dumps(ordering_report, indent=2),
        encoding="utf-8",
    )

    md = [
        "# Order Book Reconstruction Diagnostics",
        "",
        f"- Window: `{window_start}` .. `{window_end}`",
        f"- Instrument: `{args.inst_id}`",
        f"- Validated points: `{len(points_df)}`",
        "",
        "## Sanity Counters",
        f"- crossed_book_real_count: `{sanity_summary['crossed_book_real_count']}`",
        f"- crossed_book_rec_count: `{sanity_summary['crossed_book_rec_count']}`",
        f"- negative_or_zero_size_real_total: `{sanity_summary['negative_or_zero_size_real_total']}`",
        f"- negative_or_zero_size_rec_total: `{sanity_summary['negative_or_zero_size_rec_total']}`",
        f"- top1_bid_match_rate: `{sanity_summary['top1_bid_match_rate']:.4f}`",
        f"- top1_ask_match_rate: `{sanity_summary['top1_ask_match_rate']:.4f}`",
        f"- max_px_diff_worst: `{sanity_summary['max_px_diff_worst']:.8f}`",
        f"- max_sz_diff_worst: `{sanity_summary['max_sz_diff_worst']:.8f}`",
        "",
        "## Ordering Audit",
        f"- duplicated_ts_events: `{ordering_report['duplicated_ts_events']}`",
        f"- rows_inside_duplicated_ts_events: `{ordering_report['rows_inside_duplicated_ts_events']}`",
        f"- max_rows_per_single_ts_event: `{ordering_report['max_rows_per_single_ts_event']}`",
        f"- ambiguity risk: `{ordering_conclusion['is_real_source_of_drift_risk']}`",
        "",
        f"Conclusion: {ordering_conclusion['reason']}",
    ]
    (out_dir / "forensic_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Diagnostics written to: {out_dir}")
    print(f"Worst points CSV: {out_dir / 'worst_points.csv'}")
    print(f"Forensic summary: {out_dir / 'forensic_summary.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forensic diagnostics for order book reconstruction"
    )
    parser.add_argument("--reconstructed-parquet", required=True)
    parser.add_argument("--inst-id", required=True)
    parser.add_argument("--schema", default="okx_raw")
    parser.add_argument("--start", help="UTC timestamp start")
    parser.add_argument("--end", help="UTC timestamp end")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--sample-points", type=int, default=200)
    parser.add_argument("--worst-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="data/diagnostics/orderbook_forensics",
        help="Directory for diagnostics artifacts",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

