# flake8: noqa
"""
Utilities to load reconstructed order book parquet data for a given time window.

This module is intentionally "IO-focused": it finds all reconstructed grid parquet files
that overlap the requested [start, end] interval and concatenates them into a single
DataFrame sorted by `ts_event`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ParquetLoadResult:
    df: pd.DataFrame
    selected_files: list[str]


def _parse_hourly_filename_starts_ends(
    inst_id: str,
    stem: str,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """
    Filename format example:
      book_grid100ms_<INST>_YYYY-MM-DD_HH-MM-SS__YYYY-MM-DD_HH-MM-SS
    """
    pattern = re.compile(
        rf"^book_grid100ms_{re.escape(inst_id)}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{2}}-\d{{2}}-\d{{2}})__"
        rf"(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{2}}-\d{{2}}-\d{{2}})$"
    )
    m = pattern.match(stem)
    if not m:
        return None

    s_date, s_time, e_date, e_time = m.groups()
    file_start = pd.Timestamp(f"{s_date} {s_time.replace('-', ':')}", tz="UTC")
    file_end = pd.Timestamp(f"{e_date} {e_time.replace('-', ':')}", tz="UTC")
    return file_start, file_end


def load_reconstructed_grid_parquet(
    *,
    base_dir: Path,
    inst_id: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    grid_dirname: str = "grid100ms",
    file_prefix: str | None = None,
) -> ParquetLoadResult:
    """
    Load reconstructed parquet files for `inst_id` under:
      {base_dir}/{inst_id}/{grid_dirname}/<YYYY-MM-DD>/*.parquet

    The loader:
    - finds all parquet files matching the expected naming convention,
    - selects those overlapping [start_ts, end_ts],
    - concatenates and filters strictly to [start_ts, end_ts].
    """
    if end_ts < start_ts:
        raise ValueError("end_ts must be >= start_ts")

    mode_dir = base_dir / inst_id / grid_dirname
    if not mode_dir.exists():
        raise FileNotFoundError(f"Mode directory does not exist: {mode_dir}")

    prefix = file_prefix or f"book_grid100ms_{inst_id}_"

    selected_files: list[tuple[pd.Timestamp, Path]] = []
    for p in mode_dir.rglob("*.parquet"):
        if not p.stem.startswith(prefix.replace(".parquet", "")):
            # keep loose: rely on our parser below
            pass

        parsed = _parse_hourly_filename_starts_ends(inst_id=inst_id, stem=p.stem)
        if parsed is None:
            continue

        file_start, file_end = parsed
        # overlap check
        if file_end >= start_ts and file_start <= end_ts:
            selected_files.append((file_start, p))

    if not selected_files:
        raise FileNotFoundError(
            f"No reconstructed grid parquet files overlap [{start_ts} .. {end_ts}] for {inst_id} under {mode_dir}"
        )

    selected_files_sorted = sorted(selected_files, key=lambda x: x[0])
    files_for_read = [p for _, p in selected_files_sorted]

    dfs: list[pd.DataFrame] = []
    for p in files_for_read:
        d = pd.read_parquet(p)
        if "ts_event" not in d.columns:
            raise ValueError(f"Parquet missing ts_event: {p}")
        dfs.append(d)

    df = pd.concat(dfs, ignore_index=True)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"]).sort_values("ts_event").reset_index(drop=True)

    df = df[(df["ts_event"] >= start_ts) & (df["ts_event"] <= end_ts)].copy()
    selected_names = [str(p) for p in files_for_read]
    return ParquetLoadResult(df=df, selected_files=selected_names)

