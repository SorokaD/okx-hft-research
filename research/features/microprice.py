"""
Microprice: volume-weighted price between bid and ask at L1.
"""

from __future__ import annotations

import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def compute_microprice(df: pd.DataFrame) -> pd.Series:
    """
    Microprice at L1: (bid_px_01 * ask_sz_01 + ask_px_01 * bid_sz_01) / (bid_sz_01 + ask_sz_01).
    Requires columns: bid_px_01, ask_px_01, bid_sz_01, ask_sz_01.
    """
    _require_cols(df, ["bid_px_01", "ask_px_01", "bid_sz_01", "ask_sz_01"])
    bpx, aspx = df["bid_px_01"], df["ask_px_01"]
    bsz, asz = df["bid_sz_01"], df["ask_sz_01"]
    total = bsz + asz
    return (bpx * asz + aspx * bsz) / total.replace(0, float("nan"))
