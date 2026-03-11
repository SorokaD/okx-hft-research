"""
Spread features: absolute and relative bid-ask spread.
"""

from __future__ import annotations

import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def compute_spread(df: pd.DataFrame) -> pd.Series:
    """
    Absolute spread: ask_px_01 - bid_px_01.
    Requires columns: bid_px_01, ask_px_01.
    """
    _require_cols(df, ["bid_px_01", "ask_px_01"])
    return df["ask_px_01"] - df["bid_px_01"]


def compute_relative_spread(df: pd.DataFrame) -> pd.Series:
    """
    Relative spread: (ask_px_01 - bid_px_01) / mid, where mid = (bid_px_01 + ask_px_01) / 2.
    Requires columns: bid_px_01, ask_px_01.
    """
    _require_cols(df, ["bid_px_01", "ask_px_01"])
    mid = (df["bid_px_01"] + df["ask_px_01"]) / 2
    spread = df["ask_px_01"] - df["bid_px_01"]
    return spread / mid
