"""
Order book imbalance features: L1 and L10.
"""

from __future__ import annotations

import pandas as pd


def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def compute_l1_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    L1 imbalance: (bid_sz_01 - ask_sz_01) / (bid_sz_01 + ask_sz_01).
    Range [-1, 1]. Positive = more bid pressure.
    Requires columns: bid_sz_01, ask_sz_01.
    """
    _require_cols(df, ["bid_sz_01", "ask_sz_01"])
    b, a = df["bid_sz_01"], df["ask_sz_01"]
    total = b + a
    return (b - a) / total.replace(0, float("nan"))


def compute_l10_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    L10 imbalance: (sum bid_sz_01..10 - sum ask_sz_01..10) / (sum bid + sum ask).
    Requires columns: bid_sz_01..bid_sz_10, ask_sz_01..ask_sz_10.
    """
    bid_cols = [f"bid_sz_{i:02d}" for i in range(1, 11)]
    ask_cols = [f"ask_sz_{i:02d}" for i in range(1, 11)]
    _require_cols(df, bid_cols + ask_cols)
    bid_sum = df[bid_cols].sum(axis=1)
    ask_sum = df[ask_cols].sum(axis=1)
    total = bid_sum + ask_sum
    return (bid_sum - ask_sum) / total.replace(0, float("nan"))
