"""
Future mid price move label: {-1, 0, 1} based on mid change over horizon.
"""

from __future__ import annotations

import pandas as pd


def build_future_mid_move_label(
    df: pd.DataFrame,
    horizon_rows: int = 10,
    mid_col: str = "mid_px",
    bucket_threshold: float = 1e-4,
) -> pd.Series:
    """
    Build label: compare current mid_px to mid_px shifted forward by horizon_rows.
    Returns: -1 (down), 0 (flat), 1 (up) based on relative move vs bucket_threshold.
    """
    if mid_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: {mid_col}")
    mid = df[mid_col]
    future_mid = mid.shift(-horizon_rows)
    ret = (future_mid - mid) / mid.replace(0, float("nan"))
    label = pd.Series(0, index=df.index, dtype="int8")
    label[ret > bucket_threshold] = 1
    label[ret < -bucket_threshold] = -1
    return label
