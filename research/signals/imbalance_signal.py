"""
Simple imbalance-based signal: {-1, 0, 1}.
"""

from __future__ import annotations

import pandas as pd

from research.features.imbalance import compute_l1_imbalance


def build_imbalance_signal(
    df: pd.DataFrame,
    threshold: float = 0.2,
) -> pd.Series:
    """
    Signal from L1 imbalance: 1 if imbalance > threshold, -1 if < -threshold, else 0.
    """
    imb = compute_l1_imbalance(df)
    signal = pd.Series(0, index=df.index, dtype="int8")
    signal[imb > threshold] = 1
    signal[imb < -threshold] = -1
    return signal
