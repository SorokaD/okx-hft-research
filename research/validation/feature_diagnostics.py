"""
Feature diagnostics: basic stats, missing values, correlation.
"""

from __future__ import annotations

import pandas as pd


def run_feature_diagnostics(df: pd.DataFrame, feature_cols: list[str] | None = None) -> dict:
    """
    Basic diagnostics for feature columns: mean, std, null count, min, max.
    """
    cols = feature_cols or [c for c in df.columns if c.startswith(("bid_", "ask_", "mid_", "spread_"))]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return {}
    sub = df[cols]
    return {
        "mean": sub.mean().to_dict(),
        "std": sub.std().to_dict(),
        "null_count": sub.isnull().sum().to_dict(),
        "min": sub.min().to_dict(),
        "max": sub.max().to_dict(),
    }
