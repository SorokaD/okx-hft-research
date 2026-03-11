"""
Basic backtest metrics: hit rate, precision long/short.
"""

from __future__ import annotations

import pandas as pd


def hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Fraction of correct predictions (where y_true == y_pred) among non-zero predictions.
    """
    mask = y_pred != 0
    if mask.sum() == 0:
        return 0.0
    return (y_true[mask] == y_pred[mask]).mean()


def precision_long(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Precision for long predictions (y_pred == 1)."""
    mask = y_pred == 1
    if mask.sum() == 0:
        return 0.0
    return (y_true[mask] == 1).mean()


def precision_short(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Precision for short predictions (y_pred == -1)."""
    mask = y_pred == -1
    if mask.sum() == 0:
        return 0.0
    return (y_true[mask] == -1).mean()
