# flake8: noqa
"""
Tick size handling.

Goal: provide a robust tick-size inference (or validation), without hardcoding an
instrument-specific constant in business logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TickSizeResult:
    tick_size: float
    inferred_from: str
    sample_rows: int
    notes: str


def infer_tick_size(
    df: pd.DataFrame,
    *,
    price_col: str = "mid_px",
    sample_n: int = 200_000,
    min_positive_quantile: float = 0.01,
) -> TickSizeResult:
    """
    Infer tick size from observed price changes.

    Heuristic:
    - take consecutive absolute deltas in the selected price column,
    - keep only positive deltas,
    - find the most common "small delta cluster" after rounding.

    This works well for reconstructed grid data where prices should change in tick steps.
    """
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")

    if len(df) < 2:
        raise ValueError("Not enough rows to infer tick size")

    x = df[price_col]
    if sample_n and len(x) > sample_n:
        x = x.sample(sample_n, random_state=42).sort_index()

    x_np = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64)
    x_np = x_np[~np.isnan(x_np)]
    if x_np.size < 2:
        raise ValueError("Not enough numeric samples to infer tick size")

    deltas = np.abs(np.diff(x_np))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        raise ValueError("All consecutive deltas are zero; cannot infer tick size")

    # Round deltas to avoid float noise; the rounding precision is chosen from delta magnitude.
    # Example: if deltas are ~0.1, rounding to 1e-8 is safe; if deltas are larger, still safe.
    deltas_round = np.round(deltas, 8)

    # Use the lower tail to capture the smallest non-zero step.
    q = float(np.quantile(deltas_round, min_positive_quantile))
    candidate_mask = deltas_round <= q if q > 0 else deltas_round > 0
    candidates = deltas_round[candidate_mask]
    if candidates.size == 0:
        candidates = deltas_round

    # Mode among candidates.
    vals, counts = np.unique(candidates, return_counts=True)
    tick = float(vals[np.argmax(counts)])

    notes = "inferred from mode of small positive price deltas"
    return TickSizeResult(
        tick_size=tick,
        inferred_from=price_col,
        sample_rows=int(x_np.size),
        notes=notes,
    )


def validate_tick_size(tick_size: float, *, min_tick: float = 1e-12, max_tick: float = 1e12) -> float:
    tick_size = float(tick_size)
    if tick_size < min_tick or tick_size > max_tick:
        raise ValueError(f"tick_size seems invalid: {tick_size}")
    return tick_size

