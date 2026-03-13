"""
Grid sampling helpers for order book reconstruction.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterator


def grid_timestamps(
    start: datetime,
    end: datetime,
    step_ms: int,
) -> Iterator[datetime]:
    """
    Yield timestamps on a regular grid from start to end (inclusive).

    Step is in milliseconds.
    """
    step_delta = timedelta(milliseconds=step_ms)
    t = start
    while t <= end:
        yield t
        t = t + step_delta
