"""
Time utilities for research: parsing, timezone handling.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Current UTC datetime."""
    return datetime.now(timezone.utc)
