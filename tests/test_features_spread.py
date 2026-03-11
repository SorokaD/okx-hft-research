"""Tests for spread features."""

import pandas as pd
import pytest

from research.features.spread import compute_spread, compute_relative_spread


def test_compute_spread() -> None:
    df = pd.DataFrame({
        "bid_px_01": [100.0, 99.9],
        "ask_px_01": [100.1, 100.0],
    })
    spread = compute_spread(df)
    assert spread.iloc[0] == 0.1
    assert spread.iloc[1] == 0.1


def test_compute_spread_missing_cols() -> None:
    df = pd.DataFrame({"bid_px_01": [100.0]})
    with pytest.raises(ValueError, match="ask_px_01"):
        compute_spread(df)


def test_compute_relative_spread() -> None:
    df = pd.DataFrame({
        "bid_px_01": [100.0],
        "ask_px_01": [100.2],
    })
    rel = compute_relative_spread(df)
    assert abs(rel.iloc[0] - 0.002) < 1e-6  # 0.2/100.1
