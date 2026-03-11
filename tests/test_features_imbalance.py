"""Tests for imbalance features."""

import pandas as pd
import pytest

from research.features.imbalance import compute_l1_imbalance, compute_l10_imbalance


def test_compute_l1_imbalance() -> None:
    df = pd.DataFrame({
        "bid_sz_01": [3.0, 1.0],
        "ask_sz_01": [1.0, 3.0],
    })
    imb = compute_l1_imbalance(df)
    assert abs(imb.iloc[0] - 0.5) < 1e-6  # (3-1)/(3+1)=0.5
    assert abs(imb.iloc[1] + 0.5) < 1e-6


def test_compute_l10_imbalance() -> None:
    cols = {f"bid_sz_{i:02d}": [1.0] for i in range(1, 11)}
    cols.update({f"ask_sz_{i:02d}": [1.0] for i in range(1, 11)})
    df = pd.DataFrame(cols)
    df["bid_sz_01"] = 3.0
    df["ask_sz_01"] = 1.0
    imb = compute_l10_imbalance(df)
    assert imb.iloc[0] > 0


def test_l1_imbalance_missing_cols() -> None:
    df = pd.DataFrame({"bid_sz_01": [1.0]})
    with pytest.raises(ValueError, match="ask_sz_01"):
        compute_l1_imbalance(df)
