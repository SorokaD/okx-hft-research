# flake8: noqa
"""
Profitability-aware target generation for maker-like scenarios.

We approximate a simple maker entry / taker exit:
- Long: buy (maker) at current best_bid, sell (taker) at future best_ask.
- Short: sell (maker) at current best_ask, buy (taker) at future best_bid.

The output targets are binary indicators whether the net profit covers
fees + a minimum profit threshold (all expressed in ticks).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MakerTakerFeeModel:
    # Fee rates as fractions of notional (e.g. 0.0002 = 2 bps)
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0007

    # Extra minimum profit threshold in ticks
    min_profit_ticks: float = 0.0

    # When inferring net profitability, rounding is unnecessary;
    # use raw floats but keep configuration transparent.


def _fees_ticks_from_notional(
    *,
    entry_px: np.ndarray,
    exit_px: np.ndarray,
    tick_size: float,
    maker_fee_rate: float,
    taker_fee_rate: float,
) -> np.ndarray:
    fees_price_units = entry_px * maker_fee_rate + exit_px * taker_fee_rate
    return fees_price_units / tick_size


def build_gross_profitability_targets(
    df: pd.DataFrame,
    *,
    tick_size: float,
    future_best_bid_px: pd.Series,
    future_best_ask_px: pd.Series,
    horizon_tag: str,
    gross_profit_min_ticks: float,
) -> pd.DataFrame:
    """
    Gross profitability: ignores fees, checks whether price excursion >= threshold.
    """
    entry_best_bid = df["bid_px_01"]
    entry_best_ask = df["ask_px_01"]

    # Long gross profit in ticks: exit (ask) - entry (bid)
    long_gross_ticks = (future_best_ask_px - entry_best_bid) / tick_size
    short_gross_ticks = (entry_best_ask - future_best_bid_px) / tick_size

    out = pd.DataFrame(index=df.index)
    out[f"target_gross_profitable_long_{horizon_tag}"] = (long_gross_ticks >= gross_profit_min_ticks).astype(
        np.int8
    )
    out[f"target_gross_profitable_short_{horizon_tag}"] = (short_gross_ticks >= gross_profit_min_ticks).astype(
        np.int8
    )
    out[f"target_gross_profitable_{horizon_tag}"] = np.maximum(
        out[f"target_gross_profitable_long_{horizon_tag}"], out[f"target_gross_profitable_short_{horizon_tag}"]
    ).astype(np.int8)
    return out


def build_net_profitability_targets(
    df: pd.DataFrame,
    *,
    tick_size: float,
    future_best_bid_px: pd.Series,
    future_best_ask_px: pd.Series,
    horizon_tag: str,
    fee_model: MakerTakerFeeModel,
) -> pd.DataFrame:
    """
    Net profitability: gross profit minus maker+taker fees in ticks.

    Targets:
    - positive class if net profit in ticks >= fee-covered threshold.
    """
    entry_best_bid = df["bid_px_01"]
    entry_best_ask = df["ask_px_01"]

    entry_best_bid_np = entry_best_bid.to_numpy(dtype=np.float64)
    entry_best_ask_np = entry_best_ask.to_numpy(dtype=np.float64)
    future_best_bid_np = future_best_bid_px.to_numpy(dtype=np.float64)
    future_best_ask_np = future_best_ask_px.to_numpy(dtype=np.float64)

    # Long: buy at bid, sell at future ask
    long_gross_ticks = (future_best_ask_np - entry_best_bid_np) / tick_size
    long_fees_ticks = _fees_ticks_from_notional(
        entry_px=entry_best_bid_np,
        exit_px=future_best_ask_np,
        tick_size=tick_size,
        maker_fee_rate=fee_model.maker_fee_rate,
        taker_fee_rate=fee_model.taker_fee_rate,
    )
    long_net_ticks = long_gross_ticks - long_fees_ticks - fee_model.min_profit_ticks

    # Short: sell at ask, buy at future bid
    short_gross_ticks = (entry_best_ask_np - future_best_bid_np) / tick_size
    short_fees_ticks = _fees_ticks_from_notional(
        entry_px=entry_best_ask_np,
        exit_px=future_best_bid_np,
        tick_size=tick_size,
        maker_fee_rate=fee_model.maker_fee_rate,
        taker_fee_rate=fee_model.taker_fee_rate,
    )
    short_net_ticks = short_gross_ticks - short_fees_ticks - fee_model.min_profit_ticks

    out = pd.DataFrame(index=df.index)
    out[f"target_net_profitable_long_{horizon_tag}"] = (long_net_ticks >= 0).astype(np.int8)
    out[f"target_net_profitable_short_{horizon_tag}"] = (short_net_ticks >= 0).astype(np.int8)
    out[f"target_net_profitable_{horizon_tag}"] = np.maximum(
        out[f"target_net_profitable_long_{horizon_tag}"], out[f"target_net_profitable_short_{horizon_tag}"]
    ).astype(np.int8)
    return out

