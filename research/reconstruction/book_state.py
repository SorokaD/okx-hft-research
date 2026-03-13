"""
Order book state for reconstruction: holds bids/asks and applies updates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class OrderBookState:
    """
    In-memory order book state for offline reconstruction.

    Bids: dict[price, size], sorted descending by price.
    Asks: dict[price, size], sorted ascending by price.
    """

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    @classmethod
    def from_snapshot_row(cls, row: pd.Series) -> OrderBookState:
        """
        Initialize book state from a snapshot row.

        Expects columns: bid_px_01..bid_px_10, bid_sz_01..bid_sz_10,
        ask_px_01..ask_px_10, ask_sz_01..ask_sz_10.
        Levels with size <= 0 or missing are skipped.
        """
        book = cls()
        for i in range(1, 11):
            px_col = f"bid_px_{i:02d}"
            sz_col = f"bid_sz_{i:02d}"
            if px_col in row.index and sz_col in row.index:
                px = row.get(px_col)
                sz = row.get(sz_col)
                if pd.notna(px) and pd.notna(sz) and float(sz) > 0:
                    book.bids[float(px)] = float(sz)
        for i in range(1, 11):
            px_col = f"ask_px_{i:02d}"
            sz_col = f"ask_sz_{i:02d}"
            if px_col in row.index and sz_col in row.index:
                px = row.get(px_col)
                sz = row.get(sz_col)
                if pd.notna(px) and pd.notna(sz) and float(sz) > 0:
                    book.asks[float(px)] = float(sz)
        return book

    def apply_update(self, side: str, price: float, size: float) -> None:
        """
        Apply a single update.

        - size > 0: set level at price to size.
        - size == 0: remove level at price.
        - Unknown side: raises ValueError.
        """
        if side.lower() == "bid":
            if size > 0:
                self.bids[price] = size
            else:
                self.bids.pop(price, None)
        elif side.lower() == "ask":
            if size > 0:
                self.asks[price] = size
            else:
                self.asks.pop(price, None)
        else:
            raise ValueError(f"Unknown side: {side!r}. Expected 'bid' or 'ask'.")

    def get_top_levels(self, depth: int = 10) -> dict[str, list[tuple[float, float]]]:
        """
        Return top L levels: bids descending, asks ascending.
        """
        bids_sorted = sorted(self.bids.items(), key=lambda x: -x[0])[:depth]
        asks_sorted = sorted(self.asks.items(), key=lambda x: x[0])[:depth]
        return {"bids": bids_sorted, "asks": asks_sorted}

    def to_record(
        self,
        ts_event: Any,
        inst_id: str,
        anchor_snapshot_ts: Any,
        reconstruction_mode: str,
        depth: int = 10,
    ) -> dict[str, Any]:
        """
        Convert current state to a flat record for DataFrame.

        Missing levels (fewer than depth) are filled with None.
        """
        levels = self.get_top_levels(depth)
        record: dict[str, Any] = {
            "ts_event": ts_event,
            "inst_id": inst_id,
            "anchor_snapshot_ts": anchor_snapshot_ts,
            "reconstruction_mode": reconstruction_mode,
        }
        best_bid = levels["bids"][0][0] if levels["bids"] else None
        best_ask = levels["asks"][0][0] if levels["asks"] else None
        if best_bid is not None and best_ask is not None:
            record["mid_px"] = (best_bid + best_ask) / 2
            record["spread_px"] = best_ask - best_bid
        else:
            record["mid_px"] = None
            record["spread_px"] = None
        for i in range(1, depth + 1):
            record[f"bid_px_{i:02d}"] = levels["bids"][i - 1][0] if i <= len(levels["bids"]) else None
            record[f"bid_sz_{i:02d}"] = levels["bids"][i - 1][1] if i <= len(levels["bids"]) else None
        for i in range(1, depth + 1):
            record[f"ask_px_{i:02d}"] = levels["asks"][i - 1][0] if i <= len(levels["asks"]) else None
            record[f"ask_sz_{i:02d}"] = levels["asks"][i - 1][1] if i <= len(levels["asks"]) else None
        return record
