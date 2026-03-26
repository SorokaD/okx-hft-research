"""
Order book state for reconstruction: holds bids/asks and applies updates.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


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

        Expects wide snapshot columns like:
        - bid_px_XX / bid_sz_XX
        - ask_px_XX / ask_sz_XX

        Depth is arbitrary: we discover all bid/ask level indices present
        in the row and load them into the internal dicts.

        Levels with size <= 0 or missing are skipped.
        """
        book = cls()

        # Bid levels: discover all bid_px_XX columns
        bid_px_cols = [c for c in row.index if str(c).startswith("bid_px_")]
        for px_col in bid_px_cols:
            lvl = str(px_col).split("bid_px_")[-1]
            sz_col = f"bid_sz_{lvl}"
            if sz_col not in row.index:
                continue
            px = row.get(px_col)
            sz = row.get(sz_col)
            if pd.notna(px) and pd.notna(sz) and float(sz) > 0:
                book.bids[float(px)] = float(sz)

        # Ask levels: discover all ask_px_XX columns
        ask_px_cols = [c for c in row.index if str(c).startswith("ask_px_")]
        for px_col in ask_px_cols:
            lvl = str(px_col).split("ask_px_")[-1]
            sz_col = f"ask_sz_{lvl}"
            if sz_col not in row.index:
                continue
            px = row.get(px_col)
            sz = row.get(sz_col)
            if pd.notna(px) and pd.notna(sz) and float(sz) > 0:
                book.asks[float(px)] = float(sz)

        # Basic consistency check (top-of-book)
        best_bid = max(book.bids.keys()) if book.bids else None
        best_ask = min(book.asks.keys()) if book.asks else None
        if (
            best_bid is not None
            and best_ask is not None
            and best_bid >= best_ask
        ):
            log.warning(
                "Snapshot consistency violated: best_bid >= best_ask (best_bid=%s, best_ask=%s)",
                best_bid,
                best_ask,
            )
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
            raise ValueError(
                f"Unknown side: {side!r}. Expected 'bid' or 'ask'."
            )

    def get_top_levels(
        self, depth: int = 10
    ) -> dict[str, list[tuple[float, float]]]:
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
            if best_bid >= best_ask:
                log.warning(
                    "to_record consistency violated: best_bid >= best_ask (best_bid=%s, best_ask=%s)",
                    best_bid,
                    best_ask,
                )
            record["mid_px"] = (best_bid + best_ask) / 2
            record["spread_px"] = best_ask - best_bid
        else:
            record["mid_px"] = None
            record["spread_px"] = None
        bids = levels["bids"]
        asks = levels["asks"]
        for i in range(1, depth + 1):
            bid_px_key = f"bid_px_{i:02d}"
            bid_sz_key = f"bid_sz_{i:02d}"
            if i <= len(bids):
                record[bid_px_key] = bids[i - 1][0]
                record[bid_sz_key] = bids[i - 1][1]
            else:
                record[bid_px_key] = None
                record[bid_sz_key] = None

        for i in range(1, depth + 1):
            ask_px_key = f"ask_px_{i:02d}"
            ask_sz_key = f"ask_sz_{i:02d}"
            if i <= len(asks):
                record[ask_px_key] = asks[i - 1][0]
                record[ask_sz_key] = asks[i - 1][1]
            else:
                record[ask_px_key] = None
                record[ask_sz_key] = None
        return record
