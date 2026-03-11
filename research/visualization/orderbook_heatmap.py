"""
Orderbook heatmap: transform wide-format L10 orderbook to long for visualization.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def orderbook_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform wide-format L10 orderbook to long format for heatmap.
    Expects columns: bid_px_01..bid_sz_10, ask_px_01..ask_sz_10, plus ts_event or index.
    Returns long DataFrame with: level, side, price, size, ts (if present).
    """
    time_col = "ts_event" if "ts_event" in df.columns else None
    index_col = df.index.name or "index"
    if time_col is None:
        df = df.reset_index()

    rows = []
    for i in range(1, 11):
        bid_px = f"bid_px_{i:02d}"
        bid_sz = f"bid_sz_{i:02d}"
        ask_px = f"ask_px_{i:02d}"
        ask_sz = f"ask_sz_{i:02d}"
        if bid_px not in df.columns:
            break
        for _, row in df.iterrows():
            ts = row.get(time_col, row.get(index_col))
            rows.append({"level": i, "side": "bid", "price": row[bid_px], "size": row[bid_sz], "ts": ts})
            rows.append({"level": i, "side": "ask", "price": row[ask_px], "size": row[ask_sz], "ts": ts})

    return pd.DataFrame(rows)


def plot_orderbook_heatmap(df_long: pd.DataFrame) -> go.Figure:
    """
    Plot heatmap of orderbook size by level and time.
    df_long: output of orderbook_wide_to_long (columns: level, side, size, ts).
    """
    pivot = df_long.pivot_table(index="ts", columns=["level", "side"], values="size", aggfunc="sum", fill_value=0)
    fig = px.imshow(pivot.T, aspect="auto", labels=dict(x="Time", y="Level-Side", color="Size"))
    return fig
