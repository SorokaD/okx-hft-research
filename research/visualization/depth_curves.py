"""
Depth curves: cumulative bid/ask size vs price level.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_depth_curves(df_row: pd.Series) -> go.Figure:
    """
    Plot cumulative bid and ask depth for a single orderbook snapshot.
    df_row: one row of wide-format L10 orderbook.
    """
    bid_px = [df_row.get(f"bid_px_{i:02d}") for i in range(1, 11) if f"bid_px_{i:02d}" in df_row.index]
    bid_sz = [df_row.get(f"bid_sz_{i:02d}", 0) for i in range(1, len(bid_px) + 1)]
    ask_px = [df_row.get(f"ask_px_{i:02d}") for i in range(1, 11) if f"ask_px_{i:02d}" in df_row.index]
    ask_sz = [df_row.get(f"ask_sz_{i:02d}", 0) for i in range(1, len(ask_px) + 1)]
    bid_cum = pd.Series(bid_sz).cumsum().tolist()
    ask_cum = pd.Series(ask_sz).cumsum().tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bid_cum, y=bid_px, mode="lines", name="Bid"))
    fig.add_trace(go.Scatter(x=ask_cum, y=ask_px, mode="lines", name="Ask"))
    fig.update_layout(
        title="Orderbook depth curves",
        xaxis_title="Cumulative size",
        yaxis_title="Price",
    )
    return fig
