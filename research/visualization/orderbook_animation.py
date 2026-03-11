"""
Orderbook animation: replay orderbook state over time.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_orderbook_animation_frames(
    df: pd.DataFrame,
    ts_col: str = "ts_event",
) -> list[go.Frame]:
    """
    Build Plotly frames for orderbook replay by time.
    df: wide-format L10 orderbook with ts_col.
    Returns list of go.Frame for animation.
    """
    frames = []
    for i, row in df.iterrows():
        bid_prices = [row.get(f"bid_px_{j:02d}") for j in range(1, 11) if f"bid_px_{j:02d}" in row]
        bid_sizes = [row.get(f"bid_sz_{j:02d}", 0) for j in range(1, len(bid_prices) + 1)]
        ask_prices = [row.get(f"ask_px_{j:02d}") for j in range(1, 11) if f"ask_px_{j:02d}" in row]
        ask_sizes = [row.get(f"ask_sz_{j:02d}", 0) for j in range(1, len(ask_prices) + 1)]
        ts = row.get(ts_col, i)
        frame = go.Frame(
            data=[
                go.Bar(x=bid_sizes, y=bid_prices, orientation="h", name="Bid", marker_color="green"),
                go.Bar(x=ask_sizes, y=ask_prices, orientation="h", name="Ask", marker_color="red"),
            ],
            layout=go.Layout(title=f"Orderbook @ {ts}"),
            name=str(ts),
        )
        frames.append(frame)
    return frames


def plot_orderbook_replay(df: pd.DataFrame, ts_col: str = "ts_event") -> go.Figure:
    """
    Create animated orderbook replay figure.
    """
    frames = build_orderbook_animation_frames(df, ts_col)
    if not frames:
        return go.Figure()
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(updatemenus=[{"type": "buttons", "buttons": [{"label": "Play", "method": "animate"}]}])
    return fig
