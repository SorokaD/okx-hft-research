"""
Spread charts: absolute and relative spread over time.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_spread_over_time(
    df: pd.DataFrame,
    spread_series: pd.Series,
    ts_col: str = "ts_event",
    title: str = "Spread over time",
) -> go.Figure:
    """
    Line chart of spread (or any series) vs time.
    """
    ts = df[ts_col] if ts_col in df.columns else df.index
    fig = go.Figure(data=go.Scatter(x=ts, y=spread_series, mode="lines"))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Spread")
    return fig


def plot_relative_spread_over_time(
    df: pd.DataFrame,
    spread_series: pd.Series,
    ts_col: str = "ts_event",
) -> go.Figure:
    """Relative spread (in bps) over time."""
    return plot_spread_over_time(
        df, spread_series * 1e4, ts_col, title="Relative spread (bps)"
    )
