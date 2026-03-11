"""
Imbalance analysis page.
"""

import streamlit as st
import pandas as pd

from research.features.imbalance import compute_l1_imbalance
from research.visualization.spread_charts import plot_spread_over_time
from apps.streamlit.components.filters import render_instrument_filter
from apps.streamlit.components.charts import render_plotly

st.title("Imbalance Analysis")
inst = render_instrument_filter()

# Mock data for structure
sample = pd.DataFrame({
    "bid_px_01": [100.0] * 10, "ask_px_01": [100.1] * 10,
    "bid_sz_01": [1.0 + i * 0.1 for i in range(10)],
    "ask_sz_01": [0.5 + i * 0.05 for i in range(10)],
})
sample["ts_event"] = pd.date_range("2024-01-01", periods=10, freq="1s")

try:
    imb = compute_l1_imbalance(sample)
    fig = plot_spread_over_time(sample, imb, title="L1 Imbalance over time")
    render_plotly(fig)
except Exception as e:
    st.error(str(e))
