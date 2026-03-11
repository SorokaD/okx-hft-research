"""
Orderbook heatmap page.
"""

import streamlit as st

from research.visualization.orderbook_heatmap import orderbook_wide_to_long, plot_orderbook_heatmap
from apps.streamlit.components.filters import render_instrument_filter
from apps.streamlit.components.charts import render_plotly

st.title("Orderbook Heatmap")
inst = render_instrument_filter()

# Placeholder: load data when DB is configured
# df = load_orderbook_snapshot(inst, ...)
# For now show message
st.info("Загрузите данные из БД или используйте экспортированный sample. После загрузки DataFrame в df_long — вызывайте plot_orderbook_heatmap(df_long).")

# Example with mock data for structure (replace with load_orderbook_snapshot when DB is ready)
import pandas as pd
sample = pd.DataFrame({
    "bid_px_01": [100.0, 100.0], "ask_px_01": [100.1, 100.1],
    "bid_sz_01": [1.0, 1.1], "ask_sz_01": [0.5, 0.6],
    "bid_px_02": [99.9, 99.9], "ask_px_02": [100.2, 100.2],
    "bid_sz_02": [2.0, 2.1], "ask_sz_02": [1.0, 1.0],
})
sample["ts_event"] = pd.date_range("2024-01-01 12:00:00", periods=2, freq="1s")
df_long = orderbook_wide_to_long(sample)
if not df_long.empty:
    try:
        fig = plot_orderbook_heatmap(df_long)
        render_plotly(fig)
    except Exception as e:
        st.warning(f"Heatmap placeholder: {e}")
