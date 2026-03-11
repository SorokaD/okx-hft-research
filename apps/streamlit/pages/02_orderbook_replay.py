"""
Orderbook replay page.
"""

import streamlit as st
import pandas as pd

from research.visualization.orderbook_animation import plot_orderbook_replay
from apps.streamlit.components.filters import render_instrument_filter
from apps.streamlit.components.charts import render_plotly

st.title("Orderbook Replay")
inst = render_instrument_filter()

st.info("Replay анимация стакана. Загрузите данные из БД. Структура готова: plot_orderbook_replay(df).")
