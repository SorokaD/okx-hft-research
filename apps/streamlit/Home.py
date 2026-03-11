"""
OKX HFT Research — Streamlit home page.
Run: streamlit run apps/streamlit/Home.py
"""

import streamlit as st

st.set_page_config(
    page_title="OKX HFT Research",
    page_icon="📊",
    layout="wide",
)

st.title("OKX HFT Research")
st.markdown("""
Исследовательский слой для анализа рыночной микроструктуры OKX.

**Страницы:**
- **Orderbook Heatmap** — тепловая карта стакана
- **Orderbook Replay** — анимация стакана во времени
- **Imbalance** — анализ L1/L10 imbalance

Данные загружаются из TimescaleDB (okx_core). Настройте `.env` для подключения к БД.
""")
