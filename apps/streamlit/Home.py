"""
OKX HFT Research — Streamlit home page.
Run: streamlit run apps/streamlit/Home.py
"""
import sys
from pathlib import Path

# Add project root for research import
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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
