"""
Streamlit filters: instrument, time range.
"""

import streamlit as st


def render_instrument_filter(default: str = "BTC-USDT-SWAP") -> str:
    """Instrument selector sidebar widget."""
    instruments = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
    return st.sidebar.selectbox("Instrument", instruments, index=0)
