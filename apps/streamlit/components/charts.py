"""
Streamlit chart wrappers.
"""

import streamlit as st


def render_plotly(fig):
    """Render Plotly figure in Streamlit."""
    st.plotly_chart(fig, use_container_width=True)
