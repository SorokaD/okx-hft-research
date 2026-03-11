#!/bin/bash
# Run Streamlit app
cd "$(dirname "$0")/.."
streamlit run apps/streamlit/Home.py --server.port "${STREAMLIT_SERVER_PORT:-8501}"
