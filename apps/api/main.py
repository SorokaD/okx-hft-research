"""
Minimal FastAPI app for OKX HFT Research.
Run: uvicorn apps.api.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="OKX HFT Research API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok", "service": "okx-hft-research"}
