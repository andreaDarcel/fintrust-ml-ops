# src/api/main.py
"""FinTrust Prediction API."""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_validation import TransactionRow
from src.pipeline import FinTrustPipeline
from src import config, __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

state: dict = {"pipeline": None}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the prediction pipeline once when the API starts."""
    logger.debug("Starting lifespan for %s", application.title)
    try:
        state["pipeline"] = FinTrustPipeline()
        logger.info("Pipeline chargé ✅")
    except FileNotFoundError:
        logger.warning("⚠️ Modèle absent → mode dégradé")
        state["pipeline"] = None
    yield
    state["pipeline"] = None


app = FastAPI(
    title="FinTrust ML API",
    version=__version__,
    description="API d’estimation du besoin de revue d’une transaction",
    lifespan=lifespan,
)


# ============================================================
# RESPONSE SCHEMAS
# ============================================================
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class PredictionResponse(BaseModel):
    transaction_id: str
    prediction: int
    probability: float
    requires_review: bool
    model_version: str


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/health", response_model=HealthResponse)
def health():
    model_loaded = state["pipeline"] is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=__version__,
    )


@app.get("/info")
def info():
    """Return the model metadata."""
    if not config.MODEL_CONFIG_FILE.exists():
        raise HTTPException(404, "Config modèle absente")
    return json.loads(config.MODEL_CONFIG_FILE.read_text(encoding="utf-8"))


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRow):
    """Return a review prediction for one transaction."""
    if state["pipeline"] is None:
        raise HTTPException(503, "Modèle non chargé")
    try:
        return state["pipeline"].predict_one(transaction.model_dump())
    except Exception:
        logger.exception("Erreur inférence")
        raise HTTPException(500, "Prediction failed") from None
