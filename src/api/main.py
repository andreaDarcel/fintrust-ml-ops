from fastapi import FastAPI, HTTPException, status
from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel
from src import config, __version__
from src.preprocessing import load_customers, load_transactions, feature_preparation
from src.data_validation import DataValidationError, TransactionRow, CustomerRow

## 1. RESPONSE SCHEMAS
class PredictionResponse(BaseModel):
    customer_id: str 
    transaction_id: str
    prediction: int
    probability: float
    model_status: str
    model_version: str
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool = True
    version: str

## 2. MODEL LOADING
ml_state: dict = {
    "model_loaded": True,
    "loaded_at": datetime.now(timezone.utc).isoformat(),
    "model_version": __version__,
}

## 3. APP
app = FastAPI(
    title="FinTrust ML Ops API",
    description="API for FinTrust ML Operations",
    version=__version__,
)

## 4. ENDPOINTS
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok", 
        model_loaded=ml_state["model_loaded"], 
        version=ml_state["model_version"],
    )