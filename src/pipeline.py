# src/pipeline.py
"""
Unified FinTrust prediction pipeline.

Usage:
    pipeline = FinTrustPipeline()
    result = pipeline.predict_one({...})
    results = pipeline.predict_batch(df)
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.data_validation import TransactionRow, validate_transactions
from src.models.model import load_model
from src.preprocessing import build_features

logger = logging.getLogger(__name__)


class FinTrustPipeline:
    """End-to-end transaction review pipeline."""

    def __init__(self, model_path=None):
        self.model: Pipeline = load_model(model_path)
        logger.info("Pipeline initialisé avec %s", config.MODEL_FILE)

    # ---------- 1. SINGLE-TRANSACTION PREDICTION ----------
    def predict_one(self, row: dict[str, Any]) -> dict[str, Any]:
        """Validate one transaction and return its review prediction."""
        # 1. Validate the transaction with Pydantic.
        validated = TransactionRow(**row)

        # 2. Create a single-row DataFrame.
        df = pd.DataFrame([validated.model_dump()])

        # 3. Build model features.
        df = build_features(df)

        # 4. Predict the review status.
        proba = float(self.model.predict_proba(df)[0, 1])
        prediction = int(proba >= 0.5)

        # 5. Build the structured response.
        return {
            "transaction_id": validated.transaction_id,
            "prediction": prediction,
            "probability": round(proba, 4),
            "requires_review": bool(prediction == 1),
            "model_version": config.MODEL_VERSION,
        }

    # ---------- 2. BATCH PREDICTION ----------
    def predict_batch(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate a DataFrame, predict valid rows, and report rejected rows.
        """
        report = validate_transactions(df, raise_on_error=False)
        valid_df = report["valid_rows"]

        if valid_df.empty:
            return {
                "predictions": [],
                "n_input": len(df),
                "n_valid": 0,
                "n_rejected": len(df),
                "validation_report": {
                    "summary": report["summary"],
                    "issues": report["issues"],
                },
            }

        # Build features for valid transactions.
        features_df = build_features(valid_df)

        # Predict all valid rows in one batch.
        probas = self.model.predict_proba(features_df)[:, 1]
        preds = (probas >= 0.5).astype(int)

        predictions = [
            {
                "transaction_id": valid_df.iloc[i]["transaction_id"],
                "prediction": int(preds[i]),
                "probability": round(float(probas[i]), 4),
                "requires_review": bool(preds[i]),
            }
            for i in range(len(valid_df))
        ]

        return {
            "predictions": predictions,
            "n_input": len(df),
            "n_valid": len(valid_df),
            "n_rejected": len(df) - len(valid_df),
            "validation_report": {
                "summary": report["summary"],
                "issues": report["issues"],
            },
        }
