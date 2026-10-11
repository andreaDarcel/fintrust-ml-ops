# src/pipeline.py
"""
Pipeline unifié FinTrust.

Utilisation :
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
    """Pipeline de prédiction intégré de bout en bout."""

    def __init__(self, model_path=None):
        self.model: Pipeline = load_model(model_path)
        logger.info("Pipeline initialisé avec %s", config.MODEL_FILE)

    # ---------- 1. PRÉDICTION UNIQUE ----------
    def predict_one(self, row: dict[str, Any]) -> dict[str, Any]:
        """Input dict → validation → features → prédiction → dict."""
        # 1. Validation (Pydantic)
        validated = TransactionRow(**row)

        # 2. DataFrame d'une ligne
        df = pd.DataFrame([validated.model_dump()])

        # 3. Features
        df = build_features(df)

        # 4. Prédiction
        proba = float(self.model.predict_proba(df)[0, 1])
        prediction = int(proba >= 0.5)

        # 5. Sortie structurée
        return {
            "transaction_id": validated.transaction_id,
            "prediction": prediction,
            "probability": round(proba, 4),
            "requires_review": bool(prediction == 1),
            "model_version": config.MODEL_VERSION,
        }

    # ---------- 2. PRÉDICTION EN LOT ----------
    def predict_batch(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Input DataFrame → validation → features → prédictions → dict.
        Retourne les prédictions + un rapport sur les lignes rejetées.
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

        # Features
        features_df = build_features(valid_df)

        # Prédictions vectorisées
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
