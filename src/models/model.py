"""Train, evaluate, save, and load the risk-review model."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.preprocessing import build_features

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "amount_ngn",
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_is_weekend",
]
CATEGORICAL_FEATURES = [
    "transaction_type",
    "channel",
    "device_type",
    "location",
    "international_transaction",
    "transaction_status",
]


def build_preprocessor() -> ColumnTransformer:
    """Build the fitted-at-training preprocessing steps."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def build_model() -> Pipeline:
    """Build the complete preprocessing and classifier pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            random_state=config.RANDOM_SEED,
            n_jobs=1,
            class_weight="balanced",
        )),
    ])


def train_model(
    df: pd.DataFrame,
    target_col: str = config.TRAINING_TARGET,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train on a real binary target and return the model and holdout metrics."""
    if target_col not in df.columns:
        raise ValueError(f"Training target '{target_col}' is missing")

    target = df[target_col]
    if target_col == "risk_review_flag":
        target = target.map({"No": 0, "Yes": 1})
    else:
        target = pd.to_numeric(target, errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Training target must contain only binary 0/1 or Yes/No values")
    if target.value_counts().min() < 2:
        raise ValueError("Training requires at least two rows for each target class")

    features = build_features(df.drop(columns=[target_col]))
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target.astype(int),
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=target,
    )

    pipeline = build_model()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }
    logger.info(
        "Trained risk-review model: f1=%.4f, roc_auc=%.4f",
        metrics["f1"],
        metrics["roc_auc"],
    )
    return pipeline, metrics


def save_model(
    pipeline: Pipeline,
    metrics: dict[str, Any],
    model_path: str | Path = config.MODEL_FILE,
) -> None:
    """Save the model and JSON metadata beside the model artifact."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "fintrust_risk_review",
        "version": config.MODEL_VERSION,
        "algorithm": "RandomForestClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": pipeline.named_steps["classifier"].get_params(deep=False),
        "features": pipeline.named_steps["preprocessor"].get_feature_names_out().tolist(),
        "target": config.TRAINING_TARGET,
        "training_data": {
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
        },
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"n_train", "n_test"}
        },
    }
    metadata_path = model_path.parent / "config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model and metadata to %s", model_path.parent)


def load_model(model_path: str | Path | None = None) -> Pipeline:
    """Load a trained model or raise FileNotFoundError when absent."""
    path = Path(model_path or config.MODEL_FILE)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
