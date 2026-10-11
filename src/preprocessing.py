"""Load validated datasets and create consistent model features."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config
from src.data_validation import (
    DataValidationError,
    validate_customers,
    validate_transactions,
)

logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model inputs without identifiers or target columns."""
    features = df.rename(columns=lambda column: str(column).strip().lower())
    features = features.rename(columns={"transaction_datetime": "transaction_date"})
    features = features.copy()

    if "transaction_date" in features.columns:
        transaction_date = pd.to_datetime(
            features.pop("transaction_date"), format="mixed", errors="coerce", utc=True
        )
        features["transaction_hour"] = transaction_date.dt.hour
        features["transaction_day_of_week"] = transaction_date.dt.dayofweek
        features["transaction_is_weekend"] = transaction_date.dt.dayofweek.isin([5, 6])

    return features.drop(
        columns=[
            "transaction_id",
            "customer_id",
            "risk_review_flag",
            "is_fraud",
        ],
        errors="ignore",
    )


def load_customers(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the customer CSV."""
    return _load_dataset(path or config.CUSTOMER_FILE, validate_customers)


def load_transactions(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the transaction CSV."""
    return _load_dataset(path or config.TRANSACTION_FILE, validate_transactions)


def _load_dataset(path: str | Path, validator) -> pd.DataFrame:
    try:
        report = validator(pd.read_csv(path), raise_on_error=True)
        return report["valid_rows"]
    except DataValidationError as error:
        _log_validation_failure(error)
        raise


def _log_validation_failure(error: DataValidationError) -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"validation_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    log_file.write_text(json.dumps(error.report, indent=2, default=str), encoding="utf-8")
    logger.error("Dataset validation failed; report saved to %s", log_file)
