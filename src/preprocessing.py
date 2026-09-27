# src/preprocessing.py
import pandas as pd
from src import config
from src.data_validation import (
    validate_customers, validate_transactions,
    DataValidationError,
)


def load_customers(path=None) -> pd.DataFrame:
    path = path or config.CUSTOMER_FILE
    df = pd.read_csv(path)
    try:
        report = validate_customers(df, raise_on_error=True)
        return report["valid_rows"]
    except DataValidationError as e:
        # LOG obligatoire en production
        _log_validation_failure(e)
        raise


def load_transactions(path=None) -> pd.DataFrame:
    path = path or config.TRANSACTION_FILE
    df = pd.read_csv(path)
    try:
        report = validate_transactions(df, raise_on_error=True)
        return report["valid_rows"]
    except DataValidationError as e:
        _log_validation_failure(e)
        raise


def _log_validation_failure(error: DataValidationError) -> None:
    """Écrit le rapport dans un fichier de log (audit trail)."""
    import json
    from datetime import datetime
    from pathlib import Path

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"validation_{datetime.utcnow():%Y%m%d_%H%M%S}.json"

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(error.report, f, indent=2, default=str)

    print(f"🚨 {error}")
    print(f"📄 Rapport détaillé : {log_file}")