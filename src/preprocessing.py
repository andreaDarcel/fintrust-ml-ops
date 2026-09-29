# src/preprocessing.py
import pandas as pd
from src import config
from datetime import datetime, timezone
from src.data_validation import validate_customers, validate_transactions, DataValidationError

def missing_values_handling(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    if dataset_name == "customers":
        # For customers, we can drop rows with missing critical values
        critical_columns = ["customer_id", "customer_name", "age", "gender"]
        df = df.dropna(subset=critical_columns)
    elif dataset_name == "transactions":
        # For transactions, we can fill missing values with defaults or drop
        df["amount_ngn"] = df["amount_ngn"].fillna(0)  # Fill missing amounts with 0
        df = df.dropna(subset=["transaction_id", "customer_id"])  # Drop rows with missing IDs
    return df

def categorical_encoding(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Encode categorical variables in the dataset."""
    if dataset_name == "customers":
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "Other": -1})
        df = pd.get_dummies(df, columns=["customer_segment", "account_type", "preferred_channel"], drop_first=True)
    elif dataset_name == "transactions":
        df = pd.get_dummies(df, columns=["transaction_type", "channel", "device_type", "location"], drop_first=True)
    return df

def numerical_processing(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Process numerical variables in the dataset."""
    if dataset_name == "customers":
        # Normalize age and digital engagement score
        df["age"] = (df["age"] - df["age"].mean()) / df["age"].std()
        df["digital_engagement_score"] = (df["digital_engagement_score"] - df["digital_engagement_score"].mean()) / df["digital_engagement_score"].std()
    elif dataset_name == "transactions":
        # Normalize amount
        df["amount_ngn"] = (df["amount_ngn"] - df["amount_ngn"].mean()) / df["amount_ngn"].std()
    return df

def feature_preparation(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Prepare features for modeling."""
    df = missing_values_handling(df, dataset_name)
    df = categorical_encoding(df, dataset_name)
    df = numerical_processing(df, dataset_name)
    return df

def load_customers(path=None) -> pd.DataFrame:
    path = path or config.CUSTOMER_FILE
    df = pd.read_csv(path)
    try:
        report = validate_customers(df, raise_on_error=True)
        return report["valid_rows"]
    except DataValidationError as e:
        # LOG in production
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
    """report in a log file (audit trail)."""
    import json
    from datetime import datetime
    from pathlib import Path

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"validation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(error.report, f, indent=2, default=str)

    print(f"🚨 {error}")
    print(f"📄 Rapport détaillé : {log_file}")