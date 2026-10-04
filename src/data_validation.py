# src/data_validation.py
"""
Validation data Fintrust

- CustomerRow       : one row of the customers file
- TransactionRow    : one row of the transactions file

- validate_customers(df)
- validate_transactions(df)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src import config


# ============================================================
# 1. CLIENT SCHEMA
# ============================================================
class CustomerRow(BaseModel):
    """Validated customer record from the customers dataset."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)            ##saying that we don't want any extra fields and we want to strip whitespace from strings
    customer_id: str = Field(..., pattern=config.CUSTOMER_ID_PATTERN)
    customer_name: str = Field(..., min_length=2, max_length=100)  ##the customer name must be between 2 and 100 characters
    age: int = Field(..., ge=config.MIN_AGE, le=config.MAX_AGE)
    gender: str
    city: str = Field(..., min_length=2, max_length=56)
    customer_segment: str
    monthly_income_band: str
    preferred_channel: str
    tenure_months: int = Field(..., ge=config.MIN_TENURE_MONTHS, le=config.MAX_TENURE_MONTHS)
    account_type: str
    digital_engagement_score: float = Field(..., ge=10, le=100)
    account_status: str

    @field_validator("gender")
    @classmethod
    def gender_must_be_valid(cls, v: str) -> str:
        """Validate that the gender matches the allowed values."""
        if v not in config.VALID_GENDERS:
            raise ValueError(
                f"Genre '{v}' inconnu. Attendu : {sorted(config.VALID_GENDERS)}"
            )
        return v

    @field_validator(
        "customer_segment", "monthly_income_band", "preferred_channel",
        "account_type", "account_status",
    )
    @classmethod
    def customer_categories_must_be_valid(cls, value: str, info) -> str:
        allowed_values = {
            "customer_segment": config.VALID_CUSTOMER_SEGMENTS,
            "monthly_income_band": config.MONTHLY_INCOME_BANDS,
            "preferred_channel": config.VALID_PREFERRED_CHANNELS,
            "account_type": config.VALID_ACCOUNT_TYPES,
            "account_status": config.VALID_ACCOUNT_STATUS,
        }[info.field_name]
        if value not in allowed_values:
            expected = sorted(allowed_values)
            raise ValueError(f"Valeur '{value}' invalide. Attendu : {expected}")
        return value


# ============================================================
# 2. TRANSACTION SCHEMA
# ============================================================
class TransactionRow(BaseModel):
    """Validated transaction record from the transactions dataset."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: str = Field(..., pattern=config.TRANSACTION_ID_PATTERN)
    customer_id: str = Field(..., pattern=config.CUSTOMER_ID_PATTERN)
    amount_ngn: float = Field(..., ge=100.0)
    transaction_type: str
    channel: str
    device_type: str
    location: str = Field(..., min_length=2, max_length=56)
    transaction_date: datetime
    international_transaction: str
    transaction_status: str
    risk_review_flag: str

    @field_validator("transaction_type")
    @classmethod
    def type_must_be_supported(cls, v: str) -> str:
        """Validate that the transaction type is supported."""
        if v not in config.VALID_TRANSACTION_TYPES:
            raise ValueError(
                f"Type '{v}' inconnu. "
                f"Attendu : {sorted(config.VALID_TRANSACTION_TYPES)}"
            )
        return v

    @field_validator(
        "channel", "device_type", "international_transaction",
        "transaction_status", "risk_review_flag",
    )
    @classmethod
    def transaction_categories_must_be_valid(cls, value: str, info) -> str:
        allowed_values = {
            "channel": config.VALID_CHANNEL,
            "device_type": config.VALID_DEVICE_TYPE,
            "international_transaction": config.VALID_INTERNATIONAL_TRANSACTION,
            "transaction_status": config.VALID_TRANSACTION_STATUS,
            "risk_review_flag": config.VALID_RISK_REVIEW_FLAG,
        }[info.field_name]
        if value not in allowed_values:
            expected = sorted(allowed_values)
            raise ValueError(f"Valeur '{value}' invalide. Attendu : {expected}")
        return value

    @field_validator("transaction_date")
    @classmethod
    def timestamp_not_in_future(cls, v: datetime) -> datetime:
        """Ensure the transaction timestamp is not set in the future."""
        if v > datetime.now(timezone.utc):
            raise ValueError("timestamp ne peut pas être dans le futur")
        return v


# ============================================================
# 3. PERSONNALIZED ERROR
# ============================================================
class DataValidationError(Exception):
    """Raised when a dataset fails validation. Contains a detailed report."""
    def __init__(self, dataset_name: str, report: dict):
        self.dataset_name = dataset_name
        self.report = report
        super().__init__(f"[{dataset_name}] {report['summary']}")


# ============================================================
# 4. FONCTION OF VALIDATION
# ============================================================
def _validate_dataframe(
    df: pd.DataFrame,
    schema: type[BaseModel],
    required_columns: set[str],
    dataset_name: str,
    raise_on_error: bool = True,
) -> dict:
    """
    Validate a dataframe against a pydantic schema and a set of required columns.
    Returns a report dictionary with validation results.
    """
    report: dict[str, Any] = {
        "dataset": dataset_name,
        "is_valid": True,
        "summary": "",
        "issues": {
            "empty_dataset": [],
            "missing_columns": [],
            "unexpected_columns": [],
            "missing_values": [],
            "invalid_rows": [],
        },
        "valid_rows": pd.DataFrame(),
        "stats": {"total_rows": 0, "valid_rows": 0, "invalid_rows": 0},
    }

    # ---------- CHECK 1 : dataset empty----------
    if df is None or df.empty:
        report["is_valid"] = False
        report["issues"]["empty_dataset"].append("Dataset is empty or None")
        report["summary"] = "❌ Empty Dataset"
        if raise_on_error:
            raise DataValidationError(dataset_name, report)
        return report

    report["stats"]["total_rows"] = len(df)

    # ---------- CHECK 2 : missing columns ----------
    actual_cols = set(df.columns)
    missing_cols = required_columns - actual_cols
    if missing_cols:
        report["is_valid"] = False
        report["issues"]["missing_columns"] = sorted(missing_cols)

    # ---------- CHECK 3 : unexpected columns ----------
    unexpected_cols = actual_cols - required_columns
    if unexpected_cols:
        report["is_valid"] = False
        report["issues"]["unexpected_columns"] = sorted(unexpected_cols)

    # If there are missing or unexpected columns, we can stop here and return the report
    if missing_cols or unexpected_cols:
        report["summary"] = (
            f"❌ Columns issue : "
            f"{len(missing_cols)} missing."
            f"{len(unexpected_cols)} unexpected."
        )
        if raise_on_error:
            raise DataValidationError(dataset_name, report)
        return report

    # ---------- CHECK 4 : missing values ----------
    null_counts = df.isnull().sum()
    for col, n in null_counts.items():
        if n > 0:
            report["issues"]["missing_values"].append({
                "column": col,
                "count": int(n),
                "percentage": round(100 * n / len(df), 2),
            })
    if report["issues"]["missing_values"]:
        report["is_valid"] = False

    # ---------- CHECK 5 : validation line by line ----------
    valid_rows = []
    for idx, row in df.iterrows():
        try:
            schema(**row.to_dict())
            valid_rows.append(row)
        except ValidationError as e:
            report["issues"]["invalid_rows"].append({
                "row_index": int(idx),
                "error": str(e).split("\n", maxsplit=1)[0],
            })

    report["valid_rows"] = pd.DataFrame(valid_rows)
    report["stats"]["valid_rows"] = len(valid_rows)
    report["stats"]["invalid_rows"] = len(report["issues"]["invalid_rows"])

    if report["issues"]["invalid_rows"]:
        report["is_valid"] = False

    # ---------- Resume ----------
    n_issues = sum(len(v) for v in report["issues"].values())
    if report["is_valid"]:
        report["summary"] = f"✅ Dataset '{dataset_name}' valide ({len(df)} lignes)."
    else:
        report["summary"] = (
            f"❌ Dataset '{dataset_name}' invalide : "
            f"{n_issues} problème(s) sur {len(df)} lignes "
            f"({report['stats']['invalid_rows']} lignes rejetées)."
        )

    if raise_on_error and not report["is_valid"]:
        raise DataValidationError(dataset_name, report)

    return report


# ============================================================
# 5. WRAPPERS
# ============================================================
def validate_customers(df: pd.DataFrame, raise_on_error: bool = True) -> dict:
    """Validate a customers dataframe against the expected schema.

    Args:
        df: DataFrame containing customer records.
        raise_on_error: If True, raise DataValidationError when validation fails.

    Returns:
        A report dictionary with validation status, issues, and valid rows.
    """
    return _validate_dataframe(
        df, CustomerRow, config.CUSTOMER_COLUMNS,
        "customers", raise_on_error,
    )


def validate_transactions(df: pd.DataFrame, raise_on_error: bool = True) -> dict:
    """Validate a transactions dataframe against the expected schema.

    Args:
        df: DataFrame containing transaction records.
        raise_on_error: If True, raise DataValidationError when validation fails.

    Returns:
        A report dictionary with validation status, issues, and valid rows.
    """
    return _validate_dataframe(
        df, TransactionRow, config.TRANSACTION_COLUMNS,
        "transactions", raise_on_error,
    )