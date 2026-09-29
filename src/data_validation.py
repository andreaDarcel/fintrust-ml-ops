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
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src import config


# ============================================================
# 1. CLIENT SCHEMA
# ============================================================
class CustomerRow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)            ##saying that we don't want any extra fields and we want to strip whitespace from strings

    customer_id: str = Field(..., validation_alias=config.CUSTOMER_ID_PATTERN)  ##the customer id must match the pattern defined in config.py
    customer_name: str = Field(..., min_length=2, max_length=100)  ##the customer name must be between 2 and 100 characters
    age: int = Field(..., ge=config.MIN_AGE, le=config.MAX_AGE)
    gender: str = Field(..., validation_alias=config.VALID_GENDERS)
    city: str = Field(..., min_length=2, max_length=56)
    customer_segment: str = Field(..., validation_alias=config.VALID_CUSTOMER_SEGMENTS)
    monthly_income_band: str = Field(..., validation_alias=config.MONTHLY_INCOME_BANDS)
    preferred_channel: str = Field(..., validation_alias=config.VALID_PREFERRED_CHANNELS)
    tenure_months: int = Field(..., ge=config.MIN_TENURE_MONTHS, le=config.MAX_TENURE_MONTHS)
    account_type: str = Field(..., validation_alias=config.VALID_ACCOUNT_TYPES)
    digital_engagement_score: float = Field(..., ge=10, le=100)
    account_status: str = Field(..., validation_alias=config.VALID_ACCOUNT_STATUS)

    @field_validator("gender")
    @classmethod
    def gender_must_be_valid(cls, v: str) -> str:
        if v not in config.VALID_GENDERS:
            raise ValueError(
                f"Genre '{v}' inconnu. Attendu : {sorted(config.VALID_GENDERS)}"
            )
        return v


# ============================================================
# 2. TRANSACTION SCHEMA
# ============================================================
class TransactionRow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: str = Field(..., validation_alias=config.TRANSACTION_ID_PATTERN)
    customer_id: str = Field(..., validation_alias=config.CUSTOMER_ID_PATTERN)
    amount_ngn: float = Field(..., ge=100.0)
    transaction_type: str = Field(..., validation_alias=config.VALID_TRANSACTION_TYPES)
    channel: str = Field(..., validation_alias=config.VALID_CHANNEL)
    device_type: str = Field(..., validation_alias=config.VALID_DEVICE_TYPE)
    location: str = Field(..., min_length=2, max_length=56)
    transaction_date: datetime = Field(..., validation_alias=config.TRANSACTION_DATE_FORMAT)
    international_transaction: str = Field(..., validation_alias=config.VALID_INTERNATIONAL_TRANSACTION)
    transaction_status: str = Field(..., validation_alias=config.VALID_TRANSACTION_STATUS)
    risk_review_flag: str = Field(..., validation_alias=config.VALID_RISK_REVIEW_FLAG)

    @field_validator("transaction_type")
    @classmethod
    def type_must_be_supported(cls, v: str) -> str:
        if v not in config.VALID_TRANSACTION_TYPES:
            raise ValueError(
                f"Type '{v}' inconnu. "
                f"Attendu : {sorted(config.VALID_TRANSACTION_TYPES)}"
            )
        return v

    @field_validator("transaction_date")
    @classmethod
    def timestamp_not_in_future(cls, v: datetime) -> datetime:
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
        except Exception as e:
            report["issues"]["invalid_rows"].append({
                "row_index": int(idx),
                "error": str(e).split("\n")[0],  # 1ère ligne du message
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
    return _validate_dataframe(
        df, CustomerRow, config.CUSTOMER_COLUMNS,
        "customers", raise_on_error,
    )


def validate_transactions(df: pd.DataFrame, raise_on_error: bool = True) -> dict:
    return _validate_dataframe(
        df, TransactionRow, config.TRANSACTION_COLUMNS,
        "transactions", raise_on_error,
    )