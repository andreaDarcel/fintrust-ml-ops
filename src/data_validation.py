# src/data_validation.py
"""
FinTrust data validation.

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
# 1. CUSTOMER SCHEMA
# ============================================================
class CustomerRow(BaseModel):
    """Validated customer record from the customers dataset."""
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
    customer_id: str = Field(..., pattern=config.CUSTOMER_ID_PATTERN)
    customer_name: str = Field(..., min_length=2, max_length=100)
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
    device_type: str | None = None
    location: str | None = Field(default=None, min_length=2, max_length=56)
    transaction_date: datetime
    international_transaction: str
    transaction_status: str
    risk_review_flag: str | None = None

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
    def transaction_categories_must_be_valid(cls, value: str | None, info) -> str | None:
        if value is None:
            return value
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
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        if v > datetime.now(timezone.utc):
            raise ValueError("timestamp ne peut pas être dans le futur")
        return v


# ============================================================
# 3. CUSTOM VALIDATION ERROR
# ============================================================
class DataValidationError(Exception):
    """Raised when a dataset fails validation. Contains a detailed report."""
    def __init__(self, dataset_name: str, report: dict):
        self.dataset_name = dataset_name
        self.report = report
        super().__init__(f"[{dataset_name}] {report['summary']}")


# ============================================================
# 4. DATAFRAME VALIDATION
# ============================================================
def _normalize_dataframe(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    normalized = df.rename(columns=lambda column: str(column).strip().lower())
    normalized = normalized.rename(columns={"transaction_datetime": "transaction_date"})
    if "transaction_date" in normalized.columns:
        normalized = normalized.copy()
        normalized["transaction_date"] = pd.to_datetime(
            normalized["transaction_date"], format="mixed", errors="coerce", utc=True
        )
    return normalized.astype(object).where(pd.notna(normalized), None)


def _new_report(dataset_name: str) -> dict[str, Any]:
    return {
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


def _column_issues(
    df: pd.DataFrame,
    schema: type[BaseModel],
) -> tuple[list[str], list[str], set[str]]:
    schema_columns = set(schema.model_fields)
    required_columns = {
        name for name, field in schema.model_fields.items() if field.is_required()
    }
    actual_columns = set(df.columns)
    return (
        sorted(required_columns - actual_columns),
        sorted(actual_columns - schema_columns),
        schema_columns - required_columns,
    )


def _missing_value_issues(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues = []
    for column, count in df.isna().sum().items():
        if count:
            issues.append({
                "column": column,
                "count": int(count),
                "percentage": round(100 * count / len(df), 2),
            })
    return issues


def _validate_rows(
    df: pd.DataFrame,
    schema: type[BaseModel],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    valid_rows = []
    invalid_rows = []
    for index, row in df.iterrows():
        try:
            validated_row = schema(**row.to_dict())
            valid_rows.append(pd.Series(validated_row.model_dump()))
        except ValidationError as error:
            invalid_rows.append({
                "row_index": int(index),
                "error": str(error).split("\n", maxsplit=1)[0],
            })
    return pd.DataFrame(valid_rows), invalid_rows


def _finish_report(
    report: dict[str, Any],
    dataset_name: str,
    raise_on_error: bool,
) -> dict[str, Any]:
    if raise_on_error and not report["is_valid"]:
        raise DataValidationError(dataset_name, report)
    return report


def _validate_dataframe(
    df: pd.DataFrame | None,
    schema: type[BaseModel],
    dataset_name: str,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate columns and rows, retaining valid records in tolerant mode."""
    df = _normalize_dataframe(df)
    report = _new_report(dataset_name)

    if df is None or df.empty:
        report["is_valid"] = False
        report["issues"]["empty_dataset"].append("Dataset is empty or None")
        report["summary"] = "Empty dataset"
        return _finish_report(report, dataset_name, raise_on_error)

    report["stats"]["total_rows"] = len(df)
    missing, unexpected, optional_columns = _column_issues(df, schema)
    report["issues"]["missing_columns"] = missing
    report["issues"]["unexpected_columns"] = unexpected
    if missing or unexpected:
        report["is_valid"] = False
        report["summary"] = f"Columns issue: {len(missing)} missing, {len(unexpected)} unexpected."
        return _finish_report(report, dataset_name, raise_on_error)

    missing_values = _missing_value_issues(df)
    report["issues"]["missing_values"] = missing_values
    valid_rows, invalid_rows = _validate_rows(df, schema)
    report["valid_rows"] = valid_rows
    report["issues"]["invalid_rows"] = invalid_rows
    report["stats"]["valid_rows"] = len(valid_rows)
    report["stats"]["invalid_rows"] = len(invalid_rows)
    has_required_missing_values = any(
        issue["column"] not in optional_columns for issue in missing_values
    )
    report["is_valid"] = not has_required_missing_values and not invalid_rows

    if report["is_valid"]:
        report["summary"] = f"Dataset '{dataset_name}' valide ({len(df)} lignes)."
    else:
        issue_count = sum(len(items) for items in report["issues"].values())
        report["summary"] = (
            f"Dataset '{dataset_name}' invalide : {issue_count} problème(s) "
            f"sur {len(df)} lignes ({len(invalid_rows)} lignes rejetées)."
        )
    return _finish_report(report, dataset_name, raise_on_error)


# ============================================================
# 5. VALIDATION WRAPPERS
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
        df, CustomerRow, "customers", raise_on_error,
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
        df, TransactionRow, "transactions", raise_on_error,
    )
