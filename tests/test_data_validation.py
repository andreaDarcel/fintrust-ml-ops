# tests/test_data_validation.py
import pandas as pd
import pytest
from datetime import datetime

from src.data_validation import (
    validate_customers, validate_transactions,
    DataValidationError,
)


# --- Fixtures : minimal valid datasets ---
@pytest.fixture
def valid_customers() -> pd.DataFrame:
    return pd.DataFrame([{
        "customer_id": "FT-C00001",
        "customer_name": "John Doe",
        "age": 30,
        "gender": "M",
        "city": "Kano",
        "customer_segment": "Premium",
        "account_type": "Savings",
        "tenure_months": 12,
        "monthly_income_band": "Below 100k",
        "digital_engagement_score": 50.0,
        "preferred_channel": "Mobile App",
        "account_status": "Active",
    }])


@pytest.fixture
def valid_transactions() -> pd.DataFrame:
    return pd.DataFrame([{
        "transaction_id": "FT-T000001",
        "customer_id": "FT-C00001",
        "channel": "Mobile App",
        "Device_Type": "Mobile",
        "Location": "Lagos",
        "amount_ngn": 1000.0,
        "transaction_type": "transfer",
        "transaction_date": datetime(2024, 1, 1),
        "international_transaction": "No",
        "transaction_status": "success",
        "risk_review_flag": "No",
    }])


# ============================================================
# CAS 1 : DATASET EMPTY
# ============================================================
def test_empty_transactions_rejected():
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(pd.DataFrame())
    assert exc.value.report["issues"]["empty_dataset"]


# ============================================================
# CAS 2 : MISSED COLUMNS
# ============================================================
def test_missing_column_transactions(valid_transactions):
    df = valid_transactions.drop(columns=["channel"])
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(df)
    assert "channel" in exc.value.report["issues"]["missing_columns"]


# ============================================================
# CAS 3 : UNEXPECTED COLUMNS
# ============================================================
def test_unexpected_column_transactions(valid_transactions):
    df = valid_transactions.copy()
    df["malicious_field"] = "hack"
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(df)
    assert "malicious_field" in exc.value.report["issues"]["unexpected_columns"]


# ============================================================
# CAS 4 : MISSED VALUES
# ============================================================
def test_missing_value_in_amount(valid_transactions):
    df = valid_transactions.copy()
    df.loc[0, "amount_ngn"] = None
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(df)
    cols = [i["column"] for i in exc.value.report["issues"]["missing_values"]]
    assert "amount_ngn" in cols


# ============================================================
# CAS 5 : INCORRECT TYPES
# ============================================================
def test_wrong_type_amount(valid_transactions):
    df = valid_transactions.copy()
    df["amount_ngn"] = "pas_un_nombre"
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(df)
    assert len(exc.value.report["issues"]["invalid_rows"]) == 1


# ============================================================
# CAS 6 : UNEXPECTED CATEGORIES
# ============================================================
@pytest.mark.parametrize("bad_channel", ["ATC","Mac", ""])
def test_invalid_channel(valid_transactions, bad_channel):
    df = valid_transactions.copy()
    df.loc[0, "channel"] = bad_channel
    with pytest.raises(DataValidationError):
        validate_transactions(df)


def test_invalid_gender(valid_customers):
    df = valid_customers.copy()
    df.loc[0, "gender"] = "Alien"
    with pytest.raises(DataValidationError):
        validate_customers(df)


# ============================================================
# CAS 7 :INVALIDS ENTER
# ============================================================
def test_negative_amount(valid_transactions):
    df = valid_transactions.copy()
    df.loc[0, "amount_ngn"] = -1000.0
    with pytest.raises(DataValidationError):
        validate_transactions(df)


def test_age_out_of_range(valid_customers):
    df = valid_customers.copy()
    df.loc[0, "age"] = 5  # < 18
    with pytest.raises(DataValidationError):
        validate_customers(df)


def test_future_timestamp(valid_transactions):
    df = valid_transactions.copy()
    df.loc[0, "timestamp"] = datetime(2099, 1, 1)
    with pytest.raises(DataValidationError):
        validate_transactions(df)


# ============================================================
# CAS 8 : VALID DATASET → PASS
# ============================================================
def test_valid_transactions_pass(valid_transactions):
    report = validate_transactions(valid_transactions, raise_on_error=False)
    assert report["is_valid"] is True
    assert report["stats"]["valid_rows"] == 1


def test_valid_customers_pass(valid_customers):
    report = validate_customers(valid_customers, raise_on_error=False)
    assert report["is_valid"] is True


# ============================================================
# CAS 9 : TOLERANT MODE → take valid lines
# ============================================================
def test_tolerant_mode_keeps_valid_rows(valid_transactions):
    df = pd.concat([valid_transactions, valid_transactions], ignore_index=True)
    df.loc[1, "amount"] = -999  # 2ème ligne invalide
    report = validate_transactions(df, raise_on_error=False)
    assert report["is_valid"] is False
    assert len(report["valid_rows"]) == 1