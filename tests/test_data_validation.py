# tests/test_data_validation.py
import pandas as pd
import pytest
from datetime import datetime

from src.data_validation import (
    validate_customers, validate_transactions,
    DataValidationError,
)


# --- Fixtures : datasets valides minimaux ---
@pytest.fixture
def valid_customers() -> pd.DataFrame:
    return pd.DataFrame([{
        "customer_id": "C0001",
        "age": 30,
        "gender": "M",
        "country": "France",
        "signup_date": datetime(2023, 1, 1),
        "income": 45000.0,
    }])


@pytest.fixture
def valid_transactions() -> pd.DataFrame:
    return pd.DataFrame([{
        "transaction_id": "T00001",
        "customer_id": "C0001",
        "amount": 100.0,
        "currency": "EUR",
        "transaction_type": "transfer",
        "timestamp": datetime(2024, 1, 1),
    }])


# ============================================================
# CAS 1 : DATASET VIDE
# ============================================================
def test_empty_transactions_rejected():
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(pd.DataFrame())
    assert exc.value.report["issues"]["empty_dataset"]


# ============================================================
# CAS 2 : COLONNES MANQUANTES
# ============================================================
def test_missing_column_transactions(valid_transactions):
    df = valid_transactions.drop(columns=["currency"])
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(df)
    assert "currency" in exc.value.report["issues"]["missing_columns"]


# ============================================================
# CAS 3 : COLONNES INATTENDUES
# ============================================================
def test_unexpected_column_transactions(valid_transactions):
    df = valid_transactions.copy()
    df["malicious_field"] = "hack"
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(df)
    assert "malicious_field" in exc.value.report["issues"]["unexpected_columns"]


# ============================================================
# CAS 4 : VALEURS MANQUANTES
# ============================================================
def test_missing_value_in_amount(valid_transactions):
    df = valid_transactions.copy()
    df.loc[0, "amount"] = None
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(df)
    cols = [i["column"] for i in exc.value.report["issues"]["missing_values"]]
    assert "amount" in cols


# ============================================================
# CAS 5 : TYPES INCORRECTS
# ============================================================
def test_wrong_type_amount(valid_transactions):
    df = valid_transactions.copy()
    df["amount"] = "pas_un_nombre"
    with pytest.raises(DataValidationError) as exc:
        validate_transactions(df)
    assert len(exc.value.report["issues"]["invalid_rows"]) == 1


# ============================================================
# CAS 6 : CATÉGORIES INATTENDUES
# ============================================================
@pytest.mark.parametrize("bad_currency", ["XYZ", "eu", "EURO", ""])
def test_invalid_currency(valid_transactions, bad_currency):
    df = valid_transactions.copy()
    df.loc[0, "currency"] = bad_currency
    with pytest.raises(DataValidationError):
        validate_transactions(df)


def test_invalid_gender(valid_customers):
    df = valid_customers.copy()
    df.loc[0, "gender"] = "Alien"
    with pytest.raises(DataValidationError):
        validate_customers(df)


# ============================================================
# CAS 7 : ENTRÉES INVALIDES (règles métier)
# ============================================================
def test_negative_amount(valid_transactions):
    df = valid_transactions.copy()
    df.loc[0, "amount"] = -10
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
# CAS 8 : DATASET VALIDE → PASSE
# ============================================================
def test_valid_transactions_pass(valid_transactions):
    report = validate_transactions(valid_transactions, raise_on_error=False)
    assert report["is_valid"] is True
    assert report["stats"]["valid_rows"] == 1


def test_valid_customers_pass(valid_customers):
    report = validate_customers(valid_customers, raise_on_error=False)
    assert report["is_valid"] is True


# ============================================================
# CAS 9 : MODE TOLÉRANT → récupère les lignes valides
# ============================================================
def test_tolerant_mode_keeps_valid_rows(valid_transactions):
    df = pd.concat([valid_transactions, valid_transactions], ignore_index=True)
    df.loc[1, "amount"] = -999  # 2ème ligne invalide
    report = validate_transactions(df, raise_on_error=False)
    assert report["is_valid"] is False
    assert len(report["valid_rows"]) == 1