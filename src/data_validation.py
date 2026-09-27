# src/data_validation.py
"""
Validation des données FinTrust.

Deux datasets → deux schémas Pydantic :
- CustomerRow       : une ligne du fichier client
- TransactionRow    : une ligne du fichier transactions

Plus deux fonctions de validation DataFrame :
- validate_customers(df)
- validate_transactions(df)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src import config


# ============================================================
# 1. SCHÉMA CLIENT
# ============================================================
class CustomerRow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    customer_id: str = Field(..., min_length=4, max_length=32)
    age: int = Field(..., ge=config.MIN_AGE, le=config.MAX_AGE)
    gender: str
    country: str = Field(..., min_length=2, max_length=56)
    signup_date: datetime
    income: float = Field(..., ge=0)

    @field_validator("gender")
    @classmethod
    def gender_must_be_valid(cls, v: str) -> str:
        if v not in config.VALID_GENDERS:
            raise ValueError(
                f"Genre '{v}' inconnu. Attendu : {sorted(config.VALID_GENDERS)}"
            )
        return v


# ============================================================
# 2. SCHÉMA TRANSACTION
# ============================================================
class TransactionRow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: str = Field(..., min_length=4, max_length=64)
    customer_id: str = Field(..., min_length=4, max_length=32)
    amount: float = Field(..., gt=config.MIN_AMOUNT, le=config.MAX_AMOUNT)
    currency: str = Field(..., pattern=r"^[A-Z]{3}$")
    transaction_type: str
    timestamp: datetime

    @field_validator("currency")
    @classmethod
    def currency_must_be_supported(cls, v: str) -> str:
        if v not in config.VALID_CURRENCIES:
            raise ValueError(
                f"Devise '{v}' non supportée. "
                f"Attendu : {sorted(config.VALID_CURRENCIES)}"
            )
        return v

    @field_validator("transaction_type")
    @classmethod
    def type_must_be_supported(cls, v: str) -> str:
        if v not in config.VALID_TRANSACTION_TYPES:
            raise ValueError(
                f"Type '{v}' inconnu. "
                f"Attendu : {sorted(config.VALID_TRANSACTION_TYPES)}"
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def timestamp_not_in_future(cls, v: datetime) -> datetime:
        if v > datetime.utcnow():
            raise ValueError("timestamp ne peut pas être dans le futur")
        return v


# ============================================================
# 3. ERREUR PERSONNALISÉE
# ============================================================
class DataValidationError(Exception):
    """Erreur levée quand un dataset est invalide (contient le rapport complet)."""
    def __init__(self, dataset_name: str, report: dict):
        self.dataset_name = dataset_name
        self.report = report
        super().__init__(f"[{dataset_name}] {report['summary']}")


# ============================================================
# 4. FONCTION GÉNÉRIQUE DE VALIDATION
# ============================================================
def _validate_dataframe(
    df: pd.DataFrame,
    schema: type[BaseModel],
    required_columns: set[str],
    dataset_name: str,
    raise_on_error: bool = True,
) -> dict:
    """
    Valide un DataFrame entier contre un schéma Pydantic.

    Détecte :
    - dataset vide
    - colonnes manquantes
    - colonnes inattendues
    - valeurs manquantes
    - lignes invalides (types, catégories, règles métier)
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

    # ---------- CHECK 1 : dataset vide ----------
    if df is None or df.empty:
        report["is_valid"] = False
        report["issues"]["empty_dataset"].append("Le dataset est vide ou None.")
        report["summary"] = "❌ Dataset vide."
        if raise_on_error:
            raise DataValidationError(dataset_name, report)
        return report

    report["stats"]["total_rows"] = len(df)

    # ---------- CHECK 2 : colonnes manquantes ----------
    actual_cols = set(df.columns)
    missing_cols = required_columns - actual_cols
    if missing_cols:
        report["is_valid"] = False
        report["issues"]["missing_columns"] = sorted(missing_cols)

    # ---------- CHECK 3 : colonnes inattendues ----------
    unexpected_cols = actual_cols - required_columns
    if unexpected_cols:
        report["is_valid"] = False
        report["issues"]["unexpected_columns"] = sorted(unexpected_cols)

    # Si problèmes de colonnes → on ne peut pas valider les lignes
    if missing_cols or unexpected_cols:
        report["summary"] = (
            f"❌ Problème de colonnes : "
            f"{len(missing_cols)} manquante(s), "
            f"{len(unexpected_cols)} inattendue(s)."
        )
        if raise_on_error:
            raise DataValidationError(dataset_name, report)
        return report

    # ---------- CHECK 4 : valeurs manquantes ----------
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

    # ---------- CHECK 5 : validation ligne par ligne ----------
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

    # ---------- Résumé ----------
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
# 5. WRAPPERS SPÉCIFIQUES
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