# scripts/audit_data.py
import pandas as pd
from src.config import CUSTOMER_FILE, TRANSACTION_FILE


def audit(path, name):
    """Print a quick schema and quality summary for a CSV dataset."""
    print(f"\n{'='*60}\n📊 AUDIT : {name}\n{'='*60}")
    df = pd.read_csv(path)

    print(f"Lignes : {len(df)} | Colonnes : {len(df.columns)}")
    print(f"\n📋 Colonnes :\n{list(df.columns)}")

    print(f"\n🔤 Types :\n{df.dtypes}")

    print(f"\n❓ Valeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    print("\n🎯 Valeurs uniques (colonnes object) :")
    for col in df.select_dtypes(include="object").columns:
        uniques = df[col].unique()[:10]
        print(f"  {col}: {list(uniques)} ({df[col].nunique()} uniques)")

    print(f"\n📈 Stats numériques :\n{df.describe()}")


if __name__ == "__main__":
    audit(CUSTOMER_FILE, "CUSTOMERS")
    audit(TRANSACTION_FILE, "TRANSACTIONS")