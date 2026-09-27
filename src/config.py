from pathlib import path       ##for the path         

ROOT_DIR = path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"    



## source files 

CUSTOMER_FILE = DATA_RAW / "customer.csv"
TRANSACTION_FILE = DATA_RAW / "transaction.csv"
DATA_DICTIONARY_FILE = DATA_RAW / "data_dictionary.csv"



## expected columns in the data files
CUSTOMER_COLUMNS = {
    "customer_ID",
    "customer_Name",
    "Age",
    "Gender",
    "city",
    "Customer_Segment",
    "Account_Type",
    "Tenure_Months",
    "Digital_Engagement_Score",
    "Montly_Income_Band",
    "Preferred_Channel",
    "Account_Status"
}

TRANSACTION_COLUMNS ={
    "Transaction_ID",
    "customer_ID",
    "Transaction_DateTime",
    "Transaction_Type",
    "Amount_NGN",
    "Channel",
    "Device_Type",
    "Location",
    "International_Transaction",
    "Transaction_Status",
    "Risk_Review_Flag"
}

VALID_CURRENCIES = {"EUR", "USD", "GBP", "XAF", "XOF"}      # ← adapter
VALID_TRANSACTION_TYPES = {"transfer", "payment", "withdrawal", "deposit"}  # ← adapter
VALID_GENDERS = {"M", "F", "Other"}                          # ← adapter

# --- Règles métier ---
MIN_AMOUNT = 0.01
MAX_AMOUNT = 1_000_000.0
MIN_AGE = 18
MAX_AGE = 120

# --- Politique de gestion des valeurs manquantes ---
# "strict" = rejet, "warn" = log mais continue, "drop" = supprime les lignes
MISSING_VALUES_POLICY = "strict"