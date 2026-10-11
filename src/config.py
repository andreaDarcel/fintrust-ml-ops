"""Project paths and validation rules."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODEL_FILE = MODELS_DIR / "fintrust_model.joblib"
MODEL_CONFIG_FILE = MODELS_DIR / "config.json"
MODEL_VERSION = "0.1.0"
TRAINING_TARGET = "risk_review_flag"



# Source files

CUSTOMER_FILE = DATA_RAW / "customer.csv"
TRANSACTION_FILE = DATA_RAW / "transaction.csv"
DATA_DICTIONARY_FILE = DATA_RAW / "data_dictionary.csv"


## Reproducubility
RANDOM_SEED = 42
TEST_SIZE = 0.2

## expected columns in the data files
CUSTOMER_COLUMNS = {
    "customer_id", "customer_name", "age", "gender", "city",
    "customer_segment", "account_type", "tenure_months",
    "digital_engagement_score", "monthly_income_band",
    "preferred_channel", "account_status",
}

TRANSACTION_COLUMNS ={
    "transaction_id", "customer_id", "transaction_date",
    "transaction_type", "amount_ngn", "channel", "device_type",
    "location", "international_transaction", "transaction_status",
    "risk_review_flag",
}

# --- customer table rules ---

VALID_PREFERRED_CHANNELS = {"Mobile App", "USSD", "Web"}
VALID_GENDERS = {"Male", "Female", "Prefer not to say", "Other"}
VALID_CUSTOMER_SEGMENTS = {"Premium", "Everyday", "Student", "SME"}
VALID_ACCOUNT_TYPES = {"Savings", "Premium", "Current"}
VALID_ACCOUNT_STATUS = {"Active", "Dormant", "Restricted"}

CUSTOMER_ID_PATTERN = r"^FT-C0\d{4}$"
MIN_TENURE_MONTHS = 1
MAX_TENURE_MONTHS = 99
DIGITAL_ENGAGEMENT = (10, 100)
MONTHLY_INCOME_BANDS = {
    "Below 100k": (0, 100000),
    "100k-249k": (100000, 249000),
    "250k-499k": (250000, 499000),
    "500k-999k": (500000, 999000),
    "1m+": (1000000, float("inf")),
}
MIN_AGE = 18
MAX_AGE = 99


# --- transaction table rules ---

VALID_TRANSACTION_TYPES = {
    "Card Purchase", "Airtime/Data", "Transfer", "Bill Payment",
    "Cash Withdrawal", "Deposit",
}
VALID_CHANNEL = {"ATM", "POS", "Mobile App", "USSD", "Web"}
VALID_DEVICE_TYPE = {"POS Terminal", "iOS", "Android", "ATM Terminal", "Web Browser"}
VALID_INTERNATIONAL_TRANSACTION = {"Yes", "No"}
VALID_TRANSACTION_STATUS = {"Failed", "Pending", "Reversed", "Successful"}
VALID_RISK_REVIEW_FLAG = {"Yes", "No"}

TRANSACTION_ID_PATTERN = r"^FT-T\d{6}$"
TRANSACTION_DATE_FORMAT = "%Y/%m/%d %H:%M"  ##Example : 2024/01/01 12:00

# --- gestion of missing values ---
MISSING_VALUES_POLICY = "strict"
LOG_BUT_CONTINUE = "warn"
DROP_ROWS = "drop"
