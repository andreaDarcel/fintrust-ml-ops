from pathlib import Path       ##for the path         

ROOT_DIR = Path(__file__).resolve().parent.parent
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
    "Monthly_Income_Band",
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

# --- customer table rules ---

VALID_PREFERRED_CHANNELS = {"Mobile App", "USSD", "Web"}  
VALID_GENDERS = {"Male", "Female", "Prefer not to say", "Other"} 
VALID_CUSTOMER_SEGMENTS ={"Premium", "Everyday", "student", "SME"}   
VALID_ACCOUNT_TYPES = {"Savings", "Premium", "Current"}              
VALID_ACCOUNT_STATUS = {"Active", "Dormant"}              

CUSTOMER_ID_PATTERN = r"^FT-C0\d{4}"    ##Example : FT-C01234
MIN_TENURE_MONTHS = 1
MAX_TENURE_MONTHS = 99
DIGITAL_ENGAGEMENT = (10,100)  ##the digital engagement score must be between 10 and 100
MONTHLY_INCOME_BANDS = {
    "Below 100000": (0, 100000),
    "100k - 249k": (100000, 249000),
    "250k - 499k": (250000, 499000),
    "500k - 999k": (500000, 999000),
    "1m+": (1000000, float("inf"))
}
MIN_AGE = 18
MAX_AGE = 99


# --- transaction table rules ---

VALID_TRANSACTION_TYPES = {"Card Purchase","Airtime/Data","Transfer", "Bill Payment", "Cash Withdrawal", "Deposit"}
VALID_CHANNEL = {"ATM ", "POS", "Mobile App", "USSD", "Web"}
VALID_DEVICE_TYPE ={"POS Terminal", "iOS", "Android","ATM Terminal", "Web Browser"}
VALID_INTERNATIONAL_TRANSACTION = {"Yes", "No"}
VALID_TRANSACTION_STATUS = {"Successful", "Reversed"}
VALID_RISK_REVIEW_FLAG = {"Yes", "No"}

TRANSACTION_ID_PATTERN = r"^FT-T00\d{4}"  ##Example : FT-T01234
TRANSACTION_DATE_FORMAT = "%Y/%m/%d %H:%M"  ##Example : 2024/01/01 12:00

# --- gestion of missing values ---
MISSING_VALUES_POLICY = "strict"
LOG_BUT_CONTINUE = "warn"
DROP_ROWS = "drop"