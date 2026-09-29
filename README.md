# FinTrust Digital Bank — ML Risk Intelligence Workflow

Automated risk evaluation engine and reproducible ML workflow for transaction monitoring.

## Project Structure
```text
fintrust-ml-ops/
|-- .github/
|   |__workflow
|-- data/
|   |--proceed/
|   |--raw/
|       |--customer.csv
|       |--data_dictionary.csv
|       |__transaction.csv
| 
|--models/
|
|--notebooks/
|   |__dataexplore.py
|      
├── src/
│   ├── data_validation.py   # Pydantic data schemas & validation rules
│   ├── preprocessing.py     # Scikit-learn feature transformers
│   ├── pipeline.py          # End-to-end execution workflow
│   └── api/
│       └── main.py          # FastAPI REST service
├── tests/
|   |--__init__.py
|   |--test_api.py
|   |--test_data_validation.py
│   └── test_workflow.py     # Pytest test suite
├── requirements.txt         # Dependencies
└── README.md

##1. Run Technical Tests
###Execute the test suite to verify pipeline functionality:

pytest tests/

##2. Launch Local API Service

uvicorn src.api.main:app --reload
