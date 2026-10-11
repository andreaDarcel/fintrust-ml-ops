# src/predict.py
"""Command-line prediction script: ``python -m src.predict``."""
import json
import sys

import pandas as pd

from src import config
from src.pipeline import FinTrustPipeline


def main(path: str | None = None) -> None:
    path = path or str(config.TRANSACTION_FILE)
    df = pd.read_csv(path).head(10)

    pipeline = FinTrustPipeline()
    result = pipeline.predict_batch(df)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
