"""Train the risk-review model with ``python -m src.train``."""

import logging
from pathlib import Path

from src import config
from src.models.model import save_model, train_model
from src.preprocessing import load_transactions

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main(model_path: str | Path = config.MODEL_FILE) -> None:
    """Validate source transactions, train, and persist model plus metadata."""
    transactions = load_transactions()
    logger.info("Validated %s transactions", len(transactions))
    pipeline, metrics = train_model(transactions, config.TRAINING_TARGET)
    save_model(pipeline, metrics, model_path)
    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    main()
