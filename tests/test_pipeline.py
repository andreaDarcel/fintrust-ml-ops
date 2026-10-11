import pandas as pd
import pytest

from src import config
from src.models.model import load_model, save_model, train_model
from src.pipeline import FinTrustPipeline
from src.preprocessing import build_features, load_transactions


def make_training_data(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": [f"FT-T{index:06d}" for index in range(rows)],
        "customer_id": ["FT-C00001"] * rows,
        "transaction_date": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
        "transaction_type": ["Transfer" if index % 2 else "Card Purchase" for index in range(rows)],
        "amount_ngn": [200.0 + index for index in range(rows)],
        "channel": ["Mobile App"] * rows,
        "device_type": [None if index % 10 == 0 else "Android" for index in range(rows)],
        "location": ["Lagos"] * rows,
        "international_transaction": ["No"] * rows,
        "transaction_status": ["Successful"] * rows,
        "risk_review_flag": ["Yes" if index % 4 == 0 else "No" for index in range(rows)],
    })


def make_transaction(index: int = 12345) -> dict:
    return {
        "transaction_id": f"FT-T{index:06d}",
        "customer_id": "FT-C00001",
        "amount_ngn": 500.0,
        "transaction_type": "Transfer",
        "channel": "Mobile App",
        "transaction_date": "2026-01-15T10:30:00Z",
        "international_transaction": "No",
        "transaction_status": "Successful",
    }


@pytest.fixture(scope="module")
def trained_pipeline() -> FinTrustPipeline:
    model, _ = train_model(make_training_data())
    pipeline = FinTrustPipeline.__new__(FinTrustPipeline)
    pipeline.model = model
    return pipeline


def test_source_transactions_validate():
    assert len(load_transactions()) == 12_000


def test_predict_one_returns_review_signal(trained_pipeline):
    result = trained_pipeline.predict_one(make_transaction())

    assert result["prediction"] in (0, 1)
    assert result["requires_review"] == (result["prediction"] == 1)
    assert 0 <= result["probability"] <= 1
    assert result["model_version"] == config.MODEL_VERSION


def test_predict_batch_accepts_unlabeled_rows(trained_pipeline):
    transactions = pd.DataFrame([make_transaction(), make_transaction(12346)])
    result = trained_pipeline.predict_batch(transactions)

    assert result["n_input"] == 2
    assert result["n_valid"] == 2
    assert result["n_rejected"] == 0
    assert len(result["predictions"]) == 2


def test_predict_batch_reports_rejected_rows(trained_pipeline):
    valid = make_transaction()
    invalid = make_transaction(12346)
    invalid["channel"] = "Unknown"
    result = trained_pipeline.predict_batch(pd.DataFrame([valid, invalid]))

    assert result["n_valid"] == 1
    assert result["n_rejected"] == 1


def test_predict_batch_handles_empty_input(trained_pipeline):
    result = trained_pipeline.predict_batch(pd.DataFrame())

    assert result["n_input"] == 0
    assert result["predictions"] == []


def test_training_requires_a_real_target():
    transactions = make_training_data().drop(columns=["risk_review_flag"])

    with pytest.raises(ValueError, match="target"):
        train_model(transactions)


def test_model_and_metadata_roundtrip(trained_pipeline, tmp_path):
    model_path = tmp_path / "model.joblib"
    save_model(trained_pipeline.model, {"n_train": 80, "n_test": 20, "f1": 0.5}, model_path)
    loaded_model = load_model(model_path)
    features = build_features(make_training_data().drop(columns=["risk_review_flag"]))
    metadata = (tmp_path / "config.json").read_text(encoding="utf-8")

    assert loaded_model.predict_proba(features.head(2)).shape == (2, 2)
    assert '"target": "risk_review_flag"' in metadata
