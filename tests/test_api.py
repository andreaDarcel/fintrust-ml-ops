from fastapi.testclient import TestClient

from src import __version__, config
from src.api import main as api


TRANSACTION = {
    "transaction_id": "FT-T012345",
    "customer_id": "FT-C00001",
    "amount_ngn": 250.0,
    "transaction_type": "Transfer",
    "channel": "Mobile App",
    "transaction_date": "2026-01-01T10:00:00Z",
    "international_transaction": "No",
    "transaction_status": "Successful",
}


def test_health_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "missing.joblib")
    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "model_loaded": False,
        "version": __version__,
    }


def test_prediction_without_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "missing.joblib")
    with TestClient(api.app) as client:
        response = client.post("/predict", json=TRANSACTION)

    assert response.status_code == 503


def test_prediction_with_loaded_model(monkeypatch):
    class StubPipeline:
        def predict_one(self, transaction):
            return {
                "transaction_id": transaction["transaction_id"],
                "prediction": 1,
                "probability": 0.8,
                "requires_review": True,
                "model_version": __version__,
            }

    monkeypatch.setattr(api, "FinTrustPipeline", StubPipeline)
    with TestClient(api.app) as client:
        response = client.post("/predict", json=TRANSACTION)

    assert response.status_code == 200
    assert response.json()["requires_review"] is True
