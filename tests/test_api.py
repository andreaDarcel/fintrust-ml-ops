from fastapi.testclient import TestClient
from src import __version__
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    playload = r.json()
    assert r.status_code == 200
    assert playload["status"] == "ok"
    assert playload["model_loaded"] == True
    assert playload["version"] == __version__