from fastapi.testclient import Testclient
from src.api.main import app

client = Testclient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] == True
    assert r.json()["version"] == __version__