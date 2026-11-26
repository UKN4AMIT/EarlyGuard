from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_status():
    response = client.get("/api/model/status")
    assert response.status_code == 200
    body = response.json()
    assert body["loaded"] is True
    assert body["model_features"] == 34
    assert body["preprocessor_output_features"] == 34

def test_prediction_endpoint():
    patient = {
        "age": 48,
        "bp": 80,
        "sg": 1.020,
        "al": 0,
        "su": 0,
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "bgr": 100,
        "bu": 20,
        "sc": 1.0,
        "sod": 140,
        "pot": 4.5,
        "hemo": 15,
        "pcv": 44,
        "wc": 7800,
        "rc": 5.2,
        "htn": "no",
        "dm": "no",
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no",
    }
    response = client.post("/api/predict/kidney", json=patient)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in ("CKD", "Not CKD")
    assert 0 <= body["ckd_probability"] <= 1
    assert 0 <= body["not_ckd_probability"] <= 1
