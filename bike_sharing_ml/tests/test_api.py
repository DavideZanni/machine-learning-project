"""Test dell'endpoint FastAPI /predict, con un artefatto fittizio (no training reale)."""
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from bike_sharing.apps import api as api_module
from bike_sharing.data.preprocessing import build_preprocessing_pipeline
from bike_sharing.models.train import wrap_with_log_target

FEATURE_COLUMNS = [
    "temp", "atemp", "hum", "windspeed", "yr", "holiday", "workingday",
    "mnth", "weekday", "season", "weathersit", "hr",
]

VALID_PAYLOAD = {
    "season": 1, "yr": 0, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 2,
    "workingday": 1, "weathersit": 1, "temp": 0.5, "atemp": 0.5, "hum": 0.5, "windspeed": 0.2,
}


def _train_fixture_model():
    X = pd.DataFrame({
        "temp": [0.3, 0.5, 0.7, 0.4], "atemp": [0.3, 0.5, 0.6, 0.4],
        "hum": [0.5, 0.4, 0.6, 0.5], "windspeed": [0.1, 0.2, 0.15, 0.1],
        "yr": [0, 0, 1, 1], "holiday": [0, 0, 0, 1], "workingday": [1, 1, 0, 0],
        "mnth": [1, 2, 3, 4], "weekday": [0, 1, 2, 3], "season": [1, 1, 2, 2],
        "weathersit": [1, 2, 1, 3], "hr": [8, 12, 18, 22],
    })
    y = pd.Series([50.0, 80.0, 120.0, 30.0])
    preprocessor = build_preprocessing_pipeline(granularity="hour", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})
    model = wrap_with_log_target(preprocessor, LinearRegression())
    model.fit(X, y)
    return model


def test_health_endpoint_reports_model_loaded():
    with TestClient(api_module.app) as client:
        # Impostato DOPO l'ingresso nel context manager: il lifespan gira
        # all'__enter__ e, non trovando un artefatto reale, azzera lo stato —
        # sovrascriverlo prima verrebbe perso.
        api_module._state["model"] = _train_fixture_model()
        api_module._state["metadata"] = {"model_name": "linear", "feature_columns": FEATURE_COLUMNS}

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_returns_nonnegative_prediction():
    with TestClient(api_module.app) as client:
        api_module._state["model"] = _train_fixture_model()
        api_module._state["metadata"] = {"model_name": "linear", "feature_columns": FEATURE_COLUMNS}

        response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_cnt"] >= 0.0
    assert body["model_name"] == "linear"


def test_predict_endpoint_rejects_out_of_range_input():
    with TestClient(api_module.app) as client:
        api_module._state["model"] = _train_fixture_model()
        api_module._state["metadata"] = {"model_name": "linear", "feature_columns": FEATURE_COLUMNS}

        invalid_payload = {**VALID_PAYLOAD, "season": 9}
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
