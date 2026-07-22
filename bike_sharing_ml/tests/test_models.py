"""Test per metriche, salvataggio artefatti e predizioni."""
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from bike_sharing.models.evaluate import compute_metrics, save_artifact, save_metrics


def test_compute_metrics_returns_expected_keys_and_nonnegative_values():
    y_true = np.array([10.0, 20.0, 0.0, 5.0])
    y_pred = np.array([12.0, 18.0, 1.0, 4.0])

    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"rmse", "mae", "mape", "r2", "rmsle"}
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["mape"] >= 0.0


def test_compute_metrics_mape_handles_near_zero_actuals_without_exploding():
    y_true = np.array([0.0, 0.0, 100.0])
    y_pred = np.array([1.0, 2.0, 90.0])

    metrics = compute_metrics(y_true, y_pred)

    assert np.isfinite(metrics["mape"])


def test_save_artifact_and_metrics_round_trip(tmp_path):
    model = LinearRegression().fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))

    save_artifact(model, feature_columns=["x"], model_name="linear", models_dir=tmp_path, granularity="hour")
    save_metrics({"linear": {"rmse": 1.0}}, models_dir=tmp_path, granularity="hour")

    loaded_model = joblib.load(tmp_path / "hour" / "production.joblib")
    metadata = json.loads((tmp_path / "hour" / "production_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "hour" / "metrics.json").read_text(encoding="utf-8"))

    assert loaded_model.predict([[4.0]])[0] == pytest.approx(4.0, rel=1e-6)
    assert metadata == {"model_name": "linear", "granularity": "hour", "feature_columns": ["x"]}
    assert metrics == {"linear": {"rmse": 1.0}}
