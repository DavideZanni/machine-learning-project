"""Test per metriche, salvataggio artefatti e predizioni."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from bike_sharing.data.preprocessing import build_preprocessing_pipeline
from bike_sharing.models.evaluate import compute_metrics, save_artifact, save_metrics
from bike_sharing.models.train import wrap_with_log_target


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


def test_wrapped_baseline_predicts_nonnegative_and_finite_values():
    X = pd.DataFrame({
        "temp": [0.3, 0.5, 0.7, 0.4, 0.6, 0.2],
        "atemp": [0.3, 0.5, 0.6, 0.4, 0.6, 0.25],
        "hum": [0.5, 0.4, 0.6, 0.5, 0.3, 0.7],
        "windspeed": [0.1, 0.2, 0.15, 0.1, 0.2, 0.3],
        "yr": [0, 0, 1, 1, 0, 1],
        "holiday": [0, 0, 0, 1, 0, 0],
        "workingday": [1, 1, 0, 0, 1, 1],
        "mnth": [1, 2, 3, 4, 5, 6],
        "weekday": [0, 1, 2, 3, 4, 5],
        "season": [1, 1, 2, 2, 2, 3],
        "weathersit": [1, 2, 1, 3, 1, 2],
    })
    y = pd.Series([50.0, 80.0, 120.0, 30.0, 200.0, 60.0])

    preprocessor = build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})

    model = wrap_with_log_target(preprocessor, LinearRegression())
    model.fit(X, y)
    predictions = model.predict(X)

    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0.0)


from sklearn.model_selection import TimeSeriesSplit

from bike_sharing.data.preprocessing import build_preprocessing_pipeline
from bike_sharing.models.train import tune_boosting_model


def _tiny_synthetic_dataset(n_rows: int = 60):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "temp": rng.uniform(0, 1, n_rows),
        "atemp": rng.uniform(0, 1, n_rows),
        "hum": rng.uniform(0, 1, n_rows),
        "windspeed": rng.uniform(0, 1, n_rows),
        "yr": rng.integers(0, 2, n_rows),
        "holiday": rng.integers(0, 2, n_rows),
        "workingday": rng.integers(0, 2, n_rows),
        "mnth": rng.integers(1, 13, n_rows),
        "weekday": rng.integers(0, 7, n_rows),
        "season": rng.integers(1, 5, n_rows),
        "weathersit": rng.integers(1, 4, n_rows),
    })
    y = pd.Series(rng.uniform(10, 300, n_rows))
    return X, y


def test_tune_boosting_model_lightgbm_returns_fitted_capable_estimator():
    X, y = _tiny_synthetic_dataset()
    cv = TimeSeriesSplit(n_splits=2)

    def preprocessor_factory():
        return build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})

    estimator, best_params, best_rmse = tune_boosting_model(
        "lightgbm", preprocessor_factory, X, y, cv, n_trials=2, seed=42
    )
    estimator.fit(X, y)
    predictions = estimator.predict(X)

    assert isinstance(best_params, dict)
    assert np.isfinite(best_rmse)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0.0)
