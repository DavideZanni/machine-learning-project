"""Test per le utility di plotting."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from bike_sharing.data.preprocessing import build_preprocessing_pipeline
from bike_sharing.models.train import wrap_with_log_target
from bike_sharing.utils.visualization import plot_feature_importance, plot_residuals


def test_plot_residuals_creates_file(tmp_path):
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    out_path = tmp_path / "residuals.png"

    plot_residuals(y_true, y_pred, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_feature_importance_creates_file_for_tree_model(tmp_path):
    X = pd.DataFrame({
        "temp": [0.3, 0.5, 0.7, 0.4], "atemp": [0.3, 0.5, 0.6, 0.4],
        "hum": [0.5, 0.4, 0.6, 0.5], "windspeed": [0.1, 0.2, 0.15, 0.1],
        "yr": [0, 0, 1, 1], "holiday": [0, 0, 0, 1], "workingday": [1, 1, 0, 0],
        "mnth": [1, 2, 3, 4], "weekday": [0, 1, 2, 3], "season": [1, 1, 2, 2],
        "weathersit": [1, 2, 1, 3],
    })
    y = pd.Series([50.0, 80.0, 120.0, 30.0])
    preprocessor = build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})
    model = wrap_with_log_target(preprocessor, RandomForestRegressor(n_estimators=10, random_state=42))
    model.fit(X, y)

    out_path = tmp_path / "importance.png"
    plot_feature_importance(model, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
