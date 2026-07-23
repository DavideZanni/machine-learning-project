"""Esperimento offline: confronto RMSE con/senza lag e rolling features su 'cnt'.

Le lag/rolling features (vedi bike_sharing.features.build_features.LagRollingFeatures)
NON sono usate nella pipeline di produzione (l'API /predict non ha accesso allo
storico reale per uno scenario ipotetico). Questo script quantifica il costo di
questa scelta architetturale sul dataset hour.csv, a scopo di trasparenza nel README.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import FeatureUnion

from bike_sharing.config import AppConfig, Settings
from bike_sharing.data.loader import load_dataset
from bike_sharing.data.preprocessing import build_preprocessing_pipeline, chronological_split
from bike_sharing.features.build_features import LagRollingFeatures
from bike_sharing.models.evaluate import compute_metrics
from bike_sharing.models.train import BUILD_ESTIMATOR, wrap_with_log_target

logger = logging.getLogger(__name__)


def _build_pipeline_with_lag(granularity: str, cyclical_periods: dict, lag_periods: list[int], rolling_windows: list[int]) -> ColumnTransformer:
    base = build_preprocessing_pipeline(granularity, cyclical_periods)
    lag = ColumnTransformer(
        transformers=[("lag_rolling", LagRollingFeatures(lag_periods, rolling_windows), ["cnt"])],
        remainder="drop",
    )
    return FeatureUnion([("base", base), ("lag", lag)])


def run_lag_feature_experiment(config: AppConfig) -> dict:
    """Confronta, sulla granularità hour, l'RMSE test del miglior modello (LightGBM,
    per costo computazionale contenuto) con e senza lag/rolling features."""
    csv_path = Path(config.data.raw_dir) / config.data.hour_file
    df = load_dataset(csv_path, granularity="hour")
    train_val_df, test_df = chronological_split(df, config.split.train_val_fraction)

    drop_cols = ["cnt", "casual", "registered", "instant", "dteday"]
    feature_columns_no_lag = [c for c in train_val_df.columns if c not in drop_cols]
    feature_columns_with_lag = feature_columns_no_lag + ["cnt"]

    cyclical_periods = config.features.cyclical.model_dump()
    seed = config.project.random_seed
    tscv = TimeSeriesSplit(n_splits=config.split.n_cv_splits)

    y_train_val, y_test = train_val_df[config.target.column], test_df[config.target.column]

    estimator_no_lag = wrap_with_log_target(
        build_preprocessing_pipeline("hour", cyclical_periods),
        BUILD_ESTIMATOR["lightgbm"]({}, seed),
    )
    estimator_no_lag.fit(train_val_df[feature_columns_no_lag], y_train_val)
    metrics_no_lag = compute_metrics(y_test.to_numpy(), estimator_no_lag.predict(test_df[feature_columns_no_lag]))

    estimator_with_lag = wrap_with_log_target(
        _build_pipeline_with_lag("hour", cyclical_periods, config.features.lag_periods, config.features.rolling_windows),
        BUILD_ESTIMATOR["lightgbm"]({}, seed),
    )
    estimator_with_lag.fit(train_val_df[feature_columns_with_lag], y_train_val)
    metrics_with_lag = compute_metrics(y_test.to_numpy(), estimator_with_lag.predict(test_df[feature_columns_with_lag]))

    result = {"without_lag_features": metrics_no_lag, "with_lag_features": metrics_with_lag}
    out_path = Path(config.serving.model_dir) / "hour" / "lag_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Esperimento lag features: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_lag_feature_experiment(Settings().load())
