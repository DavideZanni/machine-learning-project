"""CLI di training: split cronologico, tuning Optuna, stacking, salvataggio artefatti."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from bike_sharing.config import AppConfig, Settings
from bike_sharing.data.loader import load_dataset
from bike_sharing.data.preprocessing import build_preprocessing_pipeline, chronological_split
from bike_sharing.models.evaluate import compute_metrics, save_artifact, save_metrics

logger = logging.getLogger(__name__)

BASELINE_MODELS: dict[str, Any] = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
}


def wrap_with_log_target(preprocessor: ColumnTransformer, estimator: Any) -> TransformedTargetRegressor:
    """Incapsula preprocessing + modello in un TransformedTargetRegressor con
    target log1p/expm1: un solo oggetto sklearn, un solo artefatto joblib.

    Nota: expm1(x) >= 0 solo per x >= 0. Uno stimatore lineare non vincolato
    (LinearRegression, Ridge) può produrre output log-space negativi sotto
    extrapolazione (input anomali, fuori dal range di training), risultando
    in una predizione finale leggermente negativa (asintoticamente -> -1).
    Questo non è un bug: è una proprietà nota di TransformedTargetRegressor
    con stimatori lineari. Un eventuale layer di serving (API) può applicare
    un clip difensivo np.maximum(pred, 0) se necessario.
    """
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def _build_lightgbm(params: dict, seed: int) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params, random_state=seed, verbosity=-1)


def _lightgbm_param_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _build_xgboost(params: dict, seed: int) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**params, random_state=seed, verbosity=0)


def _xgboost_param_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def _build_catboost(params: dict, seed: int) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**params, random_seed=seed, verbose=False, allow_writing_files=False)


def _catboost_param_space(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 800),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }


PARAM_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "lightgbm": _lightgbm_param_space,
    "xgboost": _xgboost_param_space,
    "catboost": _catboost_param_space,
}

BUILD_ESTIMATOR: dict[str, Callable[[dict, int], Any]] = {
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "catboost": _build_catboost,
}


def _make_objective(
    model_name: str,
    preprocessor_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    seed: int,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = PARAM_SPACES[model_name](trial)
        estimator = BUILD_ESTIMATOR[model_name](params, seed)
        ttr = wrap_with_log_target(preprocessor_factory(), estimator)
        scores = cross_val_score(ttr, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return float(-scores.mean())

    return objective


def tune_boosting_model(
    model_name: str,
    preprocessor_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    n_trials: int,
    seed: int,
) -> tuple[TransformedTargetRegressor, dict, float]:
    """Tuning bayesiano (Optuna/TPE) di un modello di boosting su TimeSeriesSplit.

    Ritorna un TransformedTargetRegressor NON fittato con i migliori iperparametri
    trovati: va fittato sul train/val completo prima della valutazione finale.
    """
    if model_name not in BUILD_ESTIMATOR:
        raise ValueError(f"model_name non registrato: {model_name!r}; disponibili: {list(BUILD_ESTIMATOR)}")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = _make_objective(model_name, preprocessor_factory, X, y, cv, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_estimator_raw = BUILD_ESTIMATOR[model_name](study.best_params, seed)
    best_ttr = wrap_with_log_target(preprocessor_factory(), best_estimator_raw)
    return best_ttr, study.best_params, study.best_value


def tune_random_forest(
    preprocessor_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    n_iter: int,
    seed: int,
) -> tuple[TransformedTargetRegressor, dict, float]:
    """RandomizedSearchCV per RandomForest (più economico di uno study Optuna
    dedicato: il guadagno atteso da una ricerca bayesiana su RF è minore che sui
    modelli di boosting)."""
    ttr = wrap_with_log_target(preprocessor_factory(), RandomForestRegressor(random_state=seed))

    param_distributions = {
        "regressor__model__n_estimators": randint(100, 600),
        "regressor__model__max_depth": randint(3, 25),
        "regressor__model__min_samples_split": randint(2, 20),
        "regressor__model__min_samples_leaf": randint(1, 10),
        "regressor__model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        ttr,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=seed,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(-search.best_score_)


def build_stacking_regressor(tuned_models: dict[str, Any], cv: TimeSeriesSplit) -> StackingRegressor:
    """Stacking: base = modelli già tunati su TimeSeriesSplit (Task 9-12, mai
    shuffle), meta-learner = RidgeCV sulle predizioni out-of-fold.

    Nota: internamente usa KFold(shuffle=False, n_splits=cv.n_splits) invece del
    `cv` (TimeSeriesSplit) ricevuto in input. `StackingRegressor.fit` usa
    `cross_val_predict`, che richiede che i fold di test partizionino l'intero
    dataset senza buchi; `TimeSeriesSplit` esclude per costruzione il segmento
    iniziale da ogni fold di test (nessun train precede il primo campione) e fa
    fallire `cross_val_predict` con `ValueError: cross_val_predict only works
    for partitions`. `KFold(shuffle=False)` copre l'intero dataset e mantiene
    l'ordine originale delle righe (nessuno shuffle), ma alcuni fold di train
    contengono righe temporalmente successive al proprio fold di test: introduce
    un leakage limitato e circoscritto alla sola costruzione delle feature del
    meta-learner. Non riguarda il tuning dei modelli base (Task 9-12, che restano
    su TimeSeriesSplit) né la fase di serving finale."""
    estimators = list(tuned_models.items())
    stacking_cv = KFold(n_splits=cv.n_splits, shuffle=False)
    return StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=stacking_cv)


def run_training(config: AppConfig, granularity: str, use_lag_features: bool = False) -> dict[str, dict[str, float]]:
    """Esegue l'intero training: split cronologico, tuning Optuna, stacking,
    selezione finale, salvataggio artefatto e metriche. Ritorna le metriche di
    tutti i modelli candidati (per il confronto nel README)."""
    data_file = config.data.hour_file if granularity == "hour" else config.data.day_file
    csv_path = Path(config.data.raw_dir) / data_file
    df = load_dataset(csv_path, granularity=granularity)

    train_val_df, test_df = chronological_split(df, config.split.train_val_fraction)
    drop_cols = ["cnt", "casual", "registered", "instant", "dteday"]
    feature_columns = [c for c in train_val_df.columns if c not in drop_cols]

    X_train_val, y_train_val = train_val_df[feature_columns], train_val_df[config.target.column]
    X_test, y_test = test_df[feature_columns], test_df[config.target.column]

    cyclical_periods = config.features.cyclical.model_dump()

    def preprocessor_factory():
        return build_preprocessing_pipeline(granularity, cyclical_periods)

    tscv = TimeSeriesSplit(n_splits=config.split.n_cv_splits)
    seed = config.project.random_seed

    tuned_models: dict[str, Any] = {}
    for model_name in config.optuna.models:
        estimator, best_params, cv_rmse = tune_boosting_model(
            model_name, preprocessor_factory, X_train_val, y_train_val, tscv, config.optuna.n_trials, seed
        )
        logger.info("Optuna %s: RMSE CV=%.3f, params=%s", model_name, cv_rmse, best_params)
        tuned_models[model_name] = estimator

    rf_estimator, rf_params, rf_cv_rmse = tune_random_forest(
        preprocessor_factory, X_train_val, y_train_val, tscv, config.random_forest.n_iter, seed
    )
    logger.info("RandomForest: RMSE CV=%.3f, params=%s", rf_cv_rmse, rf_params)
    tuned_models["random_forest"] = rf_estimator

    stacking = build_stacking_regressor(tuned_models, tscv)

    candidates: dict[str, Any] = dict(tuned_models)
    candidates["stacking"] = stacking
    for name, estimator in candidates.items():
        estimator.fit(X_train_val, y_train_val)

    results = {name: compute_metrics(y_test.to_numpy(), estimator.predict(X_test)) for name, estimator in candidates.items()}
    best_name = min(results, key=lambda name: results[name]["rmse"])
    logger.info("Modello vincente per granularità=%s: %s (RMSE test=%.3f)", granularity, best_name, results[best_name]["rmse"])

    models_dir = Path(config.serving.model_dir)
    save_artifact(candidates[best_name], feature_columns, best_name, models_dir, granularity)
    save_metrics(results, models_dir, granularity)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Training pipeline Bike Sharing ML")
    parser.add_argument("--granularity", choices=["day", "hour"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings() if args.config is None else Settings(config_path=args.config)
    config = settings.load()
    granularity = args.granularity or config.granularity

    run_training(config, granularity)


if __name__ == "__main__":
    main()
