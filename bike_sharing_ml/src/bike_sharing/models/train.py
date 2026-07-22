"""CLI di training: split cronologico, tuning Optuna, stacking, salvataggio artefatti."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

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


from typing import Callable

import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


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
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = _make_objective(model_name, preprocessor_factory, X, y, cv, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_estimator_raw = BUILD_ESTIMATOR[model_name](study.best_params, seed)
    best_ttr = wrap_with_log_target(preprocessor_factory(), best_estimator_raw)
    return best_ttr, study.best_params, study.best_value


from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


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
