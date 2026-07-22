"""CLI di training: split cronologico, tuning Optuna, stacking, salvataggio artefatti."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

BASELINE_MODELS: dict[str, Any] = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
}


def wrap_with_log_target(preprocessor, estimator: Any) -> TransformedTargetRegressor:
    """Incapsula preprocessing + modello in un TransformedTargetRegressor con
    target log1p/expm1: un solo oggetto sklearn, un solo artefatto joblib,
    predizioni sempre >= 0 grazie a expm1.
    """
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
