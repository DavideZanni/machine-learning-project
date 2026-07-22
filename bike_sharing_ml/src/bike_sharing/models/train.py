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
