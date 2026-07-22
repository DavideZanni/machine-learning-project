"""Trasformatori scikit-learn custom per feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Codifica seno/coseno per variabili cicliche (mese, giorno settimana, ora).

    Stateless: `periods` è un iperparametro passato al costruttore, `fit` non impara
    nulla dal training set (nessun rischio di leakage) — ma resta un Transformer
    sklearn valido, incluso in Pipeline/ColumnTransformer e serializzabile con joblib.
    """

    def __init__(self, periods: dict[str, int]):
        self.periods = periods

    def fit(self, X: pd.DataFrame, y=None) -> "CyclicalEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for column, period in self.periods.items():
            values = X[column].astype(float)
            angle = 2 * np.pi * values / period
            out[f"{column}_sin"] = np.sin(angle)
            out[f"{column}_cos"] = np.cos(angle)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = []
        for column in self.periods:
            names += [f"{column}_sin", f"{column}_cos"]
        return np.array(names)
