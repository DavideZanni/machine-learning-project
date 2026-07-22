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


class WeatherInteractionFeatures(BaseEstimator, TransformerMixin):
    """Feature derivate meteo: differenza temp/percepita, indice di disagio,
    interazione giorno lavorativo x condizione meteo.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "WeatherInteractionFeatures":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        temp = X["temp"].astype(float)
        atemp = X["atemp"].astype(float)
        hum = X["hum"].astype(float)
        workingday = X["workingday"].astype(float)
        weathersit = X["weathersit"].astype(float)

        out = pd.DataFrame(index=X.index)
        out["temp_atemp_diff"] = temp - atemp
        # Indice di disagio semplificato: temperatura percepita pesata dall'umidità.
        # Il dataset usa scale normalizzate (0-1), non gradi reali: è un indice
        # adimensionale a fini comparativi, non un vero Humidex fisico.
        out["discomfort_index"] = atemp * (1 + hum)
        out["workingday_x_weathersit"] = workingday * weathersit
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(["temp_atemp_diff", "discomfort_index", "workingday_x_weathersit"])
