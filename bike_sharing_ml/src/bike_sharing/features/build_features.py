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


class LagRollingFeatures(BaseEstimator, TransformerMixin):
    """Lag e rolling statistics su 'cnt' — SOLO uso sperimentale/offline.

    Non utilizzabile nella pipeline servita via API: richiede la colonna 'cnt'
    (lo storico reale dei noleggi), che non è disponibile per uno scenario
    ipotetico "what-if" a singola richiesta. Vedi README, sezione "Esperimento
    lag features", per il confronto RMSE offline che giustifica l'esclusione dal
    modello di produzione. Ogni lag/rolling è calcolato con shift >= 1 per non
    includere mai il valore corrente nel proprio calcolo.
    """

    def __init__(self, lag_periods: list[int], rolling_windows: list[int]):
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows

    def fit(self, X: pd.DataFrame, y=None) -> "LagRollingFeatures":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cnt = X["cnt"].astype(float)
        out = pd.DataFrame(index=X.index)
        for lag in self.lag_periods:
            out[f"cnt_lag_{lag}"] = cnt.shift(lag)
        for window in self.rolling_windows:
            shifted = cnt.shift(1)
            out[f"cnt_rolling_mean_{window}"] = shifted.rolling(window).mean()
            out[f"cnt_rolling_std_{window}"] = shifted.rolling(window).std()
        return out.fillna(0.0)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = [f"cnt_lag_{lag}" for lag in self.lag_periods]
        for window in self.rolling_windows:
            names += [f"cnt_rolling_mean_{window}", f"cnt_rolling_std_{window}"]
        return np.array(names)
