"""Split cronologico e ColumnTransformer di preprocessing (nessun leakage)."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from bike_sharing.features.build_features import CyclicalEncoder, WeatherInteractionFeatures

NUMERIC_PASSTHROUGH = ["temp", "atemp", "hum", "windspeed"]
BINARY_PASSTHROUGH = ["yr", "holiday", "workingday"]
LOW_CARDINALITY_CATEGORICAL = ["season", "weathersit"]


def chronological_split(
    df: pd.DataFrame, train_val_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split cronologico rigoroso: primo `train_val_fraction` -> train/val, resto -> test.

    Il DataFrame deve già essere ordinato cronologicamente (vedi loader.load_dataset).
    Non usa mai shuffle: la porzione di test è sempre temporalmente successiva al train.
    """
    if not 0.0 < train_val_fraction < 1.0:
        raise ValueError("train_val_fraction deve essere in (0, 1)")

    split_idx = int(len(df) * train_val_fraction)
    train_val = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train_val, test


def build_preprocessing_pipeline(
    granularity: str, cyclical_periods: dict[str, int]
) -> ColumnTransformer:
    """Costruisce il ColumnTransformer di preprocessing, non fittato.

    Va sempre fittato esclusivamente dentro ogni fold di CV o sul solo train set
    finale — mai su tutto il dataset prima dello split (era il bug originale).

    `remainder="drop"` esclude implicitamente `instant`, `dteday`, `casual`,
    `registered`: le ultime due sono componenti dirette di `cnt` e includerle
    come feature costituirebbe target leakage.
    """
    cyclical_columns = {"mnth": cyclical_periods["mnth"], "weekday": cyclical_periods["weekday"]}
    if granularity == "hour":
        cyclical_columns["hr"] = cyclical_periods["hr"]

    transformers = [
        ("numeric", "passthrough", NUMERIC_PASSTHROUGH),
        ("binary", "passthrough", BINARY_PASSTHROUGH),
        ("cyclical", CyclicalEncoder(periods=cyclical_columns), list(cyclical_columns.keys())),
        (
            "weather_interactions",
            WeatherInteractionFeatures(),
            ["temp", "atemp", "hum", "workingday", "weathersit"],
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            LOW_CARDINALITY_CATEGORICAL,
        ),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")
