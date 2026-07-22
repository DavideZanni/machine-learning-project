"""Caricamento del dataset Bike Sharing (day.csv o hour.csv)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CATEGORICAL_COLUMNS = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
HOURLY_ONLY_CATEGORICAL = ["hr"]


def load_dataset(csv_path: Path, granularity: str) -> pd.DataFrame:
    """Carica il csv, effettua il parsing della data e tipizza le colonne categoriche.

    Args:
        csv_path: percorso a day.csv o hour.csv.
        granularity: "day" o "hour" — determina se la colonna 'hr' è attesa e come
            ordinare cronologicamente il DataFrame.

    Returns:
        DataFrame ordinato cronologicamente (per dteday, e per hr se granularity="hour").
    """
    if granularity not in ("day", "hour"):
        raise ValueError(f"granularity deve essere 'day' o 'hour', ricevuto: {granularity!r}")

    df = pd.read_csv(csv_path)
    df["dteday"] = pd.to_datetime(df["dteday"])

    categorical_cols = list(CATEGORICAL_COLUMNS)
    if granularity == "hour":
        if "hr" not in df.columns:
            raise ValueError(
                f"granularity='hour' richiede la colonna 'hr', assente nel file: {csv_path}"
            )
        categorical_cols += HOURLY_ONLY_CATEGORICAL
        sort_cols = ["dteday", "hr"]
    else:
        sort_cols = ["dteday"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
