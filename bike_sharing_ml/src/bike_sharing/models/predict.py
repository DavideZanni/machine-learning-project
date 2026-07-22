"""Caricamento del modello di produzione e funzione di predizione per il serving."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_production_artifact(models_dir: Path, granularity: str) -> tuple[Any, dict]:
    """Carica il modello di produzione (.joblib) e i suoi metadata per una granularità."""
    target_dir = Path(models_dir) / granularity
    model = joblib.load(target_dir / "production.joblib")
    metadata = json.loads((target_dir / "production_metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def predict_from_features(model: Any, feature_columns: list[str], features: dict) -> float:
    """Costruisce una riga di input coerente con le colonne attese dal preprocessing
    e restituisce la predizione di `cnt` (garantita >= 0 grazie a expm1 in fase di
    training; un clamp difensivo copre eventuali arrotondamenti in virgola mobile).
    """
    row = pd.DataFrame([{col: features[col] for col in feature_columns}])
    prediction = model.predict(row)[0]
    return max(float(prediction), 0.0)
