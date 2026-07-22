"""Metriche di valutazione e persistenza degli artefatti di modello."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """MAPE con guardia epsilon per y_true vicini a 0 (es. ore notturne, cnt basso)."""
    denominator = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcola RMSE, MAE, MAPE, R2, RMSLE in scala originale (non su log)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred_clipped = np.clip(np.asarray(y_pred, dtype=float), a_min=0.0, a_max=None)

    mse = mean_squared_error(y_true, y_pred_clipped)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred_clipped)),
        "mape": _safe_mape(y_true, y_pred_clipped),
        "r2": float(r2_score(y_true, y_pred_clipped)),
        "rmsle": float(np.sqrt(mean_squared_log_error(y_true, y_pred_clipped))),
    }


def save_artifact(
    model: Any,
    feature_columns: list[str],
    model_name: str,
    models_dir: Path,
    granularity: str,
) -> None:
    """Salva l'artefatto di produzione: pipeline+modello (.joblib) e metadata (.json)."""
    target_dir = Path(models_dir) / granularity
    target_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, target_dir / "production.joblib")
    metadata = {"model_name": model_name, "granularity": granularity, "feature_columns": feature_columns}
    (target_dir / "production_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_metrics(results: dict[str, dict[str, float]], models_dir: Path, granularity: str) -> None:
    """Salva le metriche di tutti i modelli candidati per una granularità."""
    target_dir = Path(models_dir) / granularity
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
