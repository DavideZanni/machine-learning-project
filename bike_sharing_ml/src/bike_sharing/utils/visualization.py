"""Utility di plotting: residui e feature importance (Gini/gain o coefficienti stacking)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Scatter Predetto-vs-Reale e Residui-vs-Predetto, salvati come PNG."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.4)
    upper_limit = max(np.max(y_true), np.max(y_pred))
    axes[0].plot([0, upper_limit], [0, upper_limit], "r--")
    axes[0].set_xlabel("Valore reale")
    axes[0].set_ylabel("Predizione")
    axes[0].set_title("Predetto vs Reale")

    axes[1].scatter(y_pred, residuals, alpha=0.4)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Predizione")
    axes[1].set_ylabel("Residuo")
    axes[1].set_title("Residui")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(model: Any, out_path: Path) -> None:
    """Feature importance del modello vincente.

    Se il modello è uno StackingRegressor: importanza = |coefficiente| del
    meta-learner (RidgeCV) per ciascun modello base. Altrimenti: importanza
    Gini/gain nativa (`feature_importances_`) dei modelli ad albero — scelta al
    posto di SHAP per semplicità e robustezza multi-libreria (LightGBM/XGBoost/
    CatBoost/RandomForest hanno tutte `feature_importances_`, ma API SHAP diverse).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    inner = getattr(model, "regressor_", getattr(model, "regressor", model))
    fitted_model = inner.named_steps["model"] if hasattr(inner, "named_steps") else inner

    if hasattr(fitted_model, "final_estimator_"):
        names = [name for name, _ in fitted_model.estimators]
        coefs = np.abs(fitted_model.final_estimator_.coef_)
        ax.barh(names, coefs)
        ax.set_xlabel("Peso assoluto nel meta-learner (RidgeCV)")
        ax.set_title("Contributo dei modelli base allo Stacking")
    elif hasattr(fitted_model, "feature_importances_"):
        preprocessor = inner.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()
        importances = fitted_model.feature_importances_
        order = np.argsort(importances)[-20:]
        ax.barh(np.array(feature_names)[order], importances[order])
        ax.set_xlabel("Importanza (Gini/gain)")
        ax.set_title("Feature Importance")
    else:
        ax.text(0.5, 0.5, "Feature importance non disponibile per questo modello", ha="center")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
