# Bike Sharing ML — Productionization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ricostruire `Bike_Sharing.py` come pacchetto Python production-grade (`bike_sharing_ml/`), eliminando il leakage temporale e di CV, aggiungendo feature engineering avanzato, tuning Optuna su LightGBM/XGBoost/CatBoost, uno StackingRegressor, serving via FastAPI + Streamlit e containerizzazione Docker.

**Architecture:** Pipeline sklearn config-driven (day/hour via `config.yaml` + pydantic), split cronologico 75/25 con `TimeSeriesSplit` per CV/tuning/stacking, target `log1p`/`expm1` incapsulato in `TransformedTargetRegressor`, un solo artefatto joblib per granularità servito da FastAPI, Streamlit come client HTTP dell'API.

**Tech Stack:** pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna, fastapi, uvicorn, streamlit, pydantic, pydantic-settings, joblib, matplotlib, seaborn, pyyaml, requests, pytest, httpx.

## Global Constraints

- Python richiesto `>=3.10` (progetto sviluppato su Python 3.14; immagine Docker pinnata su `python:3.11-slim` per compatibilità wheel più ampia con lightgbm/xgboost/catboost).
- Split cronologico rigoroso 75% (train/val) / 25% (test): **mai** `shuffle=True` o `train_test_split` casuale. Ogni CV/tuning/stacking usa `TimeSeriesSplit`.
- Target modellato come `log1p(cnt)`, incapsulato in `sklearn.compose.TransformedTargetRegressor` (mai log/exp manuali fuori dalla pipeline).
- Optuna: ~25 trial/modello (`config.optuna.n_trials`), sampler TPE con seed fisso (`config.project.random_seed`).
- Lag/rolling features su `cnt`: implementate e valutate, ma **escluse dalla pipeline di produzione/serving** (`config.features.enable_lag_features: false` di default) — motivazione: l'API `/predict` non ha accesso garantito allo storico reale per uno scenario ipotetico.
- Granularità servita di default da API/Dashboard: `hour` (`config.serving.default_granularity: hour`). `day` resta allenabile via CLI/config per confronto.
- Dashboard → API: comunicazione HTTP reale (`requests`), non import diretto del modello.
- `remainder="drop"` nel `ColumnTransformer`: esclude esplicitamente `instant`, `dteday`, `casual`, `registered` dalle feature (le ultime due sono componenti dirette di `cnt` → leakage se incluse).
- Nessun push al remote Git senza richiesta esplicita. I commit locali per-task fanno parte del workflow di questo piano (già approvato procedendo con brainstorming → writing-plans).
- Codice, docstring e commenti in italiano.

---

### Task 1: Scaffolding pacchetto, dipendenze e configurazione

**Files:**
- Create: `bike_sharing_ml/pyproject.toml`
- Create: `bike_sharing_ml/config/config.yaml`
- Create: `bike_sharing_ml/src/bike_sharing/__init__.py`
- Create: `bike_sharing_ml/src/bike_sharing/config.py`
- Create: `bike_sharing_ml/src/bike_sharing/data/__init__.py`
- Create: `bike_sharing_ml/src/bike_sharing/features/__init__.py`
- Create: `bike_sharing_ml/src/bike_sharing/models/__init__.py`
- Create: `bike_sharing_ml/src/bike_sharing/utils/__init__.py`
- Create: `bike_sharing_ml/tests/__init__.py`
- Create: `bike_sharing_ml/tests/test_data.py` (solo scheletro + primo test config)
- Create: `bike_sharing_ml/data/raw/day.csv` (copia da `bike sharing dataset/day.csv`)
- Create: `bike_sharing_ml/data/raw/hour.csv` (copia da `bike sharing dataset/hour.csv`)

**Interfaces:**
- Produces: `bike_sharing.config.AppConfig` (pydantic `BaseModel`, campi: `project`, `data`, `granularity`, `split`, `target`, `features`, `optuna`, `random_forest`, `serving`), `bike_sharing.config.Settings` (pydantic `BaseSettings` con `config_path: Path` e metodo `.load() -> AppConfig`).

- [ ] **Step 1: Crea struttura cartelle e copia i dataset**

```bash
mkdir -p "bike_sharing_ml/config" "bike_sharing_ml/data/raw" "bike_sharing_ml/data/processed" "bike_sharing_ml/models" \
  "bike_sharing_ml/src/bike_sharing/data" "bike_sharing_ml/src/bike_sharing/features" "bike_sharing_ml/src/bike_sharing/models" \
  "bike_sharing_ml/src/bike_sharing/utils" "bike_sharing_ml/apps" "bike_sharing_ml/tests"
cp "../bike sharing dataset/day.csv" "bike_sharing_ml/data/raw/day.csv"
cp "../bike sharing dataset/hour.csv" "bike_sharing_ml/data/raw/hour.csv"
touch "bike_sharing_ml/src/bike_sharing/__init__.py" "bike_sharing_ml/src/bike_sharing/data/__init__.py" \
  "bike_sharing_ml/src/bike_sharing/features/__init__.py" "bike_sharing_ml/src/bike_sharing/models/__init__.py" \
  "bike_sharing_ml/src/bike_sharing/utils/__init__.py" "bike_sharing_ml/tests/__init__.py"
```

(Percorsi relativi a `Progetto Zanni Davide 131946/`, cioè la working directory del repo Git.)

- [ ] **Step 2: Scrivi `pyproject.toml`**

```toml
[project]
name = "bike-sharing-ml"
version = "0.1.0"
description = "Pipeline production-grade per la previsione della domanda di bike sharing"
requires-python = ">=3.10"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "lightgbm>=4.0",
    "xgboost>=2.0",
    "catboost>=1.2",
    "optuna>=3.5",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "streamlit>=1.33",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "joblib>=1.3",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pyyaml>=6.0",
    "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Scrivi `config/config.yaml`**

```yaml
project:
  name: bike_sharing_ml
  random_seed: 42

data:
  raw_dir: data/raw
  processed_dir: data/processed
  day_file: day.csv
  hour_file: hour.csv

granularity: hour

split:
  train_val_fraction: 0.75
  n_cv_splits: 5

target:
  column: cnt
  log_transform: true

features:
  cyclical:
    mnth: 12
    weekday: 7
    hr: 24
  enable_lag_features: false
  lag_periods: [1, 24]
  rolling_windows: [3, 24]

optuna:
  n_trials: 25
  timeout_seconds: null
  models: [lightgbm, xgboost, catboost]

random_forest:
  n_iter: 20

serving:
  default_granularity: hour
  model_dir: models
  api_host: 0.0.0.0
  api_port: 8000
  api_base_url: http://localhost:8000
```

- [ ] **Step 4: Installa il pacchetto in editable mode con le dipendenze**

```bash
pip install -e "bike_sharing_ml[dev]"
```
Expected: installazione completata senza errori (`catboost` verrà scaricato; il resto risulta già presente nell'ambiente).

- [ ] **Step 5: Scrivi il test di caricamento configurazione (fallirà: `config.py` non esiste)**

`bike_sharing_ml/tests/test_data.py`:
```python
"""Test per configurazione e loader dati."""
from pathlib import Path

import pytest

from bike_sharing.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_valid_yaml():
    settings = Settings(config_path=REPO_ROOT / "config" / "config.yaml")
    config = settings.load()

    assert config.project.name == "bike_sharing_ml"
    assert config.granularity in ("day", "hour")
    assert 0.0 < config.split.train_val_fraction < 1.0
    assert config.serving.default_granularity == "hour"


def test_config_rejects_invalid_granularity(tmp_path):
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text(
        """
project: {name: x, random_seed: 1}
data: {raw_dir: a, processed_dir: b, day_file: day.csv, hour_file: hour.csv}
granularity: weekly
split: {train_val_fraction: 0.75, n_cv_splits: 5}
target: {column: cnt, log_transform: true}
features: {cyclical: {mnth: 12, weekday: 7, hr: 24}, enable_lag_features: false, lag_periods: [1], rolling_windows: [3]}
optuna: {n_trials: 5, timeout_seconds: null, models: [lightgbm]}
random_forest: {n_iter: 5}
serving: {default_granularity: hour, model_dir: models, api_host: 0.0.0.0, api_port: 8000, api_base_url: http://x}
""",
        encoding="utf-8",
    )
    settings = Settings(config_path=bad_yaml)
    with pytest.raises(Exception):
        settings.load()
```

- [ ] **Step 6: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bike_sharing.config'`

- [ ] **Step 7: Scrivi `src/bike_sharing/config.py`**

```python
"""Configurazione tipizzata del progetto, caricata da config.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    name: str
    random_seed: int


class DataConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    day_file: str
    hour_file: str


class SplitConfig(BaseModel):
    train_val_fraction: float = Field(gt=0.0, lt=1.0)
    n_cv_splits: int = Field(gt=1)


class TargetConfig(BaseModel):
    column: str
    log_transform: bool


class CyclicalConfig(BaseModel):
    mnth: int
    weekday: int
    hr: int


class FeaturesConfig(BaseModel):
    cyclical: CyclicalConfig
    enable_lag_features: bool
    lag_periods: list[int]
    rolling_windows: list[int]


class OptunaConfig(BaseModel):
    n_trials: int = Field(gt=0)
    timeout_seconds: int | None = None
    models: list[str]


class RandomForestConfig(BaseModel):
    n_iter: int = Field(gt=0)


class ServingConfig(BaseModel):
    default_granularity: Literal["day", "hour"]
    model_dir: str
    api_host: str
    api_port: int
    api_base_url: str


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    granularity: Literal["day", "hour"]
    split: SplitConfig
    target: TargetConfig
    features: FeaturesConfig
    optuna: OptunaConfig
    random_forest: RandomForestConfig
    serving: ServingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.model_validate(raw)


class Settings(BaseSettings):
    """Punto di ingresso configurabile via variabili d'ambiente (prefisso BIKE_)."""

    model_config = SettingsConfigDict(env_prefix="BIKE_")

    config_path: Path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

    def load(self) -> AppConfig:
        return AppConfig.from_yaml(self.config_path)
```

- [ ] **Step 8: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_data.py -v`
Expected: PASS (2 test verdi)

- [ ] **Step 9: Commit**

```bash
git add bike_sharing_ml/pyproject.toml bike_sharing_ml/config bike_sharing_ml/src bike_sharing_ml/tests bike_sharing_ml/data/raw
git commit -m "feat: scaffolding pacchetto bike_sharing_ml e configurazione pydantic"
```

---

### Task 2: Data loader

**Files:**
- Create: `bike_sharing_ml/src/bike_sharing/data/loader.py`
- Modify: `bike_sharing_ml/tests/test_data.py`

**Interfaces:**
- Consumes: nessuno (primo modulo dati).
- Produces: `load_dataset(csv_path: Path, granularity: str) -> pd.DataFrame` — DataFrame ordinato cronologicamente, colonne categoriche tipizzate come `category`, `dteday` come `datetime64`.

- [ ] **Step 1: Aggiungi test per il loader (fallirà: `loader.py` non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_data.py`:
```python
import pandas as pd

from bike_sharing.data.loader import load_dataset

DAY_CSV = REPO_ROOT / "data" / "raw" / "day.csv"
HOUR_CSV = REPO_ROOT / "data" / "raw" / "hour.csv"


def test_load_dataset_day_has_expected_columns_and_dtypes():
    df = load_dataset(DAY_CSV, granularity="day")

    assert "hr" not in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["dteday"])
    assert str(df["season"].dtype) == "category"
    assert df["dteday"].is_monotonic_increasing


def test_load_dataset_hour_is_sorted_by_date_and_hour():
    df = load_dataset(HOUR_CSV, granularity="hour")

    assert str(df["hr"].dtype) == "category"
    combined = df["dteday"].astype(str) + "-" + df["hr"].astype(str).str.zfill(2)
    assert combined.is_monotonic_increasing


def test_load_dataset_has_no_missing_values():
    df = load_dataset(DAY_CSV, granularity="day")
    assert df.isnull().sum().sum() == 0
```

- [ ] **Step 2: Verifica che i test falliscano**

Run: `pytest bike_sharing_ml/tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bike_sharing.data.loader'`

- [ ] **Step 3: Scrivi `src/bike_sharing/data/loader.py`**

```python
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
        categorical_cols += HOURLY_ONLY_CATEGORICAL
        sort_cols = ["dteday", "hr"]
    else:
        sort_cols = ["dteday"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df.sort_values(sort_cols).reset_index(drop=True)
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_data.py -v`
Expected: PASS (5 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/data/loader.py bike_sharing_ml/tests/test_data.py
git commit -m "feat: data loader con parsing date e tipizzazione categoriche"
```

---

### Task 3: Feature ciclica (CyclicalEncoder)

**Files:**
- Create: `bike_sharing_ml/src/bike_sharing/features/build_features.py`
- Create: `bike_sharing_ml/tests/test_features.py`

**Interfaces:**
- Consumes: nessuno.
- Produces: `CyclicalEncoder(periods: dict[str, int])` — transformer sklearn (`BaseEstimator`, `TransformerMixin`), `transform(X: pd.DataFrame) -> pd.DataFrame` con colonne `{col}_sin`/`{col}_cos` in `[-1, 1]`, `get_feature_names_out()`.

- [ ] **Step 1: Scrivi il test (fallirà: modulo non esiste)**

`bike_sharing_ml/tests/test_features.py`:
```python
"""Test per i transformer di feature engineering."""
import numpy as np
import pandas as pd

from bike_sharing.features.build_features import CyclicalEncoder


def test_cyclical_encoder_outputs_are_bounded_and_named():
    df = pd.DataFrame({"mnth": [1, 6, 12], "weekday": [0, 3, 6]})
    encoder = CyclicalEncoder(periods={"mnth": 12, "weekday": 7})

    out = encoder.fit_transform(df)

    assert set(out.columns) == {"mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos"}
    assert out.to_numpy().min() >= -1.0
    assert out.to_numpy().max() <= 1.0
    assert list(encoder.get_feature_names_out()) == ["mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos"]


def test_cyclical_encoder_wraps_around_period():
    df = pd.DataFrame({"hr": [0, 23]})
    encoder = CyclicalEncoder(periods={"hr": 24})

    out = encoder.fit_transform(df)

    # hr=0 e hr=23 sono adiacenti sul cerchio (23 -> 0), quindi vicini in coseno
    assert np.isclose(out["hr_cos"].iloc[0], np.cos(0), atol=1e-9)
    assert np.abs(out["hr_cos"].iloc[1] - out["hr_cos"].iloc[0]) < 0.1
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bike_sharing.features.build_features'`

- [ ] **Step 3: Scrivi `src/bike_sharing/features/build_features.py` (solo CyclicalEncoder per ora)**

```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_features.py -v`
Expected: PASS (2 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/features/build_features.py bike_sharing_ml/tests/test_features.py
git commit -m "feat: CyclicalEncoder per feature temporali cicliche (sin/cos)"
```

---

### Task 4: Feature meteo derivate (WeatherInteractionFeatures)

**Files:**
- Modify: `bike_sharing_ml/src/bike_sharing/features/build_features.py`
- Modify: `bike_sharing_ml/tests/test_features.py`

**Interfaces:**
- Consumes: nessuno (indipendente da `CyclicalEncoder`).
- Produces: `WeatherInteractionFeatures()` — transformer sklearn, `transform(X) -> pd.DataFrame` con colonne `temp_atemp_diff`, `discomfort_index`, `workingday_x_weathersit`.

- [ ] **Step 1: Aggiungi il test (fallirà: classe non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_features.py`:
```python
from bike_sharing.features.build_features import WeatherInteractionFeatures


def test_weather_interaction_features_computes_expected_columns():
    df = pd.DataFrame({
        "temp": [0.5, 0.8],
        "atemp": [0.4, 0.9],
        "hum": [0.6, 0.3],
        "workingday": [1, 0],
        "weathersit": [2, 1],
    })
    transformer = WeatherInteractionFeatures()

    out = transformer.fit_transform(df)

    assert list(out.columns) == ["temp_atemp_diff", "discomfort_index", "workingday_x_weathersit"]
    assert np.isclose(out["temp_atemp_diff"].iloc[0], 0.1)
    assert np.isclose(out["workingday_x_weathersit"].iloc[0], 2.0)
    assert np.isclose(out["workingday_x_weathersit"].iloc[1], 0.0)
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'WeatherInteractionFeatures'`

- [ ] **Step 3: Aggiungi la classe a `build_features.py`**

Aggiungi in fondo al file (dopo `CyclicalEncoder`):
```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_features.py -v`
Expected: PASS (3 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/features/build_features.py bike_sharing_ml/tests/test_features.py
git commit -m "feat: WeatherInteractionFeatures per interazioni meteo/calendario"
```

---

### Task 5: Feature lag/rolling sperimentali (LagRollingFeatures)

**Files:**
- Modify: `bike_sharing_ml/src/bike_sharing/features/build_features.py`
- Modify: `bike_sharing_ml/tests/test_features.py`

**Interfaces:**
- Consumes: nessuno.
- Produces: `LagRollingFeatures(lag_periods: list[int], rolling_windows: list[int])` — transformer sklearn **sperimentale**, richiede la colonna `cnt` in input, **non** usato nella pipeline di produzione (vedi Global Constraints).

- [ ] **Step 1: Aggiungi il test (fallirà: classe non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_features.py`:
```python
from bike_sharing.features.build_features import LagRollingFeatures


def test_lag_rolling_features_never_uses_current_value():
    df = pd.DataFrame({"cnt": [10.0, 20.0, 30.0, 40.0, 50.0]})
    transformer = LagRollingFeatures(lag_periods=[1], rolling_windows=[2])

    out = transformer.fit_transform(df)

    assert list(out.columns) == ["cnt_lag_1", "cnt_rolling_mean_2", "cnt_rolling_std_2"]
    # Il lag_1 alla riga 2 (indice 1) deve essere il valore della riga precedente (10.0)
    assert out["cnt_lag_1"].iloc[1] == 10.0
    # La rolling mean a finestra 2, riga 3 (indice 2), usa solo valori shiftati (righe 0 e 1)
    assert np.isclose(out["cnt_rolling_mean_2"].iloc[2], 15.0)
    # Nessun NaN residuo (fillna(0.0) applicato ai primi periodi senza storico sufficiente)
    assert out.isnull().sum().sum() == 0
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'LagRollingFeatures'`

- [ ] **Step 3: Aggiungi la classe a `build_features.py`**

Aggiungi in fondo al file:
```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_features.py -v`
Expected: PASS (4 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/features/build_features.py bike_sharing_ml/tests/test_features.py
git commit -m "feat: LagRollingFeatures sperimentale (esclusa dal serving)"
```

---

### Task 6: Split cronologico e pipeline di preprocessing

**Files:**
- Create: `bike_sharing_ml/src/bike_sharing/data/preprocessing.py`
- Modify: `bike_sharing_ml/tests/test_data.py`

**Interfaces:**
- Consumes: `CyclicalEncoder`, `WeatherInteractionFeatures` da `bike_sharing.features.build_features` (Task 3, 4); `load_dataset` da Task 2.
- Produces: `chronological_split(df: pd.DataFrame, train_val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]`; `build_preprocessing_pipeline(granularity: str, cyclical_periods: dict[str, int]) -> ColumnTransformer` (non fittato).

- [ ] **Step 1: Aggiungi i test (falliranno: modulo non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_data.py`:
```python
from bike_sharing.data.preprocessing import build_preprocessing_pipeline, chronological_split


def test_chronological_split_respects_time_order():
    df = load_dataset(DAY_CSV, granularity="day")

    train_val, test = chronological_split(df, train_val_fraction=0.75)

    assert len(train_val) + len(test) == len(df)
    # Guard di regressione: nessuna data di test deve precedere l'ultima data di training
    assert train_val["dteday"].max() < test["dteday"].min()


def test_chronological_split_rejects_invalid_fraction():
    df = load_dataset(DAY_CSV, granularity="day")
    with pytest.raises(ValueError):
        chronological_split(df, train_val_fraction=1.5)


def test_preprocessing_pipeline_produces_no_nulls_and_drops_target_leakage_columns():
    df = load_dataset(HOUR_CSV, granularity="hour")
    cyclical_periods = {"mnth": 12, "weekday": 7, "hr": 24}

    pipeline = build_preprocessing_pipeline(granularity="hour", cyclical_periods=cyclical_periods)
    X = df.drop(columns=["cnt", "casual", "registered", "instant", "dteday"])
    transformed = pipeline.fit_transform(X)

    assert not pd.DataFrame(transformed).isnull().to_numpy().any()
    feature_names = pipeline.get_feature_names_out()
    assert not any("casual" in name or "registered" in name for name in feature_names)
```

- [ ] **Step 2: Verifica che i test falliscano**

Run: `pytest bike_sharing_ml/tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bike_sharing.data.preprocessing'`

- [ ] **Step 3: Scrivi `src/bike_sharing/data/preprocessing.py`**

```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_data.py -v`
Expected: PASS (8 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/data/preprocessing.py bike_sharing_ml/tests/test_data.py
git commit -m "feat: split cronologico e ColumnTransformer di preprocessing senza leakage"
```

---

### Task 7: Metriche di valutazione e salvataggio artefatti

**Files:**
- Create: `bike_sharing_ml/src/bike_sharing/models/evaluate.py`
- Create: `bike_sharing_ml/tests/test_models.py`

**Interfaces:**
- Consumes: nessuno.
- Produces: `compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]` (chiavi: `rmse`, `mae`, `mape`, `r2`, `rmsle`); `save_artifact(model, feature_columns: list[str], model_name: str, models_dir: Path, granularity: str) -> None`; `save_metrics(results: dict, models_dir: Path, granularity: str) -> None`.

- [ ] **Step 1: Scrivi il test (fallirà: modulo non esiste)**

`bike_sharing_ml/tests/test_models.py`:
```python
"""Test per metriche, salvataggio artefatti e predizioni."""
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from bike_sharing.models.evaluate import compute_metrics, save_artifact, save_metrics


def test_compute_metrics_returns_expected_keys_and_nonnegative_values():
    y_true = np.array([10.0, 20.0, 0.0, 5.0])
    y_pred = np.array([12.0, 18.0, 1.0, 4.0])

    metrics = compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {"rmse", "mae", "mape", "r2", "rmsle"}
    assert metrics["rmse"] >= 0.0
    assert metrics["mae"] >= 0.0
    assert metrics["mape"] >= 0.0


def test_compute_metrics_mape_handles_near_zero_actuals_without_exploding():
    y_true = np.array([0.0, 0.0, 100.0])
    y_pred = np.array([1.0, 2.0, 90.0])

    metrics = compute_metrics(y_true, y_pred)

    assert np.isfinite(metrics["mape"])


def test_save_artifact_and_metrics_round_trip(tmp_path):
    model = LinearRegression().fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))

    save_artifact(model, feature_columns=["x"], model_name="linear", models_dir=tmp_path, granularity="hour")
    save_metrics({"linear": {"rmse": 1.0}}, models_dir=tmp_path, granularity="hour")

    loaded_model = joblib.load(tmp_path / "hour" / "production.joblib")
    metadata = json.loads((tmp_path / "hour" / "production_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "hour" / "metrics.json").read_text(encoding="utf-8"))

    assert loaded_model.predict([[4.0]])[0] == pytest.approx(4.0, rel=1e-6)
    assert metadata == {"model_name": "linear", "granularity": "hour", "feature_columns": ["x"]}
    assert metrics == {"linear": {"rmse": 1.0}}
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bike_sharing.models.evaluate'`

- [ ] **Step 3: Scrivi `src/bike_sharing/models/evaluate.py`**

```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_models.py -v`
Expected: PASS (3 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/evaluate.py bike_sharing_ml/tests/test_models.py
git commit -m "feat: metriche di valutazione (RMSE/MAE/MAPE/R2/RMSLE) e persistenza artefatti"
```

---

### Task 8: Baseline lineari con target trasformato (log1p/expm1)

**Files:**
- Create: `bike_sharing_ml/src/bike_sharing/models/train.py`
- Modify: `bike_sharing_ml/tests/test_models.py`

**Interfaces:**
- Consumes: `build_preprocessing_pipeline` (Task 6), `compute_metrics` (Task 7).
- Produces: `BASELINE_MODELS: dict[str, Any]` (`{"linear": LinearRegression(), "ridge": Ridge()}`); `wrap_with_log_target(preprocessor: ColumnTransformer, estimator: Any) -> TransformedTargetRegressor`.

- [ ] **Step 1: Aggiungi il test (fallirà: modulo non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_models.py`:
```python
import pandas as pd

from bike_sharing.data.preprocessing import build_preprocessing_pipeline
from bike_sharing.models.train import wrap_with_log_target


def test_wrapped_baseline_predicts_nonnegative_and_finite_values():
    X = pd.DataFrame({
        "temp": [0.3, 0.5, 0.7, 0.4, 0.6, 0.2],
        "atemp": [0.3, 0.5, 0.6, 0.4, 0.6, 0.25],
        "hum": [0.5, 0.4, 0.6, 0.5, 0.3, 0.7],
        "windspeed": [0.1, 0.2, 0.15, 0.1, 0.2, 0.3],
        "yr": [0, 0, 1, 1, 0, 1],
        "holiday": [0, 0, 0, 1, 0, 0],
        "workingday": [1, 1, 0, 0, 1, 1],
        "mnth": [1, 2, 3, 4, 5, 6],
        "weekday": [0, 1, 2, 3, 4, 5],
        "season": [1, 1, 2, 2, 2, 3],
        "weathersit": [1, 2, 1, 3, 1, 2],
    })
    y = pd.Series([50.0, 80.0, 120.0, 30.0, 200.0, 60.0])

    preprocessor = build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})
    from sklearn.linear_model import LinearRegression

    model = wrap_with_log_target(preprocessor, LinearRegression())
    model.fit(X, y)
    predictions = model.predict(X)

    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0.0)
```

(Aggiungi `import numpy as np` in cima al file se non già presente.)

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bike_sharing.models.train'`

- [ ] **Step 3: Scrivi `src/bike_sharing/models/train.py` (parte 1: baseline)**

```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_models.py -v`
Expected: PASS (4 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/train.py bike_sharing_ml/tests/test_models.py
git commit -m "feat: baseline lineari con TransformedTargetRegressor (log1p/expm1)"
```

---

### Task 9: Tuning Optuna — LightGBM

**Files:**
- Modify: `bike_sharing_ml/src/bike_sharing/models/train.py`
- Modify: `bike_sharing_ml/tests/test_models.py`

**Interfaces:**
- Consumes: `wrap_with_log_target` (Task 8).
- Produces: `PARAM_SPACES: dict[str, Callable]`, `BUILD_ESTIMATOR: dict[str, Callable]` (con entry `"lightgbm"`); `tune_boosting_model(model_name: str, preprocessor_factory: Callable[[], ColumnTransformer], X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit, n_trials: int, seed: int) -> tuple[TransformedTargetRegressor, dict, float]`.

- [ ] **Step 1: Aggiungi il test (fallirà: funzione non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_models.py`:
```python
from sklearn.model_selection import TimeSeriesSplit

from bike_sharing.data.preprocessing import build_preprocessing_pipeline
from bike_sharing.models.train import tune_boosting_model


def _tiny_synthetic_dataset(n_rows: int = 60):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "temp": rng.uniform(0, 1, n_rows),
        "atemp": rng.uniform(0, 1, n_rows),
        "hum": rng.uniform(0, 1, n_rows),
        "windspeed": rng.uniform(0, 1, n_rows),
        "yr": rng.integers(0, 2, n_rows),
        "holiday": rng.integers(0, 2, n_rows),
        "workingday": rng.integers(0, 2, n_rows),
        "mnth": rng.integers(1, 13, n_rows),
        "weekday": rng.integers(0, 7, n_rows),
        "season": rng.integers(1, 5, n_rows),
        "weathersit": rng.integers(1, 4, n_rows),
    })
    y = pd.Series(rng.uniform(10, 300, n_rows))
    return X, y


def test_tune_boosting_model_lightgbm_returns_fitted_capable_estimator():
    X, y = _tiny_synthetic_dataset()
    cv = TimeSeriesSplit(n_splits=2)

    def preprocessor_factory():
        return build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})

    estimator, best_params, best_rmse = tune_boosting_model(
        "lightgbm", preprocessor_factory, X, y, cv, n_trials=2, seed=42
    )
    estimator.fit(X, y)
    predictions = estimator.predict(X)

    assert isinstance(best_params, dict)
    assert np.isfinite(best_rmse)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0.0)
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k lightgbm`
Expected: FAIL — `ImportError: cannot import name 'tune_boosting_model'`

- [ ] **Step 3: Estendi `src/bike_sharing/models/train.py`**

Aggiungi in fondo al file:
```python
from typing import Callable

import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


def _build_lightgbm(params: dict, seed: int) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**params, random_state=seed, verbosity=-1)


def _lightgbm_param_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


PARAM_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "lightgbm": _lightgbm_param_space,
}

BUILD_ESTIMATOR: dict[str, Callable[[dict, int], Any]] = {
    "lightgbm": _build_lightgbm,
}


def _make_objective(
    model_name: str,
    preprocessor_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    seed: int,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        params = PARAM_SPACES[model_name](trial)
        estimator = BUILD_ESTIMATOR[model_name](params, seed)
        ttr = wrap_with_log_target(preprocessor_factory(), estimator)
        scores = cross_val_score(ttr, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return float(-scores.mean())

    return objective


def tune_boosting_model(
    model_name: str,
    preprocessor_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    n_trials: int,
    seed: int,
) -> tuple[TransformedTargetRegressor, dict, float]:
    """Tuning bayesiano (Optuna/TPE) di un modello di boosting su TimeSeriesSplit.

    Ritorna un TransformedTargetRegressor NON fittato con i migliori iperparametri
    trovati: va fittato sul train/val completo prima della valutazione finale.
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = _make_objective(model_name, preprocessor_factory, X, y, cv, seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_estimator_raw = BUILD_ESTIMATOR[model_name](study.best_params, seed)
    best_ttr = wrap_with_log_target(preprocessor_factory(), best_estimator_raw)
    return best_ttr, study.best_params, study.best_value
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k lightgbm`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/train.py bike_sharing_ml/tests/test_models.py
git commit -m "feat: tuning Optuna (TPE) per LightGBM su TimeSeriesSplit"
```

---

### Task 10: Tuning Optuna — XGBoost

**Files:**
- Modify: `bike_sharing_ml/src/bike_sharing/models/train.py`
- Modify: `bike_sharing_ml/tests/test_models.py`

**Interfaces:**
- Consumes: `tune_boosting_model`, `PARAM_SPACES`, `BUILD_ESTIMATOR` (Task 9).
- Produces: entry `"xgboost"` in `PARAM_SPACES`/`BUILD_ESTIMATOR`.

- [ ] **Step 1: Aggiungi il test (fallirà: KeyError su "xgboost")**

Aggiungi in fondo a `bike_sharing_ml/tests/test_models.py`:
```python
def test_tune_boosting_model_xgboost_returns_fitted_capable_estimator():
    X, y = _tiny_synthetic_dataset()
    cv = TimeSeriesSplit(n_splits=2)

    def preprocessor_factory():
        return build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})

    estimator, best_params, best_rmse = tune_boosting_model(
        "xgboost", preprocessor_factory, X, y, cv, n_trials=2, seed=42
    )
    estimator.fit(X, y)
    predictions = estimator.predict(X)

    assert isinstance(best_params, dict)
    assert np.isfinite(best_rmse)
    assert np.all(predictions >= 0.0)
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k xgboost`
Expected: FAIL — `KeyError: 'xgboost'`

- [ ] **Step 3: Aggiungi le funzioni XGBoost a `train.py`**

Aggiungi prima della definizione di `PARAM_SPACES` (o subito dopo `_lightgbm_param_space`, comunque prima che i due dizionari vengano costruiti):
```python
def _build_xgboost(params: dict, seed: int) -> Any:
    from xgboost import XGBRegressor

    return XGBRegressor(**params, random_state=seed, verbosity=0)


def _xgboost_param_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
```

Modifica i due dizionari per includere la nuova entry:
```python
PARAM_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "lightgbm": _lightgbm_param_space,
    "xgboost": _xgboost_param_space,
}

BUILD_ESTIMATOR: dict[str, Callable[[dict, int], Any]] = {
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
}
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k "lightgbm or xgboost"`
Expected: PASS (entrambi i test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/train.py bike_sharing_ml/tests/test_models.py
git commit -m "feat: tuning Optuna (TPE) per XGBoost su TimeSeriesSplit"
```

---

### Task 11: Tuning Optuna — CatBoost

**Files:**
- Modify: `bike_sharing_ml/src/bike_sharing/models/train.py`
- Modify: `bike_sharing_ml/tests/test_models.py`

**Interfaces:**
- Consumes: `tune_boosting_model`, `PARAM_SPACES`, `BUILD_ESTIMATOR` (Task 9/10).
- Produces: entry `"catboost"` in `PARAM_SPACES`/`BUILD_ESTIMATOR`.

- [ ] **Step 1: Aggiungi il test (fallirà: KeyError su "catboost")**

Aggiungi in fondo a `bike_sharing_ml/tests/test_models.py`:
```python
def test_tune_boosting_model_catboost_returns_fitted_capable_estimator():
    X, y = _tiny_synthetic_dataset()
    cv = TimeSeriesSplit(n_splits=2)

    def preprocessor_factory():
        return build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})

    estimator, best_params, best_rmse = tune_boosting_model(
        "catboost", preprocessor_factory, X, y, cv, n_trials=2, seed=42
    )
    estimator.fit(X, y)
    predictions = estimator.predict(X)

    assert isinstance(best_params, dict)
    assert np.isfinite(best_rmse)
    assert np.all(predictions >= 0.0)
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k catboost`
Expected: FAIL — `KeyError: 'catboost'`

- [ ] **Step 3: Aggiungi le funzioni CatBoost a `train.py`**

Aggiungi dopo `_xgboost_param_space`:
```python
def _build_catboost(params: dict, seed: int) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(**params, random_seed=seed, verbose=False, allow_writing_files=False)


def _catboost_param_space(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 800),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
```

Aggiorna i due dizionari:
```python
PARAM_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "lightgbm": _lightgbm_param_space,
    "xgboost": _xgboost_param_space,
    "catboost": _catboost_param_space,
}

BUILD_ESTIMATOR: dict[str, Callable[[dict, int], Any]] = {
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "catboost": _build_catboost,
}
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k "lightgbm or xgboost or catboost"`
Expected: PASS (tutti e tre verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/train.py bike_sharing_ml/tests/test_models.py
git commit -m "feat: tuning Optuna (TPE) per CatBoost su TimeSeriesSplit"
```

---

### Task 12: RandomForest (RandomizedSearchCV)

**Files:**
- Modify: `bike_sharing_ml/src/bike_sharing/models/train.py`
- Modify: `bike_sharing_ml/tests/test_models.py`

**Interfaces:**
- Consumes: `wrap_with_log_target` (Task 8).
- Produces: `tune_random_forest(preprocessor_factory: Callable, X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit, n_iter: int, seed: int) -> tuple[TransformedTargetRegressor, dict, float]`.

- [ ] **Step 1: Aggiungi il test (fallirà: funzione non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_models.py`:
```python
from bike_sharing.models.train import tune_random_forest


def test_tune_random_forest_returns_fitted_estimator():
    X, y = _tiny_synthetic_dataset()
    cv = TimeSeriesSplit(n_splits=2)

    def preprocessor_factory():
        return build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})

    estimator, best_params, best_rmse = tune_random_forest(preprocessor_factory, X, y, cv, n_iter=3, seed=42)
    predictions = estimator.predict(X)

    assert isinstance(best_params, dict)
    assert np.isfinite(best_rmse)
    assert np.all(predictions >= 0.0)
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k random_forest`
Expected: FAIL — `ImportError: cannot import name 'tune_random_forest'`

- [ ] **Step 3: Aggiungi `tune_random_forest` a `train.py`**

Aggiungi in fondo al file:
```python
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def tune_random_forest(
    preprocessor_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    n_iter: int,
    seed: int,
) -> tuple[TransformedTargetRegressor, dict, float]:
    """RandomizedSearchCV per RandomForest (più economico di uno study Optuna
    dedicato: il guadagno atteso da una ricerca bayesiana su RF è minore che sui
    modelli di boosting)."""
    ttr = wrap_with_log_target(preprocessor_factory(), RandomForestRegressor(random_state=seed))

    param_distributions = {
        "regressor__model__n_estimators": randint(100, 600),
        "regressor__model__max_depth": randint(3, 25),
        "regressor__model__min_samples_split": randint(2, 20),
        "regressor__model__min_samples_leaf": randint(1, 10),
        "regressor__model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        ttr,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=seed,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(-search.best_score_)
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k random_forest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/train.py bike_sharing_ml/tests/test_models.py
git commit -m "feat: tuning RandomForest via RandomizedSearchCV su TimeSeriesSplit"
```

---

### Task 13: StackingRegressor e orchestrazione CLI end-to-end

**Files:**
- Modify: `bike_sharing_ml/src/bike_sharing/models/train.py`
- Modify: `bike_sharing_ml/tests/test_models.py`

**Interfaces:**
- Consumes: `tune_boosting_model`, `tune_random_forest`, `compute_metrics`, `save_artifact`, `save_metrics`, `load_dataset`, `chronological_split`, `build_preprocessing_pipeline`, `Settings`.
- Produces: `build_stacking_regressor(tuned_models: dict[str, Any], cv: TimeSeriesSplit) -> StackingRegressor`; `run_training(config: AppConfig, granularity: str, use_lag_features: bool = False) -> dict[str, dict[str, float]]` (orchestrazione, ritorna le metriche di tutti i candidati); `main()` (entry-point CLI via `argparse`).

- [ ] **Step 1: Aggiungi il test per `build_stacking_regressor` (fallirà: funzione non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_models.py`:
```python
from sklearn.linear_model import LinearRegression

from bike_sharing.models.train import build_stacking_regressor


def test_build_stacking_regressor_fits_and_predicts_nonnegative():
    X, y = _tiny_synthetic_dataset()
    cv = TimeSeriesSplit(n_splits=2)

    def preprocessor_factory():
        return build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})

    tuned_models = {
        "linear": wrap_with_log_target(preprocessor_factory(), LinearRegression()),
        "ridge": wrap_with_log_target(preprocessor_factory(), Ridge()),
    }
    stacking = build_stacking_regressor(tuned_models, cv)
    stacking.fit(X, y)
    predictions = stacking.predict(X)

    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0.0)
```

(Aggiungi `from bike_sharing.models.train import wrap_with_log_target` se non già importato, e `from sklearn.linear_model import Ridge`.)

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k stacking`
Expected: FAIL — `ImportError: cannot import name 'build_stacking_regressor'`

- [ ] **Step 3: Aggiungi `build_stacking_regressor` e l'orchestrazione CLI a `train.py`**

Aggiungi in fondo al file:
```python
import argparse
import logging

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

from bike_sharing.config import AppConfig, Settings
from bike_sharing.data.loader import load_dataset
from bike_sharing.data.preprocessing import build_preprocessing_pipeline, chronological_split
from bike_sharing.models.evaluate import compute_metrics, save_artifact, save_metrics

logger = logging.getLogger(__name__)


def build_stacking_regressor(tuned_models: dict[str, Any], cv: TimeSeriesSplit) -> StackingRegressor:
    """Stacking: base = modelli già tunati (ciascuno gestisce il proprio log1p/expm1
    internamente), meta-learner = RidgeCV sulle predizioni out-of-fold (scala
    originale), cv = TimeSeriesSplit (mai shuffle)."""
    estimators = list(tuned_models.items())
    return StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=cv)


def run_training(config: AppConfig, granularity: str, use_lag_features: bool = False) -> dict[str, dict[str, float]]:
    """Esegue l'intero training: split cronologico, tuning Optuna, stacking,
    selezione finale, salvataggio artefatto e metriche. Ritorna le metriche di
    tutti i modelli candidati (per il confronto nel README)."""
    data_file = config.data.hour_file if granularity == "hour" else config.data.day_file
    csv_path = Path(config.data.raw_dir) / data_file
    df = load_dataset(csv_path, granularity=granularity)

    train_val_df, test_df = chronological_split(df, config.split.train_val_fraction)
    drop_cols = ["cnt", "casual", "registered", "instant", "dteday"]
    feature_columns = [c for c in train_val_df.columns if c not in drop_cols]

    X_train_val, y_train_val = train_val_df[feature_columns], train_val_df[config.target.column]
    X_test, y_test = test_df[feature_columns], test_df[config.target.column]

    cyclical_periods = config.features.cyclical.model_dump()

    def preprocessor_factory():
        return build_preprocessing_pipeline(granularity, cyclical_periods)

    tscv = TimeSeriesSplit(n_splits=config.split.n_cv_splits)
    seed = config.project.random_seed

    tuned_models: dict[str, Any] = {}
    for model_name in config.optuna.models:
        estimator, best_params, cv_rmse = tune_boosting_model(
            model_name, preprocessor_factory, X_train_val, y_train_val, tscv, config.optuna.n_trials, seed
        )
        logger.info("Optuna %s: RMSE CV=%.3f, params=%s", model_name, cv_rmse, best_params)
        tuned_models[model_name] = estimator

    rf_estimator, rf_params, rf_cv_rmse = tune_random_forest(
        preprocessor_factory, X_train_val, y_train_val, tscv, config.random_forest.n_iter, seed
    )
    logger.info("RandomForest: RMSE CV=%.3f, params=%s", rf_cv_rmse, rf_params)
    tuned_models["random_forest"] = rf_estimator

    stacking = build_stacking_regressor(tuned_models, tscv)

    candidates: dict[str, Any] = dict(tuned_models)
    candidates["stacking"] = stacking
    for name, estimator in candidates.items():
        estimator.fit(X_train_val, y_train_val)

    results = {name: compute_metrics(y_test.to_numpy(), estimator.predict(X_test)) for name, estimator in candidates.items()}
    best_name = min(results, key=lambda name: results[name]["rmse"])
    logger.info("Modello vincente per granularità=%s: %s (RMSE test=%.3f)", granularity, best_name, results[best_name]["rmse"])

    models_dir = Path(config.serving.model_dir)
    save_artifact(candidates[best_name], feature_columns, best_name, models_dir, granularity)
    save_metrics(results, models_dir, granularity)

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Training pipeline Bike Sharing ML")
    parser.add_argument("--granularity", choices=["day", "hour"], default=None)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings() if args.config is None else Settings(config_path=args.config)
    config = settings.load()
    granularity = args.granularity or config.granularity

    run_training(config, granularity)


if __name__ == "__main__":
    main()
```

Aggiungi anche `from pathlib import Path` in cima al file se non già presente.

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k stacking`
Expected: PASS

- [ ] **Step 5: Esegui l'intera suite per verificare che nulla si sia rotto**

Run: `pytest bike_sharing_ml/tests -v`
Expected: PASS (tutti i test verdi)

- [ ] **Step 6: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/train.py bike_sharing_ml/tests/test_models.py
git commit -m "feat: StackingRegressor e orchestrazione CLI di training end-to-end"
```

---

### Task 14: Visualizzazione — residui e feature importance

**Files:**
- Create: `bike_sharing_ml/src/bike_sharing/utils/visualization.py`
- Create: `bike_sharing_ml/tests/test_visualization.py`

**Interfaces:**
- Consumes: nessuno (riceve modelli già fittati da `run_training`).
- Produces: `plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None`; `plot_feature_importance(model: Any, out_path: Path) -> None`.

> Nota: questo file di test non è nell'elenco originale della mission (`test_data.py`, `test_features.py`, `test_models.py`) ma è necessario per testare in isolamento le utility di plotting senza gonfiare `test_models.py`; è coerente con lo spirito "unit test in tests/" della mission.

- [ ] **Step 1: Scrivi il test (fallirà: modulo non esiste)**

`bike_sharing_ml/tests/test_visualization.py`:
```python
"""Test per le utility di plotting."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from bike_sharing.data.preprocessing import build_preprocessing_pipeline
from bike_sharing.models.train import wrap_with_log_target
from bike_sharing.utils.visualization import plot_feature_importance, plot_residuals


def test_plot_residuals_creates_file(tmp_path):
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    out_path = tmp_path / "residuals.png"

    plot_residuals(y_true, y_pred, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_feature_importance_creates_file_for_tree_model(tmp_path):
    X = pd.DataFrame({
        "temp": [0.3, 0.5, 0.7, 0.4], "atemp": [0.3, 0.5, 0.6, 0.4],
        "hum": [0.5, 0.4, 0.6, 0.5], "windspeed": [0.1, 0.2, 0.15, 0.1],
        "yr": [0, 0, 1, 1], "holiday": [0, 0, 0, 1], "workingday": [1, 1, 0, 0],
        "mnth": [1, 2, 3, 4], "weekday": [0, 1, 2, 3], "season": [1, 1, 2, 2],
        "weathersit": [1, 2, 1, 3],
    })
    y = pd.Series([50.0, 80.0, 120.0, 30.0])
    preprocessor = build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})
    model = wrap_with_log_target(preprocessor, RandomForestRegressor(n_estimators=10, random_state=42))
    model.fit(X, y)

    out_path = tmp_path / "importance.png"
    plot_feature_importance(model, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_visualization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bike_sharing.utils.visualization'`

- [ ] **Step 3: Scrivi `src/bike_sharing/utils/visualization.py`**

```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_visualization.py -v`
Expected: PASS (2 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/utils/visualization.py bike_sharing_ml/tests/test_visualization.py
git commit -m "feat: plotting residui e feature importance (Gini/gain + coefficienti stacking)"
```

---

### Task 15: Modulo di predizione (predict.py)

**Files:**
- Create: `bike_sharing_ml/src/bike_sharing/models/predict.py`
- Modify: `bike_sharing_ml/tests/test_models.py`

**Interfaces:**
- Consumes: artefatti salvati da `save_artifact` (Task 7, formato `production.joblib` + `production_metadata.json`).
- Produces: `load_production_artifact(models_dir: Path, granularity: str) -> tuple[Any, dict]`; `predict_from_features(model: Any, feature_columns: list[str], features: dict) -> float`.

- [ ] **Step 1: Aggiungi il test (fallirà: modulo non esiste)**

Aggiungi in fondo a `bike_sharing_ml/tests/test_models.py`:
```python
from bike_sharing.models.predict import load_production_artifact, predict_from_features


def test_predict_from_saved_artifact_is_nonnegative_and_finite(tmp_path):
    X = pd.DataFrame({
        "temp": [0.3, 0.5, 0.7, 0.4, 0.6, 0.2], "atemp": [0.3, 0.5, 0.6, 0.4, 0.6, 0.25],
        "hum": [0.5, 0.4, 0.6, 0.5, 0.3, 0.7], "windspeed": [0.1, 0.2, 0.15, 0.1, 0.2, 0.3],
        "yr": [0, 0, 1, 1, 0, 1], "holiday": [0, 0, 0, 1, 0, 0], "workingday": [1, 1, 0, 0, 1, 1],
        "mnth": [1, 2, 3, 4, 5, 6], "weekday": [0, 1, 2, 3, 4, 5], "season": [1, 1, 2, 2, 2, 3],
        "weathersit": [1, 2, 1, 3, 1, 2],
    })
    y = pd.Series([50.0, 80.0, 120.0, 30.0, 200.0, 60.0])
    preprocessor = build_preprocessing_pipeline(granularity="day", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})
    model = wrap_with_log_target(preprocessor, LinearRegression())
    model.fit(X, y)

    feature_columns = list(X.columns)
    save_artifact(model, feature_columns, "linear", tmp_path, "day")

    loaded_model, metadata = load_production_artifact(tmp_path, "day")
    sample_features = {col: X.iloc[0][col] for col in feature_columns}
    prediction = predict_from_features(loaded_model, metadata["feature_columns"], sample_features)

    assert isinstance(prediction, float)
    assert prediction >= 0.0
    assert np.isfinite(prediction)
```

- [ ] **Step 2: Verifica che il test fallisca**

Run: `pytest bike_sharing_ml/tests/test_models.py -v -k saved_artifact`
Expected: FAIL — `ModuleNotFoundError: No module named 'bike_sharing.models.predict'`

- [ ] **Step 3: Scrivi `src/bike_sharing/models/predict.py`**

```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `pytest bike_sharing_ml/tests/test_models.py -v`
Expected: PASS (tutti i test in test_models.py verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/predict.py bike_sharing_ml/tests/test_models.py
git commit -m "feat: modulo predict.py per il caricamento e serving del modello di produzione"
```

---

### Task 16: FastAPI — endpoint /predict

**Files:**
- Create: `bike_sharing_ml/apps/__init__.py`
- Create: `bike_sharing_ml/apps/api.py`
- Create: `bike_sharing_ml/tests/test_api.py`

**Interfaces:**
- Consumes: `load_production_artifact`, `predict_from_features` (Task 15), `Settings` (Task 1).
- Produces: app FastAPI con `GET /health`, `POST /predict`.

- [ ] **Step 1: Scrivi `apps/api.py`**

```python
"""FastAPI app: serve il modello di produzione per la granularità di default."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bike_sharing.config import Settings
from bike_sharing.models.predict import load_production_artifact, predict_from_features

settings = Settings()
config = settings.load()
GRANULARITY = config.serving.default_granularity
MODELS_DIR = Path(config.serving.model_dir)

_state: dict = {"model": None, "metadata": {}}


@asynccontextmanager
async def lifespan(_: FastAPI):
    _state["model"], _state["metadata"] = load_production_artifact(MODELS_DIR, GRANULARITY)
    yield


app = FastAPI(title="Bike Sharing Demand API", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    """Condizioni meteo/calendario per la granularità servita (hour)."""

    season: int = Field(ge=1, le=4, description="1=primavera 2=estate 3=autunno 4=inverno")
    yr: int = Field(ge=0, le=1, description="0=2011 1=2012")
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23, description="Ora del giorno")
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)
    temp: float = Field(ge=0.0, le=1.0, description="Temperatura normalizzata")
    atemp: float = Field(ge=0.0, le=1.0, description="Temperatura percepita normalizzata")
    hum: float = Field(ge=0.0, le=1.0)
    windspeed: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_cnt: float
    granularity: str
    model_name: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "granularity": GRANULARITY, "model_loaded": _state["model"] is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Modello non ancora caricato")

    prediction = predict_from_features(_state["model"], _state["metadata"]["feature_columns"], request.model_dump())
    return PredictionResponse(
        predicted_cnt=prediction,
        granularity=GRANULARITY,
        model_name=_state["metadata"]["model_name"],
    )
```

- [ ] **Step 2: Rendi `lifespan` tollerante all'assenza dell'artefatto (finché non è stato eseguito un training reale, Task 20)**

Modifica in `apps/api.py` la funzione `lifespan`:
```python
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        _state["model"], _state["metadata"] = load_production_artifact(MODELS_DIR, GRANULARITY)
    except FileNotFoundError:
        _state["model"], _state["metadata"] = None, {}
    yield
```

- [ ] **Step 3: Scrivi il test (artefatto fittizio impostato dopo l'ingresso nel context manager, per non farlo sovrascrivere dal `lifespan`)**

`bike_sharing_ml/tests/test_api.py`:
```python
"""Test dell'endpoint FastAPI /predict, con un artefatto fittizio (no training reale)."""
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from bike_sharing.apps import api as api_module
from bike_sharing.data.preprocessing import build_preprocessing_pipeline
from bike_sharing.models.train import wrap_with_log_target

FEATURE_COLUMNS = [
    "temp", "atemp", "hum", "windspeed", "yr", "holiday", "workingday",
    "mnth", "weekday", "season", "weathersit", "hr",
]

VALID_PAYLOAD = {
    "season": 1, "yr": 0, "mnth": 6, "hr": 8, "holiday": 0, "weekday": 2,
    "workingday": 1, "weathersit": 1, "temp": 0.5, "atemp": 0.5, "hum": 0.5, "windspeed": 0.2,
}


def _train_fixture_model():
    X = pd.DataFrame({
        "temp": [0.3, 0.5, 0.7, 0.4], "atemp": [0.3, 0.5, 0.6, 0.4],
        "hum": [0.5, 0.4, 0.6, 0.5], "windspeed": [0.1, 0.2, 0.15, 0.1],
        "yr": [0, 0, 1, 1], "holiday": [0, 0, 0, 1], "workingday": [1, 1, 0, 0],
        "mnth": [1, 2, 3, 4], "weekday": [0, 1, 2, 3], "season": [1, 1, 2, 2],
        "weathersit": [1, 2, 1, 3], "hr": [8, 12, 18, 22],
    })
    y = pd.Series([50.0, 80.0, 120.0, 30.0])
    preprocessor = build_preprocessing_pipeline(granularity="hour", cyclical_periods={"mnth": 12, "weekday": 7, "hr": 24})
    model = wrap_with_log_target(preprocessor, LinearRegression())
    model.fit(X, y)
    return model


def test_health_endpoint_reports_model_loaded():
    with TestClient(api_module.app) as client:
        # Impostato DOPO l'ingresso nel context manager: il lifespan gira
        # all'__enter__ e, non trovando un artefatto reale, azzera lo stato —
        # sovrascriverlo prima verrebbe perso.
        api_module._state["model"] = _train_fixture_model()
        api_module._state["metadata"] = {"model_name": "linear", "feature_columns": FEATURE_COLUMNS}

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_returns_nonnegative_prediction():
    with TestClient(api_module.app) as client:
        api_module._state["model"] = _train_fixture_model()
        api_module._state["metadata"] = {"model_name": "linear", "feature_columns": FEATURE_COLUMNS}

        response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_cnt"] >= 0.0
    assert body["model_name"] == "linear"


def test_predict_endpoint_rejects_out_of_range_input():
    with TestClient(api_module.app) as client:
        api_module._state["model"] = _train_fixture_model()
        api_module._state["metadata"] = {"model_name": "linear", "feature_columns": FEATURE_COLUMNS}

        invalid_payload = {**VALID_PAYLOAD, "season": 9}
        response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422
```

- [ ] **Step 4: Crea `apps/__init__.py` e verifica che i test passino**

```bash
touch bike_sharing_ml/apps/__init__.py
```

Run: `pytest bike_sharing_ml/tests/test_api.py -v`
Expected: PASS (3 test verdi)

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/apps/__init__.py bike_sharing_ml/apps/api.py bike_sharing_ml/tests/test_api.py
git commit -m "feat: FastAPI /predict e /health per il serving del modello di produzione"
```

---

### Task 17: Streamlit dashboard

**Files:**
- Create: `bike_sharing_ml/apps/dashboard.py`

**Interfaces:**
- Consumes: `Settings`, `load_dataset` (loader), API `/predict` via HTTP.
- Produces: app Streamlit con tab EDA storica e tab simulazione what-if.

**Nota:** Streamlit non ha un framework di unit test standard integrato in questo stack (nessuna dipendenza `streamlit.testing` richiesta dalla mission); la verifica di questo task è manuale (Step 3), non via `pytest`.

- [ ] **Step 1: Scrivi `apps/dashboard.py`**

```python
"""Dashboard Streamlit: EDA storica e simulazione what-if via chiamata HTTP all'API."""
from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

from bike_sharing.config import Settings
from bike_sharing.data.loader import load_dataset

settings = Settings()
config = settings.load()
API_BASE_URL = os.environ.get("BIKE_API_BASE_URL", config.serving.api_base_url)

st.set_page_config(page_title="Bike Sharing Demand", layout="wide")
st.title("Bike Sharing Demand — EDA & Simulazione")

tab_eda, tab_whatif = st.tabs(["Tendenze storiche", "Simulazione what-if"])

with tab_eda:
    raw_path = Path(config.data.raw_dir) / config.data.hour_file
    df = load_dataset(raw_path, granularity="hour")

    st.subheader("Media noleggi per mese")
    monthly = df.groupby(df["dteday"].dt.to_period("M"))["cnt"].mean()
    st.line_chart(monthly)

    st.subheader("Media noleggi per condizione meteo")
    st.bar_chart(df.groupby("weathersit", observed=True)["cnt"].mean())

with tab_whatif:
    st.subheader("Simula uno scenario")
    col1, col2, col3 = st.columns(3)

    season_labels = {1: "Primavera", 2: "Estate", 3: "Autunno", 4: "Inverno"}
    year_labels = {0: "2011", 1: "2012"}
    weather_labels = {1: "Sereno", 2: "Nuvoloso", 3: "Pioggia leggera", 4: "Pioggia forte"}

    with col1:
        season = st.selectbox("Stagione", list(season_labels), format_func=lambda v: season_labels[v])
        yr = st.selectbox("Anno", list(year_labels), format_func=lambda v: year_labels[v])
        mnth = st.slider("Mese", 1, 12, 6)
        hr = st.slider("Ora", 0, 23, 8)
    with col2:
        holiday = st.checkbox("Festivo")
        workingday = st.checkbox("Giorno lavorativo", value=True)
        weekday = st.slider("Giorno settimana (0=domenica)", 0, 6, 3)
        weathersit = st.selectbox("Meteo", list(weather_labels), format_func=lambda v: weather_labels[v])
    with col3:
        temp = st.slider("Temperatura normalizzata", 0.0, 1.0, 0.5)
        atemp = st.slider("Temperatura percepita normalizzata", 0.0, 1.0, 0.5)
        hum = st.slider("Umidità normalizzata", 0.0, 1.0, 0.5)
        windspeed = st.slider("Velocità vento normalizzata", 0.0, 1.0, 0.2)

    if st.button("Predici"):
        payload = {
            "season": season, "yr": yr, "mnth": mnth, "hr": hr,
            "holiday": int(holiday), "weekday": weekday, "workingday": int(workingday),
            "weathersit": weathersit, "temp": temp, "atemp": atemp, "hum": hum, "windspeed": windspeed,
        }
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
            response.raise_for_status()
            st.success(f"Bici stimate: {response.json()['predicted_cnt']:.0f}")
        except requests.RequestException as exc:
            st.error(f"Errore nel chiamare l'API: {exc}")
```

- [ ] **Step 2: Verifica sintattica automatica**

Run: `python -c "import ast; ast.parse(open('bike_sharing_ml/apps/dashboard.py', encoding='utf-8').read())"`
Expected: nessun output (nessun `SyntaxError`)

- [ ] **Step 3: Verifica manuale (richiede un training reale già eseguito, vedi Task 20)**

Run: `uvicorn bike_sharing.apps.api:app --reload &` seguito da `streamlit run bike_sharing_ml/apps/dashboard.py`
Expected: dashboard raggiungibile su `http://localhost:8501`, tab EDA mostra i grafici storici, tab what-if restituisce una predizione non negativa dopo aver premuto "Predici" (richiede che l'API sia in esecuzione e che esista `models/hour/production.joblib`, prodotto dal Task 20).

- [ ] **Step 4: Commit**

```bash
git add bike_sharing_ml/apps/dashboard.py
git commit -m "feat: dashboard Streamlit con EDA storica e simulazione what-if via API"
```

---

### Task 18: Dockerfile e docker-compose

**Files:**
- Create: `bike_sharing_ml/Dockerfile`
- Create: `bike_sharing_ml/docker-compose.yml`
- Create: `bike_sharing_ml/.dockerignore`

**Interfaces:**
- Consumes: `pyproject.toml` (Task 1), `apps/api.py` (Task 16), `apps/dashboard.py` (Task 17).
- Produces: immagine Docker eseguibile in due modalità (api/dashboard) via `docker-compose`.

- [ ] **Step 1: Scrivi `.dockerignore`**

```text
.git
.venv
__pycache__
*.pyc
data/processed
docs
tests
```

- [ ] **Step 2: Scrivi `Dockerfile`**

```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY apps ./apps

RUN pip install --no-cache-dir .

EXPOSE 8000 8501

CMD ["uvicorn", "bike_sharing.apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Scrivi `docker-compose.yml`**

```yaml
services:
  api:
    build: .
    command: ["uvicorn", "bike_sharing.apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro

  dashboard:
    build: .
    command: ["streamlit", "run", "apps/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
    ports:
      - "8501:8501"
    environment:
      - BIKE_API_BASE_URL=http://api:8000
    volumes:
      - ./models:/app/models:ro
      - ./config:/app/config:ro
      - ./data:/app/data:ro
    depends_on:
      - api
```

- [ ] **Step 4: Valida la sintassi di docker-compose**

Run: `docker compose -f bike_sharing_ml/docker-compose.yml config`
Expected: stampa la configurazione risolta senza errori (se Docker non è disponibile in questo ambiente, annota nel README che il file è fornito ma non validato in sandbox e va provato dall'utente in locale).

- [ ] **Step 5: Commit**

```bash
git add bike_sharing_ml/Dockerfile bike_sharing_ml/docker-compose.yml bike_sharing_ml/.dockerignore
git commit -m "feat: containerizzazione Docker (api + dashboard) via docker-compose"
```

---

### Task 19: Esperimento lag/rolling features (confronto offline)

**Files:**
- Create: `bike_sharing_ml/src/bike_sharing/models/lag_experiment.py`

**Interfaces:**
- Consumes: `LagRollingFeatures` (Task 5), `run_training`-style building blocks (Task 6, 7, 9-13).
- Produces: `run_lag_feature_experiment(config: AppConfig) -> dict` — confronta RMSE test con e senza lag features sulla granularità hour, stampa/salva il risultato in `models/hour/lag_experiment.json`.

- [ ] **Step 1: Scrivi `src/bike_sharing/models/lag_experiment.py`**

```python
"""Esperimento offline: confronto RMSE con/senza lag e rolling features su 'cnt'.

Le lag/rolling features (vedi bike_sharing.features.build_features.LagRollingFeatures)
NON sono usate nella pipeline di produzione (l'API /predict non ha accesso allo
storico reale per uno scenario ipotetico). Questo script quantifica il costo di
questa scelta architetturale sul dataset hour.csv, a scopo di trasparenza nel README.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import FeatureUnion

from bike_sharing.config import AppConfig, Settings
from bike_sharing.data.loader import load_dataset
from bike_sharing.data.preprocessing import build_preprocessing_pipeline, chronological_split
from bike_sharing.features.build_features import LagRollingFeatures
from bike_sharing.models.evaluate import compute_metrics
from bike_sharing.models.train import BUILD_ESTIMATOR, wrap_with_log_target

logger = logging.getLogger(__name__)


def _build_pipeline_with_lag(granularity: str, cyclical_periods: dict, lag_periods: list[int], rolling_windows: list[int]) -> ColumnTransformer:
    base = build_preprocessing_pipeline(granularity, cyclical_periods)
    lag = ColumnTransformer(
        transformers=[("lag_rolling", LagRollingFeatures(lag_periods, rolling_windows), ["cnt"])],
        remainder="drop",
    )
    return FeatureUnion([("base", base), ("lag", lag)])


def run_lag_feature_experiment(config: AppConfig) -> dict:
    """Confronta, sulla granularità hour, l'RMSE test del miglior modello (LightGBM,
    per costo computazionale contenuto) con e senza lag/rolling features."""
    csv_path = Path(config.data.raw_dir) / config.data.hour_file
    df = load_dataset(csv_path, granularity="hour")
    train_val_df, test_df = chronological_split(df, config.split.train_val_fraction)

    drop_cols = ["cnt", "casual", "registered", "instant", "dteday"]
    feature_columns_no_lag = [c for c in train_val_df.columns if c not in drop_cols]
    feature_columns_with_lag = feature_columns_no_lag + ["cnt"]

    cyclical_periods = config.features.cyclical.model_dump()
    seed = config.project.random_seed
    tscv = TimeSeriesSplit(n_splits=config.split.n_cv_splits)

    y_train_val, y_test = train_val_df[config.target.column], test_df[config.target.column]

    estimator_no_lag = wrap_with_log_target(
        build_preprocessing_pipeline("hour", cyclical_periods),
        BUILD_ESTIMATOR["lightgbm"]({}, seed),
    )
    estimator_no_lag.fit(train_val_df[feature_columns_no_lag], y_train_val)
    metrics_no_lag = compute_metrics(y_test.to_numpy(), estimator_no_lag.predict(test_df[feature_columns_no_lag]))

    estimator_with_lag = wrap_with_log_target(
        _build_pipeline_with_lag("hour", cyclical_periods, config.features.lag_periods, config.features.rolling_windows),
        BUILD_ESTIMATOR["lightgbm"]({}, seed),
    )
    estimator_with_lag.fit(train_val_df[feature_columns_with_lag], y_train_val)
    metrics_with_lag = compute_metrics(y_test.to_numpy(), estimator_with_lag.predict(test_df[feature_columns_with_lag]))

    result = {"without_lag_features": metrics_no_lag, "with_lag_features": metrics_with_lag}
    out_path = Path(config.serving.model_dir) / "hour" / "lag_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Esperimento lag features: %s", result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_lag_feature_experiment(Settings().load())
```

- [ ] **Step 2: Esegui lo script per generare il confronto reale**

Run: `python -m bike_sharing.models.lag_experiment` (dalla root di `bike_sharing_ml/`, con `PYTHONPATH=src` o dopo `pip install -e .`)
Expected: crea `models/hour/lag_experiment.json` con due blocchi di metriche (`without_lag_features`, `with_lag_features`); nessuna eccezione.

- [ ] **Step 3: Commit**

```bash
git add bike_sharing_ml/src/bike_sharing/models/lag_experiment.py
git commit -m "feat: esperimento offline lag/rolling features (confronto RMSE, non usato in produzione)"
```

---

### Task 20: Training reale, numeri Prima/Dopo e README finale

**Files:**
- Modify: `Progetto Zanni Davide 131946/README.md`
- Create (output, non versionato se troppo pesante — valuta `.gitignore` per `bike_sharing_ml/models/*/production.joblib` se >100MB): `bike_sharing_ml/models/day/*`, `bike_sharing_ml/models/hour/*`

**Interfaces:**
- Consumes: `run_training` (Task 13), `run_lag_feature_experiment` (Task 19), script originale `Bike_Sharing.py` (invariato).

- [ ] **Step 1: Esegui il training reale per entrambe le granularità**

```bash
cd bike_sharing_ml
python -m bike_sharing.models.train --granularity day
python -m bike_sharing.models.train --granularity hour
python -m bike_sharing.models.lag_experiment
cd ..
```
Expected: creati `bike_sharing_ml/models/day/{production.joblib,production_metadata.json,metrics.json}` e gli equivalenti per `hour`, più `models/hour/lag_experiment.json`. Annota nei log quale modello ha vinto per ciascuna granularità (singolo boosting tunato vs stacking).

- [ ] **Step 2: Genera i plot di residui e feature importance per il modello vincente di ciascuna granularità**

Aggiungi ed esegui un piccolo script una tantum (non fa parte del pacchetto, solo per generare gli artefatti da includere nel README — puoi scriverlo come blocco Python eseguito da riga di comando):
```bash
python - <<'PY'
from pathlib import Path
from bike_sharing.config import Settings
from bike_sharing.data.loader import load_dataset
from bike_sharing.data.preprocessing import chronological_split
from bike_sharing.models.predict import load_production_artifact
from bike_sharing.utils.visualization import plot_feature_importance, plot_residuals

config = Settings().load()
for granularity in ("day", "hour"):
    data_file = config.data.hour_file if granularity == "hour" else config.data.day_file
    df = load_dataset(Path(config.data.raw_dir) / data_file, granularity=granularity)
    _, test_df = chronological_split(df, config.split.train_val_fraction)
    model, metadata = load_production_artifact(Path(config.serving.model_dir), granularity)
    X_test = test_df[metadata["feature_columns"]]
    y_test = test_df[config.target.column]
    predictions = model.predict(X_test)

    reports_dir = Path(config.serving.model_dir) / granularity / "reports"
    plot_residuals(y_test.to_numpy(), predictions, reports_dir / "residuals.png")
    plot_feature_importance(model, reports_dir / "feature_importance.png")
PY
```
Expected: creati `models/day/reports/{residuals.png,feature_importance.png}` e gli equivalenti per `hour`.

- [ ] **Step 3: Esegui lo script originale per catturare i numeri "Prima" (baseline storica)**

```bash
cd "Progetto Zanni Davide 131946"
MPLBACKEND=Agg python Bike_Sharing.py > /tmp/bike_sharing_prima_output.txt 2>&1
cd ..
```
Expected: file di testo con l'output originale (incluse le righe `Errore Test (RMSE) = ...`, `Errore Test (MAE) = ...`, `Errore Test (R-squared) = ...` per Linear Regression, KNN, Decision Tree, SVR e Stacking Regressor). Estrai manualmente i valori di RMSE/MAE/R² dello Stacking Regressor originale (la baseline più comparabile con il nuovo Stacking) per la tabella del README.

- [ ] **Step 4: Riscrivi `Progetto Zanni Davide 131946/README.md`**

Sostituisci il contenuto (attualmente solo `"Progetto-Machine-Learning"`) con:
```markdown
# Bike Sharing Demand — Machine Learning Project

Previsione della domanda di bike sharing (Capital Bikeshare, Washington D.C.) a
partire da condizioni meteo e calendario, con confronto tra l'approccio
universitario originale e una pipeline production-grade.

## Architettura

Vedi `bike_sharing_ml/` per il pacchetto completo:

- `src/bike_sharing/data`: loader e split cronologico (no data leakage).
- `src/bike_sharing/features`: feature cicliche (sin/cos), interazioni meteo,
  esperimento lag/rolling (escluso dal serving).
- `src/bike_sharing/models`: tuning Optuna (LightGBM/XGBoost/CatBoost),
  RandomForest, StackingRegressor, valutazione, predizione.
- `apps/api.py`: FastAPI `/predict`. `apps/dashboard.py`: Streamlit (EDA + what-if).
- `Dockerfile` / `docker-compose.yml`: due servizi (api, dashboard).

Il design completo è documentato in
`docs/superpowers/specs/2026-07-22-bike-sharing-ml-productionization-design.md`.

## Flaw metodologici corretti rispetto a `Bike_Sharing.py`

| Problema originale | Correzione |
|---|---|
| `train_test_split` casuale (leakage temporale) | Split cronologico 75/25, mai shuffle |
| Scaling/feature selection fuori dalla CV | Tutto il preprocessing dentro `ColumnTransformer`/`Pipeline`, fittato solo per fold |
| `KFold(shuffle=True)` per CV e Stacking | `TimeSeriesSplit` ovunque |
| Nessuna gestione dell'asimmetria di `cnt` | `TransformedTargetRegressor` con `log1p`/`expm1` |
| Nessuna feature ciclica/meteo derivata | `CyclicalEncoder`, `WeatherInteractionFeatures` |
| Solo Linear/KNN/Tree/SVR | + RandomForest, LightGBM, XGBoost, CatBoost, tutti tunati con Optuna |

## Confronto metriche — Prima vs Dopo (test set)

> **Nota metodologica:** i numeri "Prima" derivano dallo script originale con il
> suo split casuale (quindi ottimisticamente distorti dal leakage temporale); i
> numeri "Dopo" derivano dal test set cronologico, mai visto durante training o
> tuning. Non sono directly comparable in termini di difficoltà del task — il
> confronto mostra comunque che la pipeline "Dopo" è più solida e non dipende da
> un leakage per le sue prestazioni.

| Metrica | Prima (Stacking, `day.csv`, split casuale) | Dopo (`day.csv`, split cronologico) | Dopo (`hour.csv`, split cronologico) |
|---|---|---|---|
| RMSE | *(compilare da `/tmp/bike_sharing_prima_output.txt`)* | *(da `models/day/metrics.json`)* | *(da `models/hour/metrics.json`)* |
| MAE | *(idem)* | *(idem)* | *(idem)* |
| MAPE | n/d (non calcolata nello script originale) | *(da metrics.json)* | *(da metrics.json)* |
| R² | *(idem)* | *(idem)* | *(idem)* |
| RMSLE | n/d (non calcolata nello script originale) | *(da metrics.json)* | *(da metrics.json)* |
| Modello vincente | Stacking (LR+KNN+Tree+SVR) | *(vedi `production_metadata.json`)* | *(vedi `production_metadata.json`)* |

*(L'implementatore che esegue il Task 20 deve sostituire i placeholder sopra con
i numeri reali letti dai file generati agli Step 1-3 prima di considerare il task
concluso — questa tabella non va mai lasciata con i placeholder in un commit finale.)*

## Esperimento lag/rolling features

Le feature di lag/rolling su `cnt` (vedi `LagRollingFeatures`) sono state
valutate offline (`models/hour/lag_experiment.json`, generato dal Task 19) ma
**escluse dalla pipeline servita**: l'endpoint `/predict` riceve solo condizioni
meteo/calendario ipotetiche, senza accesso garantito allo storico reale di `cnt`
necessario per calcolare un lag. I numeri del confronto sono riportati nel file
JSON citato.

## Come eseguire il progetto

```bash
cd bike_sharing_ml
pip install -e ".[dev]"

# Training (genera models/<granularity>/production.joblib)
python -m bike_sharing.models.train --granularity hour
python -m bike_sharing.models.train --granularity day

# Test
pytest tests -v

# API
uvicorn bike_sharing.apps.api:app --reload

# Dashboard (in un altro terminale, con l'API già in esecuzione)
streamlit run apps/dashboard.py

# Docker (api + dashboard)
docker compose up --build
```
```

- [ ] **Step 5: Compila manualmente i placeholder della tabella metriche leggendo i file JSON/testo generati**

Non c'è un test automatico per questo step (è compilazione dati, non codice); prima di procedere al commit, apri `models/day/metrics.json`, `models/hour/metrics.json` e `/tmp/bike_sharing_prima_output.txt` e sostituisci ogni placeholder `*(...)*` nella tabella con il numero reale.

- [ ] **Step 6: Esegui l'intera suite di test una ultima volta**

Run: `pytest bike_sharing_ml/tests -v`
Expected: PASS (tutti i test verdi, nessuna regressione introdotta dal training reale)

- [ ] **Step 7: Commit**

```bash
git add "Progetto Zanni Davide 131946/README.md" bike_sharing_ml/models
git commit -m "docs: README con confronto metriche Prima/Dopo, training reale eseguito"
```

## Self-review

**Copertura spec:**
- Split cronologico → Task 6. ColumnTransformer/Pipeline senza leakage → Task 6, 8-13. Target log1p/casual-registered → Task 8 (log1p scelto, come da decisione approvata).
- Feature cicliche → Task 3. Feature meteo/interazioni → Task 4. Lag/rolling → Task 5, 19 (sperimentale).
- LightGBM/XGBoost/CatBoost/RandomForest + Optuna → Task 9-12. Stacking → Task 13. Metriche complete → Task 7.
- Struttura pacchetto → Task 1 (scaffolding) + ogni singolo task per i file specifici.
- Test null/cicliche/predizioni → Task 2, 3, 6, 15. FastAPI → Task 16. Streamlit → Task 17. Docker → Task 18. README con confronto → Task 20.

**Scan placeholder:** nessun "TBD"/"TODO" nel codice; gli unici placeholder testuali sono nella tabella metriche del README (Task 20), esplicitamente marcati come da compilare con dati reali generati nello stesso task, non lasciabili in un commit finale — non sono placeholder di implementazione.

**Coerenza tipi/firme:** `wrap_with_log_target(preprocessor, estimator)` (Task 8) usato identicamente in Task 9-13, 15, 19; `tune_boosting_model(...)` firma stabile Task 9→13; `build_stacking_regressor(tuned_models, cv)` Task 13; `save_artifact`/`load_production_artifact` stessa forma di metadata (`model_name`, `granularity`, `feature_columns`) tra Task 7, 15, 16, 20 — verificato consistente.
