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


def test_load_dataset_day_sorts_shuffled_input(tmp_path):
    # day.csv è già ordinato cronologicamente nel file grezzo: questo test mescola
    # le righe prima di caricarle, in modo da fallire davvero se sort_values venisse
    # rimosso per errore da loader.py (guardia contro il leakage temporale).
    raw = pd.read_csv(DAY_CSV)
    shuffled = raw.sample(frac=1, random_state=0)
    shuffled_csv = tmp_path / "shuffled_day.csv"
    shuffled.to_csv(shuffled_csv, index=False)

    df = load_dataset(shuffled_csv, granularity="day")

    assert df["dteday"].is_monotonic_increasing


def test_load_dataset_hour_sorts_shuffled_input(tmp_path):
    # Stesso principio del test precedente, ma per hour.csv: l'ordinamento combinato
    # dteday+hr deve reggere anche partendo da un file mescolato.
    raw = pd.read_csv(HOUR_CSV)
    shuffled = raw.sample(frac=1, random_state=0)
    shuffled_csv = tmp_path / "shuffled_hour.csv"
    shuffled.to_csv(shuffled_csv, index=False)

    df = load_dataset(shuffled_csv, granularity="hour")

    combined = df["dteday"].astype(str) + "-" + df["hr"].astype(str).str.zfill(2)
    assert combined.is_monotonic_increasing


def test_load_dataset_hour_granularity_without_hr_column_raises_value_error():
    with pytest.raises(ValueError):
        load_dataset(DAY_CSV, granularity="hour")
