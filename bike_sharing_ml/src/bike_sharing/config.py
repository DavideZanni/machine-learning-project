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
