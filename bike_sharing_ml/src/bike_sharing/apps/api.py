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
    try:
        _state["model"], _state["metadata"] = load_production_artifact(MODELS_DIR, GRANULARITY)
    except FileNotFoundError:
        _state["model"], _state["metadata"] = None, {}
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
