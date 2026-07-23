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
> tuning. Non sono direttamente comparabili in termini di difficoltà del task — il
> confronto mostra comunque che la pipeline "Dopo" è più solida e non dipende da
> un leakage per le sue prestazioni.

| Metrica | Prima (Stacking, `day.csv`, split casuale) | Dopo (`day.csv`, split cronologico) | Dopo (`hour.csv`, split cronologico) |
|---|---|---|---|
| RMSE | 574.52 | 1077.62 | 65.41 |
| MAE | 418.44 | 836.43 | 40.79 |
| MAPE | n/d (non calcolata nello script originale) | 48.20% | 29.19% |
| R² | 0.9147 | 0.6188 | 0.9116 |
| RMSLE | n/d (non calcolata nello script originale) | 0.3860 | 0.3435 |
| Modello vincente | Stacking (LR+KNN+Tree+SVR) | Stacking (LightGBM+XGBoost+CatBoost+RandomForest, meta RidgeCV) | Stacking (LightGBM+XGBoost+CatBoost+RandomForest, meta RidgeCV) |

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
