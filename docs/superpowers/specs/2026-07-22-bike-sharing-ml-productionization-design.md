# Design: Refactoring & Productionizzazione Bike Sharing ML

Data: 2026-07-22
Repo: `DavideZanni/machine-learning-project` (contenuto in `Progetto Zanni Davide 131946/`)

## Contesto

Il progetto universitario originale (`Bike_Sharing.py`, dataset `day.csv`) ha diversi flaw
metodologici:

- `train_test_split` casuale (`random_state=42`, 2/3-1/3) su una serie storica → data leakage
  temporale (il modello vede "il futuro" durante il training).
- Scaling (`StandardScaler`) e feature selection (`Lasso` + `SelectFromModel`) eseguiti **fuori**
  dal ciclo di Cross-Validation, sul train set intero, prima della `GridSearchCV` → leakage tra i
  fold durante la model selection.
- `KFold(shuffle=True)` usato anche per lo Stacking finale → stessa violazione della struttura
  temporale.
- Nessuna feature ciclica, nessuna feature meteo derivata, nessun modello boosting moderno,
  nessuna ricerca iperparametrica bayesiana, nessuna gestione dell'asimmetria del target.

Obiettivo: ricostruire il progetto come pacchetto Python production-grade, eliminando i flaw
sopra, con training, serving (API + dashboard) e containerizzazione.

## Decisioni approvate dall'utente

| # | Decisione | Scelta |
|---|---|---|
| 1 | Granularità dataset | Entrambe (`day.csv`, `hour.csv`) via `config.yaml` (`granularity: day\|hour`) |
| 2 | Target | `log1p(cnt)` diretto (non split casual/registered) |
| 3 | Budget tuning Optuna | Rapido, ~25 trial/modello |
| 4 | Posizione codice | Dentro il repo esistente, sotto `Progetto Zanni Davide 131946/bike_sharing_ml/` |
| 5 | Lag/rolling features | Valutate e confrontate, ma **escluse dalla pipeline servita** (API/dashboard) — solo esperimento riportato nel README |
| 6 | Granularità servita di default | `hour.csv` (day.csv resta allenabile via CLI/config) |
| 7 | Comunicazione Dashboard↔API | HTTP reale (non import diretto), coerente con `docker-compose` a due servizi |

## Architettura

```text
Progetto Zanni Davide 131946/
├── Bike_Sharing.py            # script universitario originale (invariato, riferimento storico)
├── Bike_Sharing_Report.pdf
├── day.csv                    # dataset storico già presente nel repo
├── README.md                  # riscritto: confronto Prima/Dopo + istruzioni
└── bike_sharing_ml/
    ├── config/config.yaml
    ├── data/raw/{day.csv,hour.csv}     # copiati da "bike sharing dataset/"
    ├── data/processed/
    ├── models/                          # artefatti .joblib + plot residui/importance
    ├── src/bike_sharing/
    │   ├── config.py                    # pydantic Settings, carica config.yaml
    │   ├── data/loader.py                # lettura csv, parsing date, dtype
    │   ├── data/preprocessing.py         # split cronologico, ColumnTransformer
    │   ├── features/build_features.py    # CyclicalEncoder, WeatherInteractionFeatures,
    │   │                                  # LagRollingFeatures (sperimentale, off by default)
    │   ├── models/train.py               # CLI: split → Optuna → stacking → salvataggio
    │   ├── models/evaluate.py            # RMSE/MAE/MAPE/R2/RMSLE, plot residui, SHAP
    │   ├── models/predict.py             # carica artefatto, espone funzione di predizione
    │   └── utils/visualization.py
    ├── apps/api.py                       # FastAPI /predict, /health
    ├── apps/dashboard.py                 # Streamlit, chiama api.py via HTTP
    ├── tests/{test_data,test_features,test_models}.py
    ├── pyproject.toml
    ├── Dockerfile
    ├── docker-compose.yml
    └── README (sezione dedicata nel README di repo, non duplicato)
```

## Data flow / pipeline

1. `loader.py` legge `day.csv`/`hour.csv`, parse `dteday` a `datetime`, tipizza colonne
   categoriche (`season`, `weathersit`, ecc.) come `category`.
2. `preprocessing.py`:
   - ordina per `dteday`(+`hr` se presente)
   - split cronologico 75/25 (indice, non shuffle)
   - `ColumnTransformer`: passthrough numeriche meteo, branch `CyclicalEncoder`, branch
     `WeatherInteractionFeatures`, `OneHotEncoder` per categoriche a bassa cardinalità
   - tutto wrappato in un unico `sklearn.Pipeline`, fit **solo** dentro ogni fold di CV /
     training set finale — mai su tutto il dataset prima dello split.
3. `TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)` per
   ogni modello → un solo oggetto serializzabile con joblib, niente gestione manuale del log.
4. `train.py`:
   - split cronologico
   - Optuna study (TPE sampler) per LightGBM/XGBoost/CatBoost, objective = RMSE medio su
     `TimeSeriesSplit(n_splits=5)` sul train/val (75%), ~25 trial/modello
   - RandomForest: `RandomizedSearchCV` con `TimeSeriesSplit` (più economico di uno study Optuna
     dedicato, guadagno atteso minore da bayesian search su RF)
   - `StackingRegressor(estimators=[tuned models], final_estimator=RidgeCV, cv=TimeSeriesSplit(5))`
   - valutazione finale **una sola volta** sul test holdout (25% più recente)
   - selezione del modello vincente (singolo vs stacking) per RMSE test
   - salvataggio in `models/`: pipeline+modello vincente (`.joblib`), metriche (`.json`),
     plot residui e feature importance (Gini/gain + SHAP summary del vincitore)
5. Esperimento lag/rolling: eseguito come run separata (`--use-lag-features`), risultati
   confrontati e riportati nel README, **non** usato nel modello di produzione.

## Metriche

RMSE, MAE, MAPE (con guardia su `cnt` prossimo a zero), R², RMSLE — calcolate in scala
originale (dopo `expm1`), riportate sia come media±std della CV time-series sia come singolo
numero sul test holdout finale.

## Serving

- **FastAPI** (`apps/api.py`): `POST /predict` con `pydantic.BaseModel` che valida i campi
  meteo/calendario grezzi (season, yr, mnth, hr* solo se granularity=hour, holiday, weekday,
  workingday, weathersit, temp, atemp, hum, windspeed). Carica l'artefatto joblib della
  granularità `hour` di default (configurabile). Nessuna feature lag richiesta → stateless,
  real-time-safe. Ritorna `cnt` predetto (garantito ≥ 0 grazie a `expm1`).
- **Streamlit** (`apps/dashboard.py`): tab EDA storica (riusa `loader.py`) + tab "what-if"
  che chiama l'API via HTTP (`requests`), non import diretto del modello.

## Testing

- `test_data.py`: loader corretto, niente null dopo preprocessing, split cronologico verificato
  (`max(date train) < min(date test)` — guard di regressione contro il leakage originale).
- `test_features.py`: output `CyclicalEncoder` in `[-1, 1]`, interazioni meteo coerenti.
- `test_models.py`: predizioni dal modello salvato finite, non-negative, forma coerente.

## Docker

Immagine singola multi-stage (builder + runtime slim); `docker-compose.yml` con due servizi
(`api`, `dashboard`) che condividono `models/` in sola lettura, build dalla stessa immagine,
comando differente per servizio.

## README

Architettura, tabella comparativa metriche Prima (script originale, `day.csv`, split casuale)
vs Dopo (stacking/LightGBM tunato, split cronologico) su entrambe le granularità dove
applicabile, nota esplicita sull'esperimento lag-features e perché escluso dal serving,
istruzioni per training/test/API/dashboard/Docker.

## Fuori scope (per questo giro)

- Deployment cloud reale (solo Docker locale/compose).
- Autenticazione API.
- Monitoraggio/drift detection in produzione.
- ONNX export (menzionato nell'albero originale come opzione ma non richiesto esplicitamente:
  si salva solo `.joblib`; ONNX rimandato a un giro successivo se richiesto).
