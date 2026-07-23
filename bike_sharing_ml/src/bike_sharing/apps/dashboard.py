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
