"""Test per i transformer di feature engineering."""
import numpy as np
import pandas as pd
import pytest

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


def test_weather_interaction_features_computes_expected_columns():
    from bike_sharing.features.build_features import WeatherInteractionFeatures

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
    assert np.isclose(out["discomfort_index"].iloc[0], 0.64)
    assert np.isclose(out["workingday_x_weathersit"].iloc[0], 2.0)
    assert np.isclose(out["workingday_x_weathersit"].iloc[1], 0.0)


def test_lag_rolling_features_never_uses_current_value():
    from bike_sharing.features.build_features import LagRollingFeatures

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


def test_lag_rolling_features_rolling_std_value_and_row_zero_fillna():
    from bike_sharing.features.build_features import LagRollingFeatures

    df = pd.DataFrame({"cnt": [10.0, 20.0, 30.0, 40.0, 50.0]})
    transformer = LagRollingFeatures(lag_periods=[1], rolling_windows=[2])

    out = transformer.fit_transform(df)

    # cnt_rolling_std_2 alla riga 3 (indice 2) usa i valori shiftati delle righe 0 e 1
    # (10.0, 20.0): deviazione standard campionaria pandas (ddof=1 di default) =
    # sqrt(((10-15)^2 + (20-15)^2) / (2-1)) = sqrt(50)
    assert np.isclose(out["cnt_rolling_std_2"].iloc[2], np.sqrt(50))
    # Riga 0: nessuno storico disponibile per cnt_lag_1 -> NaN sostituito da fillna(0.0)
    assert out["cnt_lag_1"].iloc[0] == 0.0


def test_lag_rolling_features_multiple_lags_and_windows_preserve_column_order():
    from bike_sharing.features.build_features import LagRollingFeatures

    df = pd.DataFrame({"cnt": [10.0, 20.0, 30.0, 40.0, 50.0]})
    transformer = LagRollingFeatures(lag_periods=[1, 2], rolling_windows=[2, 3])

    out = transformer.fit_transform(df)

    expected_columns = [
        "cnt_lag_1",
        "cnt_lag_2",
        "cnt_rolling_mean_2",
        "cnt_rolling_std_2",
        "cnt_rolling_mean_3",
        "cnt_rolling_std_3",
    ]
    assert list(out.columns) == expected_columns
    assert list(transformer.get_feature_names_out()) == expected_columns


def test_lag_rolling_features_raises_valueerror_for_periods_below_one():
    from bike_sharing.features.build_features import LagRollingFeatures

    with pytest.raises(ValueError):
        LagRollingFeatures(lag_periods=[0], rolling_windows=[2])

    with pytest.raises(ValueError):
        LagRollingFeatures(lag_periods=[-1], rolling_windows=[2])
