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
    assert np.isclose(out["workingday_x_weathersit"].iloc[0], 2.0)
    assert np.isclose(out["workingday_x_weathersit"].iloc[1], 0.0)
