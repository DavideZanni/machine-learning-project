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
