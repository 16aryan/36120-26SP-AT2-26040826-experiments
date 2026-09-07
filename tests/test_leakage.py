"""Unit tests for leakage prevention in feature engineering and temporal splitting."""

import pandas as pd
import numpy as np
from src.features import create_target_horizons, add_lag_features
from src.train import temporal_train_val_test_split


def test_target_horizons_shift():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "cci": np.arange(10, 20, 1.0),
        "whc": np.arange(10),
    })

    horizons = create_target_horizons(df)

    # target_cci_h1 at index 0 should equal cci at index 1
    assert horizons.loc[0, "target_cci_h1"] == df.loc[1, "cci"]
    # target_cci_h2 at index 0 should equal cci at index 2
    assert horizons.loc[0, "target_cci_h2"] == df.loc[2, "cci"]
    # target_cci_h3 at index 0 should equal cci at index 3
    assert horizons.loc[0, "target_cci_h3"] == df.loc[3, "cci"]
    # target_whc_h7 at index 0 should equal whc at index 7
    assert horizons.loc[0, "target_whc_h7"] == df.loc[7, "whc"]


def test_lag_features_leakage():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "temperature_2m": np.arange(20, 30, 1.0),
    })

    lagged = add_lag_features(df, ["temperature_2m"], lags=[1, 2])

    # lag 1 at index 1 must equal value at index 0
    assert lagged.loc[1, "temperature_2m_lag_1"] == df.loc[0, "temperature_2m"]
    # lag 2 at index 2 must equal value at index 0
    assert lagged.loc[2, "temperature_2m_lag_2"] == df.loc[0, "temperature_2m"]


def test_temporal_split_no_overlap():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"date": dates, "val": np.arange(len(dates))})

    train, val, test = temporal_train_val_test_split(
        df,
        date_col="date",
        train_end="2023-12-31",
        val_end="2024-12-31",
        test_end="2025-12-31",
    )

    assert train["date"].max() <= pd.to_datetime("2023-12-31")
    assert val["date"].min() > pd.to_datetime("2023-12-31")
    assert val["date"].max() <= pd.to_datetime("2024-12-31")
    assert test["date"].min() > pd.to_datetime("2024-12-31")
    assert test["date"].max() <= pd.to_datetime("2025-12-31")
