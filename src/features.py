"""Feature engineering module with strict leakage prevention."""

import numpy as np
import pandas as pd
from typing import List, Tuple

from src.targets import add_derived_targets


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar features based on the observation date."""
    res = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(res["date"]):
        res["date"] = pd.to_datetime(res["date"])

    day_of_year = res["date"].dt.dayofyear
    month = res["date"].dt.month
    day_of_week = res["date"].dt.dayofweek

    # Cyclical sin/cos encodings
    res["sin_dayofyear"] = np.sin(2 * np.pi * day_of_year / 365.25)
    res["cos_dayofyear"] = np.cos(2 * np.pi * day_of_year / 365.25)

    res["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    res["cos_month"] = np.cos(2 * np.pi * month / 12.0)

    res["sin_dayofweek"] = np.sin(2 * np.pi * day_of_week / 7.0)
    res["cos_dayofweek"] = np.cos(2 * np.pi * day_of_week / 7.0)

    # Season indicator (Southern Hemisphere: Sydney)
    # Summer: Dec-Feb (12, 1, 2), Autumn: Mar-May (3-5), Winter: Jun-Aug (6-8), Spring: Sep-Nov (9-11)
    res["is_summer"] = month.isin([12, 1, 2]).astype(int)
    res["is_winter"] = month.isin([6, 7, 8]).astype(int)

    return res


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 3, 7, 14, 30],
) -> pd.DataFrame:
    """Add lagged feature values strictly looking backwards (shift > 0)."""
    res = df.copy()
    for col in columns:
        if col in res.columns:
            for lag in lags:
                res[f"{col}_lag_{lag}"] = res[col].shift(lag)
    return res


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [3, 7, 14, 30],
) -> pd.DataFrame:
    """Add backward-looking rolling statistics (mean, std, min, max)."""
    res = df.copy()
    for col in columns:
        if col in res.columns:
            for win in windows:
                # shift(1) ensures the rolling window strictly uses past days and doesn't include future days
                rolling = res[col].rolling(window=win, min_periods=win)
                res[f"{col}_roll_mean_{win}"] = rolling.mean()
                res[f"{col}_roll_std_{win}"] = rolling.std()
                res[f"{col}_roll_min_{win}"] = rolling.min()
                res[f"{col}_roll_max_{win}"] = rolling.max()
    return res


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific meteorological features."""
    res = df.copy()

    # Temperature difference from Sydney comfortable baseline (22 C)
    if "temperature_2m" in res.columns:
        res["temp_diff_comfort"] = np.abs(res["temperature_2m"] - 22.0)

    # Extreme gust indicator (> 50 km/h)
    if "wind_gusts_10m" in res.columns:
        res["is_high_gust"] = (res["wind_gusts_10m"] >= 50.0).astype(int)

    # Rainy day indicator
    if "precipitation" in res.columns:
        res["is_rainy"] = (res["precipitation"] >= 1.0).astype(int)

    return res


def create_target_horizons(df: pd.DataFrame) -> pd.DataFrame:
    """Create forward-shifted target variables strictly matching the AT2 specifications.

    Target 1: Climate Comfort Index (CCI)
    - Next 3 days: t+1, t+2, t+3
    Target 2: Weather Hazard Category (WHC)
    - Exactly +7 days: t+7
    """
    res = df.copy()

    if "cci" in res.columns:
        res["target_cci_h1"] = res["cci"].shift(-1)
        res["target_cci_h2"] = res["cci"].shift(-2)
        res["target_cci_h3"] = res["cci"].shift(-3)

    if "whc" in res.columns:
        res["target_whc_h7"] = res["whc"].shift(-7)

    return res


def build_full_feature_dataset(daily_df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature engineering pipeline producing a complete modeling dataset."""
    df = add_derived_targets(daily_df)
    df = add_calendar_features(df)

    base_cols = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "cloud_cover",
        "precipitation",
        "wind_gusts_10m",
        "snowfall",
        "cci",
        "whi",
    ]

    df = add_lag_features(df, columns=base_cols, lags=[1, 2, 3, 7, 14])
    df = add_rolling_features(df, columns=["temperature_2m", "precipitation", "wind_gusts_10m", "cci"], windows=[3, 7, 14])
    df = add_domain_features(df)
    df = create_target_horizons(df)

    # Drop leading rows with NaNs from lags, and trailing rows with NaNs from targets
    return df
