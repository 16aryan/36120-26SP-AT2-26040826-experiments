"""Unit tests for preprocessing and daily aggregations."""

import pandas as pd
import numpy as np
from src.preprocessing import aggregate_hourly_to_daily


def test_aggregate_hourly_to_daily():
    # 24 hours of synthetic observations
    times = pd.date_range("2024-01-01 00:00:00", "2024-01-01 23:00:00", freq="h")
    data = {
        "time": times,
        "temperature_2m": np.full(24, 20.0),
        "relative_humidity_2m": np.full(24, 60.0),
        "wind_speed_10m": np.full(24, 15.0),
        "cloud_cover": np.full(24, 50.0),
        "precipitation": np.full(24, 1.0),       # Sum should be 24.0
        "wind_gusts_10m": np.arange(1, 25, 1.0),  # Max should be 24.0
        "snowfall": np.zeros(24),
    }
    hourly_df = pd.DataFrame(data)
    daily = aggregate_hourly_to_daily(hourly_df)

    assert len(daily) == 1
    assert daily.loc[0, "temperature_2m"] == 20.0
    assert daily.loc[0, "relative_humidity_2m"] == 60.0
    assert daily.loc[0, "wind_speed_10m"] == 15.0
    assert daily.loc[0, "cloud_cover"] == 50.0
    assert daily.loc[0, "precipitation"] == 24.0
    assert daily.loc[0, "wind_gusts_10m"] == 24.0
    assert daily.loc[0, "snowfall"] == 0.0
