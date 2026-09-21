"""
Unit tests for src/targets.py -- the Climate Comfort Index (CCI) calculation
functions used in the comfort_climate experimentation notebook.

Run with:  pytest tests/test_targets.py -v

These tests assume `src/targets.py` exposes:
    calculate_temp_score(temperature_2m)
    calculate_humidity_score(relative_humidity_2m)
    calculate_wind_score(wind_speed_10m)
    calculate_cloud_score(cloud_cover)
    calculate_rain_score(precipitation)
    calculate_climate_comfort_index(temperature_2m, relative_humidity_2m,
                                     wind_speed_10m, cloud_cover, precipitation)
        -> (cci_values, components_dict)

Adjust the import path below if your package layout differs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.targets import (
    calculate_temp_score,
    calculate_humidity_score,
    calculate_wind_score,
    calculate_cloud_score,
    calculate_rain_score,
    calculate_climate_comfort_index,
    interpret_cci,
    calculate_rain_hazard,
    calculate_wind_hazard,
    calculate_cloud_hazard,
    calculate_snow_hazard,
    calculate_temp_hazard,
    calculate_weather_hazard_index,
    convert_whi_to_whc,
    add_derived_targets,
)


# ---------------------------------------------------------------------------
# Component score formulas (per AT2 brief, verbatim)
# ---------------------------------------------------------------------------

def test_temp_score_optimal_at_22c():
    # TempScore = MAX(0, 1 - ABS(temp - 22) / 20) -> should be exactly 1.0 at 22C
    assert calculate_temp_score(pd.Series([22.0])).iloc[0] == pytest.approx(1.0)


def test_temp_score_symmetric_around_optimum():
    below = calculate_temp_score(pd.Series([12.0])).iloc[0]   # 22 - 10
    above = calculate_temp_score(pd.Series([32.0])).iloc[0]   # 22 + 10
    assert below == pytest.approx(above)


def test_temp_score_never_negative():
    extreme = calculate_temp_score(pd.Series([-50.0, 100.0]))
    assert (extreme >= 0).all()


def test_humidity_score_optimal_at_50pct():
    assert calculate_humidity_score(pd.Series([50.0])).iloc[0] == pytest.approx(1.0)


def test_wind_score_optimal_at_10kmh():
    assert calculate_wind_score(pd.Series([10.0])).iloc[0] == pytest.approx(1.0)


def test_cloud_score_monotonic_decreasing():
    scores = calculate_cloud_score(pd.Series([0, 25, 50, 75, 100]))
    assert list(scores) == sorted(scores, reverse=True)


def test_cloud_score_bounds():
    scores = calculate_cloud_score(pd.Series([0.0, 100.0]))
    assert scores.iloc[0] == pytest.approx(1.0)
    assert scores.iloc[1] == pytest.approx(0.0)


def test_rain_score_zero_at_20mm_and_above():
    scores = calculate_rain_score(pd.Series([20.0, 30.0, 100.0]))
    assert (scores == 0).all()


def test_rain_score_perfect_when_dry():
    assert calculate_rain_score(pd.Series([0.0])).iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Composite CCI
# ---------------------------------------------------------------------------

def test_cci_perfect_conditions_scores_100():
    cci, _ = calculate_climate_comfort_index(
        temperature_2m=pd.Series([22.0]),
        relative_humidity_2m=pd.Series([50.0]),
        wind_speed_10m=pd.Series([10.0]),
        cloud_cover=pd.Series([0.0]),
        precipitation=pd.Series([0.0]),
    )
    assert cci.iloc[0] == pytest.approx(100.0, abs=1e-6)


def test_cci_worst_conditions_scores_near_zero():
    cci, _ = calculate_climate_comfort_index(
        temperature_2m=pd.Series([100.0]),   # far from 22C -> TempScore = 0
        relative_humidity_2m=pd.Series([0.0]),  # far from 50% -> HumidityScore = 0
        wind_speed_10m=pd.Series([200.0]),   # far from 10 -> WindScore = 0
        cloud_cover=pd.Series([100.0]),      # CloudScore = 0
        precipitation=pd.Series([50.0]),     # >= 20mm -> RainScore = 0
    )
    assert cci.iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_cci_bounded_0_to_100_on_random_inputs():
    rng = np.random.default_rng(42)
    n = 500
    cci, _ = calculate_climate_comfort_index(
        temperature_2m=pd.Series(rng.uniform(-10, 50, n)),
        relative_humidity_2m=pd.Series(rng.uniform(0, 100, n)),
        wind_speed_10m=pd.Series(rng.uniform(0, 120, n)),
        cloud_cover=pd.Series(rng.uniform(0, 100, n)),
        precipitation=pd.Series(rng.uniform(0, 80, n)),
    )
    assert (cci >= 0).all() and (cci <= 100).all()


def test_cci_weights_sum_to_one_via_perfect_partial_case():
    # If exactly one component is imperfect (e.g. temp only), CCI should equal
    # 100 * (1 - 0.35) = 65 when TempScore = 0 and all else perfect.
    cci, _ = calculate_climate_comfort_index(
        temperature_2m=pd.Series([100.0]),  # TempScore -> 0
        relative_humidity_2m=pd.Series([50.0]),
        wind_speed_10m=pd.Series([10.0]),
        cloud_cover=pd.Series([0.0]),
        precipitation=pd.Series([0.0]),
    )
    assert cci.iloc[0] == pytest.approx(65.0, abs=1e-6)


def test_cci_handles_nan_gracefully():
    cci, _ = calculate_climate_comfort_index(
        temperature_2m=pd.Series([22.0, np.nan]),
        relative_humidity_2m=pd.Series([50.0, 50.0]),
        wind_speed_10m=pd.Series([10.0, 10.0]),
        cloud_cover=pd.Series([0.0, 0.0]),
        precipitation=pd.Series([0.0, 0.0]),
    )
    assert cci.iloc[0] == pytest.approx(100.0)
    assert np.isnan(cci.iloc[1])


# ---------------------------------------------------------------------------
# No-leakage checks for shift-based target construction (mirrors notebook logic)
# ---------------------------------------------------------------------------

def test_target_shift_no_leakage():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    cci_series = pd.Series(range(10), index=dates, dtype=float)
    target_h1 = cci_series.shift(-1)

    # target at day t must equal the actual cci at day t+1, never day t or earlier
    for t in range(len(cci_series) - 1):
        assert target_h1.iloc[t] == cci_series.iloc[t + 1]
    # last row's target should be NaN (no future data available)
    assert np.isnan(target_h1.iloc[-1])


# ---------------------------------------------------------------------------
# Target 2: Weather Hazard Category (WHC) formulas (per AT2 brief, verbatim)
# ---------------------------------------------------------------------------

def test_rain_hazard_formula_and_bounds():
    # RainHazard = MIN(precipitation / 30, 1)
    assert calculate_rain_hazard(pd.Series([0.0])).iloc[0] == pytest.approx(0.0)
    assert calculate_rain_hazard(pd.Series([15.0])).iloc[0] == pytest.approx(0.5)
    assert calculate_rain_hazard(pd.Series([30.0])).iloc[0] == pytest.approx(1.0)
    assert calculate_rain_hazard(pd.Series([60.0])).iloc[0] == pytest.approx(1.0)


def test_wind_hazard_formula_and_bounds():
    # WindHazard = MIN(wind_gusts_10m / 100, 1)
    assert calculate_wind_hazard(pd.Series([0.0])).iloc[0] == pytest.approx(0.0)
    assert calculate_wind_hazard(pd.Series([50.0])).iloc[0] == pytest.approx(0.5)
    assert calculate_wind_hazard(pd.Series([100.0])).iloc[0] == pytest.approx(1.0)
    assert calculate_wind_hazard(pd.Series([150.0])).iloc[0] == pytest.approx(1.0)


def test_cloud_hazard_formula_and_bounds():
    # CloudHazard = MIN(cloud_cover / 100, 1)
    assert calculate_cloud_hazard(pd.Series([0.0])).iloc[0] == pytest.approx(0.0)
    assert calculate_cloud_hazard(pd.Series([50.0])).iloc[0] == pytest.approx(0.5)
    assert calculate_cloud_hazard(pd.Series([100.0])).iloc[0] == pytest.approx(1.0)


def test_snow_hazard_formula_and_bounds():
    # SnowHazard = MIN(snowfall / 15, 1)
    assert calculate_snow_hazard(pd.Series([0.0])).iloc[0] == pytest.approx(0.0)
    assert calculate_snow_hazard(pd.Series([7.5])).iloc[0] == pytest.approx(0.5)
    assert calculate_snow_hazard(pd.Series([15.0])).iloc[0] == pytest.approx(1.0)
    assert calculate_snow_hazard(pd.Series([30.0])).iloc[0] == pytest.approx(1.0)


def test_temp_hazard_formula_and_bounds():
    # TempHazard = MIN(ABS(temperature_2m - 22) / 25, 1)
    assert calculate_temp_hazard(pd.Series([22.0])).iloc[0] == pytest.approx(0.0)
    assert calculate_temp_hazard(pd.Series([34.5])).iloc[0] == pytest.approx(0.5)
    assert calculate_temp_hazard(pd.Series([9.5])).iloc[0] == pytest.approx(0.5)
    assert calculate_temp_hazard(pd.Series([47.0])).iloc[0] == pytest.approx(1.0)
    assert calculate_temp_hazard(pd.Series([-3.0])).iloc[0] == pytest.approx(1.0)


def test_whi_composite_perfect_conditions():
    # When all hazard components are 0, WHI must be exactly 0
    whi, _ = calculate_weather_hazard_index(
        precipitation=pd.Series([0.0]),
        wind_gusts_10m=pd.Series([0.0]),
        cloud_cover=pd.Series([0.0]),
        snowfall=pd.Series([0.0]),
        temperature_2m=pd.Series([22.0]),
    )
    assert whi.iloc[0] == pytest.approx(0.0)


def test_whi_composite_extreme_conditions():
    # When all hazard components are maxed (>=1.0), WHI must be 100
    whi, _ = calculate_weather_hazard_index(
        precipitation=pd.Series([100.0]),
        wind_gusts_10m=pd.Series([200.0]),
        cloud_cover=pd.Series([100.0]),
        snowfall=pd.Series([50.0]),
        temperature_2m=pd.Series([60.0]),
    )
    assert whi.iloc[0] == pytest.approx(100.0)


def test_convert_whi_to_whc_thresholds():
    # Class 0: WHI < 25
    # Class 1: 25 <= WHI < 50
    # Class 2: 50 <= WHI < 75
    # Class 3: WHI >= 75
    whi_values = pd.Series([0.0, 24.9, 25.0, 49.9, 50.0, 74.9, 75.0, 100.0])
    whc_labels = convert_whi_to_whc(whi_values)
    expected = [0, 0, 1, 1, 2, 2, 3, 3]
    assert list(whc_labels) == expected


def test_add_derived_targets_integration():
    daily_df = pd.DataFrame({
        "temperature_2m": [22.0, 35.0],
        "relative_humidity_2m": [50.0, 80.0],
        "wind_speed_10m": [10.0, 30.0],
        "wind_gusts_10m": [15.0, 80.0],
        "cloud_cover": [0.0, 90.0],
        "precipitation": [0.0, 15.0],
        "snowfall": [0.0, 0.0],
    })
    result = add_derived_targets(daily_df)
    assert "cci" in result.columns
    assert "whi" in result.columns
    assert "whc" in result.columns
    assert "whc_name" in result.columns
    assert result.loc[0, "whc"] == 0
    assert result.loc[0, "whc_name"] == "Low Risk"
    assert result.loc[0, "cci"] == pytest.approx(100.0)

