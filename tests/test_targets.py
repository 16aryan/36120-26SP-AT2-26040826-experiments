"""Unit tests for Target 1 (CCI) and Target 2 (WHC) calculations."""

import numpy as np
import pytest
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
)


def test_temp_score():
    assert calculate_temp_score(np.array([22.0]))[0] == pytest.approx(1.0)
    assert calculate_temp_score(np.array([32.0]))[0] == pytest.approx(0.5)
    assert calculate_temp_score(np.array([12.0]))[0] == pytest.approx(0.5)
    assert calculate_temp_score(np.array([42.0]))[0] == pytest.approx(0.0)
    assert calculate_temp_score(np.array([2.0]))[0] == pytest.approx(0.0)
    # Outside boundary clipped to 0
    assert calculate_temp_score(np.array([50.0]))[0] == pytest.approx(0.0)
    assert calculate_temp_score(np.array([-5.0]))[0] == pytest.approx(0.0)


def test_humidity_score():
    assert calculate_humidity_score(np.array([50.0]))[0] == pytest.approx(1.0)
    assert calculate_humidity_score(np.array([75.0]))[0] == pytest.approx(0.5)
    assert calculate_humidity_score(np.array([25.0]))[0] == pytest.approx(0.5)
    assert calculate_humidity_score(np.array([100.0]))[0] == pytest.approx(0.0)
    assert calculate_humidity_score(np.array([0.0]))[0] == pytest.approx(0.0)


def test_wind_score():
    assert calculate_wind_score(np.array([10.0]))[0] == pytest.approx(1.0)
    assert calculate_wind_score(np.array([30.0]))[0] == pytest.approx(0.5)
    assert calculate_wind_score(np.array([50.0]))[0] == pytest.approx(0.0)
    assert calculate_wind_score(np.array([0.0]))[0] == pytest.approx(0.75)


def test_cloud_score():
    assert calculate_cloud_score(np.array([0.0]))[0] == pytest.approx(1.0)
    assert calculate_cloud_score(np.array([50.0]))[0] == pytest.approx(0.5)
    assert calculate_cloud_score(np.array([100.0]))[0] == pytest.approx(0.0)


def test_rain_score():
    assert calculate_rain_score(np.array([0.0]))[0] == pytest.approx(1.0)
    assert calculate_rain_score(np.array([10.0]))[0] == pytest.approx(0.5)
    assert calculate_rain_score(np.array([20.0]))[0] == pytest.approx(0.0)
    # Precipitation > 20 clipped to 0
    assert calculate_rain_score(np.array([50.0]))[0] == pytest.approx(0.0)


def test_climate_comfort_index_ideal():
    """Under ideal conditions, CCI must equal exactly 100."""
    temp = np.array([22.0])
    hum = np.array([50.0])
    wind = np.array([10.0])
    cloud = np.array([0.0])
    rain = np.array([0.0])

    cci, comp = calculate_climate_comfort_index(temp, hum, wind, cloud, rain)
    assert cci[0] == pytest.approx(100.0)
    assert interpret_cci(cci[0]) == "Excellent outdoor conditions"


def test_climate_comfort_index_worst():
    """Under extreme harsh conditions, CCI must equal 0."""
    temp = np.array([50.0])
    hum = np.array([0.0])
    wind = np.array([60.0])
    cloud = np.array([100.0])
    rain = np.array([50.0])

    cci, comp = calculate_climate_comfort_index(temp, hum, wind, cloud, rain)
    assert cci[0] == pytest.approx(0.0)
    assert interpret_cci(cci[0]) == "Very poor comfort"


def test_hazard_scores():
    assert calculate_rain_hazard(np.array([0.0]))[0] == pytest.approx(0.0)
    assert calculate_rain_hazard(np.array([15.0]))[0] == pytest.approx(0.5)
    assert calculate_rain_hazard(np.array([30.0]))[0] == pytest.approx(1.0)
    assert calculate_rain_hazard(np.array([60.0]))[0] == pytest.approx(1.0)

    assert calculate_wind_hazard(np.array([50.0]))[0] == pytest.approx(0.5)
    assert calculate_wind_hazard(np.array([100.0]))[0] == pytest.approx(1.0)

    assert calculate_cloud_hazard(np.array([50.0]))[0] == pytest.approx(0.5)
    assert calculate_snow_hazard(np.array([7.5]))[0] == pytest.approx(0.5)
    assert calculate_temp_hazard(np.array([22.0]))[0] == pytest.approx(0.0)
    assert calculate_temp_hazard(np.array([47.0]))[0] == pytest.approx(1.0)


def test_whi_and_whc_classification():
    """Test WHI calculation and strict classification label boundary thresholds."""
    # Test low risk
    whi_low = np.array([10.0, 24.9])
    assert list(convert_whi_to_whc(whi_low)) == [0, 0]

    # Test moderate risk [25, 50)
    whi_mod = np.array([25.0, 49.9])
    assert list(convert_whi_to_whc(whi_mod)) == [1, 1]

    # Test high risk [50, 75)
    whi_high = np.array([50.0, 74.9])
    assert list(convert_whi_to_whc(whi_high)) == [2, 2]

    # Test extreme risk >= 75
    whi_ext = np.array([75.0, 100.0])
    assert list(convert_whi_to_whc(whi_ext)) == [3, 3]
