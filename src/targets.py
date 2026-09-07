"""Target derivation module for Climate Comfort Index (CCI) and Weather Hazard Category (WHC).

Strictly implements the exact formulas defined in the AT2 specification.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


# =====================================================================
# Target 1: Climate Comfort Index (CCI) - Regression
# =====================================================================

def calculate_temp_score(temperature_2m: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """TempScore = MAX(0, 1 - ABS(temperature_2m - 22) / 20)"""
    return np.maximum(0.0, 1.0 - np.abs(temperature_2m - 22.0) / 20.0)


def calculate_humidity_score(relative_humidity_2m: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """HumidityScore = MAX(0, 1 - ABS(relative_humidity_2m - 50) / 50)"""
    return np.maximum(0.0, 1.0 - np.abs(relative_humidity_2m - 50.0) / 50.0)


def calculate_wind_score(wind_speed_10m: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """WindScore = MAX(0, 1 - ABS(wind_speed_10m - 10) / 40)"""
    return np.maximum(0.0, 1.0 - np.abs(wind_speed_10m - 10.0) / 40.0)


def calculate_cloud_score(cloud_cover: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """CloudScore = MAX(0, 1 - cloud_cover / 100)"""
    return np.maximum(0.0, 1.0 - cloud_cover / 100.0)


def calculate_rain_score(precipitation: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """RainScore = MAX(0, 1 - precipitation / 20)"""
    return np.maximum(0.0, 1.0 - precipitation / 20.0)


def calculate_climate_comfort_index(
    temperature_2m: pd.Series | np.ndarray,
    relative_humidity_2m: pd.Series | np.ndarray,
    wind_speed_10m: pd.Series | np.ndarray,
    cloud_cover: pd.Series | np.ndarray,
    precipitation: pd.Series | np.ndarray,
) -> Tuple[pd.Series | np.ndarray, Dict[str, pd.Series | np.ndarray]]:
    """Calculate the final Climate Comfort Index (CCI) and intermediate component scores.

    CCI = 100 * (0.35*TempScore + 0.20*HumidityScore + 0.15*WindScore + 0.15*CloudScore + 0.15*RainScore)
    Range: 0 to 100. Higher values indicate more comfortable outdoor conditions.

    Returns
    -------
    Tuple[cci, components_dict]
    """
    temp_score = calculate_temp_score(temperature_2m)
    humidity_score = calculate_humidity_score(relative_humidity_2m)
    wind_score = calculate_wind_score(wind_speed_10m)
    cloud_score = calculate_cloud_score(cloud_cover)
    rain_score = calculate_rain_score(precipitation)

    cci = 100.0 * (
        0.35 * temp_score
        + 0.20 * humidity_score
        + 0.15 * wind_score
        + 0.15 * cloud_score
        + 0.15 * rain_score
    )

    components = {
        "temp_score": temp_score,
        "humidity_score": humidity_score,
        "wind_score": wind_score,
        "cloud_score": cloud_score,
        "rain_score": rain_score,
    }
    return cci, components


def interpret_cci(score: float) -> str:
    """Return textual interpretation for a CCI score."""
    if score >= 80:
        return "Excellent outdoor conditions"
    elif score >= 60:
        return "Comfortable conditions"
    elif score >= 40:
        return "Moderate comfort"
    elif score >= 20:
        return "Poor comfort"
    else:
        return "Very poor comfort"


# =====================================================================
# Target 2: Weather Hazard Category (WHC) - Multiclass Classification
# =====================================================================

def calculate_rain_hazard(precipitation: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """RainHazard = MIN(precipitation / 30, 1)"""
    return np.minimum(precipitation / 30.0, 1.0)


def calculate_wind_hazard(wind_gusts_10m: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """WindHazard = MIN(wind_gusts_10m / 100, 1)"""
    return np.minimum(wind_gusts_10m / 100.0, 1.0)


def calculate_cloud_hazard(cloud_cover: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """CloudHazard = MIN(cloud_cover / 100, 1)"""
    return np.minimum(cloud_cover / 100.0, 1.0)


def calculate_snow_hazard(snowfall: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """SnowHazard = MIN(snowfall / 15, 1)"""
    return np.minimum(snowfall / 15.0, 1.0)


def calculate_temp_hazard(temperature_2m: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """TempHazard = MIN(ABS(temperature_2m - 22) / 25, 1)"""
    return np.minimum(np.abs(temperature_2m - 22.0) / 25.0, 1.0)


def calculate_weather_hazard_index(
    precipitation: pd.Series | np.ndarray,
    wind_gusts_10m: pd.Series | np.ndarray,
    cloud_cover: pd.Series | np.ndarray,
    snowfall: pd.Series | np.ndarray,
    temperature_2m: pd.Series | np.ndarray,
) -> Tuple[pd.Series | np.ndarray, Dict[str, pd.Series | np.ndarray]]:
    """Calculate Weather Hazard Index (WHI) and component hazards.

    WHI = 100 * (0.30*RainHazard + 0.30*WindHazard + 0.20*CloudHazard + 0.10*SnowHazard + 0.10*TempHazard)

    Returns
    -------
    Tuple[whi, hazards_dict]
    """
    rain_h = calculate_rain_hazard(precipitation)
    wind_h = calculate_wind_hazard(wind_gusts_10m)
    cloud_h = calculate_cloud_hazard(cloud_cover)
    snow_h = calculate_snow_hazard(snowfall)
    temp_h = calculate_temp_hazard(temperature_2m)

    whi = 100.0 * (
        0.30 * rain_h
        + 0.30 * wind_h
        + 0.20 * cloud_h
        + 0.10 * snow_h
        + 0.10 * temp_h
    )

    hazards = {
        "rain_hazard": rain_h,
        "wind_hazard": wind_h,
        "cloud_hazard": cloud_h,
        "snow_hazard": snow_h,
        "temp_hazard": temp_h,
    }
    return whi, hazards


def convert_whi_to_whc(whi: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Convert continuous WHI into 4 multiclass classification labels (0, 1, 2, 3).

    - WHI < 25      -> 0 (Low Risk)
    - 25 <= WHI < 50 -> 1 (Moderate Risk)
    - 50 <= WHI < 75 -> 2 (High Risk)
    - WHI >= 75     -> 3 (Extreme Risk)
    """
    conditions = [
        (whi < 25.0),
        (whi >= 25.0) & (whi < 50.0),
        (whi >= 50.0) & (whi < 75.0),
        (whi >= 75.0),
    ]
    choices = [0, 1, 2, 3]
    return np.select(conditions, choices, default=0)


WHC_LABEL_TO_NAME = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
    3: "Extreme Risk",
}


def add_derived_targets(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add both CCI and WHI/WHC target columns to a daily weather DataFrame."""
    df = daily_df.copy()

    # Calculate CCI
    cci, _ = calculate_climate_comfort_index(
        temperature_2m=df["temperature_2m"],
        relative_humidity_2m=df["relative_humidity_2m"],
        wind_speed_10m=df["wind_speed_10m"],
        cloud_cover=df["cloud_cover"],
        precipitation=df["precipitation"],
    )
    df["cci"] = cci

    # Calculate WHI and WHC
    whi, _ = calculate_weather_hazard_index(
        precipitation=df["precipitation"],
        wind_gusts_10m=df["wind_gusts_10m"],
        cloud_cover=df["cloud_cover"],
        snowfall=df["snowfall"],
        temperature_2m=df["temperature_2m"],
    )
    df["whi"] = whi
    df["whc"] = convert_whi_to_whc(whi)
    df["whc_name"] = df["whc"].map(WHC_LABEL_TO_NAME)

    return df
