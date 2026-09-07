"""Configuration settings and constants for AT2 Weather Intelligence experimentation."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
CCI_MODELS_DIR = MODELS_DIR / "comfort_climate"
WHC_MODELS_DIR = MODELS_DIR / "weather_hazard"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Location parameters (Sydney, NSW, Australia)
LOCATION_NAME = "Sydney, NSW, Australia"
LATITUDE = -33.8688
LONGITUDE = 151.2093
TIMEZONE = "Australia/Sydney"

# Historical dataset date boundaries
# "You can use any data prior to 2026 as training, validation and test sets.
# All the data from 2026 onwards should be used as data in production."
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

# Required Open-Meteo variables
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
    "snowfall",
]

# Forecast horizons
CCI_HORIZON_DAYS = [1, 2, 3]  # Next 3 days (+1, +2, +3)
WHC_HORIZON_DAYS = 7          # Exactly +7 days

# Classification classes for WHC
WHC_CLASSES = {
    0: "Low Risk",
    1: "Moderate Risk",
    2: "High Risk",
    3: "Extreme Risk",
}

# Reproducibility
RANDOM_SEED = 42

# Train / Validation / Test temporal cutoffs
TRAIN_END_DATE = "2023-12-31"
VAL_END_DATE = "2024-12-31"
TEST_END_DATE = "2025-12-31"
