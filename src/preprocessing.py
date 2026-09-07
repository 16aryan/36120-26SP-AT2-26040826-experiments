"""Data preprocessing and daily aggregation module."""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config import INTERIM_DATA_DIR

logger = logging.getLogger(__name__)


def aggregate_hourly_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly weather observations into daily summaries.

    Strictly implements the aggregation rules from the AT2 specification:
    - Temperature (temperature_2m): Daily Mean
    - Relative Humidity (relative_humidity_2m): Daily Mean
    - Wind Speed (wind_speed_10m): Daily Mean
    - Cloud Cover (cloud_cover): Daily Mean
    - Precipitation (precipitation): Daily Total (sum)
    - Wind Gusts (wind_gusts_10m): Daily Maximum (max)
    - Snowfall (snowfall): Daily Total (sum)

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Hourly observations with 'time' column and weather variable columns.

    Returns
    -------
    pd.DataFrame
        Daily observations indexed by date.
    """
    df = hourly_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    df["date"] = df["time"].dt.date

    # Aggregation dictionary strictly following AT2 specification
    agg_rules = {
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "wind_speed_10m": "mean",
        "cloud_cover": "mean",
        "precipitation": "sum",
        "wind_gusts_10m": "max",
        "snowfall": "sum",
    }

    # Only aggregate columns present in the input DataFrame
    actual_rules = {col: rule for col, rule in agg_rules.items() if col in df.columns}

    daily_df = df.groupby("date").agg(actual_rules).reset_index()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    # Validate data completeness
    missing_counts = daily_df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.warning("Missing values detected after daily aggregation:\n%s", missing_counts[missing_counts > 0])
        # Forward-fill any minor sporadic missing values
        daily_df = daily_df.ffill().bfill()

    logger.info("Aggregated %d hourly records into %d daily records", len(df), len(daily_df))
    return daily_df


def save_interim_data(
    daily_df: pd.DataFrame,
    filename: str = "sydney_daily.csv",
    output_dir: Path = INTERIM_DATA_DIR,
) -> Path:
    """Save aggregated daily data to the interim data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    daily_df.to_csv(out_path, index=False)
    logger.info("Saved interim daily data to %s (%d rows)", out_path, len(daily_df))
    return out_path


def load_interim_data(
    filename: str = "sydney_daily.csv",
    data_dir: Path = INTERIM_DATA_DIR,
) -> pd.DataFrame:
    """Load interim daily data from disk."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Interim daily data not found at {path}. Run preprocessing first.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df
