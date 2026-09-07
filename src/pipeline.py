"""Full execution pipeline for data fetching, preprocessing, and feature engineering."""

import logging
from pathlib import Path
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    START_DATE,
    END_DATE,
)
from src.data_fetch import load_raw_data
from src.preprocessing import aggregate_hourly_to_daily, save_interim_data
from src.features import build_full_feature_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("=== Phase 1: Data Acquisition ===")
    hourly_df = load_raw_data(start_date=START_DATE, end_date=END_DATE, data_dir=RAW_DATA_DIR)
    logger.info("Hourly dataset shape: %s", hourly_df.shape)

    logger.info("=== Phase 2: Daily Aggregation ===")
    daily_df = aggregate_hourly_to_daily(hourly_df)
    save_interim_data(daily_df, filename="sydney_daily.csv", output_dir=INTERIM_DATA_DIR)
    logger.info("Daily interim dataset shape: %s", daily_df.shape)

    logger.info("=== Phase 3: Feature Engineering & Target Derivation ===")
    full_df = build_full_feature_dataset(daily_df)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save full processed dataset
    processed_csv = PROCESSED_DATA_DIR / "sydney_features_targets.csv"
    full_df.to_csv(processed_csv, index=False)
    logger.info("Saved full processed dataset to %s (%d rows, %d cols)", processed_csv, len(full_df), full_df.shape[1])

    # Summary of targets
    logger.info("CCI summary:\n%s", full_df["cci"].describe())
    logger.info("WHC distribution:\n%s", full_df["whc_name"].value_counts())

    return full_df


if __name__ == "__main__":
    run_pipeline()
