"""Data acquisition module for retrieving historical weather data from Open-Meteo."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.parse
import urllib.request
import pandas as pd

from src.config import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    START_DATE,
    END_DATE,
    HOURLY_VARIABLES,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def build_open_meteo_url(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    hourly_vars: Optional[list] = None,
    timezone: str = TIMEZONE,
) -> str:
    """Build the Open-Meteo Historical Weather API URL with specified parameters."""
    if hourly_vars is None:
        hourly_vars = HOURLY_VARIABLES

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
    }
    query_string = urllib.parse.urlencode(params)
    return f"{BASE_URL}?{query_string}"


def fetch_historical_weather(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    hourly_vars: Optional[list] = None,
    timezone: str = TIMEZONE,
    save_raw: bool = True,
    output_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Fetch historical hourly weather data from Open-Meteo API year-by-year.

    Querying year-by-year avoids single large request timeouts and allows resuming.
    """
    if hourly_vars is None:
        hourly_vars = HOURLY_VARIABLES

    output_dir.mkdir(parents=True, exist_ok=True)
    start_year = int(start_date.split("-")[0])
    end_year = int(end_date.split("-")[0])

    yearly_dfs = []

    for year in range(start_year, end_year + 1):
        year_start = f"{year}-01-01" if year > start_year else start_date
        year_end = f"{year}-12-31" if year < end_year else end_date

        cache_csv = output_dir / f"sydney_hourly_{year_start}_{year_end}.csv"
        cache_json = output_dir / f"sydney_hourly_{year_start}_{year_end}.json"

        if cache_csv.exists():
            logger.info("Loading cached year %d from %s", year, cache_csv)
            df_year = pd.read_csv(cache_csv)
            df_year["time"] = pd.to_datetime(df_year["time"])
            yearly_dfs.append(df_year)
            continue

        url = build_open_meteo_url(
            latitude=latitude,
            longitude=longitude,
            start_date=year_start,
            end_date=year_end,
            hourly_vars=hourly_vars,
            timezone=timezone,
        )
        logger.info("Fetching year %d from Open-Meteo: %s", year, url)

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "WeatherIntelResearch/1.0 (uts-student-assignment)"},
        )

        success = False
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Open-Meteo HTTP {response.status}")
                    raw_json_bytes = response.read()
                data = json.loads(raw_json_bytes.decode("utf-8"))
                df_year = pd.DataFrame(data["hourly"])
                df_year["time"] = pd.to_datetime(df_year["time"])

                if save_raw:
                    with open(cache_json, "wb") as f:
                        f.write(raw_json_bytes)
                    df_year.to_csv(cache_csv, index=False)

                yearly_dfs.append(df_year)
                logger.info("Successfully fetched %d hourly records for %d", len(df_year), year)
                success = True
                break
            except Exception as e:
                logger.warning("Attempt %d failed for year %d: %s", attempt + 1, year, e)

        if not success:
            raise RuntimeError(f"Failed to fetch data for year {year} after 3 attempts.")

    combined_df = pd.concat(yearly_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    if save_raw:
        master_csv = output_dir / f"sydney_hourly_{start_date}_{end_date}.csv"
        combined_df.to_csv(master_csv, index=False)
        logger.info("Saved combined raw dataset to %s (%d rows)", master_csv, len(combined_df))

    return combined_df


def load_raw_data(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    data_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load cached raw hourly data if available, otherwise fetch from Open-Meteo."""
    csv_path = data_dir / f"sydney_hourly_{start_date}_{end_date}.csv"
    if csv_path.exists():
        logger.info("Loading cached raw data from %s", csv_path)
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        return df
    return fetch_historical_weather(start_date=start_date, end_date=end_date, output_dir=data_dir)


if __name__ == "__main__":
    fetch_historical_weather()
