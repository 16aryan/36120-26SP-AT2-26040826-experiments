# AT2 - Machine Learning as a Service: Experimentation Repository

**Student Name:** Aryan Goel
**Student ID:** 26040826
**Subject:** 36120 - Advanced Machine Learning Applications

## Project Overview

This repository contains the experimentation, EDA, feature engineering, model
development and evaluation work for the Weather Intelligence project. Two
targets are predicted using historical weather data for Sydney, NSW, Australia
(-33.8688, 151.2093), sourced from the Open-Meteo Historical Weather API:

- **Climate Comfort Index (CCI)** — regression, +3 day forecast horizon
- **Weather Hazard Category (WHC)** — multiclass classification, +7 day forecast horizon

## Repository Structure

```
project/
├── data/
│   ├── raw/              # untouched API pulls
│   ├── interim/          # daily-aggregated data
│   └── processed/        # final feature/target tables
├── notebooks/
│   ├── comfort_climate/  # 36120_26SP_AT2_26040826_comfort_climate_experiment.ipynb
│   └── weather_hazard/   # 36120_26SP_AT2_26040826_weather_hazard_experiment.ipynb
├── models/
│   ├── comfort_climate/  # saved CCI model artefacts (model.joblib + metadata.json)
│   └── weather_hazard/   # saved WHC model artefacts (model.joblib + metadata.json)
├── reports/              # experiment reports, figures
├── src/                  # shared helper code (used before packaging into TestPyPI package)
├── tests/                # unit test suite (pytest)
├── pyproject.toml
└── README.md
```

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/16aryan/36120-26SP-AT2-26040826-experiments.git
   cd 36120-26SP-AT2-26040826-experiments
   ```

2. Create and activate a virtual environment (using `uv`, recommended):
   ```bash
   uv venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

   Or with plain `pip`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

3. Launch Jupyter:
   ```bash
   jupyter lab
   ```

## Data Source

- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
- Location: Sydney, NSW, Australia (lat -33.8688, lon 151.2093)
- Training/validation/test data: all observations before 2026
- Production/holdout data: all observations from 2026 onwards (not used in training)

## Target Variables

See `reports/` for the full derivation methodology of CCI and WHC, following
the formulas specified in the AT2 assessment brief.

## Reproducibility & Testing

- All dependencies are declared in `pyproject.toml`.
- Run automated unit tests with `pytest tests/ -v`.
- Each model folder contains the final trained model artifact alongside its operational `metadata.json`.
- The official UTS template notebooks in `notebooks/comfort_climate/` and `notebooks/weather_hazard/` are pre-executed with all output cards and evaluation tables.
