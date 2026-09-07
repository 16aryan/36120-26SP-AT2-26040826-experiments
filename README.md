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
│   ├── comfort_climate/  # CCI regression notebooks
│   └── weather_hazard/   # WHC classification notebooks
├── models/
│   ├── comfort_climate/  # saved CCI model artefacts
│   └── weather_hazard/   # saved WHC model artefacts
├── reports/              # experiment reports, figures
├── src/                  # shared helper code (used before packaging into TestPyPI package)
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

## Reproducibility Notes

- All dependencies are pinned in `pyproject.toml`.
- Notebooks are numbered in the order they should be run within each subfolder.
- Random seeds are fixed in all modelling notebooks for reproducibility.
