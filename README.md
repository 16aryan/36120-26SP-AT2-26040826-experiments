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
│   ├── comfort_climate/  # saved CCI model artefacts (cci_model.pkl + metadata.json)
│   └── weather_hazard/   # saved WHC model artefacts (whc_model.pkl + metadata.json)
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

## Custom Python Package (`nba-prep-26040826` v0.4.0)

As required by the AT2 specification, the custom Python package originally developed for AT1 has been extended to v0.4.0 and published to **TestPyPI**. It provides shared, reusable feature engineering, aggregation logic, and target derivations across both collegiate basketball (AT1) and meteorological intelligence (AT2).

- **TestPyPI URL:** [https://test.pypi.org/project/nba-prep-26040826/0.4.0/](https://test.pypi.org/project/nba-prep-26040826/0.4.0/)
- **Package GitHub:** [16aryan/36120-26SP-group38-26040826-Python-package-Aryan-Goel](https://github.com/16aryan/36120-26SP-group38-26040826-Python-package-Aryan-Goel)
- **Install Command:**
  ```bash
  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nba-prep-26040826==0.4.0
  ```

### Dual-Domain Scope:
1. **AT1 (Collegiate Basketball Analytics):** Anthropometric cleaning (`clean_height`), collegiate year normalization, True Shooting %, defensive impact, multi-season cumulative player statistics, draft prospect tiers, and financial contract ROI modeling.
2. **AT2 (Weather Intelligence):** Open-Meteo hourly-to-daily aggregations (`aggregate_hourly_to_daily`), Climate Comfort Index calculations (`calculate_cci`, `interpret_cci`), Weather Hazard Category classification (`calculate_whc`, `interpret_whc`), consecutive dry-streak days (`calculate_dry_streak`), and cyclical calendar encodings (`add_cyclical_calendar_features`).

## Reproducibility & Verification Status

All components, data splits, model artifacts, and test suites have been verified for reproducibility and production readiness.

### 1. Automated Unit Test Suite (`pytest`)
- **Execution Result:** **19 / 19 passed** in 1.31s (`pytest -v`)
- [`tests/test_leakage.py`](tests/test_leakage.py): Verified strict temporal boundary isolation and zero-leakage backward shifting across all feature engineering pipelines.
- [`tests/test_preprocessing.py`](tests/test_preprocessing.py): Verified hourly-to-daily data aggregation algorithms and missing value imputation rules.
- [`tests/test_targets.py`](tests/test_targets.py): Verified mathematical formulation, boundary conditions ($0 \le \text{CCI} \le 100$), individual component scores, and 4-tier WHC classifications.

### 2. Model Artifacts & Operational Verification
- **Climate Comfort Index Model (`models/comfort_climate/cci_model.pkl`):**
  - Architecture: Multi-horizon regressor (+1d, +2d, +3d forecasts).
  - Validated inference output tensor shape: `(N, 3)` across 31 engineered features.
  - Accompanied by operational [`metadata.json`](models/comfort_climate/metadata.json) containing feature gain rankings, RMSE, MAE, and $R^2$ performance metrics.
- **Weather Hazard Category Model (`models/weather_hazard/whc_model.pkl`):**
  - Architecture: Balanced multi-class classifier (+7d lead time).
  - Validated inference output shape: `(N,)` over 4 categorical tiers across 28 engineered features.
  - Accompanied by operational [`metadata.json`](models/weather_hazard/metadata.json) with accuracy, balanced accuracy, and per-class Macro/Weighted F1 metrics.

### 3. Data Splits & Row Alignment
- **Strict chronological holdout partitioning (Zero data leakage):**
  - `data/processed/X_train.csv` (2,191 rows, 28 cols) aligned with `y_train.csv` (2,191 rows) — 0 missing values.
  - `data/processed/X_val.csv` (366 rows, 28 cols) aligned with `y_val.csv` (366 rows) — 0 missing values.
  - `data/processed/X_test.csv` (358 rows, 28 cols) aligned with `y_test.csv` (358 rows) — 0 missing values.

### 4. Notebook Integrity & Execution Quality
- **Validated JSON structure and AST cell compilation:**
  - `notebooks/comfort_climate/36120_26SP_AT2_26040826_comfort_climate_experiment.ipynb`: 179 cells compiled with **0 syntax errors**.
  - `notebooks/weather_hazard/36120_26SP_AT2_26040826_weather_hazard_experiment.ipynb`: 177 cells compiled with **0 syntax errors**.
  - All output display tiles (`print_tile`) and evaluation benchmark plots pre-executed and rendered.

### 5. Source Architecture Modules
- All core modular packages in `src/` (`config`, `preprocessing`, `targets`, `features`, `train`, `pipeline`, and `model_experiments`) verified importable without warnings or circular dependencies.
