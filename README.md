# AT2 — Machine Learning as a Service: Weather Intelligence Experimentation Repository

**Student Name:** Aryan Goel  
**Student ID:** 26040826  
**Subject:** 36120 — Advanced Machine Learning Applications  
**Institution:** University of Technology Sydney (UTS)  
**Live Production Microservice:** [https://three6120-26sp-at2-26040826-api.onrender.com](https://three6120-26sp-at2-26040826-api.onrender.com)  
**FastAPI Deployment Repository:** [https://github.com/16aryan/36120-26SP-AT2-26040826-api](https://github.com/16aryan/36120-26SP-AT2-26040826-api)  
**Custom TestPyPI Package:** [`nba-prep-26040826` v0.4.0](https://test.pypi.org/project/nba-prep-26040826/0.4.0/)  

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](tests/)
[![Render](https://img.shields.io/badge/Render-Live-46E3B7.svg?logo=render&logoColor=white)](https://three6120-26sp-at2-26040826-api.onrender.com)
[![TestPyPI Package](https://img.shields.io/badge/TestPyPI-nba--prep--26040826%20v0.4.0-blue.svg)](https://test.pypi.org/project/nba-prep-26040826/0.4.0/)

---

## 1. Project Overview & Scenario

Open-Meteo has modernised its global meteorological platform through an asynchronous, microservice-based architecture. As part of their enterprise transformation program, this project designs, implements, and evaluates machine learning services capable of forecasting two high-value, weather-derived indices for **Sydney, NSW, Australia (-33.8688° S, 151.2093° E)** using historical observations from the Open-Meteo Historical Weather API:

1. **Target 1: Climate Comfort Index (CCI) [Multi-Horizon Regression]:**  
   Direct multi-horizon forecasting of outdoor human biometeorological comfort across a 3-day window ($t+1, t+2, t+3$), continuous on $[0, 100]$.
2. **Target 2: Weather Hazard Category (WHC) [Multiclass Classification]:**  
   Cost-sensitive predictive classification across four operational hazard tiers (**Low, Moderate, High, Extreme Risk**) with an exact $+7$-day lead time ($t+7$).

---

## 2. Repository Structure

```
36120-26SP-AT2-26040826-experiments/
├── data/
│   ├── raw/                  # 17 historical weather observation files (2018–2025 CSV & JSON, tracked in Git)
│   ├── interim/              # Daily-aggregated observations strictly following meteorological rules
│   └── processed/            # Final feature & target matrices (X_train, y_train, X_val, y_val, X_test, y_test)
├── notebooks/
│   ├── comfort_climate/      # 36120_26SP_AT2_26040826_comfort_climate_experiment.ipynb (179 cells executed)
│   └── weather_hazard/       # 36120_26SP_AT2_26040826_weather_hazard_experiment.ipynb (177 cells executed)
├── models/
│   ├── comfort_climate/      # Serialized CCI artifacts (.joblib and .pkl) + metadata.json
│   └── weather_hazard/       # Serialized WHC artifacts (.joblib and .pkl) + metadata.json
├── reports/
│   ├── 36120-26SP-AT2-26040826-report.md     # Comprehensive academic report (< 4,000 words)
│   ├── 36120-26SP-AT2-26040826-report.docx   # Professional Word report with 15 embedded figures (1.2 MB)
│   ├── data_sourcing.md                      # Detailed data acquisition & aggregation audit
│   ├── comfort_climate_experiments.md        # CCI experiment notes and regression findings
│   ├── weather_hazard_experiments.md         # WHC experiment notes and imbalance strategies
│   └── figures/                              # 15 publication-grade visualization artifacts (PNG)
├── src/                                      # Modular source code
│   ├── config.py                             # Project paths, temporal boundaries, coordinates
│   ├── data_fetch.py                         # Open-Meteo API data ingestion with caching
│   ├── preprocessing.py                      # Hourly-to-daily atmospheric aggregation rules
│   ├── targets.py                            # Mathematical derivations for CCI and WHC
│   ├── features.py                           # Harmonic calendar, autoregressive lags, rolling statistics
│   ├── pipeline.py                           # Leakage-free end-to-end dataset transformation
│   ├── train.py                              # Baseline training script
│   └── model_experiments.py                  # Dynamic candidate model training & artifact exporter
├── tests/                                    # Unit test suite (pytest - 28 tests passing)
│   ├── test_leakage.py                       # Zero-leakage temporal backward-shifting verification
│   ├── test_preprocessing.py                 # Aggregation rules & bounds validation
│   ├── test_targets.py                       # Mathematical formulas & threshold tiers
│   └── test_models.py                        # Model inference, shapes, and metadata consistency
├── pyproject.toml                            # Build configuration & dependencies
└── README.md                                 # Complete documentation
```

---

## 3. Data Sourcing & Zero-Leakage Splitting Protocol

### 3.1 Geographic Parameters
* **Coordinates:** Sydney, NSW, Australia (`latitude: -33.8688`, `longitude: 151.2093`)
* **Timezone:** `Australia/Sydney` (AEST/AEDT, local solar day alignment)
* **Temporal Scope:** `2018-01-01` to `2025-12-31` (8 full calendar years; 70,128 hourly readings)
* **Production Boundary:** In strict adherence to assignment guidelines, **all data $\ge 2026$ is treated as strictly inaccessible production holdout data**. No data from 2026 was used in training or tuning.

### 3.2 Out-of-Time Dataset Partitions

| Partition | Time Interval | Calendar Days | Share (%) | Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **Training Set** | 2018-01-01 to 2023-12-31 | 2,191 | 75.16% | Feature learning, model parameter fitting |
| **Validation Set** | 2024-01-01 to 2024-12-31 | 366 | 12.56% | Hyperparameter tuning & model selection |
| **Holdout Test Set** | 2025-01-01 to 2025-12-24 | 358 | 12.28% | Unbiased final performance benchmark |

*Note: The final 7 days of 2025 are consumed by the 7-day lookahead target window, leaving exactly **2,915 fully labeled daily instances** with zero forward leakage.*

---

## 4. Empirical Evaluation Results

### 4.1 Target 1: Climate Comfort Index (CCI) — Multi-Horizon Regression
*(Evaluated across $h=1, 2, 3$ days on the unseen 2025 holdout test set)*

| Model Architecture | Mean RMSE ↓ | Mean MAE ↓ | Mean $R^2$ ↑ | H1 (+1d) RMSE | H2 (+2d) RMSE | H3 (+3d) RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Dummy Mean)** | 11.7309 | 9.1015 | -0.0019 | 11.7229 | 11.7372 | 11.7326 |
| **Ridge Regression ($\alpha=10$)** | 9.7640 | **7.4350** | 0.3026 | 8.8054 | 10.1194 | 10.3673 |
| **Random Forest Regressor (Selected)** | **9.7516** | 7.4652 | **0.3053** | 8.9473 | **10.0543** | **10.2533** |
| **LightGBM Regressor** | 9.7916 | 7.5575 | 0.2982 | **8.7711** | 10.2017 | 10.4020 |
| **XGBoost Regressor** | 9.8240 | 7.4780 | 0.2945 | 8.9319 | 10.1634 | 10.3768 |

* **Decision:** Random Forest Regressor selected for production deployment, reducing test RMSE by **1.9824 points** over persistence baseline.

### 4.2 Target 2: Weather Hazard Category (WHC) — 4-Class Classification
*(Evaluated at $+7$-day lead time on the 2025 holdout test set)*

| Model Architecture | Balancing Strategy | Raw Accuracy | Balanced Accuracy ↑ | Macro F1 ↑ | Weighted F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Dummy Stratified)** | Stratified Prior | 0.4330 | 0.2456 | 0.2424 | 0.4316 |
| **Logistic Regression** | `class_weight='balanced'` | 0.4050 | 0.2730 | 0.2511 | 0.4328 |
| **Random Forest Classifier (Selected)** | **Balanced Bootstrap** | **0.5503** | **0.2865** | **0.2783** | **0.5344** |
| **XGBoost Classifier** | Scale Pos Weight | 0.5391 | 0.2863 | 0.2761 | 0.5261 |
| **LightGBM Classifier** | `is_unbalance=True` | 0.5112 | 0.2671 | 0.2612 | 0.5011 |

* **Decision:** Balanced Random Forest Classifier selected, achieving the highest Macro F1 (**0.2783**) and Balanced Accuracy (**0.2865**) while mitigating majority-class collapse.

---

## 5. Report Figures Directory (`reports/figures/`)

The repository includes **15 publication-grade visualization artifacts** generated in high-resolution PNG:

1. `cci_distribution.png`: Empirical distribution and kernel density estimation of Climate Comfort Index (2018–2025).
2. `whc_class_distribution.png`: Extreme class imbalance across four Weather Hazard tiers with decision thresholds.
3. `whc_variable_boxplots.png`: Physical parameter escalation (wind gusts, rain, cloud, temp) across hazard tiers.
4. `whc_mutual_information.png`: Top 15 non-linear feature ranking by Mutual Information for 7-day storm hazard.
5. `cci_seasonal_profile.png`: Monthly comfort profile showing Southern Hemisphere summer peaks and winter dips.
6. `cci_feature_correlations.png`: Heatmap showing correlation of engineered lag/rolling features with targets.
7. `cci_model_comparison_rmse.png`: RMSE progression across horizons ($h_1, h_2, h_3$) comparing ML models against baseline.
8. `cci_actual_vs_predicted_2025.png`: Full-year 2025 trajectory comparing ground-truth actual comfort vs. predictions.
9. `cci_residual_diagnostics.png`: Residual error distribution and homoscedastic fitted-value diagnostics.
10. `cci_feature_importance.png`: Relative feature importance rankings for Climate Comfort Index regression.
11. `whc_model_comparison.png`: Macro F1 and Balanced Accuracy comparisons across candidate classifiers.
12. `whc_roc_pr_curves.png`: Multiclass One-vs-Rest ROC and Precision-Recall curves on the 2025 holdout set.
13. `whc_confusion_matrix.png`: 4x4 row-normalized confusion matrix on 358 holdout test days.
14. `whc_feature_importance.png`: Gini impurity feature attribution rankings for 7-day weather hazard prediction.
15. `api_verification_dashboard.png`: Operational inference dashboard showing live forecasts and sub-10ms latencies.

---

## 6. Custom Python Package (`nba-prep-26040826` v0.4.0)

As mandated by the assessment specifications, the custom Python package originally created for AT1 was extended to v0.4.0 and published to **TestPyPI**. It provides reusable feature engineering, aggregation logic, and target derivations across both collegiate sports analytics and meteorological intelligence:

* **TestPyPI URL:** [https://test.pypi.org/project/nba-prep-26040826/0.4.0/](https://test.pypi.org/project/nba-prep-26040826/0.4.0/)
* **Install Command:**
  ```bash
  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nba-prep-26040826==0.4.0
  ```

---

## 7. Reproduction Instructions

### 7.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/16aryan/36120-26SP-AT2-26040826-experiments.git
cd 36120-26SP-AT2-26040826-experiments

# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with development test dependencies
pip install -e ".[dev]"
```

### 7.2 Run Automated Test Suites
```bash
# Run all 28 unit tests (Target formulas, preprocessing, feature leakage, model inference)
pytest -v
```
*Expected Result:* `28 passed in ~1.40s`

### 7.3 Reproduce Model Training & Artifact Export
```bash
# Execute model experiments pipeline
python3 src/model_experiments.py
```

### 7.4 Launch Jupyter Experimentation Notebooks
```bash
jupyter lab
# Open and run:
# 1. notebooks/comfort_climate/36120_26SP_AT2_26040826_comfort_climate_experiment.ipynb
# 2. notebooks/weather_hazard/36120_26SP_AT2_26040826_weather_hazard_experiment.ipynb
```

---

## 8. Final Deliverables Summary

* **Deliverable 1 (Experiments Archive):** `36120-26SP-AT2-26040826-experiments.zip` (31 MB)
* **Deliverable 2 (API Archive):** `36120-26SP-AT2-26040826-api.zip` (7.5 MB)
* **Deliverable 3 (Academic Report):** `36120-26SP-AT2-26040826-report.docx` (1.2 MB, 3,705 words)
* **Deliverable 4 (Live Cloud API):** [https://three6120-26sp-at2-26040826-api.onrender.com](https://three6120-26sp-at2-26040826-api.onrender.com)
