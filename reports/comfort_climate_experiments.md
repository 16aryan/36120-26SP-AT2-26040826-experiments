# Target 1: Climate Comfort Index (CCI) — Experimentation Report

**Student Name:** Aryan Goel  
**Student ID:** 26040826  
**Subject:** 36120 - Advanced Machine Learning Applications  
**Assessment Task 2:** Machine Learning as a Service  

---

## 1. Problem Formulation & Objective

The **Climate Comfort Index (CCI)** is a composite meteorological metric that assesses the suitability of outdoor atmospheric conditions for human activity.
- **Task Type:** Multi-Step Direct Regression
- **Forecast Horizon:** Next 3 Days ($t+1, t+2, t+3$)
- **Scale:** Continuous bounded range from $0$ (Very Poor Comfort) to $100$ (Excellent Outdoor Conditions).

### Mathematical Definition
Component scores:
- $\text{TempScore} = \max(0, 1 - |\text{temperature\_2m} - 22| / 20)$
- $\text{HumidityScore} = \max(0, 1 - |\text{relative\_humidity\_2m} - 50| / 50)$
- $\text{WindScore} = \max(0, 1 - |\text{wind\_speed\_10m} - 10| / 40)$
- $\text{CloudScore} = \max(0, 1 - \text{cloud\_cover} / 100)$
- $\text{RainScore} = \max(0, 1 - \text{precipitation} / 20)$

$$\text{CCI} = 100 \times \left(0.35 \times \text{Temp} + 0.20 \times \text{Hum} + 0.15 \times \text{Wind} + 0.15 \times \text{Cloud} + 0.15 \times \text{Rain}\right)$$

---

## 2. Experimental Setup & Leakage Prevention

### 2.1 Temporal Splitting
To simulate real-world forecasting conditions and prevent temporal data leakage:
- **Training Set (6 years):** 2018-01-01 to 2023-12-31 ($N = 2,191$ days)
- **Validation Set (1 year):** 2024-01-01 to 2024-12-31 ($N = 366$ days)
- **Holdout Test Set (1 year):** 2025-01-01 to 2025-12-31 ($N = 365$ days)

### 2.2 Feature Engineering Architecture
All predictors are computed using strictly backward-looking operators:
- **Autoregressive Lags:** $t-1, t-2, t-3, t-7, t-14$ for all raw weather variables and CCI.
- **Rolling Aggregations:** 3-day, 7-day, and 14-day rolling mean, std, min, and max.
- **Calendar Encodings:** Sinusoidal cyclical transforms of day of year ($\sin/\cos$) and month ($\sin/\cos$).
- **Domain Indicators:** Discomfort deviation from 22°C, high gust indicators (>50 km/h), wet day flags.

---

## 3. Candidate Algorithms & Evaluation Results

Five distinct model families were evaluated on the holdout 2025 test set across all three forecast horizons:

| Model Architecture | Mean RMSE ↓ | Mean MAE ↓ | Mean $R^2$ ↑ | Horizon 1 RMSE | Horizon 2 RMSE | Horizon 3 RMSE |
|---|---|---|---|---|---|---|
| **Baseline (Dummy Mean)** | 11.7309 | 9.1015 | -0.0019 | 11.7229 | 11.7372 | 11.7326 |
| **Ridge Regression ($\alpha=10$)** | 9.7640 | 7.4350 | 0.3026 | 8.8054 | 10.1194 | 10.3673 |
| **Random Forest Regressor** | **9.7516** | 7.4652 | **0.3053** | 8.9473 | **10.0543** | **10.2533** |
| **LightGBM Regressor** | 9.7916 | 7.5575 | 0.2982 | **8.7711** | 10.2017 | 10.4020 |
| **XGBoost Regressor** | 9.8240 | 7.4780 | 0.2945 | 8.9319 | 10.1634 | 10.3768 |

---

## 4. Key Experimental Findings

1. **Predictive Lift over Baseline:**
   All machine learning architectures outperformed the baseline dummy model by **~2.0 RMSE points** and reduced Mean Absolute Error from 9.10 to 7.44 points.
2. **Horizon Error Decay:**
   As atmospheric entropy increases with time, forecast accuracy decays across all models:
   - $h=1$ (+1 day): RMSE $\approx 8.8$
   - $h=2$ (+2 days): RMSE $\approx 10.1$
   - $h=3$ (+3 days): RMSE $\approx 10.3$
3. **Primary Feature Drivers:**
   Feature importance analysis reveals that the top predictive drivers are:
   - `cci` (today's current comfort index) and `cci_roll_mean_3`
   - `temperature_2m_roll_max_14` and `temperature_2m_roll_min_14` (seasonal heatwave / cold front baselines)
   - `precipitation_roll_max_7` (synoptic rain system persistence)

---

## 5. Model Deployment Decision

The **Random Forest / XGBoost Regressor** is serialized to `models/comfort_climate/model.joblib` accompanied by `models/comfort_climate/metadata.json`, providing direct operational integration with the FastAPI service.
