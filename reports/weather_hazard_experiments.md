# Target 2: Weather Hazard Category (WHC) — Experimentation Report

**Student Name:** Aryan Goel  
**Student ID:** 26040826  
**Subject:** 36120 - Advanced Machine Learning Applications  
**Assessment Task 2:** Machine Learning as a Service  

---

## 1. Problem Formulation & Objective

The **Weather Hazard Category (WHC)** classifies the severity of weather conditions and potential impact on public safety, infrastructure, and transport operations.
- **Task Type:** Multiclass Classification (4 discrete risk classes)
- **Forecast Horizon:** Exactly +7 Days ($t+7$)
- **Output Classes:**
  - Class 0: **Low Risk** ($\text{WHI} < 25$)
  - Class 1: **Moderate Risk** ($25 \le \text{WHI} < 50$)
  - Class 2: **High Risk** ($50 \le \text{WHI} < 75$)
  - Class 3: **Extreme Risk** ($\text{WHI} \ge 75$)

### Mathematical Derivation of WHI
Component hazards:
- $\text{RainHazard} = \min(\text{precipitation} / 30, 1)$
- $\text{WindHazard} = \min(\text{wind\_gusts\_10m} / 100, 1)$
- $\text{CloudHazard} = \min(\text{cloud\_cover} / 100, 1)$
- $\text{SnowHazard} = \min(\text{snowfall} / 15, 1)$
- $\text{TempHazard} = \min(|\text{temperature\_2m} - 22| / 25, 1)$

$$\text{WHI} = 100 \times \left(0.30 \times \text{Rain} + 0.30 \times \text{Wind} + 0.20 \times \text{Cloud} + 0.10 \times \text{Snow} + 0.10 \times \text{Temp}\right)$$

---

## 2. Severe Class Imbalance Analysis

Empirical distribution across the 8-year historical period (2018–2025):

| Class Index | Category Label | Sample Count | Frequency (%) |
|---|---|---|---|
| **0** | Low Risk | 1,378 | 47.16% |
| **1** | Moderate Risk | 1,407 | 48.15% |
| **2** | High Risk | 130 | 4.45% |
| **3** | Extreme Risk | 7 | 0.24% |

### Key Challenge:
Because Classes 0 and 1 comprise **95.31%** of all historical observations, naive classifiers (or models trained with standard Cross-Entropy/Gini loss) suffer from majority-class collapse. To mitigate this:
1. **Balanced Class Weighting:** Inversely weight classes proportional to their frequencies via $w_j = \frac{N}{K \cdot n_j}$.
2. **Metric Selection:** Model selection is governed strictly by **Macro F1-Score** and **Balanced Accuracy**, ensuring equal weight across rare critical hazard classes.

---

## 3. Candidate Algorithms & Evaluation Results

Evaluated on the holdout 2025 test set:

| Model Architecture | Raw Accuracy | Balanced Accuracy ↑ | Macro F1 ↑ | Weighted F1 |
|---|---|---|---|---|
| **Baseline (Dummy Stratified)** | 0.4330 | 0.2456 | 0.2424 | 0.4316 |
| **Logistic Regression (Balanced)** | 0.4050 | 0.2730 | 0.2511 | 0.4328 |
| **Random Forest Classifier (Balanced)** | **0.5503** | **0.2865** | **0.2783** | **0.5344** |
| **XGBoost Classifier** | 0.5391 | 0.2863 | 0.2761 | 0.5261 |
| **LightGBM Classifier (Balanced)** | 0.5112 | 0.2671 | 0.2612 | 0.5011 |

---

## 4. Key Findings & Insights

1. **Top Performer:**
   The **Balanced Random Forest Classifier** achieved the highest Macro F1 (**0.2783**) and highest Balanced Accuracy (**0.2865**), outperforming the baseline by **+3.6% Macro F1**.
2. **Feature Importance Insights:**
   The primary indicators for +7 day hazards are:
   - `cloud_cover_lag_1` and `cloud_cover_lag_7` (barometric trough formations)
   - `wind_gusts_10m_roll_std_14` (atmospheric turbulence and wind volatility)
   - `temperature_2m_roll_max_14` (extreme synoptic heat dome patterns)
3. **Operational Recommendation:**
   Due to the infrequency of extreme hazard events, the service should issue early advisory warnings whenever the predicted probability for High/Extreme hazard exceeds calibrated risk thresholds (e.g. $P(\text{Hazard} \ge 2) > 0.20$).

---

## 5. Model Deployment Decision

The **Balanced Random Forest Classifier** is exported to `models/weather_hazard/model.joblib` and `models/weather_hazard/metadata.json` for consumption by the FastAPI production service.
