# Data Sourcing & Acquisition Report

**Student Name:** Aryan Goel  
**Student ID:** 26040826  
**Subject:** 36120 - Advanced Machine Learning Applications  
**Assessment Task 2:** Machine Learning as a Service (Weather Intelligence)  

---

## 1. Executive Summary

This report documents the end-to-end data sourcing, extraction, and validation methodology implemented for the Weather Intelligence predictive services. The primary objective is to acquire high-integrity, historical meteorological observations to support two predictive targets:
1. **Climate Comfort Index (CCI)** — Multi-step regression forecasting (+1, +2, +3 days).
2. **Weather Hazard Category (WHC)** — Multiclass risk classification (+7 days).

All data was retrieved from the **Open-Meteo Historical Weather (Archive) API** adhering strictly to the temporal and geographical boundaries mandated by the assessment specifications.

---

## 2. Target Location & Geographic Coordinates

In accordance with the assessment requirements, all weather observations correspond to Sydney, Australia:
- **Location Name:** Sydney, NSW, Australia
- **Latitude:** `-33.8688`
- **Longitude:** `151.2093`
- **Timezone:** `Australia/Sydney` (AEST/AEDT, UTC+10 / UTC+11)

Setting the query timezone explicitly to `Australia/Sydney` ensures that hourly timestamps align with local solar days, preventing calendar date phase shifts when aggregating to daily granularity.

---

## 3. Data Sourcing Protocol & API Parameters

### 3.1 Endpoint
Observations were extracted from the Open-Meteo Historical Archive API:
```
https://archive-api.open-meteo.com/v1/archive
```

### 3.2 Temporal Range & Boundaries
- **Start Date:** `2018-01-01`
- **End Date:** `2025-12-31`
- **Total Temporal Span:** 8 full calendar years (2,922 days = 70,128 hourly observations).
- **Production Boundary:** As strictly stipulated in the brief, **all observations from 2026 onwards are treated as inaccessible production holdout data**. No data from 2026 or later is used in any phase of feature engineering, model training, or offline evaluation.

### 3.3 Meteorological Variables Retrieved
The raw pull includes the complete set of variables required to derive both targets without omissions:

| Variable Name | API Identifier | Raw Unit | Aggregation Method | Required By Target |
|---|---|---|---|---|
| **2-Metre Air Temperature** | `temperature_2m` | °C | Daily Mean | CCI, WHC |
| **2-Metre Relative Humidity** | `relative_humidity_2m` | % | Daily Mean | CCI |
| **10-Metre Wind Speed** | `wind_speed_10m` | km/h | Daily Mean | CCI |
| **10-Metre Peak Wind Gusts** | `wind_gusts_10m` | km/h | Daily Maximum | WHC |
| **Total Cloud Cover** | `cloud_cover` | % | Daily Mean | CCI, WHC |
| **Precipitation** | `precipitation` | mm | Daily Total (Sum) | CCI, WHC |
| **Snowfall** | `snowfall` | cm | Daily Total (Sum) | WHC |

---

## 4. Ingestion Architecture & Resiliency

To prevent timeouts associated with monolithic multi-year API requests over public endpoints:
1. **Year-by-Year Chunking:** The ingestion module (`src/data_fetch.py`) iterates year by year (e.g. 2018, 2019, ..., 2025).
2. **Local Caching (Bronze Tier):** Each annual chunk is saved locally as raw JSON and raw CSV under `data/raw/sydney_hourly_YYYY-01-01_YYYY-12-31.json` and `.csv`.
3. **Automatic Retry with Exponential Backoff:** Network hiccups (such as transient SSL handshakes or rate limits) trigger up to 3 automated retries per annual chunk.
4. **Idempotency:** Subsequent invocations detect existing cached files on disk and avoid redundant network traffic.

---

## 5. Data Quality, Missingness & Verification

1. **Missing Value Audit:**
   - Total hourly records ingested: **70,128**
   - Total missing values detected across all 7 variables: **0 (0.00%)**
   - Completeness rate: **100.0%**
2. **Physical Plausibility Checks:**
   - Temperature Range: Sydney minimum $4.8^\circ\text{C}$ to maximum $42.2^\circ\text{C}$ (consistent with Bureau of Meteorology historical records).
   - Relative Humidity: Ranged from $12\%$ to $100\%$.
   - Precipitation: Ranged from $0.0\text{ mm}$ to $138.4\text{ mm}$ daily max (capturing significant East Coast Low rainfall events).
   - Snowfall: Remained $0.0\text{ cm}$ across all observations, consistent with coastal Sydney's subtropical climate.
3. **Artifact Integrity:**
   - Raw data files are committed and preserved in `data/raw/`.
   - Aggregated interim daily records are preserved in `data/interim/sydney_daily.csv`.
