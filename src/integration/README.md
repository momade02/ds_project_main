# Integration — Component README

Table of Contents  
1. [Purpose & quick summary](#1-purpose--quick-summary)  
2. [Where this fits in the pipeline](#2-where-this-fits-in-the-pipeline)  
3. [Inputs & Outputs](#3-inputs--outputs)  
4. [How it works (high level)](#4-how-it-works-high-level)  
5. [Automation hooks](#5-automation-hooks)  
6. [Validation & quality checks](#6-validation--quality-checks)  
7. [Error handling & troubleshooting](#7-error-handling--troubleshooting)  
8. [Configuration (.env.example)](#8-configuration-envexample)  
9. [Links](#9-links)  
10. [Mini file tree](#10-mini-file-tree)  

---

## 1) Purpose & quick summary

`integration` — Responsible for enriching fuel stations with historical and real-time fuel price data, matching stations found by Google places API (new) to fuel stations of the `stations` table from Tankerkönig, and producing feature vectors for downstream modeling and dashboard presentation.

For non-technical readers: This component connects route planning (Google Maps) with fuel price analytics, so the app can recommend where and when to refuel based on both route and price history.

---

## 2) Where this fits in the pipeline

- **Upstream:**  
  - Data from `src/data_pipeline` 
- **Downstream:**  
  - `src/modeling` (price forecasting)  
  - `src/decision` (recommendation logic)  
  - `src/app` (dashboard/UI)

---

## 3) Inputs & Outputs

**route_tankerkoenig_integration.py:**
- Inputs:
  - List of stations with coordinates and ETAs (from `route_stations.py` in data_pipeline)
  - Stations master list (Supabase)
  - Historical price data (Supabase)
  - Real-time price data (Tankerkönig API, optional)
- Outputs:
  - List of enriched station feature dicts (for modeling), e.g.:
    - `station_uuid`, `lat`, `lon`, `time_cell`
    - `price_lag_1d_e5`, `price_lag_2d_e5`, `price_lag_3d_e5`, `price_lag_7d_e5`
    - `price_current_e5` (realtime or fallback)
  - Route metadata (dict)

**historical_data.py:**
- Inputs:
  - Station UUID
  - Fuel type (e5, e10, diesel)
- Outputs:
  - Price history DataFrame (14-day time series)
  - Hourly statistics DataFrame (avg/min/max per hour)

---

## 4) How it works (high level)

**route_tankerkoenig_integration.py** (main pipeline):
- Stations from Google are matched to nearest Tankerkönig stations using a spatial KD-Tree (coordinates don't match exactly)
- For each station, historical price lags (1d, 2d, 3d, 7d) are fetched in parallel from Supabase
- Real-time prices are optionally retrieved via the Tankerkönig API
- Deduplication ensures only unique stations are retained, prioritizing minimal detour
- Output is a list of enriched station features, ready for modeling or dashboard display

**historical_data.py** (Page 03 support):
- Provides 14-day price history for station detail charts
- Computes hourly price statistics to identify best/worst refueling times
- Used exclusively by Page 03 (Station Details)

---

## 5) Automation hooks

- Can be called directly via Python (from project root):
  ```bash
  python -m src.integration.route_tankerkoenig_integration
  ```
- Called automatically by the dashboard when a user plans a trip
- Uses parallel database queries for faster data loading
- Station list is cached for 1 hour to reduce database load

---

## 6) Validation & quality checks

- Coordinates are matched and filtered against the Tankerkönig database (unmatched stations are excluded)
- Ensures price data is available for required lags; falls back to most recent available within a window
- Deduplication logic for route-station matches
- Output schema is enforced implicitly through typed function signatures and consistent return structures

---

## 7) Error handling & troubleshooting

- Handles missing credentials (Supabase, Tankerkönig API) with clear error messages
- Catches and reports data quality issues (invalid fuel types, missing data)
- External service errors (API failures) are surfaced with actionable remediation steps

Troubleshooting steps:
1. Check for missing environment variables in `.env`
2. Inspect logs and error messages for API failures or data gaps

---

## 8) Configuration (`.env.example`)

Create a `.env` file from the example below.
```ini
# .env.example - copy to .env and fill values
SUPABASE_URL=<https://your-project.supabase.co>
SUPABASE_SECRET_KEY=<SUPABASE_SERVICE_ROLE_KEY>

TANKERKOENIG_API_KEY=<YOUR_TANKERKOENIG_API_KEY>

GOOGLE_MAPS_API_KEY=<YOUR_GOOGLE_MAPS_API_KEY>
```

---

## 9) Links

- Root README: [../../README.md](../../README.md)
- Data Pipeline: [../data_pipeline/README.md](../data_pipeline/README.md)
- Modeling: [../modeling/README.md](../modeling/README.md)
- App/UI: [../app/README.md](../app/README.md)

---

## 10) Mini file tree

```
src/integration/
├─ __init__.py
├─ README.md
├─ historical_data.py
└─ route_tankerkoenig_integration.py
```