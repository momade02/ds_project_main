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

- **Inputs**
  - List of stations with coordinates and ETAs (more detailed description in `route_stations.py`)
  - Stations list (from Supabase)
  - Historical price data (Supabase)
  - Real-time price data (Tankerkoenig API, optional)

- **Outputs**
  - List of enriched station feature dicts, e.g.:
    - `station_uuid` (string)
    - `lat` (float)
    - `lon` (float)
    - `time_cell` (int)
    - `price_lag_1d_e5` (float)
    - `price_lag_7d_e5` (float)
    - `realtime_price_e5` (float, optional)
    - ...  
  - Route metadata (dict)

---

## 4) How it works (high level)

- Stations from Google are matched to nearest fuel stations of Tankerkönig using a spatial KD-Tree because coordinates do not match perfectly. 
- For each station, historical price lags (1d, 2d, 3d, 7d) are fetched in parallel from Supabase.
- Real-time prices are optionally retrieved via the Tankerkoenig API.
- Deduplication ensures only unique stations are retained, prioritizing minimal detour.
- Output is a list of enriched station features, ready for modeling or dashboard display.

---

``` diff
## 5) Automation hooks

- Intended triggers:
  - Can be called directly via Python:
    ```bash
    python src/integration/route_tankerkoenig_integration.py
    ```
  - Integrated into dashboard backend for on-demand enrichment

- Automation features:
  - Parallel data fetching (ThreadPoolExecutor)
  - Caching of station master list (1-hour TTL)
  - Minimal manual intervention required
```
---

## 6) Validation & quality checks

- Validates input coordinates and station data
- Ensures price data is available for required lags; falls back to most recent available within a window
- Deduplication logic for route-station matches
- Data schema enforcement for outputs

---

## 7) Error handling & troubleshooting

- Handles missing credentials (Supabase, Tankerkoenig API) with clear error messages
- Catches and reports data quality issues (invalid fuel types, missing data)
- External service errors (API failures) are surfaced with actionable remediation steps

Troubleshooting steps:
1. Check for missing environment variables in `.env`
2. Inspect logs and error messages for API failures or data gaps

---

## 8) Configuration (`.env.example`)

Create a `.env` file from the example below.
```ini
# .env.example — copy to .env and fill values
SUPABASE_URL=<https://your-project.supabase.co>
SUPABASE_SECRET_KEY=<SUPABASE_SERVICE_ROLE_KEY>

TANKERKOENIG_EMAIL=<YOUR_TANKERKOENIG_EMAIL>
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
├─ README.md
├─ historical_data.py
└─ route_tankerkoenig_integration.py
```