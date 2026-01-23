# Data Pipeline - Component README

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

## 1) Purpose & quick summary

`data_pipeline` - Responsible for automated acquisition and preparation of fuel stations along a route and price data for downstream modeling and decision modules. The component produces station records and a rolling price table for all stations in Germany.

For non-technical readers: This component gathers fuel station locations and price feeds, enriches them with route, detour & ETA context, and delivers datasets so the app and models can recommend where and when to refuel.

## 2) Where this fits in the pipeline

- **Upstream:** UI/Integration that provides starting point/destination; external data providers (Tankerkoenig CSV releases; Google Maps APIs).  
- **Downstream:** `src/integration` (create full dataset), `src/modeling` (price forecasting), `src/decision` (stop optimizer), `src/app` (dashboard and route recommendations).

## 3) Inputs & Outputs

**Inputs**

- `route_stations.py`: origin/destination address (`str`), `GOOGLE_MAPS_API_KEY` (env). Example: "Berlin, Germany".
- `update_prices_stations.py`: Tankerkoenig CSVs accessible via authenticated URL (constructed using `TANKERKOENIG_EMAIL` and `TANKERKOENIG_API_KEY` in env) and Supabase credentials (`SUPABASE_URL`, `SUPABASE_SECRET_KEY`).

**Outputs**

- `route_stations.py` returns a list of station records (dict) with fields:
  - `name` (string)
  - `lat` (float)
  - `lon` (float)
  - `eta` (ISO8601 string)
  - `detour_distance_km` (float)
  - `detour_duration_min` (float)
  - `distance_along_m` (int)
  - `fraction_of_route` (float)
  - `opening_hours` (list; human-readable)
  - `opening_periods` (list of dicts)
  - `is_open_at_eta` (bool or null)
  - `open_now` (bool)

- `update_prices_stations.py` updates two Supabase tables: `stations` (full refresh) and `prices` (rolling inserts). Logs are written to `~/logs/daily_update_YYYYMMDD.log`.

**Schema of `stations` table**

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `uuid` | uuid | NOT NULL | Primary key, unique station identifier from Tankerkoenig |
| `name` | text | YES | Station name (e.g., "Shell Tuebingen Nord") |
| `brand` | text | YES | Brand name (e.g., "Shell", "Aral", "Esso") |
| `street` | text | YES | Street name |
| `house_number` | text | YES | House/building number |
| `post_code` | text | YES | German postal code (5 digits) |
| `city` | text | YES | City name |
| `latitude` | float8 | YES | Geographic latitude (decimal degrees) |
| `longitude` | float8 | YES | Geographic longitude (decimal degrees) |
| `first_active` | timestamp | YES | First recorded date station was active (German time) |
| `openingtimes_json` | jsonb | YES | Opening hours in JSON format (often empty: `{}`) |

Example row:
```
uuid: 44e2bdb7-13e3-4156-8576-8326cdd20459
name: bft Tankstelle
brand: NULL
street: Schellengasse
house_number: 53
post_code: 36304
city: Alsfeld
latitude: 50.7520089
longitude: 9.2790394
first_active: 1970-01-01 01:00:00
openingtimes_json: {}
```

**Schema of `prices` table**

Primary Key: `(date, station_uuid)` (composite)

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `date` | timestamp | NOT NULL | Price timestamp (German local time, no timezone) |
| `station_uuid` | uuid | NOT NULL | Unique station identifier from Tankerkoenig |
| `diesel` | numeric | YES | Diesel price in EUR/liter |
| `e5` | numeric | YES | E5 (Super) price in EUR/liter |
| `e10` | numeric | YES | E10 (Super E10) price in EUR/liter |
| `dieselchange` | int2 | YES | Diesel price change flag (0=no change, 1=increased, -1=decreased) |
| `e5change` | int2 | YES | E5 price change flag |
| `e10change` | int2 | YES | E10 price change flag |
| `is_synthetic` | bool | NO | Synthetic data row yes/no |

Example row:
```
date: 2025-11-14 00:01:26
station_uuid: d4a4644c-cd86-42a5-9cfa-ec6c58204312
diesel: 1.599
e5: 1.739
e10: 1.679
dieselchange: 1
e5change: 0
e10change: 0
is_synthetic: false
```

Change flag values: `0` = no change, `1` = price increased, `-1` = price decreased

**Why no Foreign Key constraint?**

The `prices` table does not have a foreign key constraint to `stations.uuid`. This is intentional: Tankerkoenig occasionally removes stations from their master list (e.g., station closes or rebrands), but historical price records may still exist in our 14-day window. Enforcing referential integrity would break the daily ingestion. JOINs work normally; orphaned records are simply excluded.

## 4) How it works (high level)

**route_stations.py** (Google Maps integration)
- Geocoding converts human-readable addresses to coordinates
- Driving route geometry and ETA are obtained from the Directions API; step polylines are decoded for map display and downsampled for Places API
- The Places API is queried along the route corridor (search-along-route). For each candidate station, a detour is computed by measuring the path Origin -> Station -> Destination versus the original route
- ETA at the station is estimated using step timing. Opening times are parsed to indicate accessibility at ETA
- Returns station records suitable for modeling and UI ranking

<picture>
  <source media="(prefers-color-scheme: light)" srcset="../../structure_graphs/light_theme_route_stations_workflow.drawio.svg">
  <img alt="Overview of route_stations.py workflow" src="../../structure_graphs/dark_theme_route_stations_workflow.drawio.svg">
</picture>

**update_prices_stations.py** (Tankerkoenig ingestion)
- A daily job downloads latest CSVs from the Tankerkoenig raw repository (authenticated via credentials in env), normalizes types and timestamps, and replaces the `stations` table (full refresh)
- Price records are cleaned, records older than 14 days are pruned (batched by hour to avoid timeouts), and synthetic rows are generated for 00:00-06:59 to ensure modeling coverage at early hours
- Insertions are batched (500 for stations, 1000 for prices) and logged with success rates

Notes:
- Both scripts expect environment variables (see Configuration). A `.env` file at repo root may be used for local testing.

## 5) Automation hooks

**update_prices_stations.py:**
- Intended schedule: Daily at 07:00 CET (after Tankerkoenig publishes yesterday's data)
- Logs written to: `~/logs/daily_update_YYYYMMDD.log`


**route_stations.py:**
- Called on-demand by the application backend when a user requests a route recommendation
- Can be called directly via Python (from project root):
  ```bash
  python -m src.integration.route_tankerkoenig_integration
  ```
  - Called automatically by the dashboard when a user plans a trip

## 6) Validation & quality checks
**`route_stations.py` validation**

- Environment validation:
  - `GOOGLE_MAPS_API_KEY` must be set; raises `ConfigError` if missing.
- Geocoding validation:
  - Query must return at least one result; raises `GeocodingError` if no results found.
  - Result must contain valid latitude and longitude coordinates; raises `GeocodingError` if missing.
  - Geocoding results that default to geographic center of Germany (lat=51.165691, lon=10.451526) are rejected; raises `GeocodingError`. This typically happens when the user inputs gibberish.
- Route validation:
  - Google Directions API must return a route; raises `ValueError` if no route found.
  - Route distance must be greater than zero; raises `ValueError` if distance <= 0.
  - Route duration must be greater than zero; raises `ValueError` if duration <= 0.
  - Route must contain decodable step polylines; raises `ValueError` if geometry is empty.
- Station validation:
  - Station candidates without coordinates (lat/lon) or outside of Germany are skipped (no error, filtered silently).
  - The distances and time durations between "Origin -> Station" and "Station -> Destination" must be >= 0 otherwise the station is skipped (no error, filtered silently).
  

**`update_prices_stations.py` validation**

- Environment validation:
  - Required env vars (`SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `TANKERKOENIG_EMAIL`, `TANKERKOENIG_API_KEY`) are validated at start; missing keys cause early exit with `sys.exit(1)`.
- CSV schema validation:
  - Downloaded CSVs are checked against expected column sets (`EXPECTED_STATIONS_COLUMNS`, `EXPECTED_PRICES_COLUMNS`).
  - Missing columns trigger a warning in the log but do not crash the script.
  - If required columns are actually missing, downstream `KeyError` will occur when accessing them.
- Data type coercion:
  - Numeric columns (`latitude`, `longitude`, `diesel`, `e5`, `e10`) are coerced via `pd.to_numeric(errors='coerce')`; invalid values become `NULL`.
  - Timestamps are stripped of timezone info and kept as German local time; malformed timestamps become `NULL`.
  - All `NaN` values are converted to `None` for Supabase compatibility.
- Operational checks:
  - Insert success rates and row counts are logged per batch.
  - A post-run summary shows total inserted vs. failed records.
  - Success threshold: >= 95% of records must be inserted for the job to be marked successful.

## 7) Error handling & troubleshooting

**Common failure modes and mitigations:**

- Missing environment variables -> fix `.env`. Example: `GOOGLE_MAPS_API_KEY` absence causes `ConfigError`.
- Supabase auth errors -> verify `SUPABASE_URL` and `SUPABASE_SECRET_KEY` and ensure service role has insert/delete privileges.
- Tankerkoenig CSV not found (HTTP 404) -> verify `TANKERKOENIG_EMAIL` and `TANKERKOENIG_API_KEY` and check the expected date path in the remote repo. Files are published with a 1-day delay, so yesterday's date is used.
- Deletion timeout on old prices -> missing index on `prices.date` column. Run: `CREATE INDEX idx_prices_date ON prices(date);`
- Schema warning in logs ("CSV schema change detected") -> Tankerkoenig may have changed their CSV format. Check if required columns are still present.

**Troubleshooting steps:**

1. Inspect logs at `~/logs/daily_update_YYYYMMDD.log`
2. Look for schema warnings or HTTP errors in the log output
3. Verify API credentials are valid and have sufficient quota
4. For Google API issues, check the Google Cloud Console for quota/billing status


## 8) Configuration (`.env.example`)

Create a `.env` file from the example below:

```ini
# Database (required)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SECRET_KEY=your-service-role-key

# Tankerkoenig (required for update_prices_stations.py)
TANKERKOENIG_API_KEY=your-tankerkoenig-api-key
TANKERKOENIG_EMAIL=your-registered-email

# Google Maps (required for route_stations.py)
GOOGLE_MAPS_API_KEY=your-google-api-key

# Mapbox (required for map visualization)
MAPBOX_API_KEY=your-mapbox-api-key
MAPBOX_ACCESS_TOKEN=your-mapbox-access-token

# Session persistence (optional)
UPSTASH_REDIS_URL=your-redis-url
```

## 9) Links

- [Back to Root README](../../README.md)
- Related components:
  - Integration: [../integration/README.md](../integration/README.md)
  - Modeling: [../modeling/README.md](../modeling/README.md)
  - App/UI: [../app/README.md](../app/README.md)

## Mini file tree

```
src/data_pipeline/
|-- README.md
|-- route_stations.py
|-- update_prices_stations.py
```