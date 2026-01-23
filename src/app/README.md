# App/UI Layer — Component README

Table of Contents  
1. [Purpose & quick summary](#1-purpose--quick-summary)  
2. [Where this fits in the pipeline](#2-where-this-fits-in-the-pipeline)  
3. [Inputs & Outputs](#3-inputs--outputs)  
4. [How it works (high level)](#4-how-it-works-high-level)  
5. [Automation hooks](#5-automation-hooks)  
6. [Validation & Quality checks](#6-validation--quality-checks)  
7. [Error handling & troubleshooting](#7-error-handling--troubleshooting)    
8. [Links](#8-links)  
9. [Mini file tree](#9-mini-file-tree)  

## 1) Purpose & quick summary

Responsible for orchestrating the end-to-end user experience through an interactive Streamlit dashboard that collects route inputs, presents predictions and recommendations, and provides detailed station analytics - all with minimal user effort.

For non-technical readers: This is the web interface where users plan trips, see recommended refueling stops, and explore detailed price history and station information.

## 2) Where this fits in the pipeline

- **Upstream:**  
  - `src/integration` layer provides enriched station data with routing features  
  - `src/modeling` layer supplies price predictions  
  - `src/decision` layer ranks stations and computes economics  
- **Downstream:**  
  - User browser (interactive dashboard)  
  - Redis persistence layer (session continuity across refreshes)  

## 3) Inputs & Outputs

All input parameters, UI components, and data transformations are extensively commented throughout the codebase to facilitate understanding. Additionally, all user inputs are explained directly within the UI via helper text, tooltips, and inline documentation to ensure users understand what each parameter does.

**Page 01 — Trip Planner (`streamlit_app.py`):**
- **Inputs:**
  - Route parameters: start/destination addresses, optional waypoints, departure time
  - Fuel selection: e5, e10, or diesel
  - Economics settings: litres to refuel, consumption rate, value of time
  - Constraints: max detour distance/time, min/max distance along route, brand filters, open-at-ETA requirements
  
- **Outputs:**
  - Interactive Mapbox map with route polyline and ranked station markers
  - Recommended station card with price, economics breakdown and address. 
  - Session state payload (`last_run`) persisted to Redis

**Page 02 — Route Analytics:**
- **Inputs:**
  - Session state (`last_run`) and user selections from Page 01

- **Outputs:**
  - Filter/audit log (inclusion/exclusion reasons with thresholds)
  - Route visualization (baseline vs. alternative with best station)
  - Comparison charts (price distribution, distance scatter, economics breakdown)
  - Drill-down links to Station Details (Page 03)
  - Value view table (all competitive stations)

**Page 03 — Station Details:**
- **Inputs:**
  - Station UUID (from Page01 or Page 04)
  - Fuel type selection (from session state or sidebar)
  - Optional: last_run context for route-specific metrics
  
- **Outputs:**
  - Station info card (name, brand, address, opening hours, open-at-ETA status)
  - price history chart (timeline with percentile bands)
  - Hourly price pattern chart (average, min, max by hour-of-day)
  - Weekday-hour heatmap (identifies cheapest/most expensive time windows)
  - Comparison mode (compare stations on same price chart)

**Page 04 — Station Explorer:**
- **Inputs:**
  - Location query (city name or address)
  - Search radius (km)
  - Fuel type, brand filter, open-only toggle
  
- **Outputs:**
  - Summary metrics (stations found, open count, cheapest price)
  - Interactive map with all search results (best station highlighted)
  - Station selector dropdown (sorted by price)
  - Results table (name, address, price, distance, open status)

**Cross-page persistence:**
- Session state stored in Redis with 12-hour TTL (configurable)
- Shared state includes: last_run, user preferences, selected stations, comparison selections

## 4) How it works (high level)

**Code documentation:** All code throughout this component is extensively commented with inline documentation explaining function logic, variables and rules. For low-level implementation details refer directly to the source code files listed in the [Mini file tree](#9-mini-file-tree) section—each file contains detailed comments.

**Architecture:**
- Multi-page Streamlit application with shared session state and Redis-backed persistence
- Service layer orchestrates upstream components (integration → modeling → decision)
- Presenter layer transforms data structures into user-friendly display formats
- UI components provide consistent styling, maps (Mapbox GL), and interactive elements

**Page-level flow:**

*Page 01 — Trip Planner (streamlit_app.py):*
1. Collect user inputs via sidebar (addresses, fuel type, constraints)
2. Call `route_recommender.run_route_recommendation(...)` to execute full pipeline
3. Store results in `st.session_state["last_run"]` and persist to Redis
4. Render interactive map with route polyline and station markers
5. Display ranked station list with economic metrics and selection actions

*Page 02 — Route Analytics:*
1. Load `last_run` from session state
2. Display filter/audit log explaining station inclusion/exclusion
3. Render comparison charts (price, distance, economics)
4. Provide drill-down links to Station Details

*Page 03 — Station Details:*
1. Accept station UUID from page navigation or direct URL parameter
2. Fetch 14-day price history via `integration.historical_data`
3. Display interactive charts (price timeline, hourly patterns, percentile bands)
4. Show station metadata and quick route planning options

*Page 04 — Station Explorer:*
1. Accept location query and search radius
2. Fetch nearby stations via `station_explorer.search_stations_by_location(...)`
3. Display grid of stations with realtime prices (if enabled)
4. Provide quick-route buttons to launch Trip Planner with selected station

**Key services:**

- `route_recommender.py`: Orchestrates full run (integration → ranking → payload assembly)
- `presenters.py`: Transforms station dicts into display-friendly formats
- `session_store.py`: Redis-backed persistence for session continuity
- `station_explorer.py`: Location-based station search

**UI components:**

- `ui/maps.py`: Mapbox GL rendering (route polylines, station markers, popups)
- `ui/sidebar.py`: Reusable input shells, station selectors, quick settings
- `ui/formatting.py`: Price/distance/time formatting, safe text escaping
- `ui/styles.py`: CSS injection, consistent visual theming

## 5) Automation hooks

- **Direct execution:**
  ```bash
  streamlit run src/app/streamlit_app.py
  ```
- **Docker deployment:** Exposed on port 8501 (configured via `.streamlit/config.toml`)
- **Automated session recovery:** Redis persistence restores user state across refreshes

## 6) Validation & Quality checks

- **Input validation:**
  - Address geocoding verification (reject invalid/ambiguous locations)
  - Numeric bounds enforcement (litres ≥ 1, consumption ≥ 0, detour distance ≥ 0.5 km, detour time ≥ 1 min)

- **UX safeguards:**
  - Graceful degradation when Redis unavailable (session state only)
  - Fallback to price-only mode when economics inputs incomplete
  - Clear error messages for API failures (Google Maps, Tankerkönig)
  - Debug mode audit logs for transparency

## 7) Error handling & troubleshooting

**Common issues:**

1. **"No stations found along route"**
   - Cause: Route too short, no stations within search radius, route not in Germany
   - Fix: Extend route, increase search radius, use route in Germany

2. **"Unable to geocode address"**
   - Cause: Invalid/ambiguous location or no geocode results
   - Check address spelling

3. **"Redis connection failed"**
   - Cause: Azure Cache for Redis credentials missing or incorrect
   - Fix: Verify `REDIS_HOST`, `REDIS_PASSWORD` in `.env`; app will fall back to session-only mode

4. **"Missing price predictions"**
   - Cause: Model files not loaded or upstream integration failure
   - Fix: Ensure `src/modeling/models/*.joblib` files exist; check logs for modeling errors

**Debugging steps:**
1. Enable debug mode in sidebar (expands audit logs and shows raw data structures)
2. Check browser console for JavaScript errors (Mapbox GL issues)
3. Inspect `st.session_state["last_run"]` in debug view for payload structure
4. Review terminal logs for upstream errors (integration, modeling, decision)

## 8) Links

- [Back to Root README](../../README.md)
- Related components:
  - Data Pipeline: [../data_pipeline/README.md](../data_pipeline/README.md)
  - Integration: [../integration/README.md](../integration/README.md)
  - Modeling: [../modeling/README.md](../modeling/README.md)
  - Decision: [../decision/README.md](../decision/README.md)

## 9) Mini file tree

```
src/app/
├─ README.md                          # This file
├─ streamlit_app.py                   # Page 01: Trip Planner (main entry point)
├─ app_errors.py                      # Custom exception types
├─ .streamlit/
│  └─ config.toml                     # Streamlit server configuration
├─ config/
│  ├─ __init__.py
│  └─ settings.py                     # Environment/secrets resolution, Redis config
├─ pages/
│  ├─ 02_route_analytics.py          # Page 02: Audit logs, comparisons
│  ├─ 03_station_details.py          # Page 03: Price history, analytics
│  └─ 04_station_explorer.py         # Page 04: Location-based search
├─ services/
│  ├─ __init__.py
│  ├─ route_recommender.py           # Run orchestration (integration → decision)
│  ├─ presenters.py                  # Data transformation for display
│  ├─ session_store.py               # Redis-backed persistence
│  └─ station_explorer.py            # Location search logic
├─ static/
│  └─ app.css                         # Custom CSS overrides
└─ ui/
   ├─ formatting.py                   # Price/distance/time formatters
   ├─ maps.py                         # Mapbox GL rendering
   ├─ sidebar.py                      # Reusable input components
   └─ styles.py                       # CSS injection helpers
```
