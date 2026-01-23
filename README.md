# Tankstoppfinder

**Project overview**
| Step1 | Description2   | Directory3  | Ressources4     | Detailed Information5 |
|------|----------------|---------------|----------------|----------------------|
| Data acquisition  | Responsible for acquisition and preparation of station-location and fuel-price data, enriching station records with route, detour and ETA context for downstream modeling and the decision/UI layers.  |./src/data_pipeline/ | Google APIs (Geocoding, Directions, Places), Tankerkönig (Daily Station & Price CSV), Supabase (managed PostgreSQL database)      | [./src/data_pipeline/README.md](./src/data_pipeline/README.md)|
| Merge of Datasets, Create Feature Vector for Model | Responsible for enriching fuel stations with historical and real-time fuel price data and producing feature vectors for downstream modeling and dashboard presentation. | ./src/integration/   |Tankerkönig (API), Supabase (managed PostgreSQL database)| [./src/integration/README.md](./src/integration/README.md)      |
| Price prediction model    | Responsible for predictions of fuel prices at gas stations using fitted ARDL (Autoregressive Distributed Lag) models.    | ./src/modeling/     | Tankerkönig (historical data)      | [./src/modeling/README.md](./src/modeling/README.md)            |
|  Fuel station Recommendation  | Responsible for selection and ranking refueling stations using predicted prices, detour costs, and user constraints to recommend the best stop.    |   ./src/decision/  | - |  [src/decision/README.md](./src/decision/README.md)       |
| 5    | Zelle 5.2      | ./src/app/      |   Streamlit, Microsoft Azure  | [./src/app/README.md](./src/app/README.md)   |

🔗 **Live Demo:** [www.tankstoppfinder.de](https://www.tankstoppfinder.de)


## Table of Contents
1. [Our Unique Approach to Smarter Refueling](#1-our-unique-approach-to-smarter-refueling)
2. [Project Overview](#2-project-overview)
3. [Tech Stack](#3-tech-stack)
4. [Getting Started](#4-getting-started)
5. [Project Structure](#5-project-structure)
6. [Data Pipeline & Automation](#6-data-pipeline--automation)
7. [Validation](#7-validation)
8. [Component Documentation](#component-documentation)
9. [Acknowledgments](#acknowledgments)

---

## 1) Our Unique Approach to Smarter Refueling

**Problem:** Fuel prices in Germany fluctuate considerably throughout the day and vary by location, creating a complex decision for drivers: Should they refuel now, or wait for a potentially cheaper station along their route? Additionally, drivers must consider whether the detour is worthwhile for the savings on fuel costs.

**Solution:** Tankstoppfinder combines real-time and historical fuel price data with route planning to recommend the most economical refueling stop. The system:
- Predicts fuel prices at each candidate station for your estimated arrival time (ETA)
- Calculates the true cost of each option (price savings vs. detour fuel/time costs)
- Ranks stations by net economic benefit

**What makes this novel:**
- Unlike simple price comparison apps, we forecast prices at *future* arrival times using ARDL models
- Economic ranking accounts for detour costs (fuel consumption + time value)
- Our econometrics model is fully transparent and comprehensible 


## 2) Project overview

| Component | Overview   | Directory  | Readme |
|------|----------------|-------------|-----------------------|
| Data acquisition  | Responsible for acquisition and preparation of station-location and fuel-price data, enriching station records with route, detour and ETA context for downstream modeling and the decision/UI layers.  | ./src/data_pipeline/ | [./src/data_pipeline/README.md](./src/data_pipeline/README.md) |
| Merge of Datasets, Create Feature Vector for Model | Responsible for enriching fuel stations with historical and real-time fuel price data and producing feature vectors for downstream modeling and dashboard presentation. | ./src/integration/   | [./src/integration/README.md](./src/integration/README.md) |
| Price prediction model    | Responsible for predictions of fuel prices at gas stations using fitted ARDL (Autoregressive Distributed Lag) models.    | ./src/modeling/     | [./src/modeling/README.md](./src/modeling/README.md) |
| Fuel station Recommendation  | Responsible for selection and ranking refueling stations using predicted prices, detour costs, and user constraints to recommend the best stop.    | ./src/decision/  | [./src/decision/README.md](./src/decision/README.md) |
| 5    | Zelle 5.2      | ./src/app/      | [./src/app/README.md](./src/app/README.md) |

\
**Overview graph of the project structure**

The system follows a modular pipeline architecture:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./structure_graphs/light_theme_workflow.drawio.svg">
  <img alt="Overview of the project structure" src="./structure_graphs/dark_theme_workflow.drawio.svg">
</picture>
ETA = Expected time of arrival, ARDL = Autoregressive Distributed Lag

## 3) Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Streamlit (Multi-Page App), Mapbox (Map provider) |
| **Hosting** | Microsoft Azure (App Service) (Fallback: Streamlit Cloud) |
| **Database** | Supabase (PostgreSQL) |
| **Data Integration** | AWS EC2 (Daily Cronjob) |
| **Session Store** | Redis (Upstash) |
| **APIs/Data Aquisition** | Google APIs (Geocoding, Directions, Places), Tankerkönig (Daily Station & Price CSV, API) |
| **Econometric Models** | scikit-learn (ARDL via OLS), joblib serialization, Tankerkönig (historic data) |


## 4) Getting Started

### Prerequisites
- Python 3.10+
- API keys for Google Maps and Tankerkönig
- Supabase project with `stations` and `prices` tables

### Quick Start (Local Development)

```bash
# 1. Clone the repository
git clone https://github.com/[your-org]/ds_project_main.git
cd ds_project_main

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys (see Configuration below)

# 5. Run the app
streamlit run src/app/streamlit_app.py
```

### Configuration (`.env`)

```ini
# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SECRET_KEY=your-service-role-key

# APIs
GOOGLE_MAPS_API_KEY=your-google-api-key
TANKERKOENIG_API_KEY=your-tankerkoenig-api-key
TANKERKOENIG_EMAIL=your-email
MAPBOX_API_KEY=pk.eyJ1IjoibW9yaXR6bWFpZGwiLCJhIjoiY21rMTd6eWZ4MDM1ZTNzcXhwMTQ1bGoweSJ9.d4F_dBOh69xJjJhDVYW4nw
MAPBOX_ACCESS_TOKEN=pk.eyJ1IjoibW9yaXR6bWFpZGwiLCJhIjoiY21rMTd6eWZ4MDM1ZTNzcXhwMTQ1bGoweSJ9.d4F_dBOh69xJjJhDVYW4nw

# Session persistence (optional)
UPSTASH_REDIS_URL=your-redis-url
```

---

## 5) Project Structure

```
ds_project_main/
├── src/
│   ├── data_pipeline/                           # Data acquisition & preparation
│   │   ├── route_stations.py                    # Google Maps integration
│   │   └── update_prices_stations.py            # Daily Tankerkönig ingestion
│   │
│   ├── integration/                             # Data enrichment & feature engineering
│   │   ├── route_tankerkoenig_integration.py    # Station matching & lag features
│   │   └── historical_data.py                   # Price history queries
│   │
│   ├── modeling/                                # Price prediction
│   │   ├── model.py                             # Model loading & caching
│   │   ├── predict.py                           # Inference logic
│   │   └── models/                              # Serialized ARDL models (.joblib)
│   │
│   ├── decision/                                # Recommendation logic
│   │   └── recommender.py                       # Economic ranking & optimization
│   │
│   └── app/                                     # Streamlit frontend
│       ├── streamlit_app.py                     # Main page (Trip Planner)
│       ├── pages/
│       │   ├── 02_route_analytics.py
│       │   ├── 03_station_details.py
│       │   └── 04_station_explorer.py
│       ├── services/                            # Backend services
│       ├── ui/                                  # UI components
│       └── config/                              # App configuration
│
├── structure_graphs/                            # Architecture diagrams
├── requirements.txt
├── .env.example
└── README.md
```


## 6) Data Pipeline & Automation

### Daily Data Refresh

The system maintains a rolling 14-day price history via automated daily updates:

| Schedule | Script | Action |
|----------|--------|--------|
| 07:00 CET | `update_prices_stations.py` | Download Tankerkönig CSVs → Refresh `stations` table → Update `prices` table → Prune records >14 days → Generate synthetic early-morning data |

**Why synthetic data?** Tankerkönig publishes price dumps with a ~7-hour delay. To ensure the app works 24/7, we generate synthetic records for 00:00–06:59 by cloning time-matched data from 48 hours prior.

### Database Schema
 
| Table | Primary Key | Rows | Update Strategy |
|-------|-------------|------|-----------------|
| `stations` | `uuid` | ~17,700 | Full refresh daily |
| `prices` | `(date, station_uuid)` | ~6–7M | Rolling 14-day window |
 
> **Note:** No foreign key constraint exists between `prices.station_uuid` and `stations.uuid`. 
> This is intentional: Tankerkönig occasionally removes stations from their dataset while historical 
> price records for those stations still exist in the 14-day window. Enforcing referential integrity 
> would break the daily ingestion pipeline. JOINs work normally; orphaned records are simply excluded.
 
For detailed column schemas, see [Data Pipeline README](./src/data_pipeline/README.md).

## 7) Validation

Each component README contains a section dedicated to data validation. For more details, please refer to that section. The checks ensure the plausibility of data and verify the presence of required inputs.


## 8) Acknowledgments

This project was developed as part of the **DS500 Data Science Project** at the University of Tübingen (Winter Term 2025/26).

**Primary Data Sources:**
- [Tankerkönig](https://creativecommons.tankerkoenig.de/) – German fuel price data (CC BY 4.0)
- [Google Maps Platform](https://developers.google.com/maps) – Geocoding, Directions, Places APIs

---

*For questions or issues, please open a GitHub issue or contact the project team.*