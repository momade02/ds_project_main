# ds_project_main
We build a route-aware refueling recommender for Germany that forecasts station-level prices at the ETA and then optimizes “where/when to stop” by trading cheaper price vs. detour/time and fuel constraints.

**Project overview**
| Step1 | Description2   | Directory3  | Ressources4     | Detailed Information5 |
|------|----------------|---------------|----------------|----------------------|
| Data acquisition  | Responsible for acquisition and preparation of station-location and fuel-price data, enriching station records with route, detour and ETA context for downstream modeling and the decision/UI layers.  |./src/data_pipeline/ | Google APIs (Geocoding, Directions, Places), Tankerkönig (Daily Station & Price CSV), Supabase (managed PostgreSQL database)      | [./src/data_pipeline/README.md](./src/data_pipeline/README.md)|
| Merge of Datasets, Create Feature Vector for Model | Responsible for enriching fuel stations with historical and real-time fuel price data and producing feature vectors for downstream modeling and dashboard presentation. | ./src/integration/   |Tankerkönig (API), Supabase (managed PostgreSQL database)| [./src/integration/README.md](./src/integration/README.md)      |
| Price prediction model    | Responsible for predictions of fuel prices at gas stations using fitted ARDL (Autoregressive Distributed Lag) models.    | ./src/modeling/     | Tankerkönig (historical data)      | [./src/modeling/README.md](./src/modeling/README.md)            |
|  Fuel station Recommendation  | Responsible for selection and ranking refueling stations using predicted prices, detour costs, and user constraints to recommend the best stop.    |   ./src/decision/  | - |  [src/decision/README.md](./src/decision/README.md)       |
| 5    | Zelle 5.2      | ./src/app/      |   Streamlit, Microsoft Azure  | [./src/app/README.md](./src/app/README.md)   |


**Overview graph of the project structure**

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./structure_graphs/light_theme_workflow.drawio.svg">
  <img alt="Overview of the project structure" src="./structure_graphs/dark_theme_workflow.drawio.svg">
</picture>

ETA = Expected time of arrival, ARDL = Autoregressive Distributed Lag