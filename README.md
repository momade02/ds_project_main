# ds_project_main
We build a route-aware refueling recommender for Germany that forecasts station-level prices at the ETA and then optimizes “where/when to stop” by trading cheaper price vs. detour/time and fuel constraints.

**Project overview**
| Step1 | Description2   | Directory3  | Ressources4     | Detailed Information5 |
|------|----------------|---------------|----------------|----------------------|
| Data Input    | The `data_pipeline` component is responsible for acquisition and preparation of station-location and fuel-price data, enriching station records with route, detour and ETA context for downstream modeling and the decision/UI layers.  |./src/data_pipeline/ | Google APIs (Geocoding, Directions, Places), Tankerkönig (Daily Station & Price CSV), Supabase (managed PostgreSQL database)      | [./src/data_pipeline/README.md](./src/data_pipeline/README.md)|
| Training price prediction model    | Train price prediction model for each fuel type      | ./src/modeling/     | Tankerkönig (historical data)      | [./src/modeling/README.md](./src/modeling/README.md)            |
| 3    | Zelle 3.2      | ./src/app/     | Zelle 3.4|  [./src/app/README.md](./src/app/README.md)         |
| 4    | Zelle 4.2      | ./src/integration/   |Zelle 4.4| [./src/integration/README.md](./src/integration/README.md)                 |
| 5    | Zelle 5.2      | ./src/decision/     |   5.4  | [src/decision/README.md](./src/decision/README.md)          |
| 6    | Zelle 6.2      | Zelle 6.3     |  6.4   |  |


**Overview graph of the project structure**

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./structure_graphs/light_theme_workflow.drawio.svg">
  <img alt="Overview of the project structure" src="./structure_graphs/dark_theme_workflow.drawio.svg">
</picture>

ETA = Expected time of arrival, ARDL = Autoregressive Distributed Lag