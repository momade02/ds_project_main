# ds_project_main
We build a route-aware refueling recommender for Germany that forecasts station-level prices at the ETA and then optimizes “where/when to stop” by trading cheaper price vs. detour/time and fuel constraints.

**Project overview**
| Step1 | Description2   | Directory3  | Ressources4     | Detailed Information5 |
|------|----------------|---------------|----------------|----------------------|
| Find fuel stations near route   | Geocode starting point and destination, search for route, find fuel stations near route    | route_stations.py    | Google APIs (Geocoding, Directions, Places)      | Zelle 1.5            |
| Training price prediction model    | Train price prediction model for each fuel type      | ./src/modeling/     | Tankerkönig (historical data)      | [./src/modeling/README.md](./src/modeling/README.md)            |
| 3    | Zelle 3.2      | Zelle 3.3     | Zelle 3.4      | Zelle 3.5            |
| 4    | Zelle 4.2      | Zelle 4.3     | Zelle 4.4      | Zelle 4.5            |
| 5    | Zelle 5.2      | Zelle 5.3     | Zelle 5.4      | Zelle 5.5            |


**Overview graph of the project structure**

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./structure_graphs/light_theme_workflow.drawio.svg">
  <img alt="Overview of the project structure" src="./structure_graphs/dark_theme_workflow.drawio.svg">
</picture>

ETA = Expected time of arrival