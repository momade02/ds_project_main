# src/app/services/route_recommender.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.app.app_errors import AppError
# Import your existing core functions exactly as you currently do in streamlit_app.py.
# The names below must match your project.
from src.integration.route_tankerkoenig_integration import get_fuel_prices_for_route
from src.decision.recommender import rank_stations_by_predicted_price


@dataclass(frozen=True)
class RouteRunInputs:
    """
    Container for the inputs that define one route run. Keep this aligned with what
    streamlit_app.py already collects.
    """
    start: str
    end: str
    fuel_code: str
    litres_to_refuel: Optional[float]
    use_economics: bool
    debug_mode: bool

    # Pass-through knobs that already exist in your UI (optional)
    # Add fields only when you are ready to pass them in.
    car_consumption_l_per_100km: Optional[float] = None
    value_of_time_eur_per_h: Optional[float] = None
    max_detour_time_min: Optional[float] = None
    max_detour_distance_km: Optional[float] = None
    min_net_saving_eur: Optional[float] = None


def run_route_recommendation(
    inputs: RouteRunInputs,
    *,
    # keep these as dicts so you can pass exactly what you already pass today
    integration_kwargs: Optional[Dict[str, Any]] = None,
    ranking_kwargs: Optional[Dict[str, Any]] = None,
    recommendation_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Single use-case entry point for the Trip Planner page.

    Returns the 'last_run' dict in the exact structure your UI already expects
    (stations, ranked, best_station, route_coords, params, fuel_code, etc.).
    """
    # Progress reporting removed: run silently and return results.

    integration_kwargs = dict(integration_kwargs or {})
    ranking_kwargs = dict(ranking_kwargs or {})
    recommendation_kwargs = dict(recommendation_kwargs or {})

    # Force minimum required fields using the SAME parameter names as streamlit_app.py
    # (get_fuel_prices_for_route does NOT accept 'start'/'end')
    integration_kwargs.setdefault("start_locality", inputs.start)
    integration_kwargs.setdefault("end_locality", inputs.end)

    # Addresses are optional in your UI; set safe defaults if not provided
    integration_kwargs.setdefault("start_address", "")
    integration_kwargs.setdefault("end_address", "")

    # Your UI explicitly sets this to True
    integration_kwargs.setdefault("use_realtime", True)

    # IMPORTANT: do NOT inject fuel_code/debug_mode here unless your integration function
    # explicitly supports them (your UI currently does not pass them).

    try:
        # --- This call MUST stay identical to your current logic ---
        # It should return whatever your current integration function returns.
        # Core integration: geocoding, route calculation, place search and
        # enrichment with historical + realtime prices.
        stations, route_info = get_fuel_prices_for_route(**integration_kwargs)

        route_coords = route_info.get("route_coords") if isinstance(route_info, dict) else None

        # recommender.py expects 'fuel_type' (positional or keyword), not 'fuel_code'
        audit_log = {}
        
        # Ensure economics parameters are passed when economics mode is enabled
        if inputs.use_economics:
            # Only set defaults if caller did not already provide them
            ranking_kwargs.setdefault("litres_to_refuel", inputs.litres_to_refuel)
            ranking_kwargs.setdefault("consumption_l_per_100km", inputs.car_consumption_l_per_100km)
            ranking_kwargs.setdefault("value_of_time_per_hour", float(inputs.value_of_time_eur_per_h or 0.0))
            ranking_kwargs.setdefault("max_detour_km", inputs.max_detour_distance_km)
            ranking_kwargs.setdefault("max_detour_min", inputs.max_detour_time_min)
            ranking_kwargs.setdefault("min_net_saving_eur", float(inputs.min_net_saving_eur or 0.0))

        ranked = rank_stations_by_predicted_price(
            stations,
            inputs.fuel_code, 
            audit_log=audit_log,  # <- fuel_type positional argument
            **ranking_kwargs,
        )
        # Ranking complete — recommendation ready

        # recommend_best_station() is just a wrapper around rank_stations_by_predicted_price()
        # and does NOT accept 'use_economics'. Since we already ranked, pick the top element.
        best_station = ranked[0] if ranked else None


        best_uuid = (best_station.get("uuid") or best_station.get("id") or best_station.get("station_uuid")) if isinstance(best_station, dict) else None

        # IMPORTANT: Return the same structure your streamlit_app.py currently stores.
        # Add/remove keys only if your current last_run uses them.
        last_run: Dict[str, Any] = {
            "stations": stations,
            "ranked": ranked,
            "best_station": best_station,
            "best_uuid": best_uuid,
            "route_info": route_info,
            "route_coords": route_coords,
            "fuel_code": inputs.fuel_code,
            "litres_to_refuel": inputs.litres_to_refuel,
            "use_economics": inputs.use_economics,
            "debug_mode": inputs.debug_mode,
            "filter_log": audit_log,
        }

        return last_run

    except AppError:
        # Preserve your custom error types unchanged
        raise
    except Exception as e:
        # Do not change error semantics yet; wrap only if you already do so in the UI.
        # For now, re-raise to keep behavior identical (Streamlit will show stack trace in dev).
        raise e
