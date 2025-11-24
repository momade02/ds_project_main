"""
Decision layer for the route-aware refuelling recommender.

This module takes the integrated + predicted station data and derives
actionable recommendations, such as:

    - Which station along the route is best for a given fuel type?
    - How do stations rank by predicted price?

The code is deliberately lightweight and purely in-memory:

    - It operates on the list-of-dicts structure returned by
      `get_fuel_prices_for_route(...)` (real system) or `run_example()`
      (test environment) from `route_tankerkoenig_integration.py`.

    - It does NOT talk to Supabase, ORS, or any external API.
      All data access happens earlier in the pipeline.

By design, the same functions can be used:

    - in the current TEST environment (using `run_example()`)
    - in the REAL system, once ORS POI is working again

The recommendation strategy is intentionally simple and transparent:

    1) Rank stations by predicted price for the chosen fuel type
       (ascending: cheaper is better).

    2) As tie-breakers, prefer stations that appear earlier along the route
       (smaller `fraction_of_route`, then smaller `distance_along_m`).

This is easy to explain and can later be extended (e.g. with detour penalties,
user preferences, time windows, etc.) without changing the public API.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import pandas as pd

from ..modeling.model import FUEL_TYPES
from ..modeling.predict import predict_for_fuel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prediction_key(fuel_type: str) -> str:
    """
    Build the dictionary key used for predictions of a given fuel type.
    """
    return f"pred_price_{fuel_type.lower().strip()}"


def _normalise_fuel_type(fuel_type: str) -> str:
    """
    Normalise and validate a fuel type string.
    """
    if fuel_type is None:
        raise ValueError("fuel_type must not be None")

    ft = fuel_type.lower().strip()
    if ft not in FUEL_TYPES:
        valid = ", ".join(FUEL_TYPES)
        raise ValueError(f"Unsupported fuel_type '{fuel_type}'. "
                         f"Expected one of: {valid}")
    return ft


def _ensure_predictions(model_input: List[Dict], fuel_type: str) -> None:
    """
    Ensure that predictions exist for the given fuel type.

    If the prediction key is missing entirely or all values are None,
    this function calls `predict_for_fuel` to compute them.

    This allows the decision layer to work both in situations where the
    application has already run predictions explicitly and in situations
    where the caller simply hands over the raw integration output.
    """
    ft = _normalise_fuel_type(fuel_type)
    key = _prediction_key(ft)

    if not model_input:
        return

    # Check if the key exists and has at least one non-None value
    key_exists = any(key in s for s in model_input)
    if key_exists:
        any_non_null = any(s.get(key) is not None for s in model_input)
    else:
        any_non_null = False

    if not any_non_null:
        # Compute predictions in-place
        predict_for_fuel(model_input, ft, prediction_key=key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_stations_by_predicted_price(
    model_input: List[Dict],
    fuel_type: str,
) -> List[Dict]:
    """
    Rank stations along the route by predicted price for a given fuel type.

    The ranking logic is:

        1) Lowest predicted price is best.
        2) For equal predicted price, prefer stations earlier on the route
           (smaller `fraction_of_route`).
        3) If still tied, prefer stations with smaller `distance_along_m`
           (slightly shorter effective route distance).

    Parameters
    ----------
    model_input : list of dict
        Stations with integration data, as produced by
        `get_fuel_prices_for_route(...)` or `run_example()`.
        Predictions may or may not yet be present.
    fuel_type : str
        One of 'e5', 'e10', 'diesel' (case-insensitive).

    Returns
    -------
    list of dict
        New list containing the same station dicts, ordered from best to worst
        according to the ranking criteria. Stations without a valid prediction
        are dropped from the ranking.
    """
    if not model_input:
        return []

    ft = _normalise_fuel_type(fuel_type)
    pred_key = _prediction_key(ft)

    # Make sure predictions exist (compute them on the fly if necessary)
    _ensure_predictions(model_input, ft)

    # Convert to DataFrame for convenient sorting
    df = pd.DataFrame(model_input)

    if pred_key not in df.columns:
        # Nothing to rank on
        return []

    # Keep only stations with a numerical prediction
    df = df[df[pred_key].notna()].copy()
    if df.empty:
        return []

    # Ensure tie-breaker columns exist; if not, create neutral defaults.
    if "fraction_of_route" not in df.columns:
        df["fraction_of_route"] = 1.0  # all at end of route
    if "distance_along_m" not in df.columns:
        df["distance_along_m"] = 0.0

    # Sort by prediction, then by position along route
    df_sorted = df.sort_values(
        by=[pred_key, "fraction_of_route", "distance_along_m"],
        ascending=[True, True, True],
        kind="mergesort",  # stable sort (keeps original order where equal)
    )

    # Rebuild a list of dicts in sorted order
    ranked_indices = df_sorted.index.tolist()
    ranked_stations = [model_input[i] for i in ranked_indices]

    return ranked_stations


def recommend_best_station(
    model_input: List[Dict],
    fuel_type: str,
) -> Optional[Dict]:
    """
    Return the single best station for a given fuel type.

    This is simply the first element of the ranking produced by
    `rank_stations_by_predicted_price`. If no station has a valid prediction,
    None is returned.

    Parameters
    ----------
    model_input : list of dict
        Stations with integration data and (optionally) predictions.
    fuel_type : str
        One of 'e5', 'e10', 'diesel' (case-insensitive).

    Returns
    -------
    dict or None
        The station dict that minimises predicted price for the chosen fuel
        (subject to the tie-breaking rules), or None if no valid prediction
        is available.
    """
    ranked = rank_stations_by_predicted_price(model_input, fuel_type)
    if not ranked:
        return None
    return ranked[0]
