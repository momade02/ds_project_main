"""
Decision layer for the route-aware refuelling recommender.

This module operates purely in-memory on the list-of-dicts structure
returned by the integration layer (`route_tankerkoenig_integration`).

Responsibilities
----------------
* Ensure that ARDL predictions exist for the chosen fuel type.
* Rank stations along the route by predicted price.
* Select the single best station for a given fuel type.

The code is agnostic to whether we are in:
    - test mode (example route, historical-only),
    - or real mode (ORS pipeline with Supabase + Tankerkönig).

As long as the input is a list of dictionaries with the expected keys,
the functions below will work.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Any

from src.modeling.model import FUEL_TYPES
from src.modeling.predict import predict_for_fuel


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _prediction_key(fuel_type: str) -> str:
    """
    Build the dictionary key used for predictions of a given fuel type.

    Examples
    --------
    >>> _prediction_key("e5")
    'pred_price_e5'
    """
    return f"pred_price_{fuel_type.lower().strip()}"


def _normalise_fuel_type(fuel_type: str) -> str:
    """
    Normalise and validate a fuel type string.

    Returns the canonical lower-case form and raises ValueError for
    unsupported fuel types.
    """
    if fuel_type is None:
        raise ValueError("fuel_type must not be None")

    ft = fuel_type.lower().strip()
    if ft not in FUEL_TYPES:
        valid = ", ".join(FUEL_TYPES)
        raise ValueError(
            f"Unsupported fuel_type '{fuel_type}'. Expected one of: {valid}"
        )
    return ft


def _ensure_predictions(
    model_input: List[Dict[str, Any]],
    fuel_type: str,
    *,
    now: datetime | None = None,
) -> None:
    """
    Ensure that ARDL predictions exist for the given fuel type.

    If the prediction key is missing entirely or all values are None,
    this function calls :func:`predict_for_fuel` to compute them in-place.

    Parameters
    ----------
    model_input :
        List of station dictionaries produced by the integration pipeline.
    fuel_type :
        Fuel type string, e.g. 'e5', 'e10', 'diesel'.
    now :
        Optional "now" in UTC to be passed down into `predict_for_fuel`,
        so that all ETA-based horizon decisions within one ranking call
        share the same reference time.
    """
    if not model_input:
        return

    ft = _normalise_fuel_type(fuel_type)
    pred_key = _prediction_key(ft)

    # Check whether we already have at least one non-null prediction
    has_any_prediction = any(
        (station.get(pred_key) is not None) for station in model_input
    )

    if not has_any_prediction:
        # This mutates model_input in-place
        predict_for_fuel(model_input, ft, prediction_key=pred_key, now=now)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_stations_by_predicted_price(
    model_input: List[Dict[str, Any]],
    fuel_type: str,
    now: datetime | None = None,
) -> List[Dict[str, Any]]:
    """
    Rank stations along the route by predicted price for a given fuel type.

    Ranking logic
    -------------
    1. Lowest predicted price (for the chosen fuel) is best.
    2. If prices are equal, prefer stations earlier on the route
       (smaller ``fraction_of_route``).
    3. If still tied, prefer smaller ``distance_along_m``
       (slightly shorter absolute distance along the route).

    Parameters
    ----------
    model_input :
        List of station dictionaries produced by the integration pipeline.
    fuel_type :
        One of ``'e5'``, ``'e10'``, ``'diesel'`` (case-insensitive).
    now :
        Optional timestamp representing "now" in UTC. If omitted, the
        current time is taken at the moment of the call.

    Returns
    -------
    list of dict
        New list of station dictionaries sorted from cheapest to most
        expensive according to the rules above. Stations without a valid
        prediction are excluded.
    """
    if not model_input:
        return []

    ft = _normalise_fuel_type(fuel_type)
    pred_key = _prediction_key(ft)

    # Use a single "now" for all stations within this ranking call
    if now is None:
        now = datetime.now(timezone.utc)

    # Make sure we have predictions for this fuel
    _ensure_predictions(model_input, ft, now=now)

    # Filter stations with valid predictions
    valid_stations: List[Dict[str, Any]] = [
        s for s in model_input if s.get(pred_key) is not None
    ]

    if not valid_stations:
        return []

    def sort_key(station: Dict[str, Any]):
        # Predicted price (primary key)
        price = station.get(pred_key, float("inf"))
        # Fraction of route (0 = start, 1 = end)
        frac = station.get("fraction_of_route", float("inf"))
        # Absolute distance along route in metres
        dist = station.get("distance_along_m", float("inf"))
        return (price, frac, dist)

    ranked = sorted(valid_stations, key=sort_key)
    return ranked


def recommend_best_station(
    model_input: List[Dict[str, Any]],
    fuel_type: str,
    now: datetime | None = None,
) -> Dict[str, Any] | None:
    """
    Select the single best station for a given fuel type.

    The "best" station is simply the first element of the ranking produced
    by :func:`rank_stations_by_predicted_price`.

    Parameters
    ----------
    model_input :
        List of station dictionaries produced by the integration pipeline.
    fuel_type :
        One of ``'e5'``, ``'e10'``, ``'diesel'`` (case-insensitive).
    now :
        Optional timestamp representing "now" in UTC. If omitted, the
        current time is taken at the moment of the call.

    Returns
    -------
    dict or None
        The station dict that minimises predicted price for the chosen fuel
        (subject to the tie-breaking rules), or ``None`` if no valid
        prediction is available.
    """
    ranked = rank_stations_by_predicted_price(model_input, fuel_type, now=now)
    if not ranked:
        return None
    return ranked[0]
