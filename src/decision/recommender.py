"""
Decision layer for the route-aware refuelling recommender.

This module operates purely in-memory on the list-of-dicts structure
returned by the integration layer (`route_tankerkoenig_integration`).

Responsibilities
----------------
* Ensure that ARDL predictions exist for the chosen fuel type.
* Optionally apply an economic detour filter and ranking:
    - baseline price on the direct route,
    - gross saving vs baseline,
    - detour fuel and time cost,
    - net saving and break-even litres.
* Rank stations along the route.
* Select the single best station for a given fuel type.

If the economic parameters are not provided (litres_to_refuel or
consumption_l_per_100km is None), the module falls back to the
classical "cheapest predicted price" ranking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from src.modeling.model import FUEL_TYPES
from src.modeling.predict import predict_for_fuel


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# What counts as "on route" when we search for the baseline price:
# very small detour only (essentially the direct route).
ONROUTE_MAX_DETOUR_KM = 0.5
ONROUTE_MAX_DETOUR_MIN = 3.0

# Hard safety caps for detours. These are defaults; they can be
# overridden via function arguments.
DEFAULT_MAX_DETOUR_KM = 10.0
DEFAULT_MAX_DETOUR_MIN = 10.0


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


def _get_detour_metrics(station: Dict[str, Any]) -> tuple[float, float]:
    """
    Return (detour_distance_km, detour_duration_min) with sane defaults.

    Important: For economics we interpret "detour" as *extra* distance/time
    compared to the baseline route. Small negative values can occur due to
    routing/rounding artefacts; we clamp them to 0.0 to avoid confusing
    negative fuel and negative break-even outputs.
    """
    detour_km = station.get("detour_distance_km")
    detour_min = station.get("detour_duration_min")

    try:
        detour_km_f = float(detour_km) if detour_km is not None else 0.0
    except (TypeError, ValueError):
        detour_km_f = 0.0

    try:
        detour_min_f = float(detour_min) if detour_min is not None else 0.0
    except (TypeError, ValueError):
        detour_min_f = 0.0

    # Clamp: "detour" = extra distance/time (never negative for economics/UI)
    detour_km_f = max(detour_km_f, 0.0)
    detour_min_f = max(detour_min_f, 0.0)

    return detour_km_f, detour_min_f


def _find_baseline_price(
    stations: List[Dict[str, Any]],
    pred_key: str,
    *,
    onroute_max_detour_km: float,
    onroute_max_detour_min: float,
) -> Optional[float]:
    """
    Determine the baseline "on-route" price for the given fuel.

    We select the cheapest station (by predicted price) among those with a
    very small detour (essentially the direct route). If none qualify,
    we fall back to the global minimum predicted price across all stations.
    """
    onroute_candidates: List[Dict[str, Any]] = []
    for s in stations:
        price = s.get(pred_key)
        if price is None:
            continue
        detour_km, detour_min = _get_detour_metrics(s)
        if abs(detour_km) <= onroute_max_detour_km and abs(detour_min) <= onroute_max_detour_min:
            onroute_candidates.append(s)

    if onroute_candidates:
        baseline_station = min(onroute_candidates, key=lambda s: s.get(pred_key, float("inf")))
        return float(baseline_station[pred_key])

    # Fallback: global minimum across all stations
    priced = [s for s in stations if s.get(pred_key) is not None]
    if not priced:
        return None
    baseline_station = min(priced, key=lambda s: s.get(pred_key, float("inf")))
    return float(baseline_station[pred_key])


def _compute_economic_metrics(
    station: Dict[str, Any],
    *,
    baseline_price: float,
    pred_key: str,
    litres_to_refuel: float,
    consumption_l_per_100km: float,
    value_of_time_per_hour: float,
) -> Dict[str, Optional[float]]:
    """
    Compute economic detour metrics for a single station.

    Notes
    -----
    * Detour distance/time are clamped to >= 0 in _get_detour_metrics.
    * We compute costs using *unrounded* detour litres for accuracy.
    * We still return detour_fuel_l rounded to 2 decimals for display.
    """
    price = station.get(pred_key)
    if price is None:
        return {k: None for k in (
            "gross_saving_eur",
            "detour_fuel_l",
            "detour_fuel_cost_eur",
            "time_cost_eur",
            "net_saving_eur",
            "breakeven_liters",
        )}

    pA = float(price)           # station price
    pB = float(baseline_price)  # baseline on-route price

    detour_km, detour_min = _get_detour_metrics(station)

    # Extra fuel for detour (raw for computations, rounded only for display)
    raw_detour_fuel_l = detour_km * (consumption_l_per_100km / 100.0)
    detour_fuel_l_display = round(raw_detour_fuel_l, 2)

    # Cost of that extra fuel (use raw litres for accuracy)
    detour_fuel_cost_eur = raw_detour_fuel_l * pA

    # Time cost (if value_of_time_per_hour > 0)
    if value_of_time_per_hour > 0.0:
        time_cost_eur = detour_min * (value_of_time_per_hour / 60.0)
    else:
        time_cost_eur = 0.0

    # Gross saving from filling at this station vs baseline
    gross_saving_eur = (pB - pA) * litres_to_refuel

    # Net saving
    net_saving_eur = gross_saving_eur - detour_fuel_cost_eur - time_cost_eur

    # Break-even litres: litres required so that net_saving becomes 0.
    # Only meaningful if station is cheaper than baseline (pB > pA).
    if pB > pA:
        breakeven_liters = (detour_fuel_cost_eur + time_cost_eur) / (pB - pA)
        # Numerical safety (should already be >= 0 due to clamped detours)
        breakeven_liters = max(float(breakeven_liters), 0.0)
    else:
        breakeven_liters = None

    return {
        "gross_saving_eur": gross_saving_eur,
        "detour_fuel_l": detour_fuel_l_display,
        "detour_fuel_cost_eur": detour_fuel_cost_eur,
        "time_cost_eur": time_cost_eur,
        "net_saving_eur": net_saving_eur,
        "breakeven_liters": breakeven_liters,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_stations_by_predicted_price(
    model_input: List[Dict[str, Any]],
    fuel_type: str,
    now: datetime | None = None,
    *,
    # Economic detour parameters (all optional; if missing -> fall back to price-only ranking)
    litres_to_refuel: Optional[float] = None,
    consumption_l_per_100km: Optional[float] = None,
    value_of_time_per_hour: float = 0.0,
    max_detour_km: Optional[float] = None,
    max_detour_min: Optional[float] = None,
    min_net_saving_eur: float = 0.0,
    onroute_max_detour_km: float = ONROUTE_MAX_DETOUR_KM,
    onroute_max_detour_min: float = ONROUTE_MAX_DETOUR_MIN,
) -> List[Dict[str, Any]]:
    """
    Rank stations along the route for a given fuel type.

    Two modes
    ---------
    1. Economic mode (if ``litres_to_refuel`` and ``consumption_l_per_100km``
       are both provided and strictly positive):

       * apply hard caps on ``detour_distance_km`` and ``detour_duration_min``,
       * compute net savings vs. a baseline on–route price,
       * drop stations with ``net_saving < min_net_saving_eur``,
       * rank by descending net saving, then by position along the route.

       If no station passes the ``min_net_saving_eur`` threshold but there
       are stations that satisfy the detour caps, we still return a ranking
       of the *feasible* stations by net saving (which may be <= 0). This
       avoids silently reverting to a global cheapest–price ranking and keeps
       the detour caps and economic interpretation consistent for the UI.

    2. Price–only mode (fallback when economic parameters are missing or no
       baseline can be computed):

       * rank by lowest predicted price,
       * tie–break by ``fraction_of_route`` and then ``distance_along_m``.

    All rankings operate on unique ``station_uuid``s (deduplication safety).
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

    # Deduplicate by station_uuid, keeping the variant with the lowest price
    by_uuid: Dict[str, Dict[str, Any]] = {}
    for s in valid_stations:
        uuid = s.get("station_uuid")
        if uuid is None:
            by_uuid[id(s)] = s
            continue

        existing = by_uuid.get(uuid)
        if existing is None or s.get(pred_key, float("inf")) < existing.get(
            pred_key, float("inf")
        ):
            by_uuid[uuid] = s

    unique_stations = list(by_uuid.values())

    # ------------------------------------------------------------------
    # Economic mode (if we have litres + consumption)
    # ------------------------------------------------------------------
    economic_mode = (
        litres_to_refuel is not None
        and consumption_l_per_100km is not None
        and litres_to_refuel > 0
        and consumption_l_per_100km > 0
    )

    # ------------------------------------------------------------------
    # Detour feasibility caps (apply in BOTH modes)
    # ------------------------------------------------------------------
    # Backwards-compatible behaviour:
    # - In economic mode we always apply detour caps (defaults if missing).
    # - In price-only mode we only apply caps if the caller provided at least
    #   one of them (max_detour_km or max_detour_min).
    apply_detour_caps = economic_mode or (max_detour_km is not None) or (max_detour_min is not None)

    if apply_detour_caps:
        # Fill missing caps with defaults (and harden against bad types)
        try:
            max_detour_km = float(max_detour_km) if max_detour_km is not None else float(DEFAULT_MAX_DETOUR_KM)
        except (TypeError, ValueError):
            max_detour_km = float(DEFAULT_MAX_DETOUR_KM)

        try:
            max_detour_min = float(max_detour_min) if max_detour_min is not None else float(DEFAULT_MAX_DETOUR_MIN)
        except (TypeError, ValueError):
            max_detour_min = float(DEFAULT_MAX_DETOUR_MIN)

        # Filter stations outside the user's feasibility constraints
        capped: List[Dict[str, Any]] = []
        for s in unique_stations:
            detour_km, detour_min = _get_detour_metrics(s)
            if detour_km <= max_detour_km and detour_min <= max_detour_min:
                capped.append(s)

        unique_stations = capped
        if not unique_stations:
            return []

    if economic_mode:
        # Baseline price (on–route)
        baseline_price = _find_baseline_price(
            unique_stations,
            pred_key,
            onroute_max_detour_km=onroute_max_detour_km,
            onroute_max_detour_min=onroute_max_detour_min,
        )


        if baseline_price is not None:
            baseline_key = f"econ_baseline_price_{ft}"
            econ_net_key = f"econ_net_saving_eur_{ft}"
            econ_gross_key = f"econ_gross_saving_eur_{ft}"
            econ_detour_fuel_key = f"econ_detour_fuel_l_{ft}"
            econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{ft}"
            econ_time_cost_key = f"econ_time_cost_eur_{ft}"
            econ_breakeven_key = f"econ_breakeven_liters_{ft}"

            economic_candidates: List[Dict[str, Any]] = []
            feasible_stations: List[Dict[str, Any]] = []

            for s in unique_stations:
                detour_km, detour_min = _get_detour_metrics(s)

                # Hard–cap filter: only consider stations within the user's
                # max detour distance and time.
                if abs(detour_km) > max_detour_km or abs(detour_min) > max_detour_min:
                    continue

                metrics = _compute_economic_metrics(
                    s,
                    baseline_price=baseline_price,
                    pred_key=pred_key,
                    litres_to_refuel=float(litres_to_refuel),
                    consumption_l_per_100km=float(consumption_l_per_100km),
                    value_of_time_per_hour=float(value_of_time_per_hour),
                )

                # Attach metrics to station dict for later use in the UI
                s[baseline_key] = baseline_price
                s[econ_gross_key] = metrics["gross_saving_eur"]
                s[econ_detour_fuel_key] = metrics["detour_fuel_l"]
                s[econ_detour_fuel_cost_key] = metrics["detour_fuel_cost_eur"]
                s[econ_time_cost_key] = metrics["time_cost_eur"]
                s[econ_net_key] = metrics["net_saving_eur"]
                s[econ_breakeven_key] = metrics["breakeven_liters"]

                feasible_stations.append(s)

                net = metrics["net_saving_eur"]
                if net is None:
                    continue
                if net < min_net_saving_eur:
                    continue

                economic_candidates.append(s)

            if economic_candidates:
                # Rank by descending net saving, then earlier on the route
                def econ_sort_key(station: Dict[str, Any]):
                    net = station.get(econ_net_key, float("-inf"))
                    frac = station.get("fraction_of_route", float("inf"))
                    dist = station.get("distance_along_m", float("inf"))
                    # Negative first component because we want descending net saving
                    return (-net, frac, dist)

                return sorted(economic_candidates, key=econ_sort_key)

            if feasible_stations:
                # No station meets the min_net_saving_eur threshold, but there
                # *are* stations within the detour caps. Return a ranking of
                # those feasible stations by (possibly negative) net saving.
                def feasible_sort_key(station: Dict[str, Any]):
                    net = station.get(econ_net_key, float("-inf"))
                    frac = station.get("fraction_of_route", float("inf"))
                    dist = station.get("distance_along_m", float("inf"))
                    return (-net, frac, dist)

                return sorted(feasible_stations, key=feasible_sort_key)
        # If we cannot compute a baseline at all, fall through to price-only
        # ranking below.

    # ------------------------------------------------------------------
    # Fallback: classical price-only ranking
    # ------------------------------------------------------------------
    def sort_key(station: Dict[str, Any]):
        price = station.get(pred_key, float("inf"))
        frac = station.get("fraction_of_route", float("inf"))
        dist = station.get("distance_along_m", float("inf"))
        return (price, frac, dist)

    ranked = sorted(unique_stations, key=sort_key)
    return ranked


def recommend_best_station(
    model_input: List[Dict[str, Any]],
    fuel_type: str,
    now: datetime | None = None,
    *,
    litres_to_refuel: Optional[float] = None,
    consumption_l_per_100km: Optional[float] = None,
    value_of_time_per_hour: float = 0.0,
    max_detour_km: Optional[float] = None,
    max_detour_min: Optional[float] = None,
    min_net_saving_eur: float = 0.0,
    onroute_max_detour_km: float = ONROUTE_MAX_DETOUR_KM,
    onroute_max_detour_min: float = ONROUTE_MAX_DETOUR_MIN,
) -> Dict[str, Any] | None:
    """
    Select the single best station for a given fuel type.

    If economic parameters are provided, the "best" station is the one
    with highest net saving (subject to the detour hard caps and minimum
    net saving). Otherwise it is simply the station with the lowest
    predicted price according to :func:`rank_stations_by_predicted_price`.
    """
    ranked = rank_stations_by_predicted_price(
        model_input,
        fuel_type,
        now=now,
        litres_to_refuel=litres_to_refuel,
        consumption_l_per_100km=consumption_l_per_100km,
        value_of_time_per_hour=value_of_time_per_hour,
        max_detour_km=max_detour_km,
        max_detour_min=max_detour_min,
        min_net_saving_eur=min_net_saving_eur,
        onroute_max_detour_km=onroute_max_detour_km,
        onroute_max_detour_min=onroute_max_detour_min,
    )
    if not ranked:
        return None
    return ranked[0]
