"""
Module: Fuel Station Recommendation Engine.

Description:
    This module implements the decision logic for selecting the optimal refueling stop.
    It operates on enriched station data (from the integration layer) and applies
    two distinct ranking strategies based on available input:

    1. **Economic Ranking (Cost-Benefit Analysis):**
       - Active when `litres_to_refuel` and `consumption_l_per_100km` are provided.
       - Calculates the "Net Saving" of a detour compared to an "On-Route Baseline".
       - Formula: (BaselinePrice - StationPrice) * Litres - (DetourFuelCost + TimeCost).
       - Filters stations exceeding max detour constraints.

    2. **Price-Only Ranking (Fallback):**
       - Active when economic parameters are missing.
       - Simply sorts stations by the lowest predicted price.
       - Tie-breakers: Distance along route (earlier is better).

Usage:
    Called by the UI layer to populate the "Recommended Stop" card and the result list.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Final

from src.modeling.model import FUEL_TYPES
from src.modeling.predict import predict_for_fuel

# ==========================================
# Type Definitions
# ==========================================

# A dictionary representing a single station with route and price data.
StationDict: TypeAlias = Dict[str, Any]

# ==========================================
# Constants & Configuration
# ==========================================

# Thresholds to consider a station "On-Route" (used for baseline price calculation)
# Essentially the direct route with negligible deviation.
ONROUTE_MAX_DETOUR_KM: Final[float] = 1.0
ONROUTE_MAX_DETOUR_MIN: Final[float] = 5.0

# Default safety caps for detours if not specified by user
DEFAULT_MAX_DETOUR_KM: Final[float] = 10.0
DEFAULT_MAX_DETOUR_MIN: Final[float] = 10.0

# Minimum plausible €/L (guards against 0.00 or corrupted values)
MIN_VALID_PRICE_EUR_L: Final[float] = 0.50

# ==========================================
# Helper: Key Generation & Validation
# ==========================================

def _prediction_key(fuel_type: str) -> str:
    """Returns the dictionary key for the specific fuel type's price prediction."""
    return f"pred_price_{fuel_type.lower().strip()}"


def _normalise_fuel_type(fuel_type: str) -> str:
    """Validates and normalizes fuel type string."""
    if fuel_type is None:
        raise ValueError("fuel_type must not be None")
    
    ft = fuel_type.lower().strip()
    if ft not in FUEL_TYPES:
        valid_options = ", ".join(FUEL_TYPES)
        raise ValueError(f"Unsupported fuel_type '{fuel_type}'. Expected: {valid_options}")
    return ft


def _ensure_predictions(
    stations: List[StationDict],
    fuel_type: str,
    now: Optional[datetime] = None,
) -> None:
    """
    Guarantees that price predictions exist in the station dictionaries.
    Mutates the list in-place by calling the inference model if needed.
    """
    if not stations:
        return

    ft = _normalise_fuel_type(fuel_type)
    pred_key = _prediction_key(ft)

    # Check if ANY station already has a prediction. If not, batch predict for all.
    has_predictions = any(s.get(pred_key) is not None for s in stations)

    if not has_predictions:
        predict_for_fuel(stations, ft, prediction_key=pred_key, now=now)


# ==========================================
# Helper: Metric Calculation
# ==========================================

def _get_detour_metrics(station: StationDict) -> Tuple[float, float]:
    """
    Extracts detour distance (km) and duration (min) with safety clamping.
    Returns (0.0, 0.0) if values are missing or negative (routing artifacts).
    """
    d_km = station.get("detour_distance_km")
    d_min = station.get("detour_duration_min")

    try:
        val_km = float(d_km) if d_km is not None else 0.0
    except (TypeError, ValueError):
        val_km = 0.0

    try:
        val_min = float(d_min) if d_min is not None else 0.0
    except (TypeError, ValueError):
        val_min = 0.0

    return max(0.0, val_km), max(0.0, val_min)


def _find_baseline_price(
    stations: List[StationDict],
    pred_key: str,
    onroute_dist_limit: float,
    onroute_time_limit: float,
) -> Optional[float]:
    """
    Determines the "Baseline Price".
    
    Strategy:
    1. Look for the cheapest station that is effectively "On-Route" (within minimal detour limits).
    2. Fallback: If no on-route station exists, use the global minimum price of all stations.
    """
    # Filter 1: Strictly On-Route candidates
    onroute_candidates = []
    for s in stations:
        price = s.get(pred_key)
        if price is None:
            continue
            
        d_km, d_min = _get_detour_metrics(s)
        if d_km <= onroute_dist_limit and d_min <= onroute_time_limit:
            onroute_candidates.append(s)

    # Strategy 1: Cheapest On-Route
    if onroute_candidates:
        best_onroute = min(onroute_candidates, key=lambda x: x[pred_key])
        return float(best_onroute[pred_key])

    # Strategy 2: Global Cheapest (Fallback)
    all_priced = [s for s in stations if s.get(pred_key) is not None]
    if not all_priced:
        return None
    
    global_best = min(all_priced, key=lambda x: x[pred_key])
    return float(global_best[pred_key])


def _calculate_economics(
    station: StationDict,
    pred_key: str,
    baseline_price: float,
    litres: float,
    consumption: float,
    hourly_wage: float,
) -> Dict[str, Optional[float]]:
    """
    Performs the core Economic Math for a single station.
    
    Formula:
        Gross Saving = (Baseline Price - Station Price) * Litres
        Detour Cost = (Detour Km * Consumption) * Station Price + (Detour Mins * Wage)
        Net Saving = Gross Saving - Detour Cost
    """
    price = station.get(pred_key)
    if price is None:
        return {
            "gross_saving_eur": None,
            "detour_fuel_l": None,
            "detour_fuel_cost_eur": None,
            "time_cost_eur": None,
            "net_saving_eur": None,
            "breakeven_liters": None,
        }

    p_station = float(price)
    p_baseline = float(baseline_price)
    
    detour_km, detour_min = _get_detour_metrics(station)

    # 1. Cost of Detour Fuel
    # Formula: (Km * L/100km) -> Litres used
    detour_litres_raw = detour_km * (consumption / 100.0)
    detour_cost_eur = detour_litres_raw * p_station

    # 2. Cost of Time
    time_cost_eur = 0.0
    if hourly_wage > 0.0:
        time_cost_eur = detour_min * (hourly_wage / 60.0)

    # 3. Savings
    gross_saving = (p_baseline - p_station) * litres
    net_saving = gross_saving - detour_cost_eur - time_cost_eur

    # 4. Break-even Analysis
    # "How many liters must I buy to make this detour worth it?"
    breakeven_liters = None
    if p_baseline > p_station:
        total_overhead = detour_cost_eur + time_cost_eur
        price_diff = p_baseline - p_station
        breakeven_liters = max(0.0, total_overhead / price_diff)

    return {
        "gross_saving_eur": gross_saving,
        "detour_fuel_l": round(detour_litres_raw, 2), # Rounded for display
        "detour_fuel_cost_eur": detour_cost_eur,
        "time_cost_eur": time_cost_eur,
        "net_saving_eur": net_saving,
        "breakeven_liters": breakeven_liters,
    }

def _is_valid_price(value: Any, *, min_price: float = float(MIN_VALID_PRICE_EUR_L)) -> bool:
    """Return True iff value is a finite float >= min_price."""
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    # NaN check without numpy
    if v != v:  # noqa: PLR0124 (NaN is not equal to itself)
        return False
    return v >= float(min_price)

# ==========================================
# Core Logic: Ranking
# ==========================================

def rank_stations_by_predicted_price(
    model_input: List[StationDict],
    fuel_type: str,
    now: Optional[datetime] = None,
    *,
    # Economic Parameters
    litres_to_refuel: Optional[float] = None,
    consumption_l_per_100km: Optional[float] = None,
    value_of_time_per_hour: float = 0.0,
    # Constraints
    max_detour_km: Optional[float] = None,
    max_detour_min: Optional[float] = None,
    min_net_saving_eur: float = 0.0,
    # Configuration
    onroute_max_detour_km: float = ONROUTE_MAX_DETOUR_KM,
    onroute_max_detour_min: float = ONROUTE_MAX_DETOUR_MIN,
    # Distance to station filters (km)
    min_distance_km: Optional[float] = None,
    max_distance_km: Optional[float] = None,    
    
    # Auditing (optional)
    audit_log: Optional[Dict[str, Any]] = None,
) -> List[StationDict]:
    """
    Main entry point for ranking stations.
    
    Returns a list of stations sorted by optimality.
    
    Logic Flow:
    1. Validate inputs and ensure price predictions exist.
    2. Filter invalid stations (missing prices).
    3. Determine Mode: Economic vs. Price-Only.
    4. Apply Detour Constraints (Hard Caps).
    5. If Economic Mode:
       - Calculate Baseline Price.
       - Calculate Net Savings per station.
       - Sort by Net Saving (Descending).
    6. If Price-Only Mode (Fallback):
       - Sort by Absolute Price (Ascending).
    """
    if not model_input:
        return []

    # 1. Setup
    ft = _normalise_fuel_type(fuel_type)
    pred_key = _prediction_key(ft)
    if now is None:
        now = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Optional audit log (exact exclusion reasons)
    # ------------------------------------------------------------------
    audit_enabled = audit_log is not None
    if audit_enabled:
        audit_log.clear()
        audit_log["reasons_by_uuid"] = {}          # uuid -> [reasons...]
        audit_log["primary_reason_by_uuid"] = {}   # uuid -> primary reason
        audit_log["counts"] = {}                   # reason -> count
        audit_log["ranked_uuids"] = []
        audit_log["excluded_uuids"] = []
        audit_log["notes"] = []
        audit_log["thresholds"] = {
            "fuel_type": fuel_type,
            "is_economic_mode": bool(litres_to_refuel is not None and consumption_l_per_100km is not None),
            "max_detour_km": max_detour_km,
            "max_detour_min": max_detour_min,
            "min_net_saving_eur": float(min_net_saving_eur),
            "onroute_max_detour_km": float(onroute_max_detour_km),
            "onroute_max_detour_min": float(onroute_max_detour_min),
            "min_distance_km": float(min_distance_km) if min_distance_km is not None else None,
            "max_distance_km": float(max_distance_km) if max_distance_km is not None else None,
        }

    def _uid(s: StationDict) -> str:
        u = s.get("station_uuid") or s.get("uuid") or s.get("id")
        if u:
            return str(u)
        # fallback for safety
        return f"{s.get('lat')}_{s.get('lon')}"

    def _add_reason(s: StationDict, reason: str) -> None:
        if not audit_enabled:
            return
        u = _uid(s)
        audit_log["reasons_by_uuid"].setdefault(u, [])
        if reason not in audit_log["reasons_by_uuid"][u]:
            audit_log["reasons_by_uuid"][u].append(reason)

    def _finalize_audit(ranked_list: List[StationDict], all_considered: List[StationDict]) -> None:
        """Set primary reasons + counts + ranked/excluded lists."""
        if not audit_enabled:
            return

        ranked_set = {_uid(s) for s in ranked_list}
        all_set = {_uid(s) for s in all_considered}

        audit_log["ranked_uuids"] = sorted(list(ranked_set))
        audit_log["excluded_uuids"] = sorted(list(all_set - ranked_set))

        # priority order for primary reason
        priority = [
            "Detour distance above cap",
            "Detour time above cap",
            f"Invalid predicted price (< {MIN_VALID_PRICE_EUR_L:.2f} €/L)",
            "Missing predicted price",
            "Below minimum net saving",
            "Missing economics metrics",
            "Duplicate station removed",
            "Not ranked (other)",
        ]

        counts: Dict[str, int] = {}
        for u in all_set:
            reasons = audit_log["reasons_by_uuid"].get(u, [])
            if u in ranked_set:
                # Ranked: keep reason list for transparency, but primary is Ranked
                primary = "Ranked"
            else:
                # Excluded: choose best primary by priority
                primary = "Not ranked (other)"
                for p in priority:
                    if p in reasons:
                        primary = p
                        break
            audit_log["primary_reason_by_uuid"][u] = primary
            counts[primary] = counts.get(primary, 0) + 1

        audit_log["counts"] = counts

    # 2. Prediction & Validation
    _ensure_predictions(model_input, ft, now=now)

    # Filter valid stations and deduplicate by UUID (keep lowest price variant)
    # - Exclude nonsense predictions (e.g., 0.00) by treating them as missing.
    invalid_reason = f"Invalid predicted price (< {MIN_VALID_PRICE_EUR_L:.2f} €/L)"

    valid_map: Dict[str, StationDict] = {}
    for s in model_input:
        p = s.get(pred_key)

        # Filter out implausible predictions (incl. 0.00)
        if not _is_valid_price(p):
            if p is not None:
                _add_reason(s, invalid_reason)
            # Normalize to "missing" so downstream logic never treats it as numeric
            s[pred_key] = None
            continue

        uuid = s.get("station_uuid")
        key = uuid if uuid else str(id(s))

        existing = valid_map.get(key)
        # Keep if new, or if new price is lower than existing entry
        if not existing or float(s[pred_key]) < float(existing[pred_key]):
            valid_map[key] = s

    unique_stations = list(valid_map.values())
    if not unique_stations:
        return []
    
    # 3. Determine Mode
    is_economic_mode = (
        litres_to_refuel is not None
        and consumption_l_per_100km is not None
        and litres_to_refuel > 0
        and consumption_l_per_100km > 0
    )

    # 4. Apply Detour Caps
    # Logic: In economic mode, caps are mandatory (defaulting if None).
    # In price mode, caps are optional (only applied if user provided them).
    should_cap = is_economic_mode or (max_detour_km is not None) or (max_detour_min is not None)

    # Keep a stable "considered set" for audit BEFORE caps remove entries
    considered_for_audit = list(unique_stations)
    
    if should_cap:
        # Resolve caps with defaults
        cap_km = float(max_detour_km) if max_detour_km is not None else DEFAULT_MAX_DETOUR_KM
        cap_min = float(max_detour_min) if max_detour_min is not None else DEFAULT_MAX_DETOUR_MIN
        
        filtered_stations = []
        for s in unique_stations:
            d_km, d_min = _get_detour_metrics(s)

            failed = False
            if d_km > cap_km:
                _add_reason(s, "Detour distance above cap")
                failed = True
            if d_min > cap_min:
                _add_reason(s, "Detour time above cap")
                failed = True

            if not failed:
                filtered_stations.append(s)
        unique_stations = filtered_stations

    if not unique_stations:
        return []

    # Apply distance filters (min_distance_km and max_distance_km)
    if min_distance_km is not None or max_distance_km is not None:
        filtered_stations = []
        for s in unique_stations:
            distance_km = s.get("distance_along_m", 0) / 1000.0  # Convert meters to kilometers

            if min_distance_km is not None and distance_km < min_distance_km:
                _add_reason(s, "Below minimum distance")
                continue

            if max_distance_km is not None and distance_km > max_distance_km:
                _add_reason(s, "Above maximum distance")
                continue

            filtered_stations.append(s)

        unique_stations = filtered_stations

    # 5. Economic Ranking Strategy
    if is_economic_mode:
        baseline = _find_baseline_price(
            unique_stations, 
            pred_key, 
            onroute_max_detour_km, 
            onroute_max_detour_min
        )

        if baseline is not None:
            # Keys for injecting results into the station dict
            k_net = f"econ_net_saving_eur_{ft}"
            
            # Calculate metrics for all candidates
            candidates = []
            for s in unique_stations:
                metrics = _calculate_economics(
                    s, pred_key, baseline,
                    float(litres_to_refuel), # type: ignore
                    float(consumption_l_per_100km), # type: ignore
                    float(value_of_time_per_hour)
                )
                
                # Inject metrics
                s[f"econ_baseline_price_{ft}"] = baseline
                s[f"econ_gross_saving_eur_{ft}"] = metrics["gross_saving_eur"]
                s[f"econ_detour_fuel_l_{ft}"] = metrics["detour_fuel_l"]
                s[f"econ_detour_fuel_cost_eur_{ft}"] = metrics["detour_fuel_cost_eur"]
                s[f"econ_time_cost_eur_{ft}"] = metrics["time_cost_eur"]
                s[k_net] = metrics["net_saving_eur"]
                s[f"econ_breakeven_liters_{ft}"] = metrics["breakeven_liters"]
                
                candidates.append(s)

            # Sort Logic:
            # 1. Net Saving (High to Low)
            # 2. Fraction of Route (Low to High - earlier stops preferred if savings equal)
            # 3. Distance Along (Low to High - tie breaker)
            def econ_sorter(x: StationDict) -> Tuple:
                net = x.get(k_net)
                if net is None: 
                    return (float("inf"), float("inf"), float("inf"))
                return (-net, x.get("fraction_of_route", float("inf")), x.get("distance_along_m", float("inf")))

            sorted_candidates = sorted(candidates, key=econ_sorter)

            # Filter Logic:
            # Mark missing prediction/econ metrics (net_saving None)
            for c in sorted_candidates:
                if c.get(k_net) is None:
                    _add_reason(c, "Missing predicted price")

            # Apply min-net-saving as a "soft filter":
            # - If at least one station passes, exclude the rest and mark them.
            # - If none pass, keep all (but still label those below threshold for transparency).
            savings_filtered = []
            below_threshold = []
            for c in sorted_candidates:
                net = c.get(k_net)
                try:
                    net_f = float(net) if net is not None else None
                except (TypeError, ValueError):
                    net_f = None

                if net_f is not None and net_f >= float(min_net_saving_eur):
                    savings_filtered.append(c)
                else:
                    below_threshold.append(c)

            if savings_filtered:
                for c in below_threshold:
                    _add_reason(c, "Below minimum net saving")
                # ranked are those that survive
                ranked_out = savings_filtered
            else:
                # threshold not binding; keep all but label
                if audit_enabled:
                    audit_log["notes"].append(
                        "Minimum net saving threshold not met by any station; returning full sorted list."
                    )
                for c in below_threshold:
                    _add_reason(c, "Below minimum net saving")
                ranked_out = sorted_candidates

            _finalize_audit(ranked_out, considered_for_audit)
            return ranked_out

    # If we reach here, we are NOT returning from the economic branch.
    # That means either:
    # - not economic mode, OR
    # - economic mode but baseline could not be computed.
    # In both cases: fall back to price-only ranking.

    if audit_enabled and is_economic_mode:
        audit_log["notes"].append("Economic mode requested but baseline price could not be computed; falling back to price-only ranking.")

    # 6. Price-Only Ranking Strategy (Fallback)
    # Sort Logic:
    # 1. Price (Low to High)
    # 2. Fraction of Route (Low to High)
    # 3. Distance Along (Low to High)
    def price_sorter(x: StationDict) -> Tuple:
        return (
            x.get(pred_key, float("inf")),
            x.get("fraction_of_route", float("inf")),
            x.get("distance_along_m", float("inf")),
        )

    sorted_candidates = sorted(unique_stations, key=price_sorter)

    # (Optional but consistent) Mark missing predicted prices among considered set
    for c in model_input:
        if c.get(pred_key) is None:
            _add_reason(c, "Missing predicted price")

    _finalize_audit(sorted_candidates, considered_for_audit if "considered_for_audit" in locals() else model_input)
    return sorted_candidates


def recommend_best_station(
    model_input: List[StationDict],
    fuel_type: str,
    now: Optional[datetime] = None,
    *,
    litres_to_refuel: Optional[float] = None,
    consumption_l_per_100km: Optional[float] = None,
    value_of_time_per_hour: float = 0.0,
    max_detour_km: Optional[float] = None,
    max_detour_min: Optional[float] = None,
    min_net_saving_eur: float = 0.0,
    onroute_max_detour_km: float = ONROUTE_MAX_DETOUR_KM,
    onroute_max_detour_min: float = ONROUTE_MAX_DETOUR_MIN,
    min_distance_km: Optional[float] = None,
    max_distance_km: Optional[float] = None,
) -> Optional[StationDict]:
    """
    Wrapper around ranking that returns only the top-ranked station.
    Returns None if no stations qualify.
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
        min_distance_km=min_distance_km,
        max_distance_km=max_distance_km,
    )
    
    return ranked[0] if ranked else None