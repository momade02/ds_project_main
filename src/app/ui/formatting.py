# src/app/ui/formatting.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd

# --- Fuel label mapping (UI label -> internal fuel code) ---------------------

FUEL_LABEL_TO_CODE: Dict[str, str] = {
    "Diesel": "diesel",
    "E5": "e5",
    "E10": "e10",
}


def fuel_label_to_code(label: str) -> str:
    return FUEL_LABEL_TO_CODE.get(label, "diesel")


# --- Safe text / IDs --------------------------------------------------------

_UUID_RE = re.compile(r"^[0-9a-fA-F-]{8,}$")


def station_uuid(station: Any) -> str:
    """
    Best-effort station UUID extraction supporting:
    - dict-like objects
    - objects with .get
    - raw string UUIDs
    """
    if station is None:
        return ""

    if isinstance(station, str):
        return station.strip()

    try:
        if isinstance(station, dict):
            return str(
                station.get("uuid")
                or station.get("id")
                or station.get("station_uuid")
                or ""
            ).strip()
        # dict-like / pydantic-like
        if hasattr(station, "get"):
            return str(
                station.get("uuid")
                or station.get("id")
                or station.get("station_uuid")
                or ""
            ).strip()
    except Exception:
        pass

    # fallback: attribute access
    for attr in ("uuid", "id", "station_uuid"):
        try:
            v = getattr(station, attr)
            if v:
                return str(v).strip()
        except Exception:
            continue

    return ""


def safe_text(value: Any, *, fallback: str = "—") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


# --- Formatting primitives ---------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def fmt_price(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.3f} €/L" if v is not None else "—"


def fmt_eur(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.2f} €" if v is not None else "—"


def fmt_km(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.2f} km" if v is not None else "—"


def fmt_min(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.1f} min" if v is not None else "—"


def fmt_liters(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.1f} L" if v is not None else "—"


# --- “Format_*” aliases (keep current call sites stable) --------------------

def format_price(value: Any) -> str:
    return fmt_price(value)

def format_eur(value: Any) -> str:
    return fmt_eur(value)

def format_km(value: Any) -> str:
    return fmt_km(value)

def format_min(value: Any) -> str:
    return fmt_min(value)

def format_liters(value: Any) -> str:
    return fmt_liters(value)


# --- Explanatory text -------------------------------------------------------

def describe_price_basis(use_economics: bool, litres_to_refuel: Optional[float]) -> str:
    """
    Centralized explanation used in both pages.
    Keep semantics identical to your existing descriptions.
    """
    if use_economics and litres_to_refuel is not None:
        return (
            "Prices shown are Tankerkönig real-time prices; predicted prices reflect "
            "your estimated arrival time. Net savings incorporate detour time and "
            f"refuelling volume (~{fmt_liters(litres_to_refuel)})."
        )
    return (
        "Prices shown are Tankerkönig real-time prices; predicted prices reflect "
        "your estimated arrival time. Ranking is based on predicted price only."
    )


# --- Backwards-compatible names (so Step 1 requires minimal edits) ----------

# streamlit_app.py / station_details.py currently use these private names.
_station_uuid = station_uuid
_safe_text = safe_text
_fmt_price = fmt_price
_fmt_eur = fmt_eur
_fmt_km = fmt_km
_fmt_min = fmt_min
_fmt_liters = fmt_liters
_format_price = format_price
_format_eur = format_eur
_format_km = format_km
_format_min = format_min
_format_liters = format_liters
_describe_price_basis = describe_price_basis
_fuel_label_to_code = fuel_label_to_code


# =============================================================================
# Station Details Page Helpers (Page 03)
# =============================================================================

def calculate_traffic_light_status(
    station_price: Optional[float],
    ranked_stations: List[Dict[str, Any]],
    fuel_code: str,
    station_net_saving: Optional[float] = None,
    use_net_savings: bool = True,
) -> Tuple[str, str, str]:
    """
    Determine traffic light indicator (green/yellow/red) for a station.
    
    Logic (when use_net_savings=True and economics available):
        Compare station's net saving to all ranked stations' net savings.
        - Green: Top 33% (best net savings / best value)
        - Yellow: Middle 33%
        - Red: Bottom 33% (worst value)
    
    Fallback (when use_net_savings=False or no economics):
        Compare station price to all ranked stations' prices.
        - Green: Top 33% (cheapest)
        - Yellow: Middle 33%
        - Red: Bottom 33% (most expensive)
    
    Args:
        station_price: Current station's predicted price
        ranked_stations: List of ranked station dicts
        fuel_code: Fuel type ('e5', 'e10', 'diesel')
        station_net_saving: Current station's net saving (optional)
        use_net_savings: Whether to use net savings ranking (default True)
    
    Returns:
        Tuple of (status_color, status_text, css_classes)
        Example: ("green", "Excellent Value", "active inactive inactive")
    """
    net_key = f"econ_net_saving_eur_{fuel_code}"
    pred_key = f"pred_price_{fuel_code}"
    
    # Try net savings first if enabled
    if use_net_savings and station_net_saving is not None:
        # Extract all valid net savings from ranked stations
        net_savings = []
        for s in ranked_stations:
            ns = s.get(net_key)
            if ns is not None:
                try:
                    net_savings.append(float(ns))
                except (TypeError, ValueError):
                    pass
        
        if net_savings and len(net_savings) >= 3:
            # Sort descending - higher net saving is better
            net_savings_sorted = sorted(net_savings, reverse=True)
            n = len(net_savings_sorted)
            
            # Calculate percentile thresholds (higher is better)
            p33 = net_savings_sorted[n // 3]       # Top 33% threshold
            p66 = net_savings_sorted[2 * n // 3]   # Top 66% threshold
            
            net_saving = float(station_net_saving)
            
            # Determine status based on net savings
            if net_saving >= p33:
                # Top 33% - Best value
                return ("green", "Excellent Value", "active inactive inactive")
            elif net_saving >= p66:
                # Middle 33% - Fair value
                return ("yellow", "Fair Value", "inactive active inactive")
            else:
                # Bottom 33% - Poor value
                return ("red", "Poor Value", "inactive inactive active")
    
    # Fallback to price-based logic
    if not station_price or not ranked_stations:
        return ("yellow", "Unknown", "inactive inactive active")
    
    # Extract all valid prices
    prices = []
    for s in ranked_stations:
        p = s.get(pred_key)
        if p is not None:
            try:
                prices.append(float(p))
            except (TypeError, ValueError):
                pass
    
    if not prices or len(prices) < 3:
        return ("yellow", "Fair Price", "inactive active inactive")
    
    prices_sorted = sorted(prices)
    n = len(prices_sorted)
    
    # Calculate percentile thresholds (lower price is better)
    p33 = prices_sorted[n // 3]
    p66 = prices_sorted[2 * n // 3]
    
    price = float(station_price)
    
    # Determine status based on price
    if price <= p33:
        # Top 33% - Cheapest
        status_color = "green"
        status_text = "Excellent Deal"
        css_classes = "active inactive inactive"
    elif price <= p66:
        # Middle 33% - Average
        status_color = "yellow"
        status_text = "Fair Price"
        css_classes = "inactive active inactive"
    else:
        # Bottom 33% - Expensive
        status_color = "red"
        status_text = "Above Average"
        css_classes = "inactive inactive active"
    
    return (status_color, status_text, css_classes)


def calculate_trip_fuel_info(
    route_info: Dict[str, Any],
    consumption_l_per_100km: Optional[float],
    litres_to_refuel: Optional[float],
) -> Optional[Dict[str, Any]]:
    """
    Calculate trip fuel information for display.
    
    Returns dict with:
        - origin: str
        - destination: str
        - distance_km: float
        - fuel_needed_l: float (estimated)
        - refuel_amount_l: float
        - arrival_fuel_remaining_l: float (if positive)
    
    Returns None if insufficient data.
    """
    if not route_info or not consumption_l_per_100km or not litres_to_refuel:
        return None
    
    distance_km = route_info.get("distance_km")
    origin = route_info.get("origin_address", "Start")
    destination = route_info.get("destination_address", "Destination")
    
    if not distance_km:
        return None
    
    try:
        distance = float(distance_km)
        consumption = float(consumption_l_per_100km)
        refuel = float(litres_to_refuel)
    except (TypeError, ValueError):
        return None
    
    # Calculate fuel needed for trip
    fuel_needed = distance * (consumption / 100.0)
    
    # Calculate remaining fuel at arrival
    arrival_remaining = refuel - fuel_needed
    
    # Shorten addresses for display
    def shorten_address(addr: str) -> str:
        parts = str(addr).split(",")
        if len(parts) >= 2:
            return parts[0].strip()
        return str(addr)[:30]
    
    return {
        "origin": shorten_address(origin),
        "destination": shorten_address(destination),
        "distance_km": distance,
        "fuel_needed_l": fuel_needed,
        "refuel_amount_l": refuel,
        "arrival_fuel_remaining_l": arrival_remaining,
    }


def check_smart_price_alert(
    hourly_df: pd.DataFrame,
    current_hour: int,
    station_open_until_hour: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Check if there's a significant price drop coming soon.
    
    Returns dict with alert info if:
    - Price drop >€0.03/L expected within 3 hours
    - Station is still open at that time
    
    Returns None if no alert needed.
    """
    if hourly_df is None or hourly_df.empty:
        return None
    
    current_price = None
    for _, row in hourly_df.iterrows():
        if int(row.get("hour", -1)) == current_hour:
            current_price = row.get("avg_price")
            break
    
    if current_price is None or pd.isna(current_price):
        return None
    
    # Check next 3 hours
    for offset in range(1, 4):
        check_hour = (current_hour + offset) % 24
        
        # Skip if station will be closed
        if station_open_until_hour is not None:
            if check_hour > station_open_until_hour:
                continue
        
        for _, row in hourly_df.iterrows():
            if int(row.get("hour", -1)) == check_hour:
                future_price = row.get("avg_price")
                if future_price is None or pd.isna(future_price):
                    continue
                
                price_drop = float(current_price) - float(future_price)
                
                if price_drop >= 0.03:
                    return {
                        "hours_to_wait": offset,
                        "drop_hour": check_hour,
                        "price_drop": price_drop,
                    }
    
    return None