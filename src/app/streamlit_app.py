"""
Streamlit UI for the route-aware fuel price recommender.

Real route (Google route + Supabase + Tankerkönig pipeline)
   - Uses `get_fuel_prices_for_route(...)`.
   - Always uses real-time Tankerkönig prices.

High-level pipeline in both modes
---------------------------------
integration (route → stations → historical + real-time prices)
    → ARDL models with horizon logic (in `src.modeling.predict`)
    → decision layer (ranking & best station in `src.decision.recommender`)

This UI additionally implements an economic detour decision:
- user-specific litres to refuel,
- car consumption (L/100 km),
- optional value of time (€/hour),
- hard caps for max detour distance / time,
- net saving and break-even litres.
"""

from __future__ import annotations

from app_errors import AppError, ConfigError, ExternalServiceError, DataAccessError, DataQualityError, PredictionError

# ---------------------------------------------------------------------------
# Make sure the project root (containing the `src` package) is on sys.path
# ---------------------------------------------------------------------------
import sys
import math
import inspect
import html
from pathlib import Path
from typing import List, Dict, Any, Optional

import base64

import json
import hashlib

import pandas as pd
import streamlit as st
import pydeck as pdk

from datetime import datetime

from config.settings import load_env_once, ensure_persisted_state_defaults

load_env_once()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_stations import environment_check, google_route_via_waypoint
from src.decision.recommender import (
    ONROUTE_MAX_DETOUR_KM,
    ONROUTE_MAX_DETOUR_MIN,
)

from ui.maps import (
    _calculate_zoom_for_bounds,
    _supports_pydeck_selections,
    _create_map_visualization,
    create_mapbox_gl_html,
)

from ui.formatting import (
    _station_uuid,
    _safe_text,
    _fmt_price,
    _fmt_eur,
    _fmt_km,
    _fmt_min,
    _format_price,
    _format_eur,
    _format_km,
    _format_min,
    _format_liters,
    _describe_price_basis,
    _fuel_label_to_code,
)

from services.route_recommender import RouteRunInputs, run_route_recommendation

from services.session_store import init_session_context, restore_persisted_state, maybe_persist_state

from ui.styles import apply_app_css

from ui.sidebar import render_sidebar

import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _reverse_geocode_station_payload(station: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Returns the first Google reverse-geocode result payload for the station.
    Cached in session_state to avoid repeated API calls on reruns.
    """
    if not station:
        return None

    station_uuid = _station_uuid(station) or f"{station.get('lat')}_{station.get('lon')}"
    if "reverse_geocode_cache" not in st.session_state:
        st.session_state["reverse_geocode_cache"] = {}

    cache: Dict[str, Dict[str, Any]] = st.session_state["reverse_geocode_cache"]
    if station_uuid in cache:
        return cache[station_uuid]

    lat = station.get("lat")
    lon = station.get("lon")
    if lat is None or lon is None:
        return None

    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None

    try:
        api_key = environment_check()
        import googlemaps  # type: ignore[import-untyped]

        client = googlemaps.Client(key=api_key)
        results = client.reverse_geocode((lat_f, lon_f), language="de")
        if not results:
            return None

        cache[station_uuid] = results[0]
        return results[0]

    except Exception:
        return None


def _reverse_geocode_station_city(station: Dict[str, Any]) -> Optional[str]:
    """
    Extract a city/locality-like name from Google reverse-geocode payload.
    """
    payload = _reverse_geocode_station_payload(station)
    if not payload:
        return None

    comps = payload.get("address_components") or []
    # Prefer "locality", then common fallbacks used in some regions.
    preferred_type_order = [
        "locality",
        "postal_town",
        "administrative_area_level_3",
        "administrative_area_level_2",
    ]

    for t in preferred_type_order:
        for c in comps:
            types = c.get("types") or []
            if t in types:
                name = c.get("long_name")
                return str(name).strip() if name else None

    return None

def _reverse_geocode_station_address(station: Dict[str, Any]) -> Optional[str]:
    payload = _reverse_geocode_station_payload(station)
    if not payload:
        return None
    formatted = payload.get("formatted_address")
    return str(formatted).strip() if formatted else None


def _compute_onroute_worst_and_net_vs_worst(
    best_station: Dict[str, Any],
    ranked_stations: Optional[List[Dict[str, Any]]],
    fuel_code: str,
    litres_to_refuel: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    """
    Returns (onroute_worst_price, net_saving_vs_onroute_worst).
    Uses predicted prices for consistency with ranking (pred_price_*).
    Net saving subtracts detour fuel/time costs if present; otherwise treats them as 0.
    """
    if not best_station or not ranked_stations or not litres_to_refuel:
        return None, None

    pred_key = f"pred_price_{fuel_code}"

    # Define "on-route" as essentially no detour (reuse your existing constants with fallback)
    ONROUTE_KM_TH = float(ONROUTE_MAX_DETOUR_KM) if (ONROUTE_MAX_DETOUR_KM and float(ONROUTE_MAX_DETOUR_KM) > 0) else 0.5
    ONROUTE_MIN_TH = float(ONROUTE_MAX_DETOUR_MIN) if (ONROUTE_MAX_DETOUR_MIN and float(ONROUTE_MAX_DETOUR_MIN) > 0) else 3.0

    onroute_prices: List[float] = []
    for s in ranked_stations:
        p = s.get(pred_key)
        if p is None:
            continue

        # Clamp detour metrics as in your economics layer
        try:
            km = float(s.get("detour_distance_km") or 0.0)
        except (TypeError, ValueError):
            km = 0.0
        try:
            mins = float(s.get("detour_duration_min") or 0.0)
        except (TypeError, ValueError):
            mins = 0.0

        km = max(km, 0.0)
        mins = max(mins, 0.0)

        if km <= ONROUTE_KM_TH and mins <= ONROUTE_MIN_TH:
            try:
                onroute_prices.append(float(p))
            except (TypeError, ValueError):
                continue

    if len(onroute_prices) < 1:
        return None, None

    onroute_worst = max(onroute_prices)

    chosen_price = best_station.get(pred_key)
    if chosen_price is None:
        return onroute_worst, None

    try:
        chosen_price_f = float(chosen_price)
        litres_f = float(litres_to_refuel)
    except (TypeError, ValueError):
        return onroute_worst, None

    # Subtract detour costs when available; if not present, treat as 0 (still useful as an estimate).
    detour_fuel_cost = best_station.get(f"econ_detour_fuel_cost_eur_{fuel_code}") or 0.0
    time_cost = best_station.get(f"econ_time_cost_eur_{fuel_code}") or 0.0
    try:
        detour_fuel_cost_f = float(detour_fuel_cost)
    except (TypeError, ValueError):
        detour_fuel_cost_f = 0.0
    try:
        time_cost_f = float(time_cost)
    except (TypeError, ValueError):
        time_cost_f = 0.0

    gross_vs_worst = (onroute_worst - chosen_price_f) * litres_f
    net_vs_worst = gross_vs_worst - detour_fuel_cost_f - time_cost_f

    return onroute_worst, net_vs_worst

def _compute_value_view_stations(
    *,
    stations: List[Dict[str, Any]],
    fuel_code: str,
    constraints: Optional[Dict[str, Any]] = None,
    filter_thresholds: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Rebuild the Page-02 "upper table" station set (value view):

    1) Hard-pass set:
       - within detour caps (km/min) if available
       - has a predicted price for the selected fuel (pred_price_{fuel_code})

    2) Determine worst on-route price among hard-pass stations using the on-route thresholds.

    3) Value view = all hard-pass stations with predicted price <= worst on-route price.
       If no on-route stations exist, value view falls back to hard-pass.

    Returns:
      (value_view_stations, meta)
    """
    constraints = constraints or {}
    filter_thresholds = filter_thresholds or {}

    pred_key = f"pred_price_{fuel_code}"

    # Detour caps (hard constraints); fall back to "no cap" if missing.
    cap_km = constraints.get("max_detour_km")
    cap_min = constraints.get("max_detour_min")

    try:
        cap_km_f = float(cap_km) if cap_km is not None else float("inf")
    except (TypeError, ValueError):
        cap_km_f = float("inf")

    try:
        cap_min_f = float(cap_min) if cap_min is not None else float("inf")
    except (TypeError, ValueError):
        cap_min_f = float("inf")

    # On-route thresholds (prefer decision-layer thresholds if present; else reuse constants).
    onroute_km = filter_thresholds.get("onroute_max_detour_km", None)
    onroute_min = filter_thresholds.get("onroute_max_detour_min", None)

    try:
        onroute_km_th = float(onroute_km) if onroute_km is not None else (
            float(ONROUTE_MAX_DETOUR_KM) if (ONROUTE_MAX_DETOUR_KM and float(ONROUTE_MAX_DETOUR_KM) > 0) else 0.5
        )
    except (TypeError, ValueError):
        onroute_km_th = 0.5

    try:
        onroute_min_th = float(onroute_min) if onroute_min is not None else (
            float(ONROUTE_MAX_DETOUR_MIN) if (ONROUTE_MAX_DETOUR_MIN and float(ONROUTE_MAX_DETOUR_MIN) > 0) else 3.0
        )
    except (TypeError, ValueError):
        onroute_min_th = 3.0

    hard_pass: List[Dict[str, Any]] = []
    for s in stations or []:
        # must have a predicted price
        p = (s or {}).get(pred_key)
        if p is None:
            continue
        try:
            _ = float(p)
        except (TypeError, ValueError):
            continue

        # must be within caps
        try:
            km = float((s or {}).get("detour_distance_km") or 0.0)
        except (TypeError, ValueError):
            km = 0.0
        try:
            mins = float((s or {}).get("detour_duration_min") or 0.0)
        except (TypeError, ValueError):
            mins = 0.0

        km = max(km, 0.0)
        mins = max(mins, 0.0)

        if km <= cap_km_f and mins <= cap_min_f:
            hard_pass.append(s)

    # Worst on-route price among hard-pass
    onroute_prices: List[float] = []
    for s in hard_pass:
        try:
            km = float((s or {}).get("detour_distance_km") or 0.0)
        except (TypeError, ValueError):
            km = 0.0
        try:
            mins = float((s or {}).get("detour_duration_min") or 0.0)
        except (TypeError, ValueError):
            mins = 0.0
        km = max(km, 0.0)
        mins = max(mins, 0.0)

        if km <= onroute_km_th and mins <= onroute_min_th:
            try:
                onroute_prices.append(float((s or {}).get(pred_key)))
            except (TypeError, ValueError):
                continue

    onroute_worst_price = max(onroute_prices) if onroute_prices else None

    if onroute_worst_price is None:
        value_view = list(hard_pass)
    else:
        value_view = []
        for s in hard_pass:
            try:
                if float((s or {}).get(pred_key)) <= float(onroute_worst_price):
                    value_view.append(s)
            except (TypeError, ValueError):
                continue

    meta = {
        "pred_key": pred_key,
        "cap_km": cap_km_f,
        "cap_min": cap_min_f,
        "onroute_km_th": float(onroute_km_th),
        "onroute_min_th": float(onroute_min_th),
        "hard_pass_n": int(len(hard_pass)),
        "onroute_worst_price": onroute_worst_price,
        "value_view_n": int(len(value_view)),
    }
    return value_view, meta

def _render_metric_grid(items: list[tuple[str, str]]) -> None:
    # Build HTML WITHOUT leading indentation/newline, otherwise Markdown treats it as code.
    cards = []
    for label, value in items:
        cards.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{_safe_text(label)}</div>"
            f"<div class='metric-value'>{_safe_text(value)}</div>"
            "</div>"
        )

    html_block = "<div class='metric-grid'>" + "".join(cards) + "</div>"
    st.markdown(html_block, unsafe_allow_html=True)

def _display_best_station(
    best_station: Dict[str, Any],
    fuel_code: str,
    litres_to_refuel: Optional[float] = None,
    *,
    ranked_stations: Optional[List[Dict[str, Any]]] = None,
    debug_mode: bool = False,
    compact: bool = False,
) -> None:
    """
    Render a panel with information about the recommended station,
    including a short explanation whether current price or a forecast
    was used, and (if available) economic detour metrics.
    """
    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"

    econ_net_key = f"econ_net_saving_eur_{fuel_code}"
    econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
    econ_detour_fuel_key = f"econ_detour_fuel_l_{fuel_code}"
    econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
    econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
    econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
    econ_baseline_key = f"econ_baseline_price_{fuel_code}"

    if not best_station:
        st.info("No station could be recommended (no valid predictions).")
        return

    station_name = best_station.get("tk_name") or best_station.get("osm_name") or best_station.get("name")
    brand = best_station.get("brand")
    city = _reverse_geocode_station_city(best_station) or best_station.get("city")

    # Line 1: brand preferred; if missing, fall back to station name
    base_name = brand if (brand and str(brand).strip()) else station_name
    city_clean = str(city).strip() if city else ""
    title_line = f"{base_name} in {city_clean}" if (base_name and city_clean) else base_name

    # Line 2: full address (reverse geocode best station only)
    formatted_address = _reverse_geocode_station_address(best_station)
    subtitle_line = formatted_address or (f"{brand}, {city}" if (brand or city) else "")

    label_html = _safe_text("Recommended station:")
    name_html = _safe_text(title_line) if title_line else ""
    addr_html = _safe_text(subtitle_line) if subtitle_line else ""

    station_header_html = (
        "<div class='station-header'>"
        f"<div class='label'>{label_html}</div>"
        f"<div class='name'>{name_html}</div>"
        + (f"<div class='addr'>{addr_html}</div>" if addr_html else "")
        + "</div>"
    )
    st.markdown(station_header_html, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Required fields for the 6-metric hero card (avoid NameError)
    # ------------------------------------------------------------------
    pred_price = best_station.get(pred_key)
    current_price = best_station.get(current_key)

    # Detour distance (km)
    detour_km = best_station.get("detour_distance_km")
    if detour_km is None:
        detour_km = best_station.get("detour_km")

    # Detour time (min) - only used in non-compact sections, but define safely
    detour_min = best_station.get("detour_duration_min")
    if detour_min is None:
        detour_min = best_station.get("detour_min")

    # Distance along route (meters) – correct key + avoid falsy 0 being dropped
    _dist_candidates = [
        "distance_along_m",          # <-- this is what your pipeline/presenters uses
        "distance_along_route_m",
        "distance_m_along_route",
        "dist_m_along_route",
        "distance_to_station_m",
        "station_distance_m",
        "distance_m",
        "dist_m",
    ]

    dist_m = None
    for k in _dist_candidates:
        if k in best_station and best_station.get(k) is not None:
            dist_m = best_station.get(k)
            break

    # Compute "on-route worst price" + "net saving vs on-route worst"
    onroute_worst, net_vs_worst = _compute_onroute_worst_and_net_vs_worst(
        best_station=best_station,
        ranked_stations=ranked_stations,
        fuel_code=fuel_code,
        litres_to_refuel=litres_to_refuel,
    )

    # Distance along route (km string)
    if dist_m is None:
        dist_str = "—"
    else:
        try:
            dist_km = float(dist_m) / 1000.0
            dist_str = f"{dist_km:.1f} km"
        except (TypeError, ValueError):
            dist_str = "—"

    hero_items = [
        ("Safe up to:", "—" if net_vs_worst is None else _format_eur(net_vs_worst)),        
        (f"Predicted {fuel_code.upper()} price:", _format_price(pred_price)),
        (f"Current {fuel_code.upper()} price:", _format_price(current_price)),
        ("Distance to station:", dist_str),
        ("Worst on-route price:", "—" if onroute_worst is None else _format_price(onroute_worst)),
        ("Detour distance", _format_km(detour_km)),
    ]

    _render_metric_grid(hero_items)

    # ------------------------------------------------------------------
    # Compact hero mode (Page 1): stop here after the 6 required metrics
    # ------------------------------------------------------------------
    if compact:
        return

    # Economic detour metrics (if available)
    if econ_net_key in best_station:
        baseline_price = best_station.get(econ_baseline_key)
        gross_saving = best_station.get(econ_gross_key)
        detour_fuel_l = best_station.get(econ_detour_fuel_key)
        detour_fuel_cost = best_station.get(econ_detour_fuel_cost_key)
        time_cost = best_station.get(econ_time_cost_key)
        net_saving = best_station.get(econ_net_key)
        breakeven_liters = best_station.get(econ_breakeven_key)

        st.markdown("#### Economic impact of this detour")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Baseline on-route price",
                _format_price(baseline_price),
            )
        with col_b:
            st.metric(
                "Gross saving",
                _format_eur(gross_saving),
            )
        with col_c:
            st.metric(
                "Detour fuel",
                _format_liters(detour_fuel_l),
            )

        col_d, col_e, col_f = st.columns(3)
        with col_d:
            st.metric(
                "Detour fuel cost",
                _format_eur(detour_fuel_cost),
            )
        with col_e:
            st.metric(
                "Time cost",
                _format_eur(time_cost),
            )
        with col_f:
            st.metric(
                "Net saving",
                _format_eur(net_saving),
            )
        
        # Trust note: very small extra distance can still imply meaningful extra time
        # due to routing penalties (turns, access roads, traffic controls) and because
        # baseline vs via-station routes are separate routing solutions.
        try:
            _detour_km_disp = float(detour_km) if detour_km is not None else 0.0
        except (TypeError, ValueError):
            _detour_km_disp = 0.0

        try:
            _detour_min_disp = float(detour_min) if detour_min is not None else 0.0
        except (TypeError, ValueError):
            _detour_min_disp = 0.0

        if _detour_km_disp < 0.2 and _detour_min_disp > 3.0:
            st.info(
                "Note: Even if the extra detour distance is very small, the extra detour time can be larger "
                "due to routing/time penalties (turns, access roads, traffic controls) and because the baseline "
                "route and the via-station route are computed separately. Distance and time therefore do not "
                "need to scale linearly."
            )

        # ------------------------------------------------------------
        # Global comparison (upper-bound + typical): chosen vs ON-ROUTE spread
        # ------------------------------------------------------------
        if ranked_stations and litres_to_refuel is not None and litres_to_refuel > 0:
            pred_key = f"pred_price_{fuel_code}"
            econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
            econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"

            # Define "on-route" for the global comparison block.
            # If imported constants are extremely strict (or 0), fall back to a sensible,
            # user-comprehensible definition of "essentially on the route".
            ONROUTE_KM_TH = (
                float(ONROUTE_MAX_DETOUR_KM)
                if (ONROUTE_MAX_DETOUR_KM is not None and float(ONROUTE_MAX_DETOUR_KM) > 0)
                else 0.5
            )
            ONROUTE_MIN_TH = (
                float(ONROUTE_MAX_DETOUR_MIN)
                if (ONROUTE_MAX_DETOUR_MIN is not None and float(ONROUTE_MAX_DETOUR_MIN) > 0)
                else 3.0
            )

            # Collect on-route stations (based on clamped extra detour)
            onroute_prices: List[float] = []
            onroute_station_for_price: Dict[float, Dict[str, Any]] = {}

            for s in ranked_stations:
                p = s.get(pred_key)
                if p is None:
                    continue

                # Use the same "extra detour" definition as the economics layer (clamped)
                raw_km = s.get("detour_distance_km")
                raw_min = s.get("detour_duration_min")

                try:
                    km = float(raw_km) if raw_km is not None else 0.0
                except (TypeError, ValueError):
                    km = 0.0
                try:
                    mins = float(raw_min) if raw_min is not None else 0.0
                except (TypeError, ValueError):
                    mins = 0.0

                km = max(km, 0.0)
                mins = max(mins, 0.0)

                if km <= ONROUTE_KM_TH and mins <= ONROUTE_MIN_TH:
                    pf = float(p)
                    onroute_prices.append(pf)
                    onroute_station_for_price[pf] = s

            # Explain why the block might not show (print once)
            if debug_mode:
                st.caption(
                    f"DEBUG (global block): on-route thresholds km≤{ONROUTE_KM_TH:.2f}, "
                    f"min≤{ONROUTE_MIN_TH:.1f}; onroute_prices={len(onroute_prices)} "
                    f"(from ranked_stations={len(ranked_stations)})."
                )

            # Only show if we have a meaningful on-route set
            if len(onroute_prices) >= 2 and best_station.get(pred_key) is not None:
                onroute_series = pd.Series(onroute_prices)
                onroute_best = float(onroute_series.min())
                onroute_median = float(onroute_series.median())
                onroute_worst = float(onroute_series.max())

                chosen_price = float(best_station[pred_key])

                detour_fuel_cost_used = float(best_station.get(econ_detour_fuel_cost_key) or 0.0)
                time_cost_used = float(best_station.get(econ_time_cost_key) or 0.0)

                # Gross + net vs references
                gross_vs_worst = (onroute_worst - chosen_price) * float(litres_to_refuel)
                net_vs_worst = gross_vs_worst - detour_fuel_cost_used - time_cost_used

                gross_vs_median = (onroute_median - chosen_price) * float(litres_to_refuel)
                net_vs_median = gross_vs_median - detour_fuel_cost_used - time_cost_used

                gross_vs_best = (onroute_best - chosen_price) * float(litres_to_refuel)
                net_vs_best = gross_vs_best - detour_fuel_cost_used - time_cost_used

                worst_station = onroute_station_for_price.get(onroute_worst, {})
                worst_name = (
                    worst_station.get("tk_name")
                    or worst_station.get("osm_name")
                    or worst_station.get("name")
                    or worst_station.get("station_name")
                    or "Worst on-route station"
                )

                st.subheader("Potential savings versus on-route price spread")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("On-route best price", _format_price(onroute_best))
                with col2:
                    st.metric("On-route median price", _format_price(onroute_median))
                with col3:
                    st.metric("On-route worst price", _format_price(onroute_worst))

                col4, col5, col6 = st.columns(3)
                with col4:
                    st.metric("Chosen station price", _format_price(chosen_price))
                with col5:
                    st.metric("Gross saving vs on-route worst", _format_eur(gross_vs_worst))
                with col6:
                    st.metric("Net saving vs on-route worst", _format_eur(net_vs_worst))

                col7, col8, col9 = st.columns(3)
                with col7:
                    st.metric("Gross saving vs on-route median", _format_eur(gross_vs_median))
                with col8:
                    st.metric("Net saving vs on-route median", _format_eur(net_vs_median))
                with col9:
                    st.metric("Net saving vs on-route best", _format_eur(net_vs_best))

                st.caption(
                    f"On-route stations are defined here as ≤ {ONROUTE_KM_TH:.2f} km and ≤ {ONROUTE_MIN_TH:.1f} min extra detour "
                    f"(n = {len(onroute_prices)}). The “vs worst” comparison is an upper-bound scenario; “vs median” is a typical "
                    f"on-route reference; “vs best” is a lower-bound. Worst on-route station: **{worst_name}**."
                )

                if debug_mode:
                    st.caption(
                        "Debug note: This block uses predicted prices at each station's arrival time (pred_price_*), consistent with the ranking logic."
                    )
            else:
                if debug_mode:
                    st.warning(
                        "Global comparison block not shown because fewer than 2 stations qualified as 'on-route' "
                        f"(thresholds km≤{ONROUTE_KM_TH:.2f}, min≤{ONROUTE_MIN_TH:.1f})."
                    )


def _display_station_details_panel(
    station: Dict[str, Any],
    fuel_code: str,
    *,
    litres_to_refuel: Optional[float] = None,
    debug_mode: bool = False,
) -> None:
    """
    Render an extensive station details panel (triggered by map click selection).

    The panel is designed for two-level interaction:
      - hover tooltip for quick glance,
      - click selection for deep dive.
    """
    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"

    name = station.get("tk_name") or station.get("osm_name") or "Unknown"
    brand = station.get("brand") or ""

    st.markdown("### Station details")
    header = f"**{name}**"
    if brand:
        header += f"  \n{brand}"
    st.markdown(header)

    station_uuid = _station_uuid(station)
    if station_uuid:
        st.caption(f"Station UUID: {station_uuid}")

    tab_overview, tab_prices, tab_econ, tab_timing, tab_raw = st.tabs(
        ["Overview", "Prices", "Economics", "Timing", "Raw"]
    )

    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current price", _fmt_price(station.get(current_key)))
        col2.metric("Predicted price", _fmt_price(station.get(pred_key)))

        detour_km = station.get("detour_distance_km") or station.get("detour_km")
        detour_min = station.get("detour_duration_min") or station.get("detour_min")
        col3.metric("Detour distance", _fmt_km(detour_km))
        col4.metric("Detour time", _fmt_min(detour_min))

        # Fuel-specific economics keys (consistent with the rest of the app)
        econ_net_key = f"econ_net_saving_eur_{fuel_code}"
        econ_baseline_key = f"econ_baseline_price_{fuel_code}"
        econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"

        if econ_net_key in station:
            st.markdown("#### Economic summary")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Baseline on-route price", _fmt_price(station.get(econ_baseline_key)))
            col_b.metric("Net saving", _fmt_eur(station.get(econ_net_key)))
            col_c.metric("Break-even litres", "—" if station.get(econ_breakeven_key) is None else f"{float(station.get(econ_breakeven_key)):.2f}")
        else:
            st.info(
                "Economic metrics are not available for this station (missing econ_* fields). "
                "This usually means the station was not part of the ranked/evaluated set."
            )

    with tab_prices:
        rows = [
            {"Metric": "Current price", "Value": _fmt_price(station.get(current_key))},
            {"Metric": "Predicted price", "Value": _fmt_price(station.get(pred_key))},
        ]
        for lag in ("1d", "2d", "3d", "7d"):
            rows.append(
                {
                    "Metric": f"Price lag {lag}",
                    "Value": _fmt_price(station.get(f"price_lag_{lag}_{fuel_code}")),
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        debug_h_key = f"debug_{fuel_code}_horizon_used"
        debug_ucp_key = f"debug_{fuel_code}_used_current_price"
        if debug_h_key in station or debug_ucp_key in station:
            st.markdown("#### Forecast basis")
            st.write(
                {
                    "used_current_price": station.get(debug_ucp_key),
                    "horizon_used": station.get(debug_h_key),
                }
            )

    with tab_econ:
        econ_net_key = f"econ_net_saving_eur_{fuel_code}"
        econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
        econ_detour_fuel_key = f"econ_detour_fuel_l_{fuel_code}"
        econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
        econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
        econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
        econ_baseline_key = f"econ_baseline_price_{fuel_code}"

        if econ_net_key not in station:
            st.info("No economic metrics available for this station.")
        else:
            econ_rows = [
                {"Metric": "Baseline on-route price", "Value": _fmt_price(station.get(econ_baseline_key))},
                {"Metric": "Gross saving", "Value": _fmt_eur(station.get(econ_gross_key))},
                {"Metric": "Detour fuel [L]", "Value": "—" if station.get(econ_detour_fuel_key) is None else f"{float(station.get(econ_detour_fuel_key)):.2f}"},
                {"Metric": "Detour fuel cost", "Value": _fmt_eur(station.get(econ_detour_fuel_cost_key))},
                {"Metric": "Time cost", "Value": _fmt_eur(station.get(econ_time_cost_key))},
                {"Metric": "Net saving", "Value": _fmt_eur(station.get(econ_net_key))},
                {"Metric": "Break-even litres", "Value": "—" if station.get(econ_breakeven_key) is None else f"{float(station.get(econ_breakeven_key)):.2f}"},
            ]
            st.dataframe(pd.DataFrame(econ_rows), hide_index=True, use_container_width=True)

            if litres_to_refuel is not None:
                st.caption(f"Economics computed assuming refuel amount: {litres_to_refuel:.1f} L")

    with tab_timing:
        eta = station.get("eta")
        st.write({"eta_raw": eta})

        timing_fields = [
            f"debug_{fuel_code}_current_time_cell",
            f"debug_{fuel_code}_cells_ahead_raw",
            f"debug_{fuel_code}_minutes_ahead",
            f"debug_{fuel_code}_minutes_to_arrival",
            f"debug_{fuel_code}_eta_utc",
            f"debug_{fuel_code}_horizon_used",
            f"debug_{fuel_code}_used_current_price",
        ]
        timing_payload = {k: station.get(k) for k in timing_fields if k in station}
        if timing_payload:
            st.write(timing_payload)
        else:
            st.info("No timing/debug metadata available for this station.")

    with tab_raw:
        if debug_mode:
            st.json(station)
        else:
            st.info("Enable 'Debug mode' in the sidebar to see the full raw station payload.")


BRAND_FILTER_ALIASES: dict[str, list[str]] = {
    "ARAL": ["ARAL"],
    "AVIA": ["AVIA", "AVIA XPress", "AVIA Xpress"],
    "AGIP ENI": ["Agip", "AGIP ENI"],
    "Shell": ["Shell"],
    "Total": ["Total", "TotalEnergies"],
    "ESSO": ["ESSO"],
    "JET": ["JET"],
    "ORLEN": ["ORLEN"],
    "HEM": ["HEM"],
    "OMV": ["OMV"],
}


def _normalize_brand(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    # Case-insensitive + whitespace normalization
    return " ".join(s.upper().split())


def _brand_filter_allowed_set(selected: list[str]) -> tuple[set[str], dict[str, list[str]]]:
    """
    Returns:
      - allowed_norm: normalized allowed brand strings
      - aliases_used: canonical -> alias list used (for auditability on Page 02)
    """
    aliases_used: dict[str, list[str]] = {}
    allowed_norm: set[str] = set()

    for canon in selected:
        canon = str(canon)
        alias_list = BRAND_FILTER_ALIASES.get(canon, [canon])
        aliases_used[canon] = list(alias_list)
        for a in alias_list:
            allowed_norm.add(_normalize_brand(a))

    return allowed_norm, aliases_used

# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Fuel Station Recommender",
        layout="wide",
    )

    # Redis-backed persistence (best-effort)
    # IMPORTANT: preserve widget-managed keys so Redis restore does not clobber user clicks
    _preserve_top_nav = st.session_state.get("top_nav")
    _preserve_sidebar_view = st.session_state.get("sidebar_view")
    _preserve_map_style_mode = st.session_state.get("map_style_mode")  # <-- ADD THIS

    init_session_context()
    ensure_persisted_state_defaults(st.session_state)

    # Keep refresh persistence working:
    # - overwrite_existing=True restores persisted values on a cold start / hard refresh
    # - then we re-apply widget keys if the user interaction already set them for this rerun
    restore_persisted_state(overwrite_existing=True)

    if _preserve_top_nav is not None:
        st.session_state["top_nav"] = _preserve_top_nav
    if _preserve_sidebar_view is not None:
        st.session_state["sidebar_view"] = _preserve_sidebar_view
    if _preserve_map_style_mode is not None:
        st.session_state["map_style_mode"] = _preserve_map_style_mode  # <-- ADD THIS

    # Apply custom CSS styles
    apply_app_css()

    st.title("Fuel Station Recommender")
    st.caption("##### Plan a route and identify the best-value refueling stops.")

    # --- Top navigation (nicer UI) ---
    NAV_TARGETS = {
        "Home": "streamlit_app.py",
        "Analytics": "pages/02_route_analytics.py",
        "Station": "pages/03_station_details.py",
        "Explorer": "pages/04_station_explorer.py",
    }

    selected = st.segmented_control(
        label="Page navigation",
        options=list(NAV_TARGETS.keys()),
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="top_nav",  # this is the single source of truth
    )

    target = NAV_TARGETS.get(selected, "streamlit_app.py")

    # Only switch away from Home when needed
    if target != "streamlit_app.py":
        st.switch_page(target)


    # ----------------------------------------------------------------------
    # Sidebar (standardized)
    # ----------------------------------------------------------------------
    sidebar = render_sidebar()

    sidebar_view = sidebar.view
    run_clicked = sidebar.run_clicked

    start_locality = sidebar.start_locality
    end_locality = sidebar.end_locality
    start_address = sidebar.start_address
    end_address = sidebar.end_address

    fuel_label = sidebar.fuel_label
    fuel_code = sidebar.fuel_code

    use_economics = sidebar.use_economics
    litres_to_refuel = sidebar.litres_to_refuel
    consumption_l_per_100km = sidebar.consumption_l_per_100km
    value_of_time_eur_per_hour = sidebar.value_of_time_eur_per_hour
    max_detour_km = sidebar.max_detour_km
    max_detour_min = sidebar.max_detour_min
    min_net_saving_eur = sidebar.min_net_saving_eur
    filter_closed_at_eta = sidebar.filter_closed_at_eta
    brand_filter_selected = list(getattr(sidebar, "brand_filter_selected", []) or [])


    debug_mode = bool(st.session_state.get("debug_mode", False))


    # -----------------------------
    # Parameters hash (controls recompute warnings)
    # -----------------------------
    params = {
        "start_locality": start_locality,
        "end_locality": end_locality,
        "start_address": start_address,
        "end_address": end_address,
        "fuel_label": fuel_label,
        "fuel_code": fuel_code,
        "use_economics": use_economics,
        "litres_to_refuel": litres_to_refuel,
        "consumption_l_per_100km": consumption_l_per_100km,
        "value_of_time_eur_per_hour": value_of_time_eur_per_hour,
        "max_detour_km": max_detour_km,
        "max_detour_min": max_detour_min,
        "min_net_saving_eur": min_net_saving_eur,
        "filter_closed_at_eta": bool(filter_closed_at_eta),
    }
    params_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    have_cached = st.session_state.get("last_run") is not None
    cached_is_stale = have_cached and (st.session_state.get("last_params_hash") != params_hash)

    # If user changed settings, keep showing cached results but warn
    if not run_clicked and cached_is_stale:
        st.warning("Inputs changed. Showing previous results.")

    # -----------------------------
    # Compute (ONLY when run_clicked)
    # -----------------------------
    if run_clicked:
        st.session_state["selected_station_uuid"] = None  # reset selection on recompute

        # 1) Integration
        route_info = None

        if not start_locality or not end_locality:
            st.error("Please provide a Start and Destination (city or full address).")
            return

        try:
            inputs = RouteRunInputs(
                start=start_locality,
                end=end_locality,
                fuel_code=fuel_code,
                litres_to_refuel=litres_to_refuel,
                use_economics=use_economics,
                debug_mode=debug_mode,
                car_consumption_l_per_100km=consumption_l_per_100km,
                value_of_time_eur_per_h=value_of_time_eur_per_hour,
                max_detour_time_min=max_detour_min,
                max_detour_distance_km=max_detour_km,
                min_net_saving_eur=min_net_saving_eur,
            )

            integration_kwargs = {
                "start_locality": start_locality,
                "end_locality": end_locality,
                "start_address": start_address,
                "end_address": end_address,
                "use_realtime": True,
                "filter_closed_at_eta": bool(filter_closed_at_eta),
            }

            if use_economics:
                ranking_kwargs = {
                    "litres_to_refuel": litres_to_refuel,
                    "consumption_l_per_100km": consumption_l_per_100km,
                    "value_of_time_per_hour": value_of_time_eur_per_hour,
                    "max_detour_km": max_detour_km,
                    "max_detour_min": max_detour_min,
                    "min_net_saving_eur": min_net_saving_eur,
                }
                recommendation_kwargs = dict(ranking_kwargs)
            else:
                ranking_kwargs = {}
                recommendation_kwargs = {}

            last_run = run_route_recommendation(
                inputs,
                integration_kwargs=integration_kwargs,
                ranking_kwargs=ranking_kwargs,
                recommendation_kwargs=recommendation_kwargs,
            )

            # ------------------------------------------------------------
            # Advanced Settings: Brand whitelist filter (upstream, before caching)
            # ------------------------------------------------------------
            brand_filtered_out_n = 0
            brand_filter_aliases_used: dict[str, list[str]] = {}

            if brand_filter_selected:
                allowed_norm, brand_filter_aliases_used = _brand_filter_allowed_set(brand_filter_selected)

                # Filter candidate stations
                stations_before = list(last_run.get("stations") or [])

                # Keep a stable "map universe" even when brand filtering is active
                # (so brand-excluded stations can still be shown as red markers).
                last_run["stations_for_map_all"] = list(stations_before)

                kept_stations = [
                    s for s in stations_before
                    if _normalize_brand((s or {}).get("brand")) in allowed_norm
                ]

                brand_filtered_out_n = len(stations_before) - len(kept_stations)

                # Keep ranked/best consistent with filtered candidates
                ranked_before = list(last_run.get("ranked") or [])
                kept_ranked = [
                    s for s in ranked_before
                    if _normalize_brand((s or {}).get("brand")) in allowed_norm
                ]

                last_run["stations"] = kept_stations
                last_run["ranked"] = kept_ranked

                # Also keep the pre-brand ranked list for completeness (optional, but useful for debugging).
                last_run["ranked_for_map_all"] = list(ranked_before)

                # Recompute best_station / best_uuid if needed
                best_station = last_run.get("best_station")
                if best_station is not None and _normalize_brand(best_station.get("brand")) not in allowed_norm:
                    best_station = None

                if best_station is None and kept_ranked:
                    best_station = kept_ranked[0]

                last_run["best_station"] = best_station
                last_run["best_uuid"] = _station_uuid(best_station) if best_station else None

            # If no brand filter is active, the map universe is just the stations list.
            last_run.setdefault("stations_for_map_all", list(last_run.get("stations") or []))
            last_run.setdefault("ranked_for_map_all", list(last_run.get("ranked") or []))

        except AppError as exc:
            st.error(exc.user_message)
            if exc.remediation:
                st.info(exc.remediation)
            return
        except Exception as exc:
            st.error("Unexpected error. Please try again. If it persists, check logs.")
            st.caption(str(exc))
            return

        # Unpack results (same semantics as before)
        stations = last_run.get("stations") or []
        ranked = last_run.get("ranked") or []
        best_station = last_run.get("best_station")
        route_info = last_run.get("route_info")

        # ------------------------------------------------------------
        # Advanced Settings: persist open-at-ETA filter metadata for Page 2
        # ------------------------------------------------------------
        # Prefer an explicit top-level field, but fall back to route_info (route_meta) if present.
        closed_at_eta_filtered_n = last_run.get("closed_at_eta_filtered_n")

        if closed_at_eta_filtered_n is None and isinstance(route_info, dict):
            closed_at_eta_filtered_n = route_info.get("closed_at_eta_filtered_n")

        # Defensive normalization (some pipelines may return None / float / str)
        try:
            if closed_at_eta_filtered_n is not None:
                closed_at_eta_filtered_n = int(closed_at_eta_filtered_n)
        except (TypeError, ValueError):
            closed_at_eta_filtered_n = None

        # Store a dedicated block so Page 2 can read it consistently
        last_run["advanced_settings"] = {
            "filter_closed_at_eta": bool(filter_closed_at_eta),
            "closed_at_eta_filtered_n": closed_at_eta_filtered_n,
            "brand_filter_selected": list(brand_filter_selected),
            "brand_filter_aliases": dict(brand_filter_aliases_used),
            "brand_filtered_out_n": int(brand_filtered_out_n),
        }

        if not stations:
            st.warning("No stations returned by the integration pipeline.")
            return

        # If economics filtering eliminates everything, keep a SHORT message here.
        # Detailed explanation moves to Route Analytics via run_summary.
        if not ranked:
            st.error("No stations passed the current filters.")
            st.caption("Open Route Analytics for details, or relax constraints in the sidebar and run again.")
            # Still cache the run so Page 2 can explain what happened.
            last_run["ranked"] = []
            # continue to cache below (do not return early)

        # Build a run summary for Page 2 (no loss of info; just not printed on Page 1).
        filtered_count = (len(stations) - len(ranked)) if ranked else (len(stations))
        last_run["run_summary"] = {
            "stations_total": len(stations),
            "stations_ranked": len(ranked),
            "stations_filtered_out": filtered_count if use_economics else 0,
            "use_economics": bool(use_economics),
            "filter_closed_at_eta": bool(filter_closed_at_eta),
            "closed_at_eta_filtered_n": closed_at_eta_filtered_n,
            "constraints": {
                "max_detour_km": float(max_detour_km),
                "max_detour_min": float(max_detour_min),
                "min_net_saving_eur": float(min_net_saving_eur),
                "litres_to_refuel": float(litres_to_refuel),
                "consumption_l_per_100km": float(consumption_l_per_100km),
                "value_of_time_eur_per_hour": float(value_of_time_eur_per_hour),
            "brand_filter_selected": list(brand_filter_selected),
            "brand_filter_aliases": dict(brand_filter_aliases_used),
            "brand_filtered_out_n": int(brand_filtered_out_n),
            },
        }

        # Add UI-only metadata to last_run and cache
        last_run["params"] = params
        last_run["fuel_code"] = fuel_code
        last_run["litres_to_refuel"] = litres_to_refuel
        last_run["debug_mode"] = debug_mode

        last_run["computed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state["last_run"] = last_run
        st.session_state["last_params_hash"] = params_hash


    # -----------------------------
    # Render from cache (ALWAYS)
    # -----------------------------
    cached = st.session_state.get("last_run")
    if not cached:
        return

    stations = cached["stations"]
    ranked = cached["ranked"]
    best_station = cached["best_station"]
    best_uuid = cached["best_uuid"]
    route_info = cached.get("route_info")
    route_coords = cached.get("route_coords")

    # ------------------------------------------------------------
    # Page 01 metrics should use the Page-02 "upper table" logic:
    # stations priced <= worst on-route station (value view).
    # ------------------------------------------------------------
    run_summary = cached.get("run_summary") or {}
    constraints = (run_summary.get("constraints") or {}) if isinstance(run_summary, dict) else {}
    filter_log = cached.get("filter_log") or {}
    filter_thresholds = (filter_log.get("thresholds") or {}) if isinstance(filter_log, dict) else {}

    value_view_stations, value_view_meta = _compute_value_view_stations(
        stations=list(cached.get("stations") or []),
        fuel_code=fuel_code,
        constraints=constraints,
        filter_thresholds=filter_thresholds,
    )

    # Persist for Page 02 / other pages (and avoid re-computation drift)
    cached["value_view_stations"] = value_view_stations
    cached["value_view_meta"] = value_view_meta
    st.session_state["last_run"] = cached

    _display_best_station(
        best_station,
        fuel_code,
        litres_to_refuel=litres_to_refuel,
        ranked_stations=value_view_stations,  # <-- key change
        debug_mode=debug_mode,
        compact=True,
    )


    # ----------------------------------------------------------------------
    # Map visualization (only for real route mode)
    # ----------------------------------------------------------------------
    if isinstance(route_info, dict) and route_info.get("route_coords"):
        try:
            # -----------------------------
            # Map section header row
            # -----------------------------
            if "map_style_mode" not in st.session_state:
                st.session_state["map_style_mode"] = "Standard"

            h_left, h_right = st.columns([0.70, 0.30], vertical_alignment="center")
            with h_left:
                st.subheader(
                    "Route and stations map",
                    help=(
                        "See legend below map."
                    ),
                    anchor=False,
                )
                st.caption("Hover or click a station marker to show details.")

            with h_right:
                st.segmented_control(
                    label="",
                    options=["Standard", "Satellite"],
                    selection_mode="single",
                    label_visibility="collapsed",
                    width="stretch",
                    key="map_style_mode",
                )

            use_satellite = (st.session_state.get("map_style_mode") == "Satellite")

            # -----------------------------
            # Map data (SOURCE OF TRUTH: route_info)
            # -----------------------------
            route_coords = route_info.get("route_coords") or []

            # Via-route caching (persisted in last_run)
            # -----------------------------
            best_uuid = cached.get("best_uuid") or _station_uuid(best_station) if best_station else None
            departure_time = (route_info or {}).get("departure_time", "now")

            # A stable cache key for the via-route (changes only when route endpoints / best station / departure_time change)
            try:
                start_lon, start_lat = route_coords[0][0], route_coords[0][1]
                end_lon, end_lat = route_coords[-1][0], route_coords[-1][1]
                via_cache_key_desired = f"{best_uuid}|{start_lat:.6f},{start_lon:.6f}|{end_lat:.6f},{end_lon:.6f}|{departure_time}"
            except Exception:
                via_cache_key_desired = None

            # Reuse previously computed via route if available
            via_overview = cached.get("via_full_coords")
            via_cache_key_existing = cached.get("via_cache_key")

            should_recompute_via = (
                via_cache_key_desired is not None
                and via_cache_key_existing != via_cache_key_desired
            )

            # Only recompute when needed; never overwrite a previously valid route with None on transient failures
            if should_recompute_via and best_station and route_coords:
                try:
                    st_lat = float(best_station.get("lat")) if best_station.get("lat") is not None else None
                    st_lon = float(best_station.get("lon")) if best_station.get("lon") is not None else None

                    if st_lat is not None and st_lon is not None:
                        api_key = environment_check()
                        via = google_route_via_waypoint(
                            start_lat=start_lat,
                            start_lon=start_lon,
                            waypoint_lat=st_lat,
                            waypoint_lon=st_lon,
                            end_lat=end_lat,
                            end_lon=end_lon,
                            api_key=api_key,
                            departure_time=departure_time,
                        )

                        new_via = via.get("via_full_coords")

                        # Accept only a non-trivial coordinate list; otherwise keep the old one
                        if isinstance(new_via, list) and len(new_via) >= 2:
                            via_overview = new_via
                            cached["via_full_coords"] = via_overview
                            cached["via_cache_key"] = via_cache_key_desired
                            st.session_state["last_run"] = cached  # persist in-session immediately
                except Exception:
                    # Keep previous via_overview if present (do not overwrite with None)
                    pass

            # Calculate zoom level based on route extent (robust fallback)
            zoom_for_markers = 7.5
            if route_coords:
                try:
                    lons = [c[0] for c in route_coords if c and c[0] is not None]
                    lats = [c[1] for c in route_coords if c and c[1] is not None]
                    if lons and lats:
                        z = _calculate_zoom_for_bounds(
                            lon_min=min(lons),
                            lon_max=max(lons),
                            lat_min=min(lats),
                            lat_max=max(lats),
                            padding_percent=0.10,
                            map_width_px=700,
                            map_height_px=500,
                        )
                        if z is not None:
                            zoom_for_markers = float(z)
                except Exception:
                    zoom_for_markers = 7.5

            # -----------------------------
            # Map rendering (Mapbox GL JS with cooperative gestures)
            # Show ALL stations (stable universe) and color-code them.
            # -----------------------------
            stations_for_map_all = list(cached.get("stations_for_map_all") or cached.get("stations") or [])
            best_uuid = cached.get("best_uuid")

            # ------------------------------------------------------------------
            # Map click → Station Details deep-link (implemented via query params)
            #
            # maps.py sets:
            #   ?goto=Station&station_uuid=<uuid>
            # We resolve the station dict from the stable map-universe (stations_for_map_all),
            # store it for Page 03, then switch pages.
            # ------------------------------------------------------------------
            def _qp_get_first(key: str) -> str:
                try:
                    v = st.query_params.get(key)  # type: ignore[attr-defined]
                    if isinstance(v, (list, tuple)):
                        return str(v[0]) if v else ""
                    return str(v) if v is not None else ""
                except Exception:
                    try:
                        v = st.experimental_get_query_params().get(key, [])  # type: ignore[attr-defined]
                        return str(v[0]) if v else ""
                    except Exception:
                        return ""

            def _qp_clear(*keys: str) -> None:
                # Best-effort removal (avoid breaking older Streamlit versions)
                try:
                    for k in keys:
                        try:
                            if k in st.query_params:  # type: ignore[attr-defined]
                                del st.query_params[k]  # type: ignore[attr-defined]
                        except Exception:
                            pass
                except Exception:
                    try:
                        qp = st.experimental_get_query_params()  # type: ignore[attr-defined]
                        for k in keys:
                            qp.pop(k, None)
                        st.experimental_set_query_params(**qp)  # type: ignore[attr-defined]
                    except Exception:
                        return

            _goto = _qp_get_first("goto").strip()
            _clicked_uuid = _qp_get_first("station_uuid").strip()

            if _goto.lower() == "station" and _clicked_uuid:
                # Resolve station dict (so Page 03 can render even if station is not in ranked/stations)
                _station_by_uuid = {(_station_uuid(s) or ""): s for s in stations_for_map_all if _station_uuid(s)}
                st.session_state["selected_station_uuid"] = _clicked_uuid
                st.session_state["selected_station_data"] = _station_by_uuid.get(_clicked_uuid)
                st.session_state["selected_station_source"] = "map_click"

                # Keep nav consistent and switch page
                st.session_state["top_nav"] = "Station"
                _qp_clear("goto", "station_uuid")
                st.switch_page("pages/03_station_details.py")

            value_view_uuids = set()
            for s in (cached.get("value_view_stations") or []):
                u = _station_uuid(s)
                if u:
                    value_view_uuids.add(u)

            marker_category_by_uuid: Dict[str, str] = {}
            for s in stations_for_map_all:
                u = _station_uuid(s)
                if not u:
                    continue
                if best_uuid and u == best_uuid:
                    marker_category_by_uuid[u] = "best"
                elif u in value_view_uuids:
                    marker_category_by_uuid[u] = "better"
                else:
                    marker_category_by_uuid[u] = "other"

            map_html = create_mapbox_gl_html(
                route_coords=route_coords,
                stations=stations_for_map_all,  # <-- key change (was ranked)
                best_station_uuid=best_uuid,
                via_full_coords=via_overview,
                use_satellite=use_satellite,
                selected_station_uuid=st.session_state.get("selected_station_uuid"),
                marker_category_by_uuid=marker_category_by_uuid,  # <-- new
                height_px=560,
            )

            components.html(map_html, height=560, scrolling=False)

            # Legend + short explanation
            st.markdown(
                """
                <div class="map-legend">
                <div class="legend-item"><span class="legend-line baseline"></span><span>Baseline route</span></div>
                <div class="legend-item"><span class="legend-line via"></span><span>Via recommended station</span></div>

                <div class="legend-item"><span class="legend-dot best"></span><span>Best station</span></div>
                <div class="legend-item"><span class="legend-dot better"></span><span>Better than / equal to worst on-route station</span></div>
                <div class="legend-item"><span class="legend-dot other"></span><span>All other stations </span></div>
                </div>
                <div class="legend-note">
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.warning(f"Could not display map: {e}")

    # Two CTAs only: Route Analytics + Station Details
    # Station Details opens the selected station (if any), otherwise the recommended station.
    selected_uuid = st.session_state.get("selected_station_uuid")

    # Resolve station object for Station Details action
    station_for_details = None
    if selected_uuid:
        uuid_to_station: Dict[str, Dict[str, Any]] = {}
        for s in ranked:
            u = _station_uuid(s)
            if u:
                uuid_to_station[u] = s
        for s in stations:
            u = _station_uuid(s)
            if u and u not in uuid_to_station:
                uuid_to_station[u] = s
        station_for_details = uuid_to_station.get(selected_uuid)

    # Fallback to best station if no selection
    if station_for_details is None:
        station_for_details = best_station
        selected_uuid = _station_uuid(best_station) if best_station else None

    # Persist the current UX state snapshot (best-effort, writes only if changed)
    maybe_persist_state()


if __name__ == "__main__":
    main()
