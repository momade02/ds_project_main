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

from config.settings import load_env_once

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
        (f"Predicted {fuel_code.upper()} price:", _format_price(pred_price)),
        (f"Current {fuel_code.upper()} price:", _format_price(current_price)),
        ("Distance to station:", dist_str),
        ("Worst on-route price:", "—" if onroute_worst is None else _format_price(onroute_worst)),
        ("Safe up to:", "—" if net_vs_worst is None else _format_eur(net_vs_worst)),
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


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Fuel Station Recommender",
        layout="wide",
    )
# --- Permanent layout debug overlay + reduced top whitespace ---
    st.markdown(
        """
        <style>

        /* Reduce the default top padding above the first element */
        div.block-container { padding-top: 1rem; }

        /* --- Sidebar: pull content to the very top --- */

        /* Reduce padding in the sidebar header area (collapse button row) */
        section[data-testid="stSidebar"] div[data-testid="stSidebarHeader"] {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        }

        /* Reduce top padding of the actual sidebar content container */
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
        }

        /* Ensure the first element doesn't add its own top spacing */
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] > div:first-child {
        margin-top: -1rem !important;
        padding-top: 0 !important;
        }

        /* Outline the main structural containers so you see the usable layout */
        section[data-testid="stSidebar"] { outline: 2px dashed rgba(0,0,0,0.25); outline-offset: -2px; }
        section[data-testid="stMain"] { outline: 2px dashed rgba(0,0,0,0.25); outline-offset: -2px; }
        div.block-container { outline: 2px dashed rgba(255,0,0,0.25); outline-offset: -2px; }

        /* Outline each vertical block (helpful to see spacing between blocks) */
        div[data-testid="stVerticalBlock"] { outline: 1px dashed rgba(0,0,255,0.20); outline-offset: -1px; }

        /* Outline columns */
        div[data-testid="column"] { outline: 1px dashed rgba(0,128,0,0.20); outline-offset: -1px; }

        /* Tighten spacing above Streamlit caption blocks */
        div[data-testid="stCaptionContainer"] { margin-top: -1.3rem !important; }

        /* Recommended station hero card */
        .station-header {
        margin: 0.25rem auto 0.85rem auto;
        padding: 1.05rem 1.15rem;
        max-width: 980px;

        text-align: center;

        /* theme-consistent: teal-tinted surface */
        background: rgba(14, 116, 144, 0.08);                 /* based on primaryColor */
        border: 1px solid rgba(14, 116, 144, 0.22);
        border-radius: 1.05rem;
        box-shadow: 0 6px 18px rgba(31, 41, 55, 0.08);
        }

        /* --- Sidebar segmented control: equal-size buttons --- */
        section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [data-baseweb="button-group"]{
        display: flex !important;
        width: 100% !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] > div{
        flex: 1 1 0 !important;   /* equal widths */
        }

        section[data-testid="stSidebar"] div[data-testid="stSegmentedControl"] [data-baseweb="button-group"] button{
        width: 100% !important;
        justify-content: center !important; /* center text */
        }

        /* Keep dashed debug feel without being too loud */
        .station-header { outline: 1px dashed rgba(14, 116, 144, 0.35); outline-offset: -4px; }

        .station-header .label {
        margin: 0 0 0.30rem 0 !important;
        font-size: 1.10rem;
        font-weight: 650;
        letter-spacing: 0.01em;
        opacity: 0.95;
        }

        .station-header .name {
        margin: 0 0 0.25rem 0 !important;
        font-size: 1.70rem;     /* slightly larger than before */
        font-weight: 780;
        line-height: 1.15;
        }

        .station-header .addr {
        margin: 0 !important;
        font-size: 1.05rem;     /* a bit larger */
        font-weight: 600;
        opacity: 0.82;
        line-height: 1.25;
        }

        /* -----------------------------
        Responsive metric grid (3 cols desktop, 2 cols mobile)
        ------------------------------ */
        .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.25rem;
        }

        @media (max-width: 900px) {
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        }

        /* Only collapse to 1 column on very narrow screens */
        @media (max-width: 420px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
        }

        .metric-card {
        padding: 0.85rem 0.95rem;
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 0.9rem;
        }

        /* Keep your “dashed debug” feel for these custom cards too */
        .metric-grid { outline: 1px dashed rgba(0, 0, 255, 0.20); outline-offset: -1px; }
        .metric-card { outline: 1px dashed rgba(0, 128, 0, 0.20); outline-offset: -1px; }

        .metric-label {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 0 0 0.35rem 0;
        line-height: 1.2;
        }

        .metric-value {
        font-size: 1.7rem;
        font-weight: 650;
        margin: 0;
        line-height: 1.05;
        }

        /* -----------------------------
        Map header row (title + toggle inline)
        ------------------------------ */
       /* Map section header (force title + subtitle on separate lines) */
        .map-header { 
        display: block !important;
        margin-top: 0.70rem;
        margin-bottom: 0.35rem;
        }

        .map-title {
        display: block !important;
        font-size: 1.20rem;
        font-weight: 700;
        margin: 0 !important;
        line-height: 1.15;
        }

        .map-subtitle {
        display: block !important;
        margin: 0.20rem 0 0 0 !important;  /* small gap below title */
        opacity: 0.80;
        font-size: 0.98rem;
        line-height: 1.25;
}

        /* Make the map toggle button look consistent and compact */
        .st-key-toggle_basemap button {
        padding: 0.30rem 0.80rem !important;
        border: 1px solid rgba(31, 41, 55, 0.35) !important;
        border-radius: 10px !important;
        background: rgba(255, 255, 255, 0.90) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10) !important;
        }
        .st-key-toggle_basemap button:hover {
        border-color: rgba(31, 41, 55, 0.55) !important;
        }

        /* --- Top sidebar "Run recommender" button (key=run_recommender_btn_top) --- */
        section[data-testid="stSidebar"] div.st-key-run_recommender_btn_top button {
        background-color: #2E7D32 !important;  /* decent green */
        color: #ffffff !important;
        border: 1px solid #2E7D32 !important;
        }

        section[data-testid="stSidebar"] div.st-key-run_recommender_btn_top button:hover {
        background-color: #256628 !important;
        border-color: #256628 !important;
        }

        section[data-testid="stSidebar"] div.st-key-run_recommender_btn_top button:active {
        background-color: #1F5622 !important;
        border-color: #1F5622 !important;
        }

        section[data-testid="stSidebar"] div.st-key-run_recommender_btn_top button:focus {
        box-shadow: 0 0 0 0.2rem rgba(46, 125, 50, 0.35) !important;
        }

        /* Force sidebar width (unsupported CSS override) */
        section[data-testid="stSidebar"] {
        width: 380px !important;      /* pick your default */
        min-width: 380px !important;
        }

        /* --- Route input: stable icon column (no SVG) --- */

        .route-start-circle {
        width: 14px;
        height: 14px;
        border-radius: 999px;
        border: 2px solid rgba(49, 51, 63, 0.55);
        box-sizing: border-box;
        }

        .route-dots {
        display: flex;
        flex-direction: column;
        gap: 3px;
        margin: 8px 0;
        }

        .route-dots span {
        width: 3px;
        height: 3px;
        border-radius: 999px;
        background: rgba(49, 51, 63, 0.35);
        }

        /* Teardrop pin using pure CSS */
        .route-pin {
        width: 14px;
        height: 14px;
        background: #D32F2F;
        border-radius: 999px 999px 999px 0;
        transform: rotate(-45deg);
        position: relative;
        margin-top: 2px;
        }

        .route-pin::after {
        content: "";
        width: 6px;
        height: 6px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 999px;
        position: absolute;
        top: 4px;
        left: 4px;
        }

        .route-icon-col {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;     /* vertical centering */
        height: 100%;                /* take full column height */
        padding-top: 1.1rem;              /* remove manual offset */
        }


        /* Optional: reduce vertical gap between the two inputs (sidebar only). */
        section[data-testid="stSidebar"] div.st-key-start_locality { margin-bottom: -1.4rem !important; }

        /* =========================
        Sidebar Route (no columns)
        Icons on the RIGHT + inputs narrower
        ========================= */

        /* Reserve a fixed right gutter so inputs are proactively less wide */
        section[data-testid="stSidebar"] div.st-key-start_locality,
        section[data-testid="stSidebar"] div.st-key-end_locality {
        position: relative !important;
        }

        /* Narrow the visible input box by reducing its max-width */
        section[data-testid="stSidebar"] div.st-key-start_locality .stTextInput,
        section[data-testid="stSidebar"] div.st-key-end_locality .stTextInput {
        max-width: calc(100% - 46px) !important;   /* <- controls “proactively less wide” */
        }

        /* Ensure the input itself doesn’t overlap the icon area */
        section[data-testid="stSidebar"] div.st-key-start_locality input,
        section[data-testid="stSidebar"] div.st-key-end_locality input {
        padding-right: 0.75rem !important;
        }

        /* Start icon: circle on the right, vertically centered to the input */
        section[data-testid="stSidebar"] div.st-key-start_locality::after {
        content: "";
        position: absolute;
        right: 13px;
        top: 50%;
        transform: translateY(-50%);
        width: 14px;
        height: 14px;
        border-radius: 999px;
        border: 2px solid rgba(49, 51, 63, 0.55);
        box-sizing: border-box;
        pointer-events: none;
        }

        /* Destination icon: red pin on the right */
        section[data-testid="stSidebar"] div.st-key-end_locality::after {
        content: "";
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        width: 20px;
        height: 20px;
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
        pointer-events: none;

        /* Inline SVG pin */
        background-image: url("data:image/svg+xml;utf8,\
        <svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'>\
        <path fill='%23D32F2F' d='M12 2c-3.31 0-6 2.69-6 6 0 4.5 6 14 6 14s6-9.5 6-14c0-3.31-2.69-6-6-6zm0 8.5c-1.38 0-2.5-1.12-2.5-2.5S10.62 5.5 12 5.5s2.5 1.12 2.5 2.5S13.38 10.5 12 10.5z'/>\
        </svg>");
        }

        /* The three connector dots: aligned to the right gutter between inputs */
        section[data-testid="stSidebar"] .route-dots-right {
        width: 40px;
        margin-left: auto;
        margin-top: -0rem;      /* pulls dots closer to the top input */
        margin-bottom: -0.30rem;   /* pulls bottom input closer (reduce spacing) */
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 3px;
        pointer-events: none;
        }

        section[data-testid="stSidebar"] .route-dots-right span {
        width: 4px;
        height: 4px;
        border-radius: 999px;
        background: rgba(49, 51, 63, 0.35);
        display: block;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
    # --- end ---

    st.title("Fuel Station Recommender")
    st.caption("##### Plan a route and identify the best-value refueling stops.")

    # --- Top navigation (nicer UI) ---
    NAV_TARGETS = {
        "Home": "streamlit_app.py",
        "Analytics": "pages/02_route_analytics.py",
        "Station": "pages/03_station_details.py",
        "Explorer": "pages/04_explorer.py",
    }

    # Initialize only once (no default parameter needed afterwards)
    if "top_nav" not in st.session_state:
        st.session_state["top_nav"] = "Home"

    selected = st.segmented_control(
        label="",
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

    # Persist last clicked station (map selection)
    if "selected_station_uuid" not in st.session_state:
        st.session_state["selected_station_uuid"] = None
    # Persist last computed results so UI does not reset on reruns (sidebar changes / map clicks)
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None  # will store dict of computed outputs
    if "last_params_hash" not in st.session_state:
        st.session_state["last_params_hash"] = None

    # ----------------------------------------------------------------------
    # Sidebar "two-page" switcher: Settings vs Status
    # ----------------------------------------------------------------------
    if "sidebar_view" not in st.session_state:
        st.session_state["sidebar_view"] = "Settings"

    sidebar_view = st.sidebar.segmented_control(
        label="",
        options=["Settings", "Status"],
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="sidebar_view",
    )

    # Top "Run" button (always visible, full sidebar width)
    run_clicked_top = st.sidebar.button(
        "Run recommender",
        use_container_width=True,
        key="run_recommender_btn_top",
    )

    # Helper: read prior values even when Settings widgets are not rendered
    def _ss(key: str, default):
        return st.session_state.get(key, default)

    # ----------------------------------------------------------------------
    # SETTINGS VIEW (all inputs + run button)
    # ----------------------------------------------------------------------
    if sidebar_view == "Settings":

        st.sidebar.markdown("### Route")

        start_locality = st.sidebar.text_input(
            "Start",
            value=_ss("start_locality", "Tübingen"),
            key="start_locality",
            label_visibility="collapsed",
            placeholder="Start: city or full address",
        )

        # Right-side connector dots between the two inputs (visual only)
        st.sidebar.markdown(
            """
            <div class="route-dots-right" aria-hidden="true">
            <span></span><span></span><span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        end_locality = st.sidebar.text_input(
            "Destination",
            value=_ss("end_locality", "Sindelfingen"),
            key="end_locality",
            label_visibility="collapsed",
            placeholder="Destination: city or full address",
        )

        start_address = ""
        end_address = ""


        st.sidebar.subheader("Fuel type")

        fuel_label = st.sidebar.selectbox(
            "Fuel type",
            options=["E5", "E10", "Diesel"],
            index=["E5", "E10", "Diesel"].index(_ss("fuel_label", "E5")),
            label_visibility="collapsed",
            key="fuel_label",
        )

        fuel_code = _fuel_label_to_code(fuel_label)

        # Detour economics
        st.sidebar.markdown(
            "### Detour economics",
            help=(
                "Decide how detours are evaluated. The recommender combines your detour limits "
                "(extra distance/time) with detour costs (extra fuel and optional value of time) "
                "and the expected price advantage to compute a net saving for each candidate station."
            ),
            )

        use_economics = st.sidebar.checkbox(
            "Economics-based decision",
            value=_ss("use_economics", True),
            key="use_economics",
        )

        litres_to_refuel = st.sidebar.number_input(
            "Litres to refuel",
            min_value=1.0,
            max_value=1000.0,
            value=float(_ss("litres_to_refuel", 40.0)),
            step=1.0,
            key="litres_to_refuel",
        )
        consumption_l_per_100km = st.sidebar.number_input(
            "Car consumption (L/100 km)",
            min_value=0.0,
            max_value=30.0,
            value=float(_ss("consumption_l_per_100km", 7.0)),
            step=0.5,
            key="consumption_l_per_100km",
        )
        value_of_time_eur_per_hour = st.sidebar.number_input(
            "Value of time (€/hour)",
            min_value=0.0,
            max_value=200.0,
            value=float(_ss("value_of_time_eur_per_hour", 0.0)),
            step=5.0,
            key="value_of_time_eur_per_hour",
        )
        max_detour_km = st.sidebar.number_input(
            "Maximum extra distance (km)",
            min_value=0.5,
            max_value=200.0,
            value=float(_ss("max_detour_km", 5.0)),
            step=0.5,
            key="max_detour_km",
        )
        max_detour_min = st.sidebar.number_input(
            "Maximum extra time (min)",
            min_value=1.0,
            max_value=240.0,
            value=float(_ss("max_detour_min", 10.0)),
            step=1.0,
            key="max_detour_min",
            help=(
                "The maximum additional travel time you are willing to accept for a detour compared to the baseline route. "
                "Stations requiring more extra time are excluded (hard constraint)."
            ),
        )
        min_net_saving_eur = st.sidebar.number_input(
            "Min net saving (€)",
            min_value=0.0,
            max_value=100.0,
            value=float(_ss("min_net_saving_eur", 0.0)),
            step=0.5,
            key="min_net_saving_eur",
            help=(
                "Minimum required net benefit for accepting a detour. Net saving = fuel price saving − "
                "detour fuel cost − optional time cost. Set to 0 to allow any positive or zero net saving."
            ),
        )

        # Optional diagnostics
        debug_mode = st.sidebar.checkbox(
            "Debug mode",
            value=_ss("debug_mode", True),
            key="debug_mode",
        )

        run_clicked = run_clicked_top

    # ----------------------------------------------------------------------
    # STATUS VIEW (explain + show current state; no settings widgets)
    # ----------------------------------------------------------------------
    else:
        # Read current values from session_state so the app logic still works
        fuel_label = _ss("fuel_label", "E5")
        fuel_code = _fuel_label_to_code(fuel_label)

        start_locality = _ss("start_locality", "Tübingen")
        end_locality = _ss("end_locality", "Sindelfingen")

        # Kept for compatibility with downstream kwargs
        start_address = ""
        end_address = ""

        use_economics = bool(_ss("use_economics", True))
        litres_to_refuel = float(_ss("litres_to_refuel", 40.0))
        consumption_l_per_100km = float(_ss("consumption_l_per_100km", 7.0))
        value_of_time_eur_per_hour = float(_ss("value_of_time_eur_per_hour", 0.0))
        max_detour_km = float(_ss("max_detour_km", 5.0))
        max_detour_min = float(_ss("max_detour_min", 10.0))
        min_net_saving_eur = float(_ss("min_net_saving_eur", 0.0))
        debug_mode = bool(_ss("debug_mode", False))

        run_clicked = run_clicked_top

        st.sidebar.subheader("What the app currently does")
        st.sidebar.markdown(
            """
    - Builds a route (Google routing) and collects stations along/near the route.
    - Pulls current Tankerkönig prices and historical context.
    - Predicts arrival-time prices (ARDL horizon logic) and ranks stations.
    - Optionally applies an economics filter (detour fuel + time cost + thresholds).
            """.strip()
        )

        cached = st.session_state.get("last_run")
        if not cached:
            st.sidebar.info("No run cached yet. Switch to **Settings** and click **Run recommender**.")
        else:
            summary = cached.get("run_summary") or {}
            computed_at = cached.get("computed_at")

            st.sidebar.markdown("### Current state")
            if computed_at:
                st.sidebar.caption(f"Last run: {computed_at}")

            # Keep this compact; Route Analytics already has the detailed story
            st.sidebar.write(
                {
                    "fuel": fuel_label,
                    "route": f"{start_locality} → {end_locality}",
                    "use_economics": bool(summary.get("use_economics", use_economics)),
                    "stations_total": summary.get("stations_total"),
                    "stations_ranked": summary.get("stations_ranked"),
                    "stations_filtered_out": summary.get("stations_filtered_out"),
                }
            )

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
    }
    params_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    have_cached = st.session_state.get("last_run") is not None
    cached_is_stale = have_cached and (st.session_state.get("last_params_hash") != params_hash)

    # If user changed settings, keep showing cached results but warn
    if not run_clicked and cached_is_stale:
        st.warning("Settings changed. Showing previous results.")

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
            "constraints": {
                "max_detour_km": float(max_detour_km),
                "max_detour_min": float(max_detour_min),
                "min_net_saving_eur": float(min_net_saving_eur),
                "litres_to_refuel": float(litres_to_refuel),
                "consumption_l_per_100km": float(consumption_l_per_100km),
                "value_of_time_eur_per_hour": float(value_of_time_eur_per_hour),
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

    _display_best_station(
        best_station,
        fuel_code,
        litres_to_refuel=litres_to_refuel,
        ranked_stations=ranked,
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
                st.markdown(
                    "<div class='map-header'>"
                    "<div class='map-title'>Route and stations map</div>"
                    "<div class='map-subtitle'>Hover or click a station marker to show details.</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

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

            # Compute via-station route (origin → best station → destination), if possible
            via_overview = None
            if best_station and route_coords:
                try:
                    start_lon, start_lat = route_coords[0][0], route_coords[0][1]
                    end_lon, end_lat = route_coords[-1][0], route_coords[-1][1]

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
                            departure_time=route_info.get("departure_time", "now"),
                        )
                        via_overview = via.get("via_full_coords")
                except Exception:
                    via_overview = None  # non-fatal

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
            # Map rendering
            # -----------------------------
            map_provider = "mapbox"
            map_style_url = (
                "mapbox://styles/mapbox/satellite-streets-v12"
                if use_satellite
                else "mapbox://styles/mapbox/streets-v12"
            )

            deck = _create_map_visualization(
                route_coords=route_coords,
                stations=ranked,
                best_station_uuid=best_uuid,
                via_full_coords=via_overview,
                zoom_level=zoom_for_markers,
                fuel_code=fuel_code,
                selected_station_uuid=st.session_state.get("selected_station_uuid"),
                map_provider=map_provider,
                map_style=map_style_url,
            )

            selected_uuid_from_event: Optional[str] = None

            if _supports_pydeck_selections():
                event = st.pydeck_chart(
                    deck,
                    on_select="rerun",
                    selection_mode="single-object",
                    key="route_map",
                    use_container_width=True,
                    height=560,
                )

                try:
                    selection = event.selection
                    objects = getattr(selection, "objects", None)
                    if objects is None and isinstance(selection, dict):
                        objects = selection.get("objects")

                    if objects and "stations" in objects and objects["stations"]:
                        selected_obj = objects["stations"][0]
                        selected_uuid_from_event = selected_obj.get("station_uuid") or None
                except Exception:
                    selected_uuid_from_event = None
            else:
                st.pydeck_chart(deck, use_container_width=True, height=560)
                st.caption(
                    "Note: Your Streamlit version does not expose pydeck click selections. "
                    "Upgrade Streamlit to enable click-to-details interaction."
                )

            # Persist selection regardless of which branch ran
            if selected_uuid_from_event:
                st.session_state["selected_station_uuid"] = selected_uuid_from_event

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


if __name__ == "__main__":
    main()
