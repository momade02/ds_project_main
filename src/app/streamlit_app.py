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

from src.app.ui.maps import (
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

from src.app.services.route_recommender import RouteRunInputs, run_route_recommendation

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _reverse_geocode_station_address(station: Dict[str, Any]) -> Optional[str]:
    """
    Return a formatted address for the given station using Google reverse geocoding.
    This is intentionally used ONLY for the recommended station (Page 1) to keep API calls minimal.
    Caches results in session_state to avoid repeated calls on reruns.
    """
    if not station:
        return None

    station_uuid = _station_uuid(station) or f"{station.get('lat')}_{station.get('lon')}"
    if "address_cache" not in st.session_state:
        st.session_state["address_cache"] = {}

    cache: Dict[str, str] = st.session_state["address_cache"]
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
        # Use the same key + validation flow you already rely on in the app.
        api_key = environment_check()

        # Import locally to avoid import-time issues in some deployments.
        import googlemaps  # type: ignore[import-untyped]

        client = googlemaps.Client(key=api_key)
        results = client.reverse_geocode((lat_f, lon_f), language="de")

        if not results:
            return None

        formatted = results[0].get("formatted_address")
        if formatted:
            cache[station_uuid] = formatted
            return formatted

        return None

    except Exception:
        # Non-fatal: address is optional for UI rendering
        return None


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
    city = best_station.get("city")

    # Line 1: brand preferred; if missing, fall back to station name
    title_line = brand if (brand and str(brand).strip()) else station_name

    # Line 2: full address (reverse geocode best station only)
    formatted_address = _reverse_geocode_station_address(best_station)
    subtitle_line = formatted_address or (f"{brand}, {city}" if (brand or city) else "")

    st.markdown("### Recommended station")
    st.markdown(f"**{_safe_text(title_line)}**")
    if subtitle_line:
        st.caption(_safe_text(subtitle_line))

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

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label=f"Predicted {fuel_code.upper()} price",
            value=_format_price(pred_price),
        )
    with col2:
        st.metric(
            label=f"Current {fuel_code.upper()} price",
            value=_format_price(current_price),
        )
    with col3:
        st.metric("Distance along route", dist_str)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("On-route worst price", "—" if onroute_worst is None else _format_price(onroute_worst))
    with col5:
        st.metric("Net saving vs on-route worst", "—" if net_vs_worst is None else _format_eur(net_vs_worst))
    with col6:
        st.metric("Detour distance", _format_km(detour_km))

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

        /* Outline the main structural containers so you see the usable layout */
        section[data-testid="stSidebar"] { outline: 2px dashed rgba(0,0,0,0.25); outline-offset: -2px; }
        section[data-testid="stMain"] { outline: 2px dashed rgba(0,0,0,0.25); outline-offset: -2px; }
        div.block-container { outline: 2px dashed rgba(255,0,0,0.25); outline-offset: -2px; }

        /* Outline each vertical block (helpful to see spacing between blocks) */
        div[data-testid="stVerticalBlock"] { outline: 1px dashed rgba(0,0,255,0.20); outline-offset: -1px; }

        /* Outline columns */
        div[data-testid="column"] { outline: 1px dashed rgba(0,128,0,0.20); outline-offset: -1px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # --- end ---

    st.title("Fuel Station Recommender")
    st.caption("##### Plan a route and identify the best-value refueling stop based on current prices and forecasts.")

    # Persist last clicked station (map selection)
    if "selected_station_uuid" not in st.session_state:
        st.session_state["selected_station_uuid"] = None
    # Persist last computed results so UI does not reset on reruns (sidebar changes / map clicks)
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None  # will store dict of computed outputs
    if "last_params_hash" not in st.session_state:
        st.session_state["last_params_hash"] = None

    # Sidebar configuration
    st.sidebar.subheader("Fuel type")

    fuel_label = st.sidebar.selectbox(
        "Fuel type",
        options=["E5", "E10", "Diesel"],
        index=0,
        label_visibility="collapsed",
    )

    fuel_code = _fuel_label_to_code(fuel_label)

    # Route settings (only used in real mode)
    st.sidebar.markdown("### Route settings")

    start_locality = st.sidebar.text_input(
        "Start locality (city/town)", value="Tübingen"
    )
    start_address = st.sidebar.text_input(
        "Start address (optional)", value=""
    )

    end_locality = st.sidebar.text_input(
        "End locality (city/town)", value="Sindelfingen"
    )
    end_address = st.sidebar.text_input(
        "End address (optional)", value=""
    )

    # Detour economics
    st.sidebar.markdown("### Detour economics")

    use_economics = st.sidebar.checkbox(
        "Use economics-based detour decision (net saving, time cost, fuel cost)",
        value=True,
    )
    st.sidebar.caption(
        "When enabled, stations may be filtered based on detour constraints below. Disable to show all stations."
    )
    litres_to_refuel = st.sidebar.number_input(
        "Litres to refuel",
        min_value=1.0,
        max_value=1000.0,
        value=40.0,
        step=1.0,
    )
    consumption_l_per_100km = st.sidebar.number_input(
        "Car consumption (L/100 km)",
        min_value=0.0,
        max_value=30.0,
        value=7.0,
        step=0.5,
    )
    value_of_time_eur_per_hour = st.sidebar.number_input(
        "Value of time (€/hour)",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
    )
    max_detour_km = st.sidebar.number_input(
        "Maximum extra distance (km)",
        min_value=0.5,
        max_value=200.0,
        value=5.0,
        step=0.5,
    )
    max_detour_min = st.sidebar.number_input(
        "Maximum extra time (min)",
        min_value=1.0,
        max_value=240.0,
        value=10.0,
        step=1.0,
    )
    min_net_saving_eur = st.sidebar.number_input(
        "Minimum net saving to accept detour (€, 0 = no threshold)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,
    )

    # Optional diagnostics
    debug_mode = st.sidebar.checkbox(
        "Debug mode (show pipeline diagnostics)", value=False
    )

    # Run button
    run_clicked = st.sidebar.button("Run recommender")

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
        st.warning("Settings changed. Showing previous results; click **Run recommender** to recompute.")

    # -----------------------------
    # Compute (ONLY when run_clicked)
    # -----------------------------
    if run_clicked:
        st.session_state["selected_station_uuid"] = None  # reset selection on recompute

        # 1) Integration
        route_info = None

        if not start_locality or not end_locality:
            st.error("Please provide at least start and end localities (cities/towns).")
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
        st.markdown("### Route and stations map")
        
        try:
            # Use route data from integration
            route_coords = route_info.get("route_coords")

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
                            departure_time=route_info.get('departure_time', 'now'),
                        )
                        via_overview = via.get('via_full_coords')
                except Exception:
                    # Non-fatal: still show baseline route
                    via_overview = None

            # Get best station UUID for highlighting (check both possible keys)
            best_uuid = None
            if best_station:
                best_uuid = best_station.get("tk_uuid") or best_station.get("station_uuid")

            # Calculate zoom level based on route extent (for marker scaling)
            zoom_for_markers = 7.5  # Default
            if route_coords:
                lons = [coord[0] for coord in route_coords]
                lats = [coord[1] for coord in route_coords]
                
                # Use Web Mercator formula for precise zoom calculation
                zoom_for_markers = _calculate_zoom_for_bounds(
                    lon_min=min(lons),
                    lon_max=max(lons),
                    lat_min=min(lats),
                    lat_max=max(lats),
                    padding_percent=0.10,
                    map_width_px=700,
                    map_height_px=500,
                )
            
            # -----------------------------
            # Basemap toggle state (Standard <-> Satellite)
            # -----------------------------
            if "use_satellite" not in st.session_state:
                st.session_state["use_satellite"] = False

            use_satellite = bool(st.session_state["use_satellite"])

            CUSTOM_STYLE = "mapbox://styles/moritzmaidl/cmk2hdk9c000101pf7jrg9wmb"
            map_provider = "mapbox"
            map_style_url = "mapbox://styles/mapbox/satellite-streets-v12" if use_satellite else "mapbox://styles/mapbox/streets-v12"

            # Button label shows the mode you can switch TO
            toggle_label = "Standard" if use_satellite else "Satellite"

            # -----------------------------
            # Map block with overlay button (upper-right on map)
            # -----------------------------
            map_block = st.container()
            with map_block:
                # Anchor + CSS: position the first button in this block over the map
                st.markdown(
                    """
                    <style>
                    /* Make ONLY the block that contains our anchor a positioning context */
                    div[data-testid="stVerticalBlock"]:has(#map-overlay-anchor) {
                        position: relative;
                    }

                    /* Move ONLY the basemap toggle (works whether Streamlit uses id or class) */
                    #st-key-toggle_basemap, .st-key-toggle_basemap {
                        position: absolute;
                        top: 70px;     /* adjust as needed */
                        left: 12px;    /* upper-left */
                        z-index: 10000;
                    }

                    /* Button look (border/"Rand" + compact + background) */
                    #st-key-toggle_basemap button, .st-key-toggle_basemap button {
                        padding: 0.25rem 0.75rem;
                        border: 1px solid rgba(31, 41, 55, 0.35) !important;
                        border-radius: 10px !important;
                        background: rgba(255, 255, 255, 0.90) !important;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
                    }

                    /* Optional hover */
                    #st-key-toggle_basemap button:hover, .st-key-toggle_basemap button:hover {
                        border-color: rgba(31, 41, 55, 0.55) !important;
                    }
                    </style>

                    <div id="map-overlay-anchor"></div>
                    """,
                    unsafe_allow_html=True,
                )

                # IMPORTANT: keep this as the first/only button inside map_block
                if st.button(toggle_label, key="toggle_basemap"):
                    st.session_state["use_satellite"] = not use_satellite
                    st.rerun()

                # Build ONE deck (use ranked stations so markers match the filtered set)
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

                st.caption("Hover for quick info. Click a station marker to show details below.")

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

                    # Extract clicked station from selection state
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
