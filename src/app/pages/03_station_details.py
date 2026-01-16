"""
Station Details & Analysis (Page 03)

FINAL VERSION
- Trip settings in LEFT SIDEBAR "Action" tab (replaces placeholder)
- 7-day price trend (clean line, no dots)
- Hourly chart with gridlines
- All mobile optimizations
- Zoom disabled on plots

Selection sources supported (cross-page navigation):
1) st.session_state["selected_station_data"] (preferred)
2) st.session_state["selected_station_uuid"] resolved from last_run["ranked"]/["stations"]
3) st.session_state["explorer_results"] (Page 04 → Page 03 handoff)
"""

from __future__ import annotations

import sys
import json
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Plotly config (mobile-friendly + NO ZOOM)
PLOTLY_CONFIG = {
    "displayModeBar": False,  # Hides zoom/pan toolbar
    "displaylogo": False,
    "responsive": True,
}

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

from ui.sidebar import render_sidebar_shell, render_station_selector

# Path setup
APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import load_env_once
load_env_once()

from src.integration.historical_data import (
    get_station_price_history,
    calculate_hourly_price_stats,
    get_cheapest_and_most_expensive_hours,
)

from ui.formatting import (
    _station_uuid,
    _fmt_price,
    _fmt_eur,
    _fmt_km,
    _fmt_min,
    _safe_text,
    calculate_traffic_light_status,
    calculate_trip_fuel_info,
    check_smart_price_alert,
)

from ui.styles import apply_app_css
from config.settings import ensure_persisted_state_defaults
from services.session_store import (
    init_session_context,
    restore_persisted_state,
    maybe_persist_state,
)


# =============================================================================
# Address Extraction (Google Maps Reverse Geocode)
# =============================================================================

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
        from config.settings import environment_check
        import googlemaps

        api_key = environment_check()
        client = googlemaps.Client(key=api_key)
        results = client.reverse_geocode((lat_f, lon_f), language="de")
        if not results:
            return None

        cache[station_uuid] = results[0]
        return results[0]

    except Exception:
        return None


def _reverse_geocode_station_address(station: Dict[str, Any]) -> Optional[str]:
    """Extract formatted address from Google Maps reverse geocode."""
    payload = _reverse_geocode_station_payload(station)
    if not payload:
        return None
    formatted = payload.get("formatted_address")
    return str(formatted).strip() if formatted else None


def _reverse_geocode_station_city(station: Dict[str, Any]) -> Optional[str]:
    """Extract city name from Google Maps reverse geocode."""
    payload = _reverse_geocode_station_payload(station)
    if not payload:
        return None

    comps = payload.get("address_components") or []
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


# =============================================================================
# Opening Hours Parsing
# =============================================================================

def _parse_opening_hours(station: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Parse opening hours from station data.
    Returns (is_open_now, closing_time_text)
    """
    opening_times_json = station.get("openingtimes_json")
    
    if not opening_times_json or opening_times_json == "{}":
        return (False, None)
    
    try:
        if isinstance(opening_times_json, str):
            hours_data = json.loads(opening_times_json)
        else:
            hours_data = opening_times_json
        
        if not hours_data or hours_data == {}:
            return (False, None)
        
        try:
            now = datetime.now(tz=ZoneInfo("Europe/Berlin") if ZoneInfo else None)
        except:
            now = datetime.now()
        
        current_time = now.time()
        weekday = now.strftime("%A").lower()
        
        today_hours = hours_data.get(weekday, [])
        if not today_hours:
            return (False, None)
        
        for period in today_hours:
            start_str = period.get("start")
            end_str = period.get("end")
            
            if not start_str or not end_str:
                continue
            
            try:
                start_time = time.fromisoformat(start_str)
                end_time = time.fromisoformat(end_str)
                
                if start_time <= current_time <= end_time:
                    return (True, end_str)
            except Exception:
                continue
        
        for period in today_hours:
            start_str = period.get("start")
            if start_str:
                try:
                    start_time = time.fromisoformat(start_str)
                    if current_time < start_time:
                        return (False, start_str)
                except Exception:
                    pass
        
        return (False, None)
    
    except Exception:
        return (False, None)


# =============================================================================
# Helper Functions
# =============================================================================

def _safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float, return None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _station_name(station: Dict[str, Any]) -> str:
    """Extract station name with brand fallback."""
    name = station.get("tk_name") or station.get("osm_name") or station.get("name")
    brand = station.get("brand")
    
    if name:
        return _safe_text(name)
    elif brand:
        return _safe_text(brand)
    else:
        return "Unknown Station"


def _station_brand(station: Dict[str, Any]) -> str:
    """Extract station brand."""
    return _safe_text(station.get("brand", ""))


def _station_city(station: Dict[str, Any]) -> str:
    """Extract station city."""
    return _safe_text(station.get("city", ""))


def _resolve_station(
    selected_station_data: Any,
    selected_uuid: str,
    ranked: List[Dict[str, Any]],
    stations: List[Dict[str, Any]],
    explorer_results: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Resolve selected station from various sources."""
    
    if selected_station_data:
        return selected_station_data, _station_uuid(selected_station_data)
    
    if selected_uuid:
        for s in ranked:
            if _station_uuid(s) == selected_uuid:
                return s, selected_uuid
        for s in stations:
            if _station_uuid(s) == selected_uuid:
                return s, selected_uuid
        for s in explorer_results:
            if _station_uuid(s) == selected_uuid:
                return s, selected_uuid
    
    return None, ""


# =============================================================================
# Plotly Charts (MOBILE-OPTIMIZED)
# =============================================================================

def create_price_trend_chart(df: pd.DataFrame, fuel_type: str) -> Tuple[go.Figure, str]:
    """7-day price trend - MOBILE OPTIMIZED. Returns (figure, stats_text)."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(
            title=f"{fuel_type.upper()} Price History",
            height=300,
        )
        return fig, ""

    # For mobile: Sample every 3rd point if we have many data points
    df_sampled = df.iloc[::3] if len(df) > 30 else df

    fig = go.Figure()
    # Clean line only (no markers)
    fig.add_trace(go.Scatter(
        x=df_sampled["date"],
        y=df_sampled["price"],
        mode="lines",
        name="Price",
        line=dict(color="#1f77b4", width=3),
        hovertemplate="%{x|%b %d}<br>€%{y:.3f}/L<extra></extra>",
    ))

    avg_price = float(df["price"].mean())
    min_price = float(df["price"].min())
    max_price = float(df["price"].max())

    fig.update_layout(
        title=f"{fuel_type.upper()} Price History",
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
        height=300,
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    # Add average line
    fig.add_hline(
        y=avg_price,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text=f"Avg",
        annotation_position="right",
        annotation_font_size=10,
    )

    # Return stats as text
    stats_text = f"Min: €{min_price:.3f} | Avg: €{avg_price:.3f} | Max: €{max_price:.3f}"
    
    return fig, stats_text


def create_hourly_pattern_chart(hourly_df: pd.DataFrame, fuel_type: str) -> go.Figure:
    """Hourly price pattern - MOBILE OPTIMIZED with gridlines."""
    if hourly_df is None or hourly_df.empty or hourly_df["avg_price"].isna().all():
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for hourly pattern",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(
            title="Best Time to Refuel",
            height=250,
        )
        return fig

    # MOBILE: Show every 2 hours (12 bars instead of 24)
    hourly_df_mobile = hourly_df[hourly_df["hour"] % 2 == 0].copy()

    optimal = get_cheapest_and_most_expensive_hours(hourly_df)
    cheapest_hour = optimal.get("cheapest_hour")
    most_expensive_hour = optimal.get("most_expensive_hour")

    # Color bars
    colors = []
    for _, row in hourly_df_mobile.iterrows():
        if pd.isna(row["avg_price"]):
            colors.append("#e0e0e0")
        elif cheapest_hour is not None and int(row["hour"]) == int(cheapest_hour):
            colors.append("#4caf50")
        elif most_expensive_hour is not None and int(row["hour"]) == int(most_expensive_hour):
            colors.append("#f44336")
        else:
            colors.append("#ffc107")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_df_mobile["hour"],
        y=hourly_df_mobile["avg_price"],
        marker_color=colors,
        hovertemplate="Hour %{x}:00<br>€%{y:.3f}/L<extra></extra>",
        showlegend=False,
        width=1.5,
    ))

    fig.update_layout(
        title=f"Best Time to Refuel - {fuel_type.upper()}",
        xaxis_title="Hour of Day",
        yaxis_title=None,
        height=300,
        margin=dict(l=40, r=20, t=60, b=50),
    )

    # Add gridlines and configure axes
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=4)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')

    return fig


# =============================================================================
# Sidebar Trip Settings Renderer
# =============================================================================

def _render_trip_settings():
    """
    Render trip settings in the sidebar Action tab.
    Replaces "Placeholder: Action" with actual trip info.
    """
    last_run = st.session_state.get("last_run") or {}
    
    if not last_run:
        st.sidebar.info("No trip data available. Run Trip Planner first.")
        return
    
    # Extract settings
    fuel_code = str(last_run.get("fuel_code") or st.session_state.get("fuel_label") or "e5")
    fuel_label = fuel_code.upper()
    
    use_economics = bool(last_run.get("use_economics", False))
    econ_status = "Yes" if use_economics else "No"
    
    litres = _safe_float(last_run.get("litres_to_refuel"))
    refuel_amount = f"{litres:.0f} L" if litres else "—"
    
    stations = list(last_run.get("stations") or [])
    route_station_count = len(stations)
    
    explorer_results = list(st.session_state.get("explorer_results") or [])
    explorer_station_count = len(explorer_results)
    
    # Display settings
    st.sidebar.markdown("**Fuel:** " + fuel_label)
    st.sidebar.markdown("**Economics:** " + econ_status)
    st.sidebar.markdown("**Refuel amount:** " + refuel_amount)
    st.sidebar.markdown("**Route stations:** " + str(route_station_count))
    st.sidebar.markdown("**Explorer stations:** " + str(explorer_station_count))


# =============================================================================
# Main Page
# =============================================================================

def main():
    """Main entry point for Station Details page."""
    
    st.set_page_config(
        page_title="Station Details",
        page_icon="⛽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Redis-backed persistence
    _preserve_top_nav = st.session_state.get("top_nav")
    _preserve_sidebar_view = st.session_state.get("sidebar_view")
    _preserve_map_style_mode = st.session_state.get("map_style_mode")
    
    init_session_context()
    ensure_persisted_state_defaults(st.session_state)
    restore_persisted_state(overwrite_existing=True)
    
    if _preserve_top_nav is not None:
        st.session_state["top_nav"] = _preserve_top_nav
    if _preserve_sidebar_view is not None:
        st.session_state["sidebar_view"] = _preserve_sidebar_view
    if _preserve_map_style_mode is not None:
        st.session_state["map_style_mode"] = _preserve_map_style_mode
    
    apply_app_css()
    
    st.title("Station Details & Analysis")
    st.caption("Detailed analysis of individual stations")
    
    # =========================================================================
    # TOP NAVIGATION
    # =========================================================================
    NAV_TARGETS = {
        "Home": "streamlit_app.py",
        "Analytics": "pages/02_route_analytics.py",
        "Station": "pages/03_station_details.py",
        "Explorer": "pages/04_station_explorer.py",
    }
    CURRENT = "Station"
    
    if st.session_state.get("_active_page") != CURRENT:
        st.session_state["_active_page"] = CURRENT
        st.session_state["top_nav"] = CURRENT
    
    selected_nav = st.segmented_control(
        label="",
        options=list(NAV_TARGETS.keys()),
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="top_nav",
    )
    
    target = NAV_TARGETS.get(selected_nav, NAV_TARGETS[CURRENT])
    if target != NAV_TARGETS[CURRENT]:
        try:
            maybe_persist_state(force=True)
        except Exception:
            pass
        st.switch_page(target)
    
    # =========================================================================
    # SIDEBAR - WITH TRIP SETTINGS IN ACTION TAB
    # =========================================================================
    render_sidebar_shell(action_renderer=_render_trip_settings)
    
    last_run = st.session_state.get("last_run") or {}
    ranked = list(last_run.get("ranked") or [])
    stations = list(last_run.get("stations") or [])
    explorer_results = list(st.session_state.get("explorer_results") or [])
    
    selection = render_station_selector(
        last_run=last_run,
        explorer_results=explorer_results,
    )
    
    # =========================================================================
    # RESOLVE STATION
    # =========================================================================
    selected_station_data = st.session_state.get("selected_station_data")
    selected_uuid = st.session_state.get("selected_station_uuid") or ""
    
    station, station_uuid = _resolve_station(
        selected_station_data,
        selected_uuid,
        ranked,
        stations,
        explorer_results,
    )
    
    if not station or not station_uuid:
        st.info("No station selected. Please select a station from the sidebar.")
        st.markdown("---")
        st.caption("Tip: Run Trip Planner or use Station Explorer to find stations.")
        maybe_persist_state()
        return
    
    # =========================================================================
    # EXTRACT STATION INFO
    # =========================================================================
    name = _station_name(station)
    brand = _station_brand(station)
    city = _station_city(station) or _reverse_geocode_station_city(station)
    full_address = _reverse_geocode_station_address(station)
    
    route_info = last_run.get("route_info") or {}
    fuel_code = str(last_run.get("fuel_code") or st.session_state.get("fuel_label") or "e5").lower()
    use_economics = bool(last_run.get("use_economics", False))
    litres_to_refuel = _safe_float(last_run.get("litres_to_refuel")) or 40.0
    
    detour_km = _safe_float(station.get("detour_distance_km") or station.get("detour_km")) or 0.0
    detour_min = _safe_float(station.get("detour_duration_min") or station.get("detour_min")) or 0.0
    detour_km = max(0.0, detour_km)
    detour_min = max(0.0, detour_min)
    
    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"
    
    predicted_price = _safe_float(station.get(pred_key))
    current_price = _safe_float(station.get(current_key))
    display_price = predicted_price if predicted_price is not None else current_price
    
    econ_net_key = f"econ_net_saving_eur_{fuel_code}"
    econ_baseline_key = f"econ_baseline_price_{fuel_code}"
    net_saving = _safe_float(station.get(econ_net_key))
    baseline_price = _safe_float(station.get(econ_baseline_key))
    
    # Traffic light status
    traffic_status, traffic_text_raw, traffic_css = calculate_traffic_light_status(
        display_price,
        ranked,
        fuel_code,
    )
    
    # Improve text clarity
    if traffic_status == "red":
        traffic_text = "Expensive"
    elif traffic_status == "yellow":
        traffic_text = "Fair Price"
    else:
        traffic_text = "Excellent Deal"
    
    is_open, time_info = _parse_opening_hours(station)
    
    # =========================================================================
    # HERO SECTION - CYAN BOX WITH PRICE + TRAFFIC LIGHT
    # =========================================================================
    
    base_name = brand if (brand and brand != "—") else name
    city_clean = city if (city and city != "—") else ""
    title_line = f"{base_name} in {city_clean}" if (base_name and city_clean) else base_name
    
    # Clean subtitle
    if full_address:
        subtitle_line = full_address
    elif city_clean:
        subtitle_line = city_clean
    else:
        subtitle_line = ""
    
    # Traffic light circles
    if traffic_status == "green":
        circles = '<span style="color: #4ade80; font-size: 1.6rem;">●</span> <span style="color: #fbbf24; font-size: 1.6rem;">○</span> <span style="color: #f87171; font-size: 1.6rem;">○</span>'
    elif traffic_status == "yellow":
        circles = '<span style="color: #4ade80; font-size: 1.6rem;">○</span> <span style="color: #fbbf24; font-size: 1.6rem;">●</span> <span style="color: #f87171; font-size: 1.6rem;">○</span>'
    else:
        circles = '<span style="color: #4ade80; font-size: 1.6rem;">○</span> <span style="color: #fbbf24; font-size: 1.6rem;">○</span> <span style="color: #f87171; font-size: 1.6rem;">●</span>'
    
    price_display = f"€{display_price:.3f}" if display_price else "—"
    
    hero_html = f"""
    <div class='station-header'>
        <div class='label'>Selected station:</div>
        <div class='name'>{_safe_text(title_line)}</div>
        {f"<div class='addr'>{_safe_text(subtitle_line)}</div>" if subtitle_line else ""}
        <div style='margin-top: 1.5rem; text-align: center;'>
            <div style='font-size: 3rem; font-weight: 800; color: #1f2937; margin-bottom: 0.8rem;'>
                {price_display} <span style='font-size: 1.8rem; font-weight: 600; color: #6b7280;'>/L</span>
            </div>
            <div style='display: flex; align-items: center; justify-content: center; gap: 0.4rem; color: #374151; font-size: 1.1rem; font-weight: 600;'>
                {circles}
                <span style='margin-left: 0.5rem;'>{traffic_text}</span>
            </div>
        </div>
    </div>
    """
    
    st.markdown(hero_html, unsafe_allow_html=True)
    
    # =========================================================================
    # INFO CARDS
    # =========================================================================
    
    info_cols = st.columns(3)

    with info_cols[0]:
        # Detour info (same as before)
        if detour_km < 0.1:
            if detour_min > 0.5:
                st.info(f"On route • +{detour_min:.0f} min stop")
            else:
                st.success("On route")
        else:
            st.info(f"{detour_km:.1f} km detour • +{detour_min:.0f} min")

    with info_cols[1]:
        # NEW: ETA (Expected Time of Arrival)
        eta_str = station.get("eta")
        if eta_str:
            try:
                # Parse ISO format datetime
                if isinstance(eta_str, str):
                    eta_dt = datetime.fromisoformat(eta_str.replace('Z', '+00:00'))
                else:
                    eta_dt = eta_str
                
                # Format as HH:MM
                eta_time = eta_dt.strftime("%H:%M")
                st.info(f"Estimated Time of Arrival: {eta_time}")
            except Exception:
                st.info("Estimated Time of Arrival: —")
        else:
            st.info("Estimated Time of Arrival: —")

    with info_cols[2]:
        # Opening hours (same as before)
        if is_open and time_info:
            st.success(f"Opening hours: OPEN until {time_info}")
        elif not is_open and time_info:
            st.error(f"Opening hours: CLOSED, opens at {time_info}")
        else:
            st.info("Opening hours: 24/7")
    
    st.markdown("---")
    
    # =========================================================================
    # SAVINGS CALCULATOR
    # =========================================================================
    
    st.markdown("### Savings Calculator")
    st.caption("Compare savings against other stations on your route (within 1km detour)")
    
    if not baseline_price or not display_price:
        st.info("Economics not available. Run Trip Planner with refuel amount to enable.")
    else:
        slider_value = st.slider(
            "Refuel amount (liters)",
            min_value=1.0,
            max_value=80.0,
            value=float(litres_to_refuel),
            step=1.0,
            key="savings_calc_slider",
        )
        
        # Find worst on-route price
        worst_price = display_price
        for s in ranked:
            s_price = _safe_float(s.get(pred_key))
            if s_price and s_price > worst_price:
                d_km = _safe_float(s.get("detour_distance_km") or s.get("detour_km") or 0)
                if d_km < 1.0:
                    worst_price = s_price
        
        price_diff = worst_price - display_price
        total_savings = price_diff * slider_value
        
        # Weekly/monthly/yearly context
        weekly_savings = total_savings
        monthly_savings = total_savings * 4.3
        yearly_savings = total_savings * 52
        
        if abs(total_savings) < 0.01:
            st.success("You're already at the cheapest station on your route!")
        else:
            calc_cols = st.columns(2)
            
            with calc_cols[0]:
                st.markdown("**Saved by choosing this station**")
                st.markdown(f"# €{total_savings:.2f}")
            
            with calc_cols[1]:
                st.markdown("**vs most expensive (on-route)**")
                st.markdown(f"# {price_diff:.3f}€/L")
            
            if total_savings > 0.50:
                st.info(
                    f"If you refuel here weekly: **€{weekly_savings:.2f}** per week • "
                    f"**€{monthly_savings:.2f}** per month • "
                    f"**€{yearly_savings:.2f}** per year"
                )
    
    st.markdown("---")
    
    # =========================================================================
    # SMART PRICE ALERT
    # =========================================================================
    
    try:
        history_df = get_station_price_history(station_uuid, fuel_code, days=14)
        hourly_df = calculate_hourly_price_stats(history_df) if history_df is not None else None
    except Exception:
        history_df = None
        hourly_df = None
    
    if hourly_df is not None and not hourly_df.empty:
        now = datetime.now()
        current_hour = now.hour
        alert = check_smart_price_alert(hourly_df, current_hour)
        
        if alert:
            hours_wait = alert["hours_to_wait"]
            drop_hour = alert["drop_hour"]
            price_drop = alert["price_drop"]
            
            st.warning(
                f"Price usually drops at {drop_hour}:00 (−€{price_drop:.3f}/L). "
                f"Worth waiting {hours_wait} hour{'s' if hours_wait > 1 else ''}?"
            )
    
    # =========================================================================
    # BEST TIME TO REFUEL (WITH GRIDLINES)
    # =========================================================================
    
    st.markdown("### Best Time to Refuel")
    st.caption("Based on 14-day price patterns")
    
    if hourly_df is not None and not hourly_df.empty:
        fig = create_hourly_pattern_chart(hourly_df, fuel_code)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        optimal = get_cheapest_and_most_expensive_hours(hourly_df)
        cheapest_hour = optimal.get("cheapest_hour")
        
        if cheapest_hour is not None:
            st.success(f"Usually cheapest: {cheapest_hour}:00")
    else:
        st.info("Not enough historical data to show hourly patterns.")
    
    st.markdown("---")
    
    # =========================================================================
    # PRICE TREND (7 DAYS - NO DOTS)
    # =========================================================================
    
    with st.expander("Price Trend (7 days)", expanded=False):
        try:
            history_df_7d = get_station_price_history(station_uuid, fuel_code, days=7)
            if history_df_7d is not None and not history_df_7d.empty:
                fig, stats_text = create_price_trend_chart(history_df_7d, fuel_code)
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                st.caption(stats_text)
            else:
                st.info("No historical data available.")
        except Exception:
            st.info("No historical data available.")
    
    # =========================================================================
    # COMPARE STATIONS
    # =========================================================================
    
    with st.expander("Compare Stations", expanded=False):
        st.caption("Top ranked stations from your route")
        
        if len(ranked) >= 1:
            # Current station card
            # Format ETA for current station
            eta_display = ""
            eta_str = station.get("eta")
            if eta_str:
                try:
                    if isinstance(eta_str, str):
                        eta_dt = datetime.fromisoformat(eta_str.replace("Z", "+00:00"))
                    else:
                        eta_dt = eta_str
                    eta_display = f" • Estimated Time of Arrival: {eta_dt.strftime('%H:%M')}"
                except:
                    pass
            
            detour_text = "On route" if detour_min < 0.5 else f"+{detour_min:.0f} min detour"
            
            st.markdown(f"""
            <div style='background: #e0f2fe; border-left: 4px solid #0284c7; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                <div style='font-weight: 700; font-size: 1.1rem; color: #0c4a6e;'>{name} (Current)</div>
                <div style='margin-top: 0.5rem; color: #075985;'>
                    <span style='font-size: 1.3rem; font-weight: 700;'>€{display_price:.3f}</span> / L
                </div>
                <div style='margin-top: 0.3rem; color: #0369a1; font-size: 0.9rem;'>
                    {detour_text}{eta_display}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 3 from ranking
            for idx, s in enumerate(ranked[:3], start=1):
                s_uuid = _station_uuid(s)
                if s_uuid == station_uuid:
                    continue
                
                s_name = _station_name(s)
                s_price = _safe_float(s.get(pred_key))
                s_detour = _safe_float(s.get("detour_duration_min"))
                
                # Format ETA for this station
                s_eta_display = ""
                s_eta_str = s.get("eta")
                if s_eta_str:
                    try:
                        if isinstance(s_eta_str, str):
                            s_eta_dt = datetime.fromisoformat(s_eta_str.replace("Z", "+00:00"))
                        else:
                            s_eta_dt = s_eta_str
                        s_eta_display = f" • Estimated Time of Arrival: {s_eta_dt.strftime('%H:%M')}"
                    except:
                        pass
                
                s_detour_text = "On route" if (s_detour or 0) < 0.5 else f"+{s_detour:.0f} min detour"
                
                if s_price:
                    st.markdown(f"""
                    <div style='background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.8rem;'>
                        <div style='font-weight: 600; font-size: 1rem; color: #374151;'>#{idx} {s_name}</div>
                        <div style='margin-top: 0.5rem; color: #1f2937;'>
                            <span style='font-size: 1.2rem; font-weight: 700;'>€{s_price:.3f}</span> / L
                        </div>
                        <div style='margin-top: 0.3rem; color: #6b7280; font-size: 0.9rem;'>
                            {s_detour_text}{s_eta_display}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Need route ranking to show comparison.")
    
    maybe_persist_state()


if __name__ == "__main__":
    main()