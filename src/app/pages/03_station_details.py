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
3) st.session_state["explorer_results"] (Page 04 -> Page 03 handoff)
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
    "scrollZoom": False,  # Disable scroll zoom
    "doubleClick": False,  # Disable double-click zoom
}

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

from ui.sidebar import render_sidebar_shell, render_station_selector, render_comparison_selector, _render_help_station

# Path setup
APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import load_env_once
load_env_once()

# Import on-route thresholds for consistent calculations with other pages
try:
    from src.decision.recommender import ONROUTE_MAX_DETOUR_KM, ONROUTE_MAX_DETOUR_MIN
except ImportError:
    # Fallback defaults if import fails
    ONROUTE_MAX_DETOUR_KM = 1.0
    ONROUTE_MAX_DETOUR_MIN = 5.0

from src.integration.historical_data import (
    get_station_price_history,
    calculate_hourly_price_stats,
    get_cheapest_and_most_expensive_hours,
)

from ui.formatting import (
    _station_uuid,
    _safe_text,
    calculate_traffic_light_status,
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
# CACHED DATA ACCESS (Performance Optimization)
# =============================================================================
# These wrappers cache database queries so switching sidebar tabs doesn't
# re-fetch data. Cache expires after 5 minutes (ttl=300).

@st.cache_data(ttl=300, show_spinner=False)
def _cached_get_station_price_history(station_uuid: str, fuel_code: str, days: int = 14) -> Optional[pd.DataFrame]:
    """Cached wrapper for get_station_price_history."""
    try:
        return get_station_price_history(station_uuid, fuel_code, days=days)
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _cached_calculate_hourly_stats(station_uuid: str, fuel_code: str, days: int = 14) -> Optional[pd.DataFrame]:
    """Cached wrapper for hourly stats calculation."""
    try:
        history_df = get_station_price_history(station_uuid, fuel_code, days=days)
        if history_df is not None and not history_df.empty:
            return calculate_hourly_price_stats(history_df)
        return None
    except Exception:
        return None


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


def _build_tankerkoenig_address(station: Dict[str, Any]) -> Optional[str]:
    """
    Build address from Tankerkönig data fields.
    Format: "Street HouseNumber, PostCode City"
    """
    street = station.get("street")
    house_number = station.get("house_number")
    post_code = station.get("post_code")
    city = station.get("city")
    
    # Need at least street and city
    if not street or not city:
        return None
    
    # Apply title case for readability (database often has CAPS)
    street_title = street.title() if street else ""
    city_title = city.title() if city else ""
    
    # Build address string
    parts = []
    
    # Street + house number
    if house_number:
        parts.append(f"{street_title} {house_number}")
    else:
        parts.append(street_title)
    
    # Post code + city
    if post_code:
        parts.append(f"{post_code} {city_title}")
    else:
        parts.append(city_title)
    
    return ", ".join(parts)


def _reverse_geocode_station_address(station: Dict[str, Any]) -> Optional[str]:
    """Extract formatted address from Google Maps reverse geocode."""
    payload = _reverse_geocode_station_payload(station)
    if not payload:
        return None
    formatted = payload.get("formatted_address")
    return str(formatted).strip() if formatted else None


def _get_station_address(station: Dict[str, Any]) -> Optional[str]:
    """
    Get station address with priority:
    1. Tankerkönig data (street, house_number, post_code, city) - always available
    2. Google Maps reverse geocoding - only cached for best station
    3. Return None - let caller handle city-only fallback with proper formatting
    """
    # Try Tankerkönig first (works for ALL stations)
    tk_address = _build_tankerkoenig_address(station)
    if tk_address:
        return tk_address
    
    # Fallback to Google Maps (only works if cached)
    gm_address = _reverse_geocode_station_address(station)
    if gm_address:
        return gm_address
    
    # Don't return city-only here - let the caller's fallback logic handle it
    # The hero section has smarter logic that can extract address from the name field
    return None


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

def _parse_opening_hours(station: Dict[str, Any], eta_datetime: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
    """
    Parse opening hours from station data at ETA time.
    Returns (is_open_at_eta, closing_or_opening_time_text)
    
    Priority:
    1. Google's is_open_at_eta (most reliable, directly from Places API)
    2. Tankerkönig's openingtimes_json (for closing/opening time text)
    
    Args:
        station: Station dict with is_open_at_eta and/or openingtimes_json
        eta_datetime: Expected arrival time (uses current time if None)
    
    Returns:
        (is_open, time_info) where:
        - is_open=True, time_info="HH:MM" -> Open until time_info
        - is_open=False, time_info="HH:MM" -> Closed, opens at time_info
        - is_open=True, time_info=None -> Open (24/7 or unknown closing time)
        - is_open=False, time_info=None -> Closed (unknown opening time)
        - is_open=None (as bool False), time_info="UNKNOWN" -> No data available
    """
    # 1. Check Google's is_open_at_eta first (most reliable)
    google_is_open = station.get("is_open_at_eta")
    
    # 2. Try to parse Tankerkönig opening times for time details
    opening_times_json = station.get("openingtimes_json")
    time_info = None
    tk_is_open = None
    
    if opening_times_json and opening_times_json != "{}":
        try:
            if isinstance(opening_times_json, str):
                hours_data = json.loads(opening_times_json)
            else:
                hours_data = opening_times_json
            
            if hours_data and hours_data != {}:
                # Use ETA or current time
                if eta_datetime:
                    check_time = eta_datetime
                else:
                    try:
                        check_time = datetime.now(tz=ZoneInfo("Europe/Berlin") if ZoneInfo else None)
                    except Exception:
                        check_time = datetime.now()
                
                current_time = check_time.time()
                weekday = check_time.strftime("%A").lower()
                
                today_hours = hours_data.get(weekday, [])
                
                if today_hours:
                    # Check if currently within any open period
                    for period in today_hours:
                        start_str = period.get("start")
                        end_str = period.get("end")
                        
                        if not start_str or not end_str:
                            continue
                        
                        try:
                            start_time = time.fromisoformat(start_str)
                            end_time = time.fromisoformat(end_str)
                            
                            if start_time <= current_time <= end_time:
                                tk_is_open = True
                                time_info = end_str  # Closing time
                                break
                        except Exception:
                            continue
                    
                    # If not open, find next opening time
                    if tk_is_open is None:
                        tk_is_open = False
                        for period in today_hours:
                            start_str = period.get("start")
                            if start_str:
                                try:
                                    start_time = time.fromisoformat(start_str)
                                    if current_time < start_time:
                                        time_info = start_str  # Opening time
                                        break
                                except Exception:
                                    pass
        except Exception:
            pass
    
    # 3. Determine final is_open status
    # Priority: Google (explicit) > Tankerkönig (parsed)
    if google_is_open is not None:
        # Google has data - use it
        is_open = bool(google_is_open)
    elif tk_is_open is not None:
        # Tankerkönig has data - use it
        is_open = tk_is_open
    else:
        # No data from either source - return special indicator
        return (False, "UNKNOWN")
    
    return (is_open, time_info)


# =============================================================================
# Helper Functions
# =============================================================================

def _render_page03_quick_back_buttons() -> None:
    """Render Page-03-only back-navigation buttons directly under the sidebar toggle."""
    if st.sidebar.button(
        "Back to Trip Planner",
        type="primary",
        use_container_width=True,
        key="p02_back_to_trip_planner",
        help="Return to the Home page.",
    ):
        # Do NOT set st.session_state["top_nav"] here (top_nav widget already exists on this page).
        # Instead, request Home navigation and let streamlit_app.py apply it before rendering its widget.
        st.session_state["nav_request_top_nav"] = "Home"

        try:
            maybe_persist_state(force=True)
        except Exception:
            pass

        st.switch_page("streamlit_app.py")

    if st.sidebar.button(
        "Back to Station Explorer",
        type="primary",
        use_container_width=True,
        key="p03_back_to_station_explorer",
        help="Return to Station Explorer.",
    ):
        try:
            maybe_persist_state(force=True)
        except Exception:
            pass
        st.switch_page("pages/04_station_explorer.py")


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


def _get_onroute_thresholds(last_run: Dict[str, Any]) -> Tuple[float, float]:
    """
    Get on-route thresholds from last_run filter_log, or use constants.
    Ensures consistency with page 02 calculations.
    
    Returns:
        Tuple of (max_detour_km, max_detour_min)
    """
    filter_log = last_run.get("filter_log") or {}
    thresholds = filter_log.get("thresholds") or {}
    
    try:
        km = float(thresholds.get("onroute_max_detour_km", ONROUTE_MAX_DETOUR_KM) or ONROUTE_MAX_DETOUR_KM)
    except (TypeError, ValueError):
        km = float(ONROUTE_MAX_DETOUR_KM)
    
    try:
        mins = float(thresholds.get("onroute_max_detour_min", ONROUTE_MAX_DETOUR_MIN) or ONROUTE_MAX_DETOUR_MIN)
    except (TypeError, ValueError):
        mins = float(ONROUTE_MAX_DETOUR_MIN)
    
    return km, mins


def _get_exclusion_thresholds(last_run: Dict[str, Any]) -> Tuple[float, float]:
    """
    Get the actual exclusion thresholds (user's max detour settings) from last_run filter_log.
    
    These are the hard caps that determine whether a station is excluded from ranking.
    Different from on-route thresholds which define what counts as "on-route".
    
    Returns:
        Tuple of (max_detour_km, max_detour_min) - the user's constraint settings
    """
    filter_log = last_run.get("filter_log") or {}
    thresholds = filter_log.get("thresholds") or {}
    
    # Default values match sidebar defaults
    DEFAULT_MAX_KM = 5.0
    DEFAULT_MAX_MIN = 10.0
    
    try:
        km = thresholds.get("max_detour_km")
        if km is not None:
            km = float(km)
        else:
            # Fallback to session_state if not in filter_log
            km = float(st.session_state.get("max_detour_km", DEFAULT_MAX_KM))
    except (TypeError, ValueError):
        km = DEFAULT_MAX_KM
    
    try:
        mins = thresholds.get("max_detour_min")
        if mins is not None:
            mins = float(mins)
        else:
            # Fallback to session_state if not in filter_log
            mins = float(st.session_state.get("max_detour_min", DEFAULT_MAX_MIN))
    except (TypeError, ValueError):
        mins = DEFAULT_MAX_MIN
    
    return km, mins


def _compute_worst_onroute_price(
    ranked: List[Dict[str, Any]],
    fuel_code: str,
    onroute_max_km: float,
    onroute_max_min: float,
) -> Optional[float]:
    """
    Find the worst (highest) predicted price among on-route stations.
    Matches the logic in page 02's _compute_onroute_worst_price.
    """
    pred_key = f"pred_price_{fuel_code}"
    vals: List[float] = []
    
    for s in ranked:
        p = s.get(pred_key)
        if p is None:
            continue
        
        d_km = _safe_float(s.get("detour_distance_km") or s.get("detour_km") or 0)
        d_min = _safe_float(s.get("detour_duration_min") or s.get("detour_min") or 0)
        
        if d_km is None:
            d_km = 0.0
        if d_min is None:
            d_min = 0.0
        
        if d_km <= onroute_max_km and d_min <= onroute_max_min:
            try:
                vals.append(float(p))
            except (TypeError, ValueError):
                continue
    
    return max(vals) if vals else None


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
            title="",
            height=250,
        )
        return fig, ""

    # Sort by date and ensure datetime
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Calculate X-axis range: always show 7 days ending at "now"
    from datetime import datetime, timedelta
    now = datetime.now()
    x_max = now + timedelta(hours=1)  # Small buffer after "now"
    x_min = now - timedelta(days=7)
    
    # Forward-fill to extend the line to "now" if data ends before today
    # This is correct because fuel prices stay constant until changed
    last_date = df["date"].max()
    last_price = df["price"].iloc[-1]
    
    if last_date < now - timedelta(hours=1):
        # Add a point at "now" with the last known price
        now_row = pd.DataFrame({
            "date": [now],
            "price": [last_price]
        })
        df = pd.concat([df, now_row], ignore_index=True)

    # For mobile: Sample every 3rd point if we have many data points
    df_sampled = df.iloc[::3] if len(df) > 30 else df

    fig = go.Figure()
    # Step line (prices jump, not smooth) - no markers
    fig.add_trace(go.Scatter(
        x=df_sampled["date"],
        y=df_sampled["price"],
        mode="lines",
        name="Price",
        line=dict(color="#1f77b4", width=3, shape="hv"),  # hv = step graph
        hovertemplate="%{x|%b %d %H:%M}<br>€%{y:.3f}/L<extra></extra>",
    ))

    avg_price = float(df["price"].mean())
    min_price = float(df["price"].min())
    max_price = float(df["price"].max())
    
    # Calculate sensible y-axis range (not starting from 0)
    y_padding = (max_price - min_price) * 0.15 if max_price > min_price else 0.05
    y_min = max(0, min_price - y_padding)
    y_max = max_price + y_padding

    fig.update_layout(
        title="",  # Empty instead of None
        xaxis_title=None,
        yaxis_title=None,  # Remove rotated label - use annotation instead
        hovermode="x unified",
        height=250,  # More compact
        showlegend=False,
        margin=dict(l=35, r=10, t=20, b=35),  # Less left margin
        xaxis=dict(
            fixedrange=True,
            range=[x_min, x_max],  # Always show full 7-day window
        ),
        yaxis=dict(fixedrange=True, range=[y_min, y_max]),  # Don't start from 0
    )
    
    # Add €/L as small annotation at top-left
    fig.add_annotation(
        text="€/L",
        xref="paper", yref="paper",
        x=0, y=1.02,
        showarrow=False,
        font=dict(size=10, color="#666"),
        xanchor="left",
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
            title="",
            height=250,
        )
        return fig

    # Count hours with actual data
    hours_with_data = hourly_df["avg_price"].notna().sum()
    total_records = hourly_df["count"].sum() if "count" in hourly_df.columns else hours_with_data
    
    # MOBILE: Show every 2 hours (12 bars instead of 24)
    hourly_df_mobile = hourly_df[hourly_df["hour"] % 2 == 0].copy()
    
    # For display, track which hours have real data
    hourly_df_mobile["has_data"] = hourly_df_mobile["avg_price"].notna()
    
    # Get optimal hours from DISPLAYED hours only (not full 24h)
    # This ensures the green/red bar actually appears on screen
    optimal = get_cheapest_and_most_expensive_hours(hourly_df_mobile)
    cheapest_hour = optimal.get("cheapest_hour")
    most_expensive_hour = optimal.get("most_expensive_hour")

    # Color bars - gray for no data, colors only for hours WITH data
    colors = []
    for _, row in hourly_df_mobile.iterrows():
        if not row["has_data"]:
            colors.append("#e5e5e5")  # Light gray for no data
        elif cheapest_hour is not None and int(row["hour"]) == int(cheapest_hour):
            colors.append("#4caf50")  # Green
        elif most_expensive_hour is not None and int(row["hour"]) == int(most_expensive_hour):
            colors.append("#f44336")  # Red
        else:
            colors.append("#ffc107")  # Yellow/Orange

    # For bars, replace NaN with the minimum valid value (so gray bars show at minimum height)
    min_valid = hourly_df_mobile.loc[hourly_df_mobile["has_data"], "avg_price"].min()
    max_valid = hourly_df_mobile.loc[hourly_df_mobile["has_data"], "avg_price"].max()
    
    # If no valid data at all, show empty chart
    if pd.isna(min_valid):
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for hourly pattern",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="gray"),
        )
        fig.update_layout(title="", height=250)
        return fig
    
    # Fill NaN with a value below min so gray bars are shorter
    placeholder_val = min_valid - (max_valid - min_valid) * 0.5 if max_valid > min_valid else min_valid * 0.95
    hourly_df_mobile["display_price"] = hourly_df_mobile["avg_price"].fillna(placeholder_val)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_df_mobile["hour"],
        y=hourly_df_mobile["display_price"],
        marker_color=colors,
        hovertemplate=[
            f"Hour {int(row['hour'])}:00<br>€{row['avg_price']:.3f}/L<extra></extra>" 
            if row["has_data"] else f"Hour {int(row['hour'])}:00<br>No data<extra></extra>"
            for _, row in hourly_df_mobile.iterrows()
        ],
        showlegend=False,
        width=1.5,
    ))

    # Calculate sensible y-axis range
    y_padding = (max_valid - min_valid) * 0.15 if max_valid > min_valid else 0.05
    y_min = max(0, placeholder_val - y_padding * 0.5)
    y_max = max_valid + y_padding

    fig.update_layout(
        title="",
        xaxis_title="Hour",
        yaxis_title=None,
        height=250,
        margin=dict(l=35, r=10, t=20, b=35),
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True, range=[y_min, y_max]),
    )
    
    # Add €/L as small annotation at top-left
    fig.add_annotation(
        text="€/L",
        xref="paper", yref="paper",
        x=0, y=1.02,
        showarrow=False,
        font=dict(size=10, color="#666"),
        xanchor="left",
    )

    # Add gridlines and configure axes
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=4)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')

    return fig


def create_weekday_hour_heatmap(df: pd.DataFrame, fuel_type: str) -> Optional[go.Figure]:
    """
    Create a heatmap showing price patterns by weekday and hour.
    Y-axis: Weekday (Mon-Sun)
    X-axis: Hour (0-23)
    Color: Price (green=cheap, red=expensive)
    
    Uses forward-fill to show the EFFECTIVE price at each hour,
    not just when prices changed.
    """
    if df is None or df.empty:
        return None
    
    # Ensure we have datetime and can extract weekday/hour
    if "date" not in df.columns or "price" not in df.columns:
        return None
    
    try:
        df_work = df.copy()
        df_work["date"] = pd.to_datetime(df_work["date"])
        df_work = df_work.sort_values("date").reset_index(drop=True)
        
        # Forward-fill to create complete hourly time series
        if len(df_work) >= 2:
            try:
                min_date = df_work["date"].min()
                max_date = df_work["date"].max()
                
                # Set date as index for resampling
                df_work = df_work.set_index("date")
                
                # Resample to hourly and forward-fill
                hourly_prices = df_work["price"].resample('H').last().ffill()
                
                # Convert back to DataFrame
                df_work = pd.DataFrame({
                    "date": hourly_prices.index,
                    "price": hourly_prices.values
                })
            except Exception:
                # Fall back to original data
                df_work = df.copy()
                df_work["date"] = pd.to_datetime(df_work["date"])
        
        df_work["weekday"] = df_work["date"].dt.dayofweek  # 0=Mon, 6=Sun
        df_work["hour"] = df_work["date"].dt.hour
        
        # Group by weekday and hour, get mean price
        pivot = df_work.groupby(["weekday", "hour"])["price"].mean().reset_index()
        
        # Need at least SOME data (very lenient - just 5 points)
        if len(pivot) < 5:
            return None
        
        # Create pivot table for heatmap
        heatmap_data = pivot.pivot(index="weekday", columns="hour", values="price")
        
        # Reindex to ensure all 24 hours are shown (0-23), even if some are missing
        heatmap_data = heatmap_data.reindex(columns=range(24))
        
        # Weekday labels - only for weekdays we have data for
        weekday_labels_full = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekdays_present = sorted(heatmap_data.index.tolist())
        weekday_labels = [weekday_labels_full[i] for i in weekdays_present]
        
        # Always show all 24 hours on x-axis
        all_hours = list(range(24))
        
        # Create heatmap with full hour coverage
        z_data = heatmap_data.loc[weekdays_present, all_hours].values
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=all_hours,
            y=weekday_labels,
            colorscale=[
                [0, "#4caf50"],      # Green = cheapest
                [0.5, "#ffc107"],    # Yellow = medium
                [1, "#f44336"],      # Red = most expensive
            ],
            hovertemplate="<b>%{y}</b> at %{x}:00<br>€%{z:.3f}/L<extra></extra>",
            colorbar=dict(
                title=dict(text="€/L", side="right"),
                thickness=15,
                len=0.9,
            ),
            showscale=True,
            # Show white/blank for NaN values
            zauto=True,
        ))
        
        fig.update_layout(
            title="",  # Empty - we have markdown heading
            xaxis_title="Hour",
            yaxis_title=None,
            height=280,
            margin=dict(l=45, r=60, t=10, b=40),
            xaxis=dict(
                fixedrange=True,
                tickmode="linear",
                tick0=0,
                dtick=3,  # Show every 3 hours: 0, 3, 6, 9, 12, 15, 18, 21
                tickangle=0,  # Horizontal labels
            ),
            yaxis=dict(
                fixedrange=True,
                autorange="reversed",  # Mo on top, So on bottom
            ),
        )
        
        return fig
        
    except Exception:
        return None


def create_comparison_chart(
    stations_data: Dict[str, pd.DataFrame],
    fuel_type: str,
    current_station_name: Optional[str] = None,
) -> Tuple[go.Figure, str, str]:
    """
    Comparison chart: current station vs selected alternatives.
    Returns: (figure, current_station_full_name, comparison_station_full_name)
    """
    if not stations_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No stations to compare",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(height=280)
        return fig, "", ""

    fig = go.Figure()
    current_full_name = ""
    comparison_full_name = ""
    
    # Calculate time range for forward-fill and X-axis
    from datetime import datetime, timedelta
    now = datetime.now()
    x_max = now + timedelta(hours=1)
    x_min = now - timedelta(days=7)
    
    for idx, (station_name, df) in enumerate(stations_data.items()):
        if df is None or df.empty:
            continue
        
        # Forward-fill: extend line to "now" with last known price
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        last_date = df["date"].max()
        last_price = df["price"].iloc[-1]
        
        if last_date < now - timedelta(hours=1):
            now_row = pd.DataFrame({
                "date": [now],
                "price": [last_price]
            })
            df = pd.concat([df, now_row], ignore_index=True)

        is_current = ("(Current)" in station_name) or (current_station_name and station_name.startswith(current_station_name))
        
        # Store full names for caption
        full_name = station_name.replace(" (Current)", "")

        if is_current:
            color = "#16a34a"  # Green for current
            width = 2
            opacity = 1.0
            dash = "solid"
            legend_name = "Your station ★"
            current_full_name = full_name
        else:
            # Blue for comparison
            color = "#2563eb"
            width = 1.5
            opacity = 0.9
            dash = "dot"
            legend_name = "Comparison"
            comparison_full_name = full_name

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["price"],
            mode="lines",
            name=legend_name,
            line=dict(color=color, width=width, dash=dash, shape="hv"),
            opacity=opacity,
            hovertemplate=f"{full_name}<br>%{{x|%b %d %H:%M}}<br>€%{{y:.3f}}/L<extra></extra>",
        ))

    # Calculate y-axis range (don't start from 0)
    all_prices = []
    for df in stations_data.values():
        if df is not None and not df.empty:
            all_prices.extend(df["price"].dropna().tolist())
    
    if all_prices:
        y_min = min(all_prices)
        y_max = max(all_prices)
        y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.05
    else:
        y_min, y_max, y_padding = 1.5, 2.0, 0.05

    fig.update_layout(
        title="",
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
        height=280,  # Compact with legend
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,  # Move legend lower (was -0.08)
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        margin=dict(l=35, r=10, t=10, b=70),  # More bottom margin (was 55)
        xaxis=dict(
            fixedrange=True,
            range=[x_min, x_max],  # Always show full 7-day window
        ),
        yaxis=dict(
            fixedrange=True, 
            range=[y_min - y_padding, y_max + y_padding],
            tickformat=".2f",
        ),
    )
    
    # Add €/L as small annotation at top-left
    fig.add_annotation(
        text="€/L",
        xref="paper", yref="paper",
        x=0, y=1.02,
        showarrow=False,
        font=dict(size=10, color="#666"),
        xanchor="left",
    )

    return fig, current_full_name, comparison_full_name


# =============================================================================
# Sidebar Trip Settings Renderer
# =============================================================================

def _render_trip_settings():
    """
    Render trip settings in the sidebar Action tab.
    Adapts content based on whether user came from Trip Planner or Explorer.
    """
    last_run = st.session_state.get("last_run") or {}
    explorer_results = list(st.session_state.get("explorer_results") or [])
    
    # If no data at all, show info message
    if not last_run and not explorer_results:
        st.sidebar.info("No station data available. Use Trip Planner or Station Explorer to find stations.")
        return
    
    # Detect source context - check BOTH session state AND radio widget state
    source = st.session_state.get("selected_station_source", "")
    
    # Also check the radio widget directly (if it exists)
    radio_key = "station_details_source_radio"
    if radio_key in st.session_state:
        radio_label = st.session_state.get(radio_key, "")
        if "explorer" in radio_label.lower():
            source = "explorer"
        elif "route" in radio_label.lower():
            source = "route"
    
    # AUTO-DETECT: If no explicit source, infer from available data
    if source not in {"route", "explorer"}:
        if explorer_results and not last_run:
            # Only explorer data available → use explorer mode
            source = "explorer"
        elif last_run and not explorer_results:
            # Only route data available → use route mode
            source = "route"
        else:
            # Both available → default to route (Trip Planner)
            source = "route"
    
    # EDGE CASE: User selected "route" but no route data exists
    if source == "route" and not last_run:
        if explorer_results:
            # Fall back to explorer mode
            source = "explorer"
        else:
            st.sidebar.info("No Trip Planner data. Run Trip Planner on the Home page first.")
            return
    
    is_explorer_mode = (source == "explorer")
    
    # -------------------------------------------------------------------------
    # EXPLORER MODE
    # -------------------------------------------------------------------------
    if is_explorer_mode:
        # Station selector FIRST (main interaction)
        st.sidebar.markdown(
            "### Select Station",
            help="Choose a station to analyze from your Explorer search results."
        )

        render_station_selector(
            last_run=last_run,
            explorer_results=explorer_results,
            max_ranked=200,
            max_excluded=200,
        )

        # Comparison selector - include explorer stations
        current_uuid = str(st.session_state.get("selected_station_uuid") or "")
        ranked = list(last_run.get("ranked") or [])
        stations = list(last_run.get("stations") or [])

        render_comparison_selector(
            ranked,
            stations,
            current_uuid,
            max_ranked=50,
            max_excluded=100,
            explorer_results=explorer_results,
        )

        # Settings moved to BOTTOM
        st.sidebar.markdown(
            "### Station Explorer Settings",
            help="You selected a station from the Station Explorer. Trip Planner settings (detour costs, economics) don't apply here."
        )

        fuel_code = str(last_run.get("fuel_code") or st.session_state.get("fuel_label") or "e5")
        fuel_label = fuel_code.upper()
        st.sidebar.markdown(
            f"- Fuel: {fuel_label}",
            help="Fuel type from your last Trip Planner run or Explorer search. Change it on the Home page or in Explorer."
        )

        return

    
    # -------------------------------------------------------------------------
    # TRIP PLANNER MODE (default)
    # -------------------------------------------------------------------------
    # Note: We already handled the case of no last_run in the logic above,
    # so at this point we're guaranteed to have last_run data.
    
    # Extract settings
    fuel_code = str(last_run.get("fuel_code") or st.session_state.get("fuel_label") or "e5")
    fuel_label = fuel_code.upper()
    
    use_economics = bool(last_run.get("use_economics", False))
    econ_status = "Yes" if use_economics else "No"
    
    litres = _safe_float(last_run.get("litres_to_refuel"))
    refuel_amount = f"{litres:.0f} L" if litres else "-"
    
    ranked = list(last_run.get("ranked") or [])
    stations = list(last_run.get("stations") or [])
    route_station_count = len(stations)
    
    explorer_station_count = len(explorer_results)
    
    # Station selectors FIRST (main interaction)
    st.sidebar.markdown(
        "### Select Station",
        help=(
            "Choose a station to analyze.\n\n"
            "From latest route run: The list always includes ALL stations found along your last route, "
            "sorted by net saving in descending order. Stations with missing net saving appear at the bottom.\n\n"
            "From station explorer: The list shows your Explorer search results."
        )
    )

    render_station_selector(
        last_run=last_run,
        explorer_results=explorer_results,
        max_ranked=200,
        max_excluded=200,
    )

    # Comparison selector
    current_uuid = str(st.session_state.get("selected_station_uuid") or "")
    render_comparison_selector(
        ranked,
        stations,
        current_uuid,
        max_ranked=50,
        max_excluded=100,
        explorer_results=explorer_results if explorer_results else None,
    )

    # Settings moved to BOTTOM
    st.sidebar.markdown(
        "### Trip Planner Settings",
        help="These settings come from your last Trip Planner run on the Home page. To change them, go back to Home and run a new trip."
    )

    st.sidebar.markdown(f"- Fuel: {fuel_label}")
    st.sidebar.markdown(f"- Economics: {econ_status}")
    st.sidebar.markdown(f"- Refuel amount: {refuel_amount}")
    st.sidebar.markdown(f"- Route stations: {route_station_count}")




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
    st.caption("##### Deep-dive into individual station data and price patterns.")

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
    # SIDEBAR - WITH TRIP SETTINGS IN ACTION TAB AND HELP CONTENT
    # =========================================================================
    render_sidebar_shell(
        top_renderer=_render_page03_quick_back_buttons,
        action_renderer=_render_trip_settings,
        help_renderer=_render_help_station,
    )
    
    # Station selectors are now rendered inside _render_trip_settings() for faster sidebar switching
    last_run = st.session_state.get("last_run") or {}
    ranked = list(last_run.get("ranked") or [])
    stations = list(last_run.get("stations") or [])
    explorer_results = list(st.session_state.get("explorer_results") or [])

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
        # Check if there's any data available at all
        has_any_data = bool(ranked or stations or explorer_results)
        
        if has_any_data:
            st.info("No station selected. Please select a station from the sidebar.")
            st.caption("Tip: Use the 'Select Station' dropdown in the sidebar to choose a station.")
        else:
            st.info("No station data available yet. Run a route recommendation first on the home page or use the Station Explorer.")
        
        maybe_persist_state()
        return

    # Reduce vertical spacing for tighter layout (similar to page 02)
    st.markdown("""
    <style>
        /* Reduce gap between elements */
        .stMarkdown { margin-bottom: 0.25rem !important; }
        .stMetric { padding: 0.5rem 0 !important; }
        
        /* Tighter section headers */
        h3 { margin-top: 1rem !important; margin-bottom: 0.5rem !important; }
        h4 { margin-top: 0.75rem !important; margin-bottom: 0.25rem !important; }
        
        /* Reduce chart container padding */
        .stPlotlyChart { margin-bottom: 0.5rem !important; }
        
        /* Reduce expander padding */
        .streamlit-expanderHeader { padding: 0.5rem 0 !important; }
        
        /* Tighter info boxes */
        .stAlert { padding: 0.5rem !important; margin: 0.25rem 0 !important; }
        
        /* Reduce column gaps */
        [data-testid="column"] { padding: 0.25rem !important; }
        
        /* Tighter captions */
        .stCaption { margin-top: 0 !important; margin-bottom: 0.25rem !important; }
                
        /* 2) Tighten headline spacing (add h1/h2; you already have h3/h4) */
        h1 { margin-top: 0.0rem !important; margin-bottom: 0.35rem !important; }
        h2 { margin-top: 0rem !important; margin-bottom: 0.35rem !important; }
        h3 { margin-top: -1rem !important; margin-bottom: 0.35rem !important; }
        h4 { margin-top: -0.5rem !important; margin-bottom: 0.25rem !important; }
                
    </style>
    """, unsafe_allow_html=True)
    

    
    # =========================================================================
    # EXTRACT STATION INFO
    # =========================================================================
    
    # Detect source context EARLY so all sections can use it
    # Check BOTH session state AND radio widget state
    station_source = st.session_state.get("selected_station_source", "")
    radio_key = "station_details_source_radio"
    if radio_key in st.session_state:
        radio_label = st.session_state.get(radio_key, "")
        if "explorer" in radio_label.lower():
            station_source = "explorer"
        elif "route" in radio_label.lower():
            station_source = "route"
    
    # Auto-detect if not set
    if station_source not in {"route", "explorer"}:
        # Check if we actually have route data (not just empty dict)
        has_route_data = bool(last_run and (last_run.get("ranked") or last_run.get("stations")))
        has_explorer_data = bool(explorer_results)
        
        if has_explorer_data and not has_route_data:
            station_source = "explorer"
        elif has_route_data and not has_explorer_data:
            station_source = "route"
        elif has_explorer_data and has_route_data:
            # Both available - check which was used to select current station
            current_station_source = st.session_state.get("selected_station_source", "")
            station_source = current_station_source if current_station_source in {"route", "explorer"} else "route"
        else:
            station_source = "route"  # Default fallback
    
    is_from_explorer = (station_source == "explorer")
    
    name = _station_name(station)
    brand = _station_brand(station)
    city = _station_city(station) or _reverse_geocode_station_city(station)
    full_address = _get_station_address(station)  # Tankerkönig first, then Google Maps, then city
    
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
    
    # Track if we're using fallback (important for user transparency)
    using_current_as_fallback = (predicted_price is None and current_price is not None)
    
    econ_net_key = f"econ_net_saving_eur_{fuel_code}"
    econ_baseline_key = f"econ_baseline_price_{fuel_code}"
    net_saving = _safe_float(station.get(econ_net_key))
    baseline_price = _safe_float(station.get(econ_baseline_key))
    
    # -------------------------------------------------------------------------
    # Selection universe (IMPORTANT):
    # Page 01 persists the "value view" set into last_run:
    # - value_view_stations: stations considered "selected/value-comparable" (new logic)
    # - value_view_meta: metadata about the benchmark (optional)
    #
    # For route context, we treat "value_view_stations" as the authoritative set.
    # For explorer context, we keep the neutral "Explorer" behavior.
    # -------------------------------------------------------------------------
    value_view_stations = list((last_run or {}).get("value_view_stations") or [])
    value_view_meta = dict((last_run or {}).get("value_view_meta") or {})

    # Use the new selection set if available (route mode).
    # Fall back to legacy ranked if value view is missing (older cache).
    rank_universe = value_view_stations if value_view_stations else list(ranked or [])

    # "Selected" means: part of the value-view universe (new logic) when coming from route.
    # In Explorer mode, we do not apply selection status.
    is_selected = (not is_from_explorer) and any(_station_uuid(s) == station_uuid for s in rank_universe)

    # Backward-compatible alias: the rest of the page still expects `is_ranked`.
    # In route context, "ranked" now means "selected/value-comparable" (new logic).
    is_ranked = is_selected

    # Get thresholds from last_run for exclusion detection (best effort)
    # on-route thresholds: define what counts as "on-route" for benchmark (e.g. 1 km, 5 min)
    onroute_km_threshold, onroute_min_threshold = _get_onroute_thresholds(last_run)
    # exclusion thresholds: user's hard caps (e.g. 5 km, 10 min)
    max_detour_km_cap, max_detour_min_cap = _get_exclusion_thresholds(last_run)

    eta_str_for_check = station.get("eta")

    # -------------------------------------------------------------------------
    # ETA datetime + opening-hours evaluation
    # (Needed for the Opening Hours info card further below)
    # -------------------------------------------------------------------------
    eta_datetime: Optional[datetime] = None
    if eta_str_for_check:
        try:
            if isinstance(eta_str_for_check, str):
                eta_datetime = datetime.fromisoformat(eta_str_for_check.replace("Z", "+00:00"))
            else:
                eta_datetime = eta_str_for_check
        except Exception:
            eta_datetime = None

    # Always compute opening status (ETA if available, otherwise "now")
    is_open, time_info = _parse_opening_hours(station, eta_datetime)

    # Provide an explanation ONLY when a route station is not in the selected/value set.
    # Note: This is now aligned with the new logic: "selected" vs "other stations".
    exclusion_reason = None
    if (not is_from_explorer) and (not is_selected):
        # Best-effort "why not selected?" checks.
        # We keep your existing checks, but we do NOT label this as "Not Ranked" anymore.

        # 1) Detour caps
        if detour_km is not None and max_detour_km_cap is not None and detour_km > max_detour_km_cap:
            exclusion_reason = f"Detour exceeds limit ({detour_km:.1f} km > {max_detour_km_cap:.1f} km max)"
        elif detour_min is not None and max_detour_min_cap is not None and detour_min > max_detour_min_cap:
            exclusion_reason = f"Detour time exceeds limit ({detour_min:.0f} min > {max_detour_min_cap:.0f} min max)"

        # 2) Closed at ETA (only if we can evaluate opening hours)
        elif eta_str_for_check:
            try:
                eta_dt = (
                    datetime.fromisoformat(eta_str_for_check.replace("Z", "+00:00"))
                    if isinstance(eta_str_for_check, str)
                    else eta_str_for_check
                )
                is_open_check, _ = _parse_opening_hours(station, eta_dt)
                if is_open_check is False:
                    exclusion_reason = (
                        f"Station closed at arrival time ({eta_dt.strftime('%H:%M') if eta_dt else 'unknown'})"
                    )
            except Exception:
                pass

        # 3) Missing prediction (value-view requires a valid predicted price)
        if exclusion_reason is None and predicted_price is None:
            exclusion_reason = "No predicted price available (not part of the value comparison set)"

        # 4) Economics: if enabled and net_saving exists, explain directionally
        # (We intentionally do NOT enforce the old 'net_saving < 0' logic here.)
        if exclusion_reason is None and use_economics:
            if net_saving is None:
                exclusion_reason = "Savings could not be computed (missing economic inputs)"
            else:
                # If Page 01 provided a benchmark price, the value-view logic is price-based.
                # Still give a user-friendly economic interpretation:
                if float(net_saving) < 0:
                    exclusion_reason = (
                        f"Negative net savings (€{float(net_saving):.2f}) – detour cost exceeds price benefit"
                    )
                else:
                    exclusion_reason = (
                        f"Positive net savings (€{float(net_saving):.2f}), but not in the value comparison set"
                    )

        if exclusion_reason is None:
            exclusion_reason = "Not part of the selected/value comparison set for this run"

    # -------------------------------------------------------------------------
    # HERO SECTION (NO RATING)
    # - Keep the cyan box with station title, address and price only.
    # - No traffic-light / ranking / tooltip / expander.
    # -------------------------------------------------------------------------

    # =========================================================================
    # HERO SECTION - CYAN BOX WITH PRICE (NO TRAFFIC LIGHT)
    # =========================================================================

    # Check for valid brand (not empty, not dash variants, not just whitespace)
    invalid_brand_values = {"", "-", "—", "–", "None", "null", "N/A", "n/a"}
    brand_valid = brand and brand.strip() and brand.strip() not in invalid_brand_values

    base_name = brand if brand_valid else name
    city_clean = city.title() if (city and city.strip() and city.strip() not in invalid_brand_values) else ""

    # Handle case where base_name is still invalid
    if not base_name or base_name.strip() in invalid_brand_values:
        base_name = "Unknown Station"

    title_line = f"{base_name} in {city_clean}" if (base_name and city_clean) else base_name

    # Build address line (same logic as before)
    street_clean = station.get("street", "")
    house_number_clean = station.get("house_number", "")
    post_code_clean = station.get("post_code", "")
    city_clean_full = station.get("city", "")

    address_parts = []
    street_part = f"{street_clean} {house_number_clean}".strip() if street_clean else ""
    city_part = f"{post_code_clean} {city_clean_full}".strip() if (post_code_clean or city_clean_full) else ""

    if street_part:
        address_parts.append(street_part)
    if city_part:
        address_parts.append(city_part)

    address_line = ", ".join([p for p in address_parts if p]) if address_parts else ""

    # --- Price display (consistent formatting) ---
    price_display = f"€{display_price:.3f}" if display_price else "—"

    # -------------------------------------------------------------------------
    # HERO (match Page 01 design): station-header only, price below address
    # -------------------------------------------------------------------------
    label_html = _safe_text("Selected station:")
    name_html = _safe_text(title_line) if title_line else ""
    addr_html = _safe_text(address_line) if address_line else ""
    price_html = _safe_text(price_display)

    hero_html = f"""
    <div class="station-header">
    <div class="label">{label_html}</div>
    <div class="name">{name_html}</div>
    {"<div class='addr'>" + addr_html + "</div>" if addr_html else ""}
    <div class="addr" style="margin-top:0.35rem; font-weight:800;">
        {price_html} <span style="font-weight:700; opacity:0.8;">/L</span>
    </div>
    </div>
    """.strip()

    st.markdown(hero_html, unsafe_allow_html=True)




    
    # =========================================================================
    # MISSING PREDICTION WARNING (if using current price as fallback)
    # =========================================================================
    if using_current_as_fallback:
        if is_from_explorer:
            # Explorer mode: predictions are NEVER available (no route = no ETA)
            st.info(
                "**Showing live price**: Explorer searches show current prices, not predictions. "
                "Predictions require a route with an estimated arrival time.",
                icon=None
            )
        else:
            # Trip Planner mode: prediction failed for some reason
            st.warning(
                "**Using live price (no forecast)**: No prediction available for this station at your arrival time. "
                "The displayed price is the current real-time price, not a forecast.",
                icon="⚠️"
            )
            with st.expander("Why is the prediction missing?", expanded=False):
                st.markdown("""
            **Possible reasons:**
            - The station may have limited historical price data
            - The model could not generate a reliable forecast for this specific time
            - Data collection gap for this station
            
            **What this means:**
            - The live price shown is from right now - it may change by the time you arrive
            - Savings calculations use today's live price, not a forecast for your arrival time
            - Consider checking prices again closer to your departure time
            """)
    
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
            st.info(f"{detour_km:.1f} km detour or +{detour_min:.0f} min")

    with info_cols[1]:
        # ETA (Expected Time of Arrival) - show day if different from today
        eta_str = station.get("eta")
        if eta_str:
            try:
                # Parse ISO format datetime
                if isinstance(eta_str, str):
                    eta_dt = datetime.fromisoformat(eta_str.replace('Z', '+00:00'))
                else:
                    eta_dt = eta_str
                
                # Check if ETA is on a different day than today
                try:
                    today = datetime.now(tz=ZoneInfo("Europe/Berlin") if ZoneInfo else None)
                except Exception:
                    today = datetime.now()
                
                if eta_dt.date() != today.date():
                    # Show day + time for next-day arrivals (e.g., "Mon 14:33")
                    eta_display = eta_dt.strftime("%a %H:%M")
                else:
                    # Same day - just show time
                    eta_display = eta_dt.strftime("%H:%M")
                
                st.info(f"ETA: {eta_display}")
            except Exception:
                st.info("ETA: --")
        else:
            st.info("ETA: --")

    with info_cols[2]:
        # Opening hours (ETA day) — match Page 02 semantics:
        # show the weekdayDescription for the ETA weekday (NOT an inferred open/closed status).
        oh = (station or {}).get("opening_hours")  # Google weekdayDescriptions (list)
        opening_hours_eta_day = "—"

        if isinstance(oh, list) and oh:
            # Use ETA weekday when we have 7 entries; otherwise a compact join
            if eta_datetime is not None and len(oh) >= 7:
                try:
                    opening_hours_eta_day = str(oh[int(eta_datetime.weekday())]).strip() or "—"  # 0=Mon..6=Sun
                except Exception:
                    opening_hours_eta_day = "—"
            else:
                items = [str(x).strip() for x in oh if x]
                opening_hours_eta_day = "; ".join(items[:3]) + ("; …" if len(items) > 3 else "") if items else "—"

        st.info(f"Opening hours: {opening_hours_eta_day}")


    # =========================================================================
    # SAVINGS CALCULATOR / PRICE COMPARISON
    # =========================================================================
    
    # Different behavior for Explorer vs Trip Planner mode
    if is_from_explorer:
        st.markdown("### Price Comparison")
        st.caption("Compare this station against other Explorer stations.")
        
        # In Explorer mode, we don't have route context
        if not display_price:
            st.info("No price data available for this station.")
        elif len(explorer_results) < 2:
            st.info("Search for more stations in Explorer to enable comparison.")
        else:
            # Find best and worst prices from explorer results WITH station names
            explorer_prices = []
            cheapest_station_name = None
            cheapest_price = float('inf')
            expensive_station_name = None
            expensive_price = 0
            
            for s in explorer_results:
                p = _safe_float(s.get(f"price_current_{fuel_code}"))
                if p and p > 0:
                    explorer_prices.append(p)
                    s_name = _station_name(s) or "Unknown"
                    s_city = _station_city(s)
                    full_name = f"{s_name} ({s_city})" if s_city else s_name
                    
                    if p < cheapest_price:
                        cheapest_price = p
                        cheapest_station_name = full_name
                    if p > expensive_price:
                        expensive_price = p
                        expensive_station_name = full_name
            
            if explorer_prices:
                min_explorer = min(explorer_prices)
                max_explorer = max(explorer_prices)
                avg_explorer = sum(explorer_prices) / len(explorer_prices)
                
                price_cols = st.columns(3)
                with price_cols[0]:
                    st.metric(
                        label="This station",
                        value=f"€{display_price:.3f}/L",
                        help="Current price at this station"
                    )
                with price_cols[1]:
                    st.metric(
                        label="Cheapest nearby",
                        value=f"€{min_explorer:.3f}/L",
                        help=f"Cheapest: {cheapest_station_name}" if cheapest_station_name else "Cheapest station in your Explorer search"
                    )
                with price_cols[2]:
                    st.metric(
                        label="Most expensive",
                        value=f"€{max_explorer:.3f}/L",
                        help=f"Most expensive: {expensive_station_name}" if expensive_station_name else "Most expensive station in your Explorer search"
                    )
                
                diff_from_cheapest = display_price - min_explorer
                
                if diff_from_cheapest < 0.001:
                    st.success(f"This is the cheapest station in your search!")
                else:
                    st.info(f"€{diff_from_cheapest:.3f}/L more expensive than the cheapest nearby")
                
                # Show which stations are cheapest/most expensive
                st.caption(f"Based on {len(explorer_prices)} stations. Average: €{avg_explorer:.3f}/L")
                if cheapest_station_name and expensive_station_name:
                    st.caption(f"Cheapest: {cheapest_station_name} | Most expensive: {expensive_station_name}")
            else:
                st.info("No price data available for comparison.")
    else:
        # TRIP PLANNER MODE - original Savings Calculator
        st.markdown("### Savings Calculator")
        st.caption("Compare this station against the most expensive on your route.")
        
        # Initialize variable that will be used later
        total_for_context = 0
        
        # Get on-route thresholds (aligned with page 02 and recommender)
        onroute_km_threshold, onroute_min_threshold = _get_onroute_thresholds(last_run)
        
        if not display_price:
            st.info("No price data available for this station.")
        else:
            # Find the WORST on-route price for user-friendly display
            # IMPORTANT: Use the full route station universe (stations), not only `ranked`.
            # `ranked` is post-filtered and may contain no on-route candidates, which would
            # incorrectly fall back to this station's own price (baseline == predicted).
            baseline_universe = list(stations or [])
            if not baseline_universe:
                baseline_universe = list(ranked or [])

            worst_onroute_computed = _compute_worst_onroute_price(
                baseline_universe, fuel_code, onroute_km_threshold, onroute_min_threshold
            )
            worst_onroute_price = worst_onroute_computed if worst_onroute_computed is not None else display_price
            
            # Calculate price difference per liter
            price_diff_per_liter = worst_onroute_price - display_price
            
            # ---------------------------------------------------------------------
            # PRICE COMPARISON (side by side)
            # ---------------------------------------------------------------------
            price_cols = st.columns(2)
            with price_cols[0]:
                st.metric(
                    label="This station",
                    value=f"€{display_price:.3f}/L",
                    delta=None,
                    help="The predicted price at your selected station"
                )
            with price_cols[1]:
                st.metric(
                    label="Worst on-route",
                    value=f"€{worst_onroute_price:.3f}/L",
                    delta=None,
                    help="Most expensive station directly on your route (≤1km, ≤5min). We assume no detour needed for this station."
                )
            
            # Show price difference
            if price_diff_per_liter > 0:
                st.success(f"**You save €{price_diff_per_liter:.3f} per liter**")
            elif price_diff_per_liter < 0:
                st.error(f"**This station is €{abs(price_diff_per_liter):.3f}/L more expensive**")
            else:
                st.info("Same price as worst on-route")
            
            
            # Fuel amount slider
            slider_value = st.slider(
                "Refuel amount (liters)",
                min_value=1.0,
                max_value=200.0,
                value=float(litres_to_refuel),
                step=1.0,
                key="savings_calc_slider",
            )
            
            # ---------------------------------------------------------------------
            # CALCULATE AND DISPLAY RESULTS
            # ---------------------------------------------------------------------
            gross_savings = price_diff_per_liter * slider_value
            
            # ECONOMICS MODE
            if use_economics:
                # Get detour costs
                detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
                time_cost_key = f"econ_time_cost_eur_{fuel_code}"
                
                detour_fuel_cost = _safe_float(station.get(detour_fuel_cost_key))
                time_cost = _safe_float(station.get(time_cost_key))
                
                # Calculate manually if not available
                if detour_fuel_cost is None or time_cost is None:
                    consumption = _safe_float(last_run.get("consumption_l_per_100km")) or _safe_float(st.session_state.get("consumption_l_per_100km")) or 7.0
                    value_of_time = _safe_float(last_run.get("value_of_time_eur_per_hour")) or _safe_float(st.session_state.get("value_of_time_eur_per_hour")) or 0.0
                    
                    if detour_km > 0 and display_price:
                        detour_fuel_cost = (detour_km * consumption / 100.0) * display_price
                    else:
                        detour_fuel_cost = 0.0
                    
                    if detour_min > 0 and value_of_time > 0:
                        time_cost = (detour_min / 60.0) * value_of_time
                    else:
                        time_cost = 0.0
                
                detour_costs = (detour_fuel_cost or 0.0) + (time_cost or 0.0)
                net_savings = gross_savings - detour_costs
                
                # Round for display consistency
                gross_rounded = round(gross_savings, 2)
                detour_rounded = round(detour_costs, 2)
                net_rounded = gross_rounded - detour_rounded
                
                # Show results with metrics - 3 columns for clearer display
                result_cols = st.columns(3)
                with result_cols[0]:
                    st.metric(
                        label="Net Savings",
                        value=f"€{net_rounded:.2f}",
                        help="Final savings after subtracting detour costs"
                    )
                with result_cols[1]:
                    st.metric(
                        label="Detour Costs",
                        value=f"€{detour_rounded:.2f}",
                        help="Extra fuel + time cost for the detour"
                    )
                with result_cols[2]:
                    st.metric(
                        label="Price Diff",
                        value=f"€{price_diff_per_liter:.3f}/L",
                        help="How much cheaper per liter vs worst on-route"
                    )
                
                # Show breakdown formula
                if detour_rounded > 0.01:
                    st.success(f"Gross: €{gross_rounded:.2f} − Detour: €{detour_rounded:.2f} = Net: €{net_rounded:.2f}")
                
                # Result evaluation for negative/break-even
                if net_savings < -0.01:
                    st.error(f"**Net Loss:** Detour costs exceed price savings")
                    if price_diff_per_liter > 0:
                        breakeven_liters = detour_costs / price_diff_per_liter
                        st.info(f"**Break-even:** Refuel at least **{breakeven_liters:.0f}L** to make this worthwhile.")
                elif abs(net_savings) <= 0.01:
                    st.info("**Break-even:** Detour costs cancel out savings")
                
                total_for_context = net_savings
                
            else:
                # NO ECONOMICS MODE - simpler display
                result_cols = st.columns(2)
                with result_cols[0]:
                    st.metric(
                        label="Gross Savings",
                        value=f"€{gross_savings:.2f}",
                        help=f"{slider_value:.0f}L × €{price_diff_per_liter:.3f}/L"
                    )
                with result_cols[1]:
                    st.metric(
                        label="Price Difference",
                        value=f"€{price_diff_per_liter:.3f}/L",
                        help="How much cheaper per liter vs worst on-route"
                    )
                
                st.warning("Detour costs not included. Enable Economics in Trip Planner for net savings.")
                total_for_context = gross_savings
            
            # Weekly/monthly/yearly projection
            if total_for_context > 0.01:
                weekly = total_for_context
                monthly = total_for_context * 4.3
                yearly = total_for_context * 52
                st.info(f"**If you refuel here weekly:** €{weekly:.2f}/week, €{monthly:.2f}/month, €{yearly:.2f}/year")
                st.caption("Assumes you refuel the same amount once per week. Monthly = weekly × 4.3, yearly = weekly × 52.")
            
            # Rounding disclaimer
            st.caption("**Note:** Small differences (a few cents) may occur because calculations use full precision while displayed prices are rounded to 3 decimals. See Help for details.")
    
    
    # SMART PRICE ALERT
    # =========================================================================
    
    # Use cached data access for performance
    history_df = _cached_get_station_price_history(station_uuid, fuel_code, days=14)
    hourly_df = _cached_calculate_hourly_stats(station_uuid, fuel_code, days=14)
    
    if hourly_df is not None and not hourly_df.empty:
        now = datetime.now()
        current_hour = now.hour
        alert = check_smart_price_alert(hourly_df, current_hour)
        
        if alert:
            hours_wait = alert["hours_to_wait"]
            drop_hour = alert["drop_hour"]
            price_drop = alert["price_drop"]
            
            st.warning(
                f"Price usually drops at {drop_hour}:00 (-€{price_drop:.3f}/L). "
                f"Worth waiting {hours_wait} hour{'s' if hours_wait > 1 else ''}?"
            )
    
    # =========================================================================
    # BEST TIME TO REFUEL (WITH GRIDLINES)
    # =========================================================================
    
    st.markdown("### Best Time to Refuel")
    st.caption("Based on 14-day price patterns • 🟢 Cheapest 🟡 Average 🔴 Expensive")
    
    if hourly_df is not None and not hourly_df.empty:
        # Data quality thresholds
        MIN_RECORDS_FOR_PATTERN = 20  # Need at least 20 price changes for reliable patterns
        MIN_HOURS_FOR_PATTERN = 4     # Need data spread across at least 4 different hours
        
        # Check data quality using ORIGINAL history data, not forward-filled hourly stats
        # hourly_df is forward-filled so ALL hours have data - misleading for quality check
        if history_df is not None and not history_df.empty:
            original_records = len(history_df)
            original_hours_with_data = history_df["date"].dt.hour.nunique() if "date" in history_df.columns else 0
        else:
            original_records = 0
            original_hours_with_data = 0
        
        is_sparse_data = (original_records < MIN_RECORDS_FOR_PATTERN or original_hours_with_data < MIN_HOURS_FOR_PATTERN)
        
        if is_sparse_data:
            # Don't show misleading chart - just explain why
            st.info(
                f"**Not enough data for reliable patterns.** "
                f"This station rarely reports price changes. "
                f"Hourly patterns require more frequent updates to be meaningful."
            )
        else:
            # Enough data - show the chart
            fig = create_hourly_pattern_chart(hourly_df, fuel_code)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Show "Usually cheapest" 
            hourly_df_displayed = hourly_df[hourly_df["hour"] % 2 == 0].copy()
            optimal = get_cheapest_and_most_expensive_hours(hourly_df_displayed)
            cheapest_hour = optimal.get("cheapest_hour")
            
            if cheapest_hour is not None:
                st.success(f"Usually cheapest: {int(cheapest_hour)}:00")
        
        # Weekday × Hour Heatmap (below hourly chart)
        st.markdown("#### Price by Day & Hour")
        if history_df is not None and len(history_df) > 0:
            heatmap_fig = create_weekday_hour_heatmap(history_df, fuel_code)
            if heatmap_fig is not None:
                st.plotly_chart(heatmap_fig, use_container_width=True, config=PLOTLY_CONFIG)
                st.caption("White cells = no data for that day/hour in the last 14 days. Tankerkönig only records price changes, so quiet periods may have gaps.")
            else:
                st.caption(f"Not enough variety in data for weekday patterns.")
        else:
            st.caption("No historical data available for heatmap.")
    else:
        st.info("Not enough historical data to show hourly patterns.")
    
    
    # =========================================================================
    # PRICE TREND (7 DAYS - NO DOTS)
    # =========================================================================
    
    st.markdown("### Price Trend (7 days)")
    history_df_7d = _cached_get_station_price_history(station_uuid, fuel_code, days=7)
    if history_df_7d is not None and not history_df_7d.empty:
        fig, stats_text = create_price_trend_chart(history_df_7d, fuel_code)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(stats_text)
    else:
        st.info("No historical data available.")
    
    # =========================================================================
    # COMPARE STATIONS
    # =========================================================================
    
    # Detect if current station came from Explorer
    # Check BOTH session state AND radio widget state (radio is more up-to-date)
    station_source = st.session_state.get("selected_station_source", "")
    radio_key = "station_details_source_radio"
    if radio_key in st.session_state:
        radio_label = st.session_state.get(radio_key, "")
        if "explorer" in radio_label.lower():
            station_source = "explorer"
        elif "route" in radio_label.lower():
            station_source = "route"
    
    is_from_explorer = (station_source == "explorer")
    
    st.markdown("### Compare Stations")
    # In Explorer mode, skip the automatic station boxes since there's no ranking
    # Go directly to Historical Price Comparison
    if is_from_explorer:
        compare_list = []  # Empty - no automatic comparison for Explorer
        show_ranking = False
    elif ranked:
        st.caption("Top ranked stations from your route")
        compare_list = ranked
        show_ranking = True
    elif stations:
        st.caption("Stations along your route")
        compare_list = stations
        show_ranking = False
    else:
        compare_list = []
        show_ranking = False
    
    # Only show comparison boxes for Trip Planner mode (not Explorer)
    if not is_from_explorer and len(compare_list) >= 1 and display_price is not None:
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
                eta_display = f" • ETA: {eta_dt.strftime('%H:%M')}"
            except Exception:
                pass
        
        # Detour info (only relevant for Trip Planner mode)
        if is_from_explorer:
            detour_text = ""
        else:
            detour_text = "On route" if detour_min < 0.5 else f"+{detour_min:.0f} min detour"
        
        # Add city to current station name for disambiguation
        current_display_name = name
        if city:
            current_display_name += f" · {city}"
        
        # Build info line for current station
        current_info_parts = []
        if detour_text:
            current_info_parts.append(detour_text)
        if eta_display:
            current_info_parts.append(eta_display.lstrip(" • "))
        current_info_line = " • ".join(current_info_parts) if current_info_parts else ""
        
        st.markdown(f"""
        <div style='background: #e0f2fe; border-left: 4px solid #0284c7; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem;'>
            <div style='font-weight: 700; font-size: 1.1rem; color: #0c4a6e;'>{current_display_name} (Current)</div>
            <div style='margin-top: 0.5rem; color: #075985;'>
                <span style='font-size: 1.3rem; font-weight: 700;'>€{display_price:.3f}</span> / L
            </div>
            {f"<div style='margin-top: 0.3rem; color: #0369a1; font-size: 0.9rem;'>{current_info_line}</div>" if current_info_line else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # Show other stations from the list (up to 3)
        shown_count = 0
        for idx, s in enumerate(compare_list, start=1):
            if shown_count >= 3:
                break
                
            s_uuid = _station_uuid(s)
            if s_uuid == station_uuid:
                continue  # Skip current station
            
            s_name = _station_name(s)
            s_city = _station_city(s)
            
            # For Explorer stations, try current price; for Trip stations, use predicted
            if is_from_explorer:
                s_price = _safe_float(s.get(f"price_current_{fuel_code}")) or _safe_float(s.get(pred_key))
            else:
                s_price = _safe_float(s.get(pred_key))
            
            s_detour = _safe_float(s.get("detour_duration_min"))
            
            # Build display name with city for disambiguation
            s_display_name = s_name
            if s_city:
                s_display_name += f" · {s_city}"
            
            # Format ETA for this station (only for Trip Planner)
            s_eta_display = ""
            if not is_from_explorer:
                s_eta_str = s.get("eta")
                if s_eta_str:
                    try:
                        if isinstance(s_eta_str, str):
                            s_eta_dt = datetime.fromisoformat(s_eta_str.replace("Z", "+00:00"))
                        else:
                            s_eta_dt = s_eta_str
                        s_eta_display = f" • ETA: {s_eta_dt.strftime('%H:%M')}"
                    except Exception:
                        pass
            
            # Detour info (only for Trip Planner)
            if is_from_explorer:
                s_detour_text = ""
            else:
                s_detour_text = "On route" if (s_detour or 0) < 0.5 else f"+{s_detour:.0f} min detour"
            
            # Build info line
            s_info_parts = []
            if s_detour_text:
                s_info_parts.append(s_detour_text)
            if s_eta_display:
                s_info_parts.append(s_eta_display.lstrip(" • "))
            s_info_line = " • ".join(s_info_parts) if s_info_parts else ""
            
            # Display name with or without ranking
            if show_ranking:
                card_title = f"#{idx} {s_display_name}"
            else:
                card_title = s_display_name
            
            if s_price:
                st.markdown(f"""
                <div style='background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem;'>
                    <div style='font-weight: 600; font-size: 1rem; color: #374151;'>{card_title}</div>
                    <div style='margin-top: 0.5rem; color: #1f2937;'>
                        <span style='font-size: 1.2rem; font-weight: 700;'>€{s_price:.3f}</span> / L
                    </div>
                    {f"<div style='margin-top: 0.3rem; color: #6b7280; font-size: 0.9rem;'>{s_info_line}</div>" if s_info_line else ""}
                </div>
                """, unsafe_allow_html=True)
                shown_count += 1
                
    elif not is_from_explorer and len(compare_list) < 1:
        # Only show this message for Trip Planner mode - Explorer already has its caption
        st.info("Run Trip Planner to see station comparison.")
    
    # Historical Comparison Chart
    # Only show divider if we had content above (Trip Planner mode with boxes)
    if not is_from_explorer:
        pass
    
    st.markdown("#### Historical Price Comparison (7 Days)")
    
    comp_uuids = st.session_state.get("comparison_station_uuids")
    if not isinstance(comp_uuids, list):
        comp_uuids = []
    
    if not station_uuid:
        st.caption("Station UUID missing.")
    elif not comp_uuids:
        st.caption("Select comparison stations in the sidebar to see price history chart.")
    else:
        with st.spinner("Loading comparison data..."):
            stations_data: Dict[str, pd.DataFrame] = {}
            
            # Build current station label with city for consistency
            current_label = name
            if city:
                current_label += f" · {city}"
            
            # Use cached data access
            current_df = _cached_get_station_price_history(station_uuid, fuel_code, days=7)
            if current_df is not None and not current_df.empty:
                stations_data[f"{current_label} (Current)"] = current_df
            
            # Map UUIDs to readable labels using _station_name
            # Search ranked, stations, AND explorer_results lists to find all station names
            label_by_uuid: Dict[str, str] = {}
            
            # First check ranked list
            for i, s in enumerate(ranked, start=1):
                u = _station_uuid(s)
                if not u:
                    continue
                s_name = _station_name(s)
                s_brand = _station_brand(s)
                s_city = _station_city(s)
                label = f"#{i} {s_name}"
                if s_brand and s_brand != s_name:
                    label += f" ({s_brand})"
                if s_city:
                    label += f" · {s_city}"
                label_by_uuid[u] = label
            
            # Also check stations list (might have different stations)
            for i, s in enumerate(stations, start=1):
                u = _station_uuid(s)
                if not u or u in label_by_uuid:  # Don't overwrite if already found
                    continue
                s_name = _station_name(s)
                s_brand = _station_brand(s)
                s_city = _station_city(s)
                label = f"#{i} {s_name}"
                if s_brand and s_brand != s_name:
                    label += f" ({s_brand})"
                if s_city:
                    label += f" · {s_city}"
                label_by_uuid[u] = label
            
            # Also check explorer_results (for Explorer mode comparisons)
            for s in explorer_results:
                u = _station_uuid(s)
                if not u or u in label_by_uuid:  # Don't overwrite if already found
                    continue
                s_name = _station_name(s)
                s_brand = _station_brand(s)
                s_city = _station_city(s)
                label = s_name
                if s_brand and s_brand != s_name:
                    label += f" ({s_brand})"
                if s_city:
                    label += f" · {s_city}"
                label_by_uuid[u] = label
            
            for u in comp_uuids[:1]:  # Only 1 comparison for cleaner mobile view
                if not u or u == station_uuid:
                    continue
                # Use cached data access
                df = _cached_get_station_price_history(u, fuel_code, days=7)
                if df is not None and not df.empty:
                    label = label_by_uuid.get(u, f"Station {u[:8]}...")
                    stations_data[label] = df
        
        if len(stations_data) < 2:
            st.caption("Not enough historical data to compare.")
        else:
            fig, current_full_name, comparison_full_name = create_comparison_chart(
                stations_data, fuel_code, current_station_name=name
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Show full station names below chart
            st.markdown(
                f"""<div style="font-size: 0.85rem; color: #555; margin-top: -0.5rem;">
                <span style="color: #16a34a;">●</span> <b>Your station:</b> {current_full_name}<br>
                <span style="color: #2563eb;">●</span> <b>Comparison:</b> {comparison_full_name}
                </div>""",
                unsafe_allow_html=True
            )
            
            # Quick insight: average price difference over the period
            try:
                current_avg = stations_data.get(f"{current_label} (Current)", pd.DataFrame())["price"].mean()
                if pd.notna(current_avg):
                    # Get comparison station name and average
                    for label, df in stations_data.items():
                        if "(Current)" in label or df.empty:
                            continue
                        alt_avg = df["price"].mean()
                        if pd.notna(alt_avg):
                            diff = float(alt_avg) - float(current_avg)
                            
                            if diff > 0:
                                st.success(f"Your station is **€{diff:.3f}/L cheaper** on average")
                            elif diff < 0:
                                st.warning(f"Your station is **€{abs(diff):.3f}/L more expensive** on average")
                            else:
                                st.info("Prices match on average")
                            
                            # Show calculation breakdown
                            st.caption(f"Based on 7-day history: Your avg €{float(current_avg):.3f}/L vs Comparison €{float(alt_avg):.3f}/L")
                            break  # Only 1 comparison
            except Exception:
                pass
    
    maybe_persist_state()


if __name__ == "__main__":
    main()