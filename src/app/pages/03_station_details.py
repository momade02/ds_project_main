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
    "scrollZoom": False,  # Disable scroll zoom
    "doubleClick": False,  # Disable double-click zoom
}

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

from ui.sidebar import render_sidebar_shell, render_station_selector, render_comparison_selector, _render_help_action

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
                    except:
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
            height=300,
        )
        return fig, ""

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
        hovertemplate="%{x|%b %d}<br>€%{y:.3f}/L<extra></extra>",
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
        xaxis=dict(fixedrange=True),
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

    # Calculate sensible y-axis range (not starting from 0)
    min_val = hourly_df_mobile["avg_price"].min()
    max_val = hourly_df_mobile["avg_price"].max()
    y_padding = (max_val - min_val) * 0.15 if max_val > min_val else 0.05
    y_min = max(0, min_val - y_padding)
    y_max = max_val + y_padding

    fig.update_layout(
        title="",  # Empty - we have markdown heading
        xaxis_title="Hour",
        yaxis_title=None,  # Remove rotated label - use annotation
        height=250,  # More compact
        margin=dict(l=35, r=10, t=20, b=35),  # Less left margin
        xaxis=dict(fixedrange=True),
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
    """
    if df is None or df.empty:
        return None
    
    # Ensure we have datetime and can extract weekday/hour
    if "date" not in df.columns or "price" not in df.columns:
        return None
    
    try:
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
        
        # Weekday labels - only for weekdays we have data for
        weekday_labels_full = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekdays_present = sorted(heatmap_data.index.tolist())
        weekday_labels = [weekday_labels_full[i] for i in weekdays_present]
        
        # Get actual hours present in data
        hours_present = sorted(heatmap_data.columns.tolist())
        
        # Create heatmap with only the data we have
        z_data = heatmap_data.loc[weekdays_present, hours_present].values
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=hours_present,
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
    
    for idx, (station_name, df) in enumerate(stations_data.items()):
        if df is None or df.empty:
            continue

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
        height=320,  # Slightly taller to accommodate legend
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
        xaxis=dict(fixedrange=True),
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
    Replaces "Placeholder: Action" with actual trip info.
    Also renders station selectors (only in Action tab for performance).
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
    
    ranked = list(last_run.get("ranked") or [])
    stations = list(last_run.get("stations") or [])
    route_station_count = len(stations)
    
    explorer_results = list(st.session_state.get("explorer_results") or [])
    explorer_station_count = len(explorer_results)
    
    # Header with explanation
    st.sidebar.markdown(
        "**Current Parameters** (?)",
        help="These settings come from your last Trip Planner run on the Home page. To change them, go back to Home and run a new trip."
    )
    
    # Display settings
    st.sidebar.markdown("**Fuel:** " + fuel_label)
    st.sidebar.markdown("**Economics:** " + econ_status)
    st.sidebar.markdown("**Refuel amount:** " + refuel_amount)
    st.sidebar.markdown("**Route stations:** " + str(route_station_count))
    st.sidebar.markdown("**Explorer stations:** " + str(explorer_station_count))
    
    st.sidebar.markdown("---")
    
    # Station selectors with help text
    st.sidebar.markdown(
        "**Select Station** (?)",
        help="Choose a station to analyze. 'Ranked' stations passed all filters and are recommended. 'Not Ranked' stations were excluded due to detour limits, being closed, or other filters."
    )
    
    # Station selectors (inside Action tab for faster sidebar switching)
    render_station_selector(
        last_run=last_run,
        explorer_results=explorer_results,
        max_ranked=200,      # Show all ranked stations (not just top 20)
        max_excluded=200,    # Show all excluded stations too
    )
    
    # Comparison selector
    current_uuid = str(st.session_state.get("selected_station_uuid") or "")
    render_comparison_selector(ranked, stations, current_uuid, max_ranked=50, max_excluded=100)


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
    # SIDEBAR - WITH TRIP SETTINGS IN ACTION TAB
    # =========================================================================
    render_sidebar_shell(action_renderer=_render_trip_settings, help_renderer=_render_help_action)
    
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
    
    # Check if this station is in the ranked list (i.e., passed all filters)
    is_ranked = any(_station_uuid(s) == station_uuid for s in ranked)
    
    # Get thresholds from last_run for accurate exclusion detection
    # on-route thresholds: define what counts as "on-route" for baseline price (e.g., 1 km, 5 min)
    onroute_km_threshold, onroute_min_threshold = _get_onroute_thresholds(last_run)
    # exclusion thresholds: the user's hard caps that determine exclusion (e.g., 5 km, 10 min)
    max_detour_km_cap, max_detour_min_cap = _get_exclusion_thresholds(last_run)
    
    # Get ETA for exclusion checks
    eta_str_for_check = station.get("eta")
    
    # Get exclusion reason if not ranked
    exclusion_reason = None
    if not is_ranked:
        # Check conditions in order of likelihood
        
        # 1. Check detour distance (most common exclusion reason)
        # Use max_detour_km_cap (user's setting) NOT onroute_km_threshold
        if detour_km is not None and max_detour_km_cap is not None and detour_km > max_detour_km_cap:
            exclusion_reason = f"Detour exceeds limit ({detour_km:.1f} km > {max_detour_km_cap:.1f} km max)"
        
        # 2. Check detour time
        # Use max_detour_min_cap (user's setting) NOT onroute_min_threshold
        elif detour_min is not None and max_detour_min_cap is not None and detour_min > max_detour_min_cap:
            exclusion_reason = f"Detour time exceeds limit ({detour_min:.0f} min > {max_detour_min_cap:.0f} min max)"
        
        # 3. Check if station is closed at ETA
        elif eta_str_for_check:
            try:
                eta_dt = datetime.fromisoformat(eta_str_for_check.replace('Z', '+00:00')) if isinstance(eta_str_for_check, str) else eta_str_for_check
                is_open_check, _ = _parse_opening_hours(station, eta_dt)
                if not is_open_check:
                    exclusion_reason = f"Station closed at arrival time ({eta_dt.strftime('%H:%M') if eta_dt else 'unknown'})"
            except:
                pass
        
        # 4. Check if no price prediction
        if exclusion_reason is None and predicted_price is None:
            exclusion_reason = "No price prediction available for this station"
        
        # 5. Check if net savings too low (when economics enabled)
        if exclusion_reason is None and use_economics:
            if net_saving is not None and net_saving < 0:
                exclusion_reason = f"Negative net savings (€{net_saving:.2f}) - detour cost exceeds price benefit"
            elif net_saving is None:
                exclusion_reason = "Could not calculate savings (missing data)"
        
        # 6. Generic fallback with hint
        if exclusion_reason is None:
            exclusion_reason = "Did not pass ranking filters (check detour/time thresholds in settings)"
    
    # Traffic light status
    if not is_ranked:
        # Station is excluded - show neutral gray indicator
        traffic_status = "gray"
        traffic_text = "Not Ranked"
    elif use_economics and net_saving is not None:
        # Ranked station with economics - use net savings ranking
        traffic_status, traffic_text_raw, traffic_css = calculate_traffic_light_status(
            display_price,
            ranked,
            fuel_code,
            station_net_saving=net_saving,
            use_net_savings=True,
        )
        # Set economics-based text
        if traffic_status == "red":
            traffic_text = "Poor Value"
        elif traffic_status == "yellow":
            traffic_text = "Fair Value"
        else:
            traffic_text = "Excellent Value"
    else:
        # Ranked station without economics - use price ranking
        traffic_status, traffic_text_raw, traffic_css = calculate_traffic_light_status(
            display_price,
            ranked,
            fuel_code,
            station_net_saving=None,
            use_net_savings=False,
        )
        # Set price-based text
        if traffic_status == "red":
            traffic_text = "Expensive"
        elif traffic_status == "yellow":
            traffic_text = "Fair Price"
        else:
            traffic_text = "Excellent Deal"
    
    # Extract ETA for opening hours check
    eta_datetime = None
    eta_str = station.get("eta")
    if eta_str:
        try:
            if isinstance(eta_str, str):
                eta_datetime = datetime.fromisoformat(eta_str.replace('Z', '+00:00'))
            else:
                eta_datetime = eta_str
        except:
            pass
    
    is_open, time_info = _parse_opening_hours(station, eta_datetime)
    
    # =========================================================================
    # HERO SECTION - CYAN BOX WITH PRICE + TRAFFIC LIGHT
    # =========================================================================
    
    base_name = brand if (brand and brand != "—") else name
    city_clean = city.title() if (city and city != "—") else ""
    title_line = f"{base_name} in {city_clean}" if (base_name and city_clean) else base_name
    
    # Build subtitle with address - prioritize full address, then construct from available data
    if full_address:
        subtitle_line = full_address
    else:
        # No full address - construct from available fields
        # The name field often contains street info (e.g., "TUEBINGEN EUROPASTR. 5")
        street = station.get("street", "")
        house_number = station.get("house_number", "")
        post_code = station.get("post_code", "")
        
        if street:
            # Have street data - build address manually (use title case for readability)
            street_title = street.title() if street else ""
            street_part = f"{street_title} {house_number}".strip() if house_number else street_title
            if post_code and city_clean:
                subtitle_line = f"{street_part}, {post_code} {city_clean}"
            elif city_clean:
                subtitle_line = f"{street_part}, {city_clean}"
            else:
                subtitle_line = street_part
        elif name and name != base_name:
            # Use name field if it contains additional info (often has street)
            # Convert to title case for readability
            name_title = name.title() if name else ""
            if city_clean and city_clean.upper() not in name.upper():
                subtitle_line = f"{name_title}, {city_clean}"
            else:
                subtitle_line = name_title
        elif city_clean:
            subtitle_line = city_clean
        else:
            subtitle_line = ""
    
    # Traffic light circles
    if traffic_status == "green":
        circles = '<span style="color: #4ade80; font-size: 1.6rem;">●</span> <span style="color: #fbbf24; font-size: 1.6rem;">○</span> <span style="color: #f87171; font-size: 1.6rem;">○</span>'
    elif traffic_status == "yellow":
        circles = '<span style="color: #4ade80; font-size: 1.6rem;">○</span> <span style="color: #fbbf24; font-size: 1.6rem;">●</span> <span style="color: #f87171; font-size: 1.6rem;">○</span>'
    elif traffic_status == "red":
        circles = '<span style="color: #4ade80; font-size: 1.6rem;">○</span> <span style="color: #fbbf24; font-size: 1.6rem;">○</span> <span style="color: #f87171; font-size: 1.6rem;">●</span>'
    else:
        # Gray/excluded - show all circles as gray/empty
        circles = '<span style="color: #9ca3af; font-size: 1.6rem;">○</span> <span style="color: #9ca3af; font-size: 1.6rem;">○</span> <span style="color: #9ca3af; font-size: 1.6rem;">○</span>'
    
    price_display = f"€{display_price:.3f}" if display_price else "—"
    
    # Build tooltip based on station status
    if not is_ranked:
        # Excluded station
        reason_text = exclusion_reason if exclusion_reason else "Station did not meet ranking criteria"
        traffic_tooltip = (
            f"⚪ This station is not ranked&#10;&#10;"
            f"Reason: {reason_text}&#10;&#10;"
            "Only ranked stations are compared for value.&#10;"
            "Check the ranked stations list for recommendations."
        )
    elif use_economics and net_saving is not None:
        traffic_tooltip = (
            "🟢 Excellent Value = Top 33% net savings (price minus detour costs)&#10;"
            "🟡 Fair Value = Middle 33%&#10;"
            "🔴 Poor Value = Bottom 33%&#10;&#10;"
            "Compared against all ranked stations on your route.&#10;"
            "Accounts for both price and detour time/fuel costs."
        )
    else:
        traffic_tooltip = (
            "🟢 Excellent Deal = Cheapest 33% of stations&#10;"
            "🟡 Fair Price = Middle 33%&#10;"
            "🔴 Expensive = Most expensive 33%&#10;&#10;"
            "Based on predicted prices. Enable economics for value-based ranking."
        )
    
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
                <span style='margin-left: 0.3rem; color: #9ca3af; cursor: help; font-size: 0.9rem;' title='{traffic_tooltip}'>(?)</span>
            </div>
        </div>
    </div>
    """
    
    st.markdown(hero_html, unsafe_allow_html=True)
    
    # =========================================================================
    # MISSING PREDICTION WARNING (if using current price as fallback)
    # =========================================================================
    if using_current_as_fallback:
        st.warning(
            "⚠️ **Using current price** — No prediction available for this station at your arrival time. "
            "The displayed price is the current real-time price, not a forecast.",
            icon="⚠️"
        )
        with st.expander("Why is the prediction missing?", expanded=False):
            st.markdown("""
            **Possible reasons:**
            - The station may have limited historical price data
            - The model couldn't generate a reliable forecast for this specific time
            - Data collection gap for this station
            
            **What this means:**
            - The current price shown may differ from the actual price when you arrive
            - Savings calculations are based on the current price, not a predicted one
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
                except:
                    today = datetime.now()
                
                if eta_dt.date() != today.date():
                    # Show day + time for next-day arrivals (e.g., "Mon 14:33")
                    eta_display = eta_dt.strftime("%a %H:%M")
                else:
                    # Same day - just show time
                    eta_display = eta_dt.strftime("%H:%M")
                
                st.info(f"ETA: {eta_display}")
            except Exception:
                st.info("ETA: —")
        else:
            st.info("ETA: —")

    with info_cols[2]:
        # Opening hours - show day if ETA is different from today
        day_prefix = ""
        if eta_datetime:
            try:
                today = datetime.now(tz=ZoneInfo("Europe/Berlin") if ZoneInfo else None)
            except:
                today = datetime.now()
            
            if eta_datetime.date() != today.date():
                day_prefix = f"{eta_datetime.strftime('%a')} "  # e.g., "Mon "
        
        # Handle different opening hours states
        if time_info == "UNKNOWN":
            # No data from Google or Tankerkönig - show as unknown
            st.info("Opening hours: Unknown (assumed 24/7)")
        elif is_open and time_info:
            # Open with known closing time
            st.success(f"Opening hours: {day_prefix}OPEN until {time_info}")
        elif not is_open and time_info:
            # Closed with known opening time
            st.error(f"Opening hours: {day_prefix}CLOSED, opens at {time_info}")
        elif is_open:
            # Open but no closing time known (likely 24/7)
            st.success(f"Opening hours: {day_prefix}OPEN (24/7)")
        else:
            # Closed but no opening time known
            st.error(f"Opening hours: {day_prefix}CLOSED")
    
    st.markdown("---")
    
    # =========================================================================
    # SAVINGS CALCULATOR
    # =========================================================================
    
    # Get on-route thresholds (aligned with page 02 and recommender)
    onroute_km_threshold, onroute_min_threshold = _get_onroute_thresholds(last_run)
    
    st.markdown("### Savings Calculator")
    st.caption("Compare against the worst-priced station directly on your route (no detour needed)")
    
    # Initialize variable that will be used later
    total_for_context = 0
    
    if not display_price:
        st.info("No price data available for this station.")
    else:
        slider_value = st.slider(
            "Refuel amount (liters)",
            min_value=1.0,
            max_value=1000.0,
            value=float(litres_to_refuel),
            step=1.0,
            key="savings_calc_slider",
        )
        
        # Find the WORST on-route price for user-friendly display
        # Uses consistent thresholds from recommender (aligned with page 02)
        worst_onroute_computed = _compute_worst_onroute_price(
            ranked, fuel_code, onroute_km_threshold, onroute_min_threshold
        )
        worst_onroute_price = worst_onroute_computed if worst_onroute_computed is not None else display_price
        
        # Calculate display metrics using WORST price
        price_diff_display = worst_onroute_price - display_price
        gross_savings_display = price_diff_display * slider_value
        
        # ECONOMICS MODE: Check user setting, not whether recommender calculated values
        # For non-ranked stations, we calculate detour costs manually
        if use_economics:
            # Try to get pre-computed detour costs from station data first
            detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
            time_cost_key = f"econ_time_cost_eur_{fuel_code}"
            
            detour_fuel_cost = _safe_float(station.get(detour_fuel_cost_key))
            time_cost = _safe_float(station.get(time_cost_key))
            
            # If not available (non-ranked station), calculate manually
            if detour_fuel_cost is None or time_cost is None:
                # Get parameters from last_run or session_state
                consumption = _safe_float(last_run.get("consumption_l_per_100km")) or _safe_float(st.session_state.get("consumption_l_per_100km")) or 7.0
                value_of_time = _safe_float(last_run.get("value_of_time_eur_per_hour")) or _safe_float(st.session_state.get("value_of_time_eur_per_hour")) or 0.0
                
                # Calculate detour fuel cost: (detour_km * consumption / 100) * fuel_price
                if detour_km > 0 and display_price:
                    detour_fuel_cost = (detour_km * consumption / 100.0) * display_price
                else:
                    detour_fuel_cost = 0.0
                
                # Calculate time cost: (detour_min / 60) * value_of_time
                if detour_min > 0 and value_of_time > 0:
                    time_cost = (detour_min / 60.0) * value_of_time
                else:
                    time_cost = 0.0
            
            detour_costs_constant = (detour_fuel_cost or 0.0) + (time_cost or 0.0)
            
            # Calculate gross and net using WORST on-route price
            # This matches the Analytics page calculation (02_route_analytics.py)
            gross_savings_display = price_diff_display * slider_value
            net_savings_display = gross_savings_display - detour_costs_constant
            
            # Calculate break-even
            breakeven_liters = None
            if price_diff_display > 0 and detour_costs_constant > 0:
                breakeven_liters = detour_costs_constant / price_diff_display
            
            # Price comparison header
            st.markdown("**Comparing prices:**")
            comp_cols = st.columns(2)
            with comp_cols[0]:
                st.markdown(f"🟢 **This station:** €{display_price:.3f}/L")
            with comp_cols[1]:
                st.markdown(f"🔴 **Worst on-route:** €{worst_onroute_price:.3f}/L")
            
            st.markdown("---")
            
            # NEGATIVE SAVINGS - Red warning with break-even
            if net_savings_display < -0.01:
                # Show numbers FIRST with tooltips
                calc_cols = st.columns(2)
                with calc_cols[0]:
                    st.metric(
                        "Gross Savings",
                        f"€{gross_savings_display:.2f}",
                        help=(
                            "Price advantage before detour costs. "
                            "Formula: (Worst on-route - This station) × Litres. "
                            f"Example: (€{worst_onroute_price:.3f} - €{display_price:.3f}) × {slider_value:.0f}L = €{gross_savings_display:.2f}"
                        )
                    )
                with calc_cols[1]:
                    st.metric(
                        "Detour Costs",
                        f"€{detour_costs_constant:.2f}",
                        help=(
                            "Extra fuel for detour plus optional time cost. "
                            "Formula: (Detour km × Consumption/100) × Price + Time cost. "
                            f"Your detour: {detour_km:.1f}km, {detour_min:.0f} minutes. "
                            f"Breakdown: Fuel €{detour_fuel_cost:.2f} + Time €{time_cost:.2f}"
                        )
                    )
                
                # Then show evaluation
                st.error(f"**Net Loss: €{abs(net_savings_display):.2f}**")
                
                # Check if prices are equal (no price advantage)
                if abs(price_diff_display) < 0.001:
                    st.caption("**This station has the same price as the worst on-route option.** There's no price advantage, but the detour still costs fuel.")
                else:
                    st.caption(f"Detour costs (€{detour_costs_constant:.2f}) exceed price advantage (€{gross_savings_display:.2f})")
                
                if breakeven_liters:
                    st.info(f"**Break-even:** Refuel at least **{breakeven_liters:.0f}L** to save money")
            
            # NEAR ZERO - Break-even
            elif abs(net_savings_display) <= 0.01:
                st.info("Detour costs cancel out price savings. No real advantage.")
                
            # POSITIVE SAVINGS - Green success
            else:
                calc_cols = st.columns(2)
                
                with calc_cols[0]:
                    st.metric(
                        "Net Savings",
                        f"€{net_savings_display:.2f}",
                        help=(
                            "Real savings after all costs. "
                            "Formula: Gross savings - Detour costs. "
                            f"Example: €{gross_savings_display:.2f} - €{detour_costs_constant:.2f} = €{net_savings_display:.2f}"
                        )
                    )
                    st.caption(f"for {slider_value:.0f} liters")
                
                with calc_cols[1]:
                    st.metric(
                        "Price Difference",
                        f"€{price_diff_display:.3f}/L",
                        help=(
                            "How much cheaper per liter vs worst on-route. "
                            "Formula: Worst - This station. "
                            f"Example: €{worst_onroute_price:.3f} - €{display_price:.3f} = €{price_diff_display:.3f}/L"
                        )
                    )
                
                # Show breakdown
                if detour_costs_constant > 0.01:
                    st.success(f"Gross: €{gross_savings_display:.2f} - Detour: €{detour_costs_constant:.2f} = Net: €{net_savings_display:.2f}")
            
            # Set for weekly/monthly context
            total_for_context = net_savings_display
        
        else:
            # NO ECONOMICS - Simple comparison with warning
            # Use same logic: find worst on-route price (using consistent thresholds)
            worst_price = display_price
            for s in ranked:
                s_price = _safe_float(s.get(pred_key))
                if s_price and s_price > worst_price:
                    d_km = _safe_float(s.get("detour_distance_km") or s.get("detour_km") or 0)
                    d_min = _safe_float(s.get("detour_duration_min") or 0)
                    # Use consistent thresholds (not hardcoded)
                    if d_km <= onroute_km_threshold and d_min <= onroute_min_threshold:
                        worst_price = s_price
            
            price_diff = worst_price - display_price
            total_savings = price_diff * slider_value
            
            st.markdown("**Comparing prices:**")
            comp_cols = st.columns(2)
            with comp_cols[0]:
                st.markdown(f"🟢 **This station:** €{display_price:.3f}/L")
            with comp_cols[1]:
                st.markdown(f"🔴 **Worst on-route:** €{worst_price:.3f}/L")
            
            st.markdown("---")
            
            calc_cols = st.columns(2)
            
            with calc_cols[0]:
                st.metric(
                    "Total Savings",
                    f"€{total_savings:.2f}",
                    help=(
                        "Price difference without detour costs. "
                        f"Formula: Price diff × Litres = €{price_diff:.3f} × {slider_value:.0f}L = €{total_savings:.2f}"
                    )
                )
                st.caption(f"for {slider_value:.0f} liters")
            
            with calc_cols[1]:
                st.metric(
                    "Price Difference",
                    f"€{price_diff:.3f}/L",
                    help=(
                        "Cheaper per liter vs worst on-route. "
                        f"Example: €{worst_price:.3f} - €{display_price:.3f} = €{price_diff:.3f}/L"
                    )
                )
            
            st.warning("Detour costs not included. Enable Economics in Trip Planner for accurate net savings.")
            
            # Set for weekly/monthly context
            total_for_context = total_savings
        
        # Weekly/monthly/yearly projection (only for meaningful positive savings)
        if total_for_context > 0.01:  # Show for any positive savings
            weekly = total_for_context
            monthly = total_for_context * 4.3
            yearly = total_for_context * 52
            st.info(
                f"If you refuel here weekly: **€{weekly:.2f}** per week • "
                f"**€{monthly:.2f}** per month • **€{yearly:.2f}** per year"
            )
    
    st.markdown("---")
    
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
                f"Price usually drops at {drop_hour}:00 (−€{price_drop:.3f}/L). "
                f"Worth waiting {hours_wait} hour{'s' if hours_wait > 1 else ''}?"
            )
    
    # =========================================================================
    # BEST TIME TO REFUEL (WITH GRIDLINES)
    # =========================================================================
    
    st.markdown("### Best Time to Refuel")
    st.caption("Based on 14-day price patterns • 🟢 Cheapest 🟡 Average 🔴 Expensive")
    
    if hourly_df is not None and not hourly_df.empty:
        fig = create_hourly_pattern_chart(hourly_df, fuel_code)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        optimal = get_cheapest_and_most_expensive_hours(hourly_df)
        cheapest_hour = optimal.get("cheapest_hour")
        
        if cheapest_hour is not None:
            st.success(f"Usually cheapest: {cheapest_hour}:00")
        
        # Weekday × Hour Heatmap (below hourly chart)
        st.markdown("#### Price by Day & Hour")
        if history_df is not None and len(history_df) > 0:
            heatmap_fig = create_weekday_hour_heatmap(history_df, fuel_code)
            if heatmap_fig is not None:
                st.plotly_chart(heatmap_fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.caption(f"Not enough variety in data for weekday patterns.")
        else:
            st.caption("No historical data available for heatmap.")
    else:
        st.info("Not enough historical data to show hourly patterns.")
    
    st.markdown("---")
    
    # =========================================================================
    # PRICE TREND (7 DAYS - NO DOTS)
    # =========================================================================
    
    with st.expander("Price Trend (7 days)", expanded=False):
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
    
    with st.expander("Compare Stations", expanded=False):
        st.caption("Top ranked stations from your route")
        
        # Use ranked if available, otherwise fall back to stations
        compare_list = ranked if ranked else stations
        
        if len(compare_list) >= 1:
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
                except:
                    pass
            
            detour_text = "On route" if detour_min < 0.5 else f"+{detour_min:.0f} min detour"
            
            # Add city to current station name for disambiguation
            current_display_name = name
            if city:
                current_display_name += f" · {city}"
            
            st.markdown(f"""
            <div style='background: #e0f2fe; border-left: 4px solid #0284c7; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                <div style='font-weight: 700; font-size: 1.1rem; color: #0c4a6e;'>{current_display_name} (Current)</div>
                <div style='margin-top: 0.5rem; color: #075985;'>
                    <span style='font-size: 1.3rem; font-weight: 700;'>€{display_price:.3f}</span> / L
                </div>
                <div style='margin-top: 0.3rem; color: #0369a1; font-size: 0.9rem;'>
                    {detour_text}{eta_display}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 3 from ranking
            for idx, s in enumerate(compare_list[:3], start=1):
                s_uuid = _station_uuid(s)
                if s_uuid == station_uuid:
                    continue
                
                s_name = _station_name(s)
                s_city = _station_city(s)
                s_price = _safe_float(s.get(pred_key))
                s_detour = _safe_float(s.get("detour_duration_min"))
                
                # Build display name with city for disambiguation
                s_display_name = s_name
                if s_city:
                    s_display_name += f" · {s_city}"
                
                # Format ETA for this station
                s_eta_display = ""
                s_eta_str = s.get("eta")
                if s_eta_str:
                    try:
                        if isinstance(s_eta_str, str):
                            s_eta_dt = datetime.fromisoformat(s_eta_str.replace("Z", "+00:00"))
                        else:
                            s_eta_dt = s_eta_str
                        s_eta_display = f" • ETA: {s_eta_dt.strftime('%H:%M')}"
                    except:
                        pass
                
                s_detour_text = "On route" if (s_detour or 0) < 0.5 else f"+{s_detour:.0f} min detour"
                
                if s_price:
                    st.markdown(f"""
                    <div style='background: #f3f4f6; border-left: 4px solid #9ca3af; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.8rem;'>
                        <div style='font-weight: 600; font-size: 1rem; color: #374151;'>#{idx} {s_display_name}</div>
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
        
        # Historical Comparison Chart
        st.markdown("---")
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
                # Search both ranked AND stations lists to find all station names
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
                    <span style="color: #16a34a;">●</span> <b>Your station:</b> {current_full_name}<br>
                    <span style="color: #2563eb;">●</span> <b>Comparison:</b> {comparison_full_name}
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