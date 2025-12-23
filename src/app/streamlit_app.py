"""
Streamlit UI for the route-aware fuel price recommender.

Two environments
----------------
1) Test mode (example route, no Google calls)
   - Uses `run_example()` from `route_tankerkoenig_integration`.

2) Real route (Google route + Supabase + Tankerkönig pipeline)
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

from urllib.parse import quote

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

from functools import lru_cache

@lru_cache(maxsize=1)
def _load_env_once() -> None:
    """Load local .env once for local runs. No-op in deployments."""
    try:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

_load_env_once()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.route_tankerkoenig_integration import (
    run_example,
    get_fuel_prices_for_route,
)
from route_stations import environment_check, google_route_via_waypoint
from src.decision.recommender import (
    recommend_best_station,
    rank_stations_by_predicted_price,
    ONROUTE_MAX_DETOUR_KM,
    ONROUTE_MAX_DETOUR_MIN,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _calculate_zoom_for_bounds(lon_min: float, lon_max: float, lat_min: float, lat_max: float, 
                                padding_percent: float = 0.10, 
                                map_width_px: int = 700, map_height_px: int = 500) -> float:
    """
    Calculate optimal zoom level to fit bounds with padding using Web Mercator projection.
    
    Parameters
    ----------
    lon_min, lon_max, lat_min, lat_max : float
        Bounding box coordinates
    padding_percent : float
        Padding around bounds (0.05 = 5%)
    map_width_px, map_height_px : int
        Map container size in pixels
    
    Returns
    -------
    float
        Zoom level (1-15), clamped to valid range
    """
    try:
        # Add padding to bounds
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        
        # Handle edge case: point (no range)
        if lon_range < 0.0001 and lat_range < 0.0001:
            return 15.0  # Zoom in on single point
        
        # Handle edge case: line (only one dimension has range)
        if lon_range < 0.0001:
            lon_range = 0.1
        if lat_range < 0.0001:
            lat_range = 0.1
        
        lon_min -= lon_range * padding_percent / 2
        lon_max += lon_range * padding_percent / 2
        lat_min -= lat_range * padding_percent / 2
        lat_max += lat_range * padding_percent / 2
        
        # Clamp latitude to valid Web Mercator range (avoid poles)
        lat_min = max(-85.05, min(85.05, lat_min))
        lat_max = max(-85.05, min(85.05, lat_max))
        
        # Calculate zoom for longitude
        lon_delta = lon_max - lon_min
        if lon_delta <= 0:
            lon_delta = 0.1
        zoom_lon = math.log2(360 * map_width_px / (256 * lon_delta))
        
        # Calculate zoom for latitude using Web Mercator projection
        lat_min_rad = math.radians(lat_min)
        lat_max_rad = math.radians(lat_max)
        
        y_min = math.log(math.tan(math.pi / 4 + lat_max_rad / 2))
        y_max = math.log(math.tan(math.pi / 4 + lat_min_rad / 2))
        
        y_delta = y_max - y_min
        if abs(y_delta) < 0.0001:
            y_delta = 0.1
        
        zoom_lat = math.log2(math.pi * map_height_px / (256 * abs(y_delta)))
        
        # Take minimum zoom to fit both dimensions
        zoom = min(zoom_lon, zoom_lat)
        
        # Clamp to valid range
        return max(1, min(zoom, 15))
    
    except Exception as e:
        # Fallback to default zoom if calculation fails
        return 7.5

def _supports_pydeck_selections() -> bool:
    """Return True if the installed Streamlit supports pydeck selection events."""
    try:
        sig = inspect.signature(st.pydeck_chart)
        return "on_select" in sig.parameters and "selection_mode" in sig.parameters
    except Exception:
        return False


def _station_uuid(station: Dict[str, Any]) -> Optional[str]:
    """Return the station UUID used throughout the project (Tankerkönig UUID preferred)."""
    return station.get("tk_uuid") or station.get("station_uuid")


def _safe_text(value: Any) -> str:
    """HTML-escape user/external text for safe tooltip rendering."""
    if value is None:
        return ""
    return html.escape(str(value))


def _fmt_price(value: Any) -> str:
    try:
        if value is None:
            return "—"
        return f"{float(value):.3f} €/L"
    except Exception:
        return "—"


def _fmt_eur(value: Any) -> str:
    try:
        if value is None:
            return "—"
        return f"{float(value):.2f} €"
    except Exception:
        return "—"


def _fmt_km(value: Any) -> str:
    try:
        if value is None:
            return "—"
        return f"{float(value):.2f} km"
    except Exception:
        return "—"


def _fmt_min(value: Any) -> str:
    try:
        if value is None:
            return "—"
        return f"{float(value):.1f} min"
    except Exception:
        return "—"

def _create_map_visualization(
    route_coords: List[List[float]],
    stations: List[Dict[str, Any]],
    best_station_uuid: Optional[str] = None,
    via_full_coords: Optional[List[List[float]]] = None,
    zoom_level: float = 7.5,
    *,
    fuel_code: Optional[str] = None,
    selected_station_uuid: Optional[str] = None,
    map_style: Optional[str] = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    show_station_labels: bool = True,
) -> pdk.Deck:
    """
    Create a pydeck map showing:
      - baseline route (and optional via-station route),
      - start/end markers,
      - all stations (best station highlighted),
      - hover tooltip (compact),
      - selection support via Streamlit's pydeck selection API.

    Notes
    -----
    - Click-selection is handled by Streamlit (st.pydeck_chart on_select=...).
      This function provides a stable layer id ("stations") and station UUIDs.
    """

    # ------------------------------------------------------------------
    # Pin icon (SVG data URL) for IconLayer
    # NOTE: Must be URL-encoded, otherwise deck.gl often fails to load it,
    #       resulting in large placeholder circles.
    # ------------------------------------------------------------------
    def _pin_icon_data_url(fill_hex: str, stroke_hex: str) -> str:
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
            f"<path d='M32 2C21 2 12 11 12 22c0 17 20 40 20 40s20-23 20-40C52 11 43 2 32 2z' "
            f"fill='{fill_hex}' stroke='{stroke_hex}' stroke-width='2'/>"
            "<circle cx='32' cy='22' r='7' fill='white'/>"
            "</svg>"
        )
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"

    # ------------------------------------------------------------------
    # Marker sizing: pixel-based, scaled by zoom
    # ------------------------------------------------------------------
    base_size_other = 2.0
    base_size_best = 7.0
    base_size_selected = 9.0

    zoom_factor = max(0.6, min(1.6, zoom_level / 10.0))
    radius_other = base_size_other * zoom_factor
    radius_best = base_size_best * zoom_factor
    radius_selected = base_size_selected * zoom_factor

    pred_key = f"pred_price_{fuel_code}" if fuel_code else None
    curr_key = f"price_current_{fuel_code}" if fuel_code else None

    # ------------------------------------------------------------------
    # Build station data used by deck.gl (tooltip + selection object payload)
    # ------------------------------------------------------------------
    station_data: List[Dict[str, Any]] = []

    for s in stations:
        # 1) Robust coordinate parsing
        lat_raw, lon_raw = s.get("lat"), s.get("lon")
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except (TypeError, ValueError):
            continue

        # Reject NaN/inf and out-of-range coordinates
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue

        # 2) Identity + flags
        station_uuid = _station_uuid(s)
        is_best = bool(best_station_uuid and station_uuid and station_uuid == best_station_uuid)
        is_selected = bool(selected_station_uuid and station_uuid and station_uuid == selected_station_uuid)

        # 3) Safe text for tooltip (HTML-escaped)
        name = _safe_text(s.get("tk_name") or s.get("osm_name") or "Unknown")
        brand = _safe_text(s.get("brand") or "")

        # 4) Keys for prices
        pred_key = f"pred_price_{fuel_code}" if fuel_code else None
        curr_key = f"price_current_{fuel_code}" if fuel_code else None

        current_price = _fmt_price(s.get(curr_key)) if curr_key else "—"
        predicted_price = _fmt_price(s.get(pred_key)) if pred_key else "—"

        # 5) Optional metrics (if present)
        detour_km = _fmt_km(s.get("detour_distance_km") or s.get("detour_km"))
        detour_min = _fmt_min(s.get("detour_duration_min") or s.get("detour_min"))
        # Net saving is fuel-specific in this project (econ_*_{fuel_code})
        econ_net_key = f"econ_net_saving_eur_{fuel_code}" if fuel_code else None

        raw_net = None
        if econ_net_key:
            raw_net = s.get(econ_net_key)

        # Backward compatible fallback (older payloads)
        if raw_net is None:
            raw_net = s.get("econ_net_saving_eur")

        net_saving = _fmt_eur(raw_net)

        # 6) Label (only used when zoomed in)
        label = ""
        try:
            if pred_key and s.get(pred_key) is not None:
                label = f"{float(s[pred_key]):.3f}"
            elif curr_key and s.get(curr_key) is not None:
                label = f"{float(s[curr_key]):.3f}"
        except Exception:
            label = ""

        # 7) Visual hierarchy
        # Use RGB (3 values) for maximum compatibility; opacity handled by layer opacity.
        fill_color = [0, 200, 0] if is_best else [255, 165, 0]

        if is_selected:
            line_color = [255, 255, 255]   # selected ring
            line_width = 3
            radius = float(radius_selected)
        elif is_best:
            line_color = [0, 0, 0]         # best ring
            line_width = 2
            radius = float(radius_best)
        else:
            line_color = [0, 0, 0]         # subtle outline
            line_width = 1
            radius = float(radius_other)

        # --- Pin icon fields for IconLayer ---
        # Use hex colors in the SVG; keep it simple and stable.
        fill_hex = "#00C800" if is_best else "#FFA500"
        stroke_hex = "#FFFFFF" if is_selected else "#000000"

        icon_url = _pin_icon_data_url(fill_hex, stroke_hex)

        # Pixel size of the icon (bigger for best/selected)
        icon_size = 30 if is_selected else (26 if is_best else 20)

        station_data.append(
            {
                "lon": lon,
                "lat": lat,
                "station_uuid": station_uuid or "",
                "name": name,
                "brand": brand,
                "tag": "RECOMMENDED" if is_best else "",
                "current_price": current_price,
                "predicted_price": predicted_price,
                "detour_km": detour_km,
                "detour_min": detour_min,
                "net_saving": net_saving,
                "label": label,
                "fill_color": fill_color,
                "line_color": line_color,
                "line_width": int(line_width),
                "radius": radius,
                "icon": {
                    "url": icon_url,
                    "width": 64,
                    "height": 64,
                    "anchorY": 64,   # bottom tip
                },
                "icon_size": icon_size,
            }
        )

    # Optional: sanity check (highly recommended during development)
    if station_data:
        uniq = {(round(d["lon"], 5), round(d["lat"], 5)) for d in station_data}
        if len(uniq) <= max(1, int(0.2 * len(station_data))):
            st.warning(
                f"Map sanity check: {len(uniq)} unique coordinates for {len(station_data)} stations. "
                "Many markers overlap; check upstream lat/lon generation."
            )

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------
    route_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": route_coords}],
        get_path="path",
        get_width=4,
        get_color=[30, 144, 255, 255],  # blue
        width_min_pixels=2,
        pickable=False,
    )

    extra_layers: List[pdk.Layer] = []

    if via_full_coords:
        extra_layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": via_full_coords}],
                get_path="path",
                get_width=4,
                get_color=[148, 0, 211, 255],  # violet
                width_min_pixels=3,
                pickable=False,
            )
        )

    use_pins = True  # set to False to force dots

    if use_pins:
        stations_layer = pdk.Layer(
            "IconLayer",
            id="stations",
            data=station_data,
            get_position=["lon", "lat"],
            get_icon="icon",             # IMPORTANT: read per-row icon dict
            get_size="icon_size",
            size_units="pixels",
            size_scale=1,
            size_min_pixels=14,
            size_max_pixels=36,
            pickable=True,
            auto_highlight=True,
        )
    else:
        stations_layer = pdk.Layer(
            "ScatterplotLayer",
            id="stations",
            data=station_data,
            get_position=["lon", "lat"],
            get_fill_color="fill_color",
            get_line_color="line_color",
            get_line_width="line_width",
            stroked=True,
            filled=True,
            get_radius="radius",
            radius_units="pixels",
            radius_scale=1,
            radius_min_pixels=2,
            radius_max_pixels=14,
            pickable=True,
            auto_highlight=True,
            opacity=1.0,
        )

    label_layers: List[pdk.Layer] = []
    if show_station_labels and zoom_level >= 9.5:
        # Only show labels when sufficiently zoomed in to avoid clutter.
        label_layers.append(
            pdk.Layer(
                "TextLayer",
                id="station-labels",
                data=station_data,
                get_position=["lon", "lat"],
                size_scale=1,
                size_min_pixels=10,
                size_max_pixels=18,
                get_text="label",
                get_size=12,
                size_units="pixels",
                get_color=[0, 0, 0, 220],
                get_text_anchor="start",
                get_alignment_baseline="center",
                get_pixel_offset=[10, 0],
                pickable=False,
            )
        )

    # ------------------------------------------------------------------
    # View state (fit to route if possible)
    # ------------------------------------------------------------------
    if route_coords:
        lons = [coord[0] for coord in route_coords]
        lats = [coord[1] for coord in route_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)

        zoom = _calculate_zoom_for_bounds(
            lon_min=min(lons),
            lon_max=max(lons),
            lat_min=min(lats),
            lat_max=max(lats),
            padding_percent=0.10,
            map_width_px=700,
            map_height_px=500,
        )
    else:
        center_lat, center_lon = 51.1657, 10.4515
        zoom = 6

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
        controller=True,
    )

    tooltip = {
        "html": (
            "<div style='font-size: 12px;'>"
            "<div style='font-weight: 600; margin-bottom: 4px;'>{name}</div>"
            "<div style='opacity: 0.85; margin-bottom: 6px;'>{brand} "
            "<span style='color: #6ee7b7; font-weight: 600;'>{tag}</span></div>"
            "<div><b>Current</b>: {current_price} &nbsp; <b>Pred</b>: {predicted_price}</div>"
            "<div><b>Detour</b>: {detour_km} / {detour_min}</div>"
            "<div><b>Net saving</b>: {net_saving}</div>"
            "</div>"
        ),
        "style": {
            "backgroundColor": "rgba(0, 0, 0, 0.85)",
            "color": "white",
            "padding": "10px",
            "borderRadius": "6px",
            "maxWidth": "320px",
        },
    }

    deck = pdk.Deck(
        layers=[route_layer, *extra_layers, stations_layer, *label_layers],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=map_style,
    )
    return deck


def _fuel_label_to_code(label: str) -> str:
    """
    Map human-readable fuel label to internal fuel code.

    Parameters
    ----------
    label : str
        One of 'E5', 'E10', 'Diesel'.

    Returns
    -------
    str
        'e5', 'e10' or 'diesel'.
    """
    mapping = {"E5": "e5", "E10": "e10", "Diesel": "diesel"}
    code = mapping.get(label)
    if code is None:
        raise ValueError(f"Unsupported fuel label '{label}'.")
    return code


def _format_price(x: Any) -> str:
    """Format a numeric price as a string with 3 decimals (or '-' for missing)."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "-"


def _format_eur(x: Any) -> str:
    """Format a numeric value as 'x.xx €' (or '-' for missing)."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.2f} €"
    except (TypeError, ValueError):
        return "-"


def _format_km(x: Any) -> str:
    """Format kilometres with one decimal."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.1f} km"
    except (TypeError, ValueError):
        return "-"


def _format_min(x: Any) -> str:
    """Format minutes with one decimal."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.1f} min"
    except (TypeError, ValueError):
        return "-"


def _format_liters(x: Any) -> str:
    """Format litres with two decimals (or '-' for missing)."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.2f} L"
    except (TypeError, ValueError):
        return "-"


def _describe_price_basis(
    station: Dict[str, Any],
    fuel_code: str,
) -> str:
    """
    Turn the debug fields for a station into a human-readable explanation.

    Uses:
        debug_<fuel>_used_current_price
        debug_<fuel>_horizon_used
        debug_<fuel>_cells_ahead_raw
    """
    used_current = bool(station.get(f"debug_{fuel_code}_used_current_price"))
    horizon = station.get(f"debug_{fuel_code}_horizon_used")

    if used_current:
        # Now reflects the refined rule with the ETA threshold
        return "Current price (arrival in ≤ 10 min)"

    if horizon is None:
        # No explicit horizon; either we could not model or we only used daily info.
        return "No forecast available (fallback)"

    try:
        h_int = int(horizon)
    except (TypeError, ValueError):
        h_int = None

    if h_int is None or h_int < 0:
        return "Forecast (ARDL model)"

    approx_min = h_int * 30
    if approx_min == 0:
        return "Forecast (same block, daily lags only)"
    else:
        return f"Forecast (~{approx_min} min ahead, horizon {h_int})"


def _build_ranking_dataframe(
    stations: List[Dict[str, Any]],
    fuel_code: str,
    debug_mode: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame with the most relevant columns for ranking display.

    Parameters
    ----------
    stations :
        List of station dictionaries (already ranked).
    fuel_code :
        'e5', 'e10' or 'diesel'.

    Returns
    -------
    pandas.DataFrame
    """
    if not stations:
        return pd.DataFrame()

    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"
    lag1_key = f"price_lag_1d_{fuel_code}"
    lag2_key = f"price_lag_2d_{fuel_code}"
    lag3_key = f"price_lag_3d_{fuel_code}"
    lag7_key = f"price_lag_7d_{fuel_code}"

    # Economic keys (may or may not exist, depending on how ranking was called)
    econ_net_key = f"econ_net_saving_eur_{fuel_code}"
    econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
    econ_detour_fuel_key = f"econ_detour_fuel_l_{fuel_code}"
    econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
    econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
    econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
    econ_baseline_key = f"econ_baseline_price_{fuel_code}"

    rows = []
    for s in stations:
        # --- 1. PRE-CALCULATE VALUES ---
        
        # Detour geometry
        # The raw route delta can be slightly negative due to routing/rounding artefacts.
        # For user-facing "detour" we display *extra* distance/time (clamped to >= 0),
        # consistent with the economics layer.
        _raw_detour_km = s.get("detour_distance_km")
        _raw_detour_min = s.get("detour_duration_min")

        try:
            _raw_detour_km_f = float(_raw_detour_km) if _raw_detour_km is not None else 0.0
        except (TypeError, ValueError):
            _raw_detour_km_f = 0.0

        try:
            _raw_detour_min_f = float(_raw_detour_min) if _raw_detour_min is not None else 0.0
        except (TypeError, ValueError):
            _raw_detour_min_f = 0.0

        # Clamp for display (extra detour only)
        _detour_km_display = max(_raw_detour_km_f, 0.0)
        _detour_min_display = max(_raw_detour_min_f, 0.0)

        # --- 2. BUILD THE DICTIONARY ---
        row = {
            "Station name": s.get("tk_name") or s.get("osm_name"),
            "Brand": s.get("brand"),
            "City": s.get("city"),
            "OSM name": s.get("osm_name"),
            "Fraction of route": s.get("fraction_of_route"),
            "Distance along route [m]": s.get("distance_along_m"),
            
            # Insert the pre-calculated values here
            "Detour distance [km]": _detour_km_display,
            "Detour time [min]": _detour_min_display,
            
            # human-readable explanation based on debug_* fields
            "Price basis": _describe_price_basis(s, fuel_code),
            f"Current {fuel_code.upper()} price": s.get(current_key),
            f"Lag 1d {fuel_code.upper()}": s.get(lag1_key),
            f"Lag 2d {fuel_code.upper()}": s.get(lag2_key),
            f"Lag 3d {fuel_code.upper()}": s.get(lag3_key),
            f"Lag 7d {fuel_code.upper()}": s.get(lag7_key),
            f"Predicted {fuel_code.upper()} price": s.get(pred_key),
        }

        # Economic metrics (only added if present)
        if econ_net_key in s:
            row["Baseline on-route price"] = s.get(econ_baseline_key)
            row["Gross saving [€]"] = s.get(econ_gross_key)
            row["Detour fuel [L]"] = s.get(econ_detour_fuel_key)
            row["Detour fuel cost [€]"] = s.get(econ_detour_fuel_cost_key)
            row["Time cost [€]"] = s.get(econ_time_cost_key)
            row["Net saving [€]"] = s.get(econ_net_key)
            row["Break-even litres"] = s.get(econ_breakeven_key)

        if debug_mode:
            # Raw signed deltas from routing (can be negative)
            row["DEBUG raw detour distance [km]"] = _raw_detour_km_f
            row["DEBUG raw detour time [min]"] = _raw_detour_min_f
            # Raw diagnostic fields from the prediction layer
            row[f"DEBUG current_time_cell_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_current_time_cell"
            )
            row[f"DEBUG cells_ahead_raw_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_cells_ahead_raw"
            )
            row[f"DEBUG minutes_to_arrival_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_minutes_to_arrival"
            )
            row[f"DEBUG horizon_used_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_horizon_used"
            )
            row[f"DEBUG eta_utc_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_eta_utc"
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    # Only these columns are numeric prices – do NOT touch "Price basis"
    numeric_price_cols = [
        f"Current {fuel_code.upper()} price",
        f"Lag 1d {fuel_code.upper()}",
        f"Lag 2d {fuel_code.upper()}",
        f"Lag 3d {fuel_code.upper()}",
        f"Lag 7d {fuel_code.upper()}",
        f"Predicted {fuel_code.upper()} price",
        "Baseline on-route price",
    ]

    for col in numeric_price_cols:
        if col in df.columns:
            df[col] = df[col].map(_format_price)

    # Format economic + detour columns if present
    if "Detour distance [km]" in df.columns:
        df["Detour distance [km]"] = df["Detour distance [km]"].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.1f}"
        )
    if "Detour time [min]" in df.columns:
        df["Detour time [min]"] = df["Detour time [min]"].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.1f}"
        )

    # Format raw routing deltas (debug-only)
    if "DEBUG raw detour distance [km]" in df.columns:
        df["DEBUG raw detour distance [km]"] = df["DEBUG raw detour distance [km]"].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.1f}"
        )
    if "DEBUG raw detour time [min]" in df.columns:
        df["DEBUG raw detour time [min]"] = df["DEBUG raw detour time [min]"].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.1f}"
        )

    if "Gross saving [€]" in df.columns:
        df["Gross saving [€]"] = df["Gross saving [€]"].map(_format_eur)
    if "Detour fuel [L]" in df.columns:
        df["Detour fuel [L]"] = df["Detour fuel [L]"].map(_format_liters)
    if "Detour fuel cost [€]" in df.columns:
        df["Detour fuel cost [€]"] = df["Detour fuel cost [€]"].map(_format_eur)
    if "Time cost [€]" in df.columns:
        df["Time cost [€]"] = df["Time cost [€]"].map(_format_eur)
    if "Net saving [€]" in df.columns:
        df["Net saving [€]"] = df["Net saving [€]"].map(_format_eur)
    if "Break-even litres" in df.columns:
        df["Break-even litres"] = df["Break-even litres"].map(
            lambda v: "-" if v is None or pd.isna(v) else f"{float(v):.2f}"
        )
        
    return df


def _display_best_station(
    best_station: Dict[str, Any],
    fuel_code: str,
    litres_to_refuel: Optional[float] = None,
    *,
    ranked_stations: Optional[List[Dict[str, Any]]] = None,
    debug_mode: bool = False,
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

    station_name = best_station.get("tk_name") or best_station.get("osm_name")
    brand = best_station.get("brand") or "-"
    city = best_station.get("city") or "-"
    frac = best_station.get("fraction_of_route")
    dist_m = best_station.get("distance_along_m")
    pred_price = best_station.get(pred_key)
    current_price = best_station.get(current_key)

    detour_km = best_station.get("detour_distance_km")
    detour_min = best_station.get("detour_duration_min")

    # Clamp for display (extra detour only), consistent with the economics layer
    try:
        detour_km_f = float(detour_km) if detour_km is not None else 0.0
    except (TypeError, ValueError):
        detour_km_f = 0.0

    try:
        detour_min_f = float(detour_min) if detour_min is not None else 0.0
    except (TypeError, ValueError):
        detour_min_f = 0.0

    detour_km = max(detour_km_f, 0.0)
    detour_min = max(detour_min_f, 0.0)

    frac_str = "-" if frac is None else f"{float(frac):.3f}"
    if dist_m is None:
        dist_str = "-"
    else:
        try:
            dist_km = float(dist_m) / 1000.0
            dist_str = f"{dist_km:.1f} km"
        except (TypeError, ValueError):
            dist_str = "-"

    st.markdown("### Recommended station")

    st.markdown(
        f"**{station_name}**  \n"
        f"{brand}, {city}"
    )

    col1, col2, col3, col4 = st.columns(4)
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
        st.metric("Fraction of route", frac_str)
    with col4:
        st.metric("Distance along route", dist_str)

    # Detour metrics row
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Detour distance", _format_km(detour_km))
    with col6:
        st.metric("Detour time", _format_min(detour_min))

    # Human-readable explanation of what the model actually used
    explanation = _describe_price_basis(best_station, fuel_code)
    st.caption(
        f"How this price was determined: {explanation} "
        "(based on arrival time and available history for this station)."
    )

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
        page_title="Route-aware Fuel Price Recommender",
        layout="wide",
    )

    st.title("Route-aware Fuel Price Recommender (Prototype)")

    # Persist last clicked station (map selection)
    if "selected_station_uuid" not in st.session_state:
        st.session_state["selected_station_uuid"] = None
    # Persist last computed results so UI does not reset on reruns (sidebar changes / map clicks)
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None  # will store dict of computed outputs
    if "last_params_hash" not in st.session_state:
        st.session_state["last_params_hash"] = None

    st.markdown(
        """
This UI wraps the existing pipeline:

- **Integration** (route → stations → historical + real-time prices)
- **ARDL prediction models** for E5, E10 and Diesel (15 models total)
- **Decision layer** to rank and recommend stations along the route

The recommendation logic can optionally incorporate your own
refuelling amount, car consumption and value of time to decide
whether a detour is economically worthwhile.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    env_label = st.sidebar.radio(
        "Environment",
        options=["Test mode (example route)", "Real route (Google pipeline)"],
        index=1,
    )

    fuel_label = st.sidebar.selectbox(
        "Fuel type",
        options=["E5", "E10", "Diesel"],
        index=0,
    )
    fuel_code = _fuel_label_to_code(fuel_label)

    # Route settings (only used in real mode)
    st.sidebar.markdown("### Route settings (real mode)")

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
    litres_to_refuel = st.sidebar.number_input(
        "Litres to refuel",
        min_value=1.0,
        max_value=200.0,
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

    # Map settings
    st.sidebar.markdown("### Map")
    map_style_label = st.sidebar.selectbox(
        "Basemap style",
        options=["Light (Positron)", "Dark (Dark Matter)", "Detailed (Voyager)"],
        index=0,
    )
    map_style_url = {
        "Light (Positron)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "Dark (Dark Matter)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "Detailed (Voyager)": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    }[map_style_label]

    show_station_labels = st.sidebar.checkbox(
        "Show station labels when zoomed in", value=True
    )

    run_clicked = st.sidebar.button("Run recommender")

    # -----------------------------
    # Parameters hash (controls recompute warnings)
    # -----------------------------
    params = {
        "env_label": env_label,
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

    # First-time user gating
    if not run_clicked and not have_cached:
        st.info("Configure the settings on the left and click **Run recommender**.")
        return

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

        if env_label.startswith("Test"):
            st.subheader("Mode: Test route (example data)")
            try:
                result = run_example()
                if isinstance(result, tuple) and len(result) == 2:
                    stations, route_info = result
                else:
                    stations = result
                    route_info = None
            except Exception as exc:
                st.error(f"Error while running example integration: {exc}")
                return
        else:
            st.subheader("Mode: Real route (Google pipeline with real-time prices)")

            if not start_locality or not end_locality:
                st.error("Please provide at least start and end localities (cities/towns).")
                return

            try:
                stations, route_info = get_fuel_prices_for_route(
                    start_locality=start_locality,
                    end_locality=end_locality,
                    start_address=start_address,
                    end_address=end_address,
                    use_realtime=True,
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

        if not stations:
            st.warning("No stations returned by the integration pipeline.")
            return

        st.markdown(f"**Total stations with complete price data:** {len(stations)}")

        # 2) Ranking + recommendation
        if use_economics:
            ranked = rank_stations_by_predicted_price(
                stations,
                fuel_code,
                litres_to_refuel=litres_to_refuel,
                consumption_l_per_100km=consumption_l_per_100km,
                value_of_time_per_hour=value_of_time_eur_per_hour,
                max_detour_km=max_detour_km,
                max_detour_min=max_detour_min,
                min_net_saving_eur=min_net_saving_eur,
            )
            best_station = recommend_best_station(
                stations,
                fuel_code,
                litres_to_refuel=litres_to_refuel,
                consumption_l_per_100km=consumption_l_per_100km,
                value_of_time_per_hour=value_of_time_eur_per_hour,
                max_detour_km=max_detour_km,
                max_detour_min=max_detour_min,
                min_net_saving_eur=min_net_saving_eur,
            )
        else:
            ranked = rank_stations_by_predicted_price(stations, fuel_code)
            best_station = recommend_best_station(stations, fuel_code)

        if not ranked:
            st.warning("No stations with valid predictions for the selected fuel and constraints.")
            return

        best_uuid = None
        if best_station:
            best_uuid = best_station.get("tk_uuid") or best_station.get("station_uuid")

        # 3) Precompute route_coords for map (if available)
        route_coords = None
        if route_info is not None and isinstance(route_info, dict):
            route_coords = route_info.get("route_coords")

        # 4) Cache everything for reruns (map clicks/sidebar changes)
        st.session_state["last_run"] = {
            "stations": stations,
            "ranked": ranked,
            "best_station": best_station,
            "best_uuid": best_uuid,
            "route_info": route_info,
            "route_coords": route_coords,
            "params": params,
        }
        st.session_state["last_params_hash"] = params_hash

    # -----------------------------
    # Render from cache (ALWAYS)
    # -----------------------------
    cached = st.session_state.get("last_run")
    if not cached:
        st.info("Configure the settings and click **Run recommender**.")
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
    )

    # ----------------------------------------------------------------------
    # Full ranking table
    # ----------------------------------------------------------------------
    st.markdown("### Ranking of stations (highest net saving → lowest)")
    st.caption(
        "Stations are ordered by **net economic benefit** of the detour "
        "(gross saving minus detour fuel and time cost), subject to your "
        "detour distance/time caps and the minimum net saving threshold. "
        "The **Price basis** column shows whether the recommendation uses "
        "the observed current price or a model forecast."
    )

    df_ranked = _build_ranking_dataframe(ranked, fuel_code, debug_mode=debug_mode)
    if df_ranked.empty:
        st.info("No stations with valid predictions to display.")
    else:
        st.dataframe(df_ranked.reset_index(drop=True))

    # ----------------------------------------------------------------------
    # Map visualization (only for real route mode)
    # ----------------------------------------------------------------------
    if not env_label.startswith("Test") and route_info is not None:
        st.markdown("### Route and stations map")
        
        try:
            # Use route data from integration
            route_coords = route_info['route_coords']

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

            # Create and display map with ALL stations
            deck = _create_map_visualization(
                route_coords,
                ranked,
                best_station_uuid=best_uuid,
                via_full_coords=via_overview,
                zoom_level=zoom_for_markers,
                fuel_code=fuel_code,
                selected_station_uuid=st.session_state.get("selected_station_uuid"),
                map_style=map_style_url,
                show_station_labels=show_station_labels,
            )

            st.caption("Hover for quick info. Click a station marker to open the details panel.")

            selected_uuid_from_event: Optional[str] = None
            if _supports_pydeck_selections():
                event = st.pydeck_chart(
                    deck,
                    on_select="rerun",
                    selection_mode="single-object",
                    key="route_map",
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
                st.pydeck_chart(deck)
                st.caption(
                    "Note: Your Streamlit version does not expose pydeck click selections. "
                    "Upgrade Streamlit to enable click-to-details interaction."
                )

            if selected_uuid_from_event:
                st.session_state["selected_station_uuid"] = selected_uuid_from_event

            # Details panel (based on last selection)
            selected_uuid = st.session_state.get("selected_station_uuid")
            if selected_uuid:
                # Prefer the ranked station object (contains economics), fall back to the raw station.
                uuid_to_station: Dict[str, Dict[str, Any]] = {}
                for s in ranked:
                    u = _station_uuid(s)
                    if u:
                        uuid_to_station[u] = s
                for s in stations:
                    u = _station_uuid(s)
                    if u and u not in uuid_to_station:
                        uuid_to_station[u] = s

                if st.button("Clear station selection", key="clear_station_selection"):
                    st.session_state["selected_station_uuid"] = None
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()

                station_obj = uuid_to_station.get(selected_uuid)
                if station_obj:
                    with st.expander("Selected station details", expanded=True):
                        _display_station_details_panel(
                            station_obj,
                            fuel_code,
                            litres_to_refuel=litres_to_refuel,
                            debug_mode=debug_mode,
                        )
                else:
                    st.info("Selected station could not be resolved. Please click another marker.")
            else:
                st.info("Click a station marker to show details here.")
            
            # Legend for routes and markers
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Routes:**  \n🔵 Blue = Direct route  \n🟣 Violet = Via recommended station")
            with col2:
                st.markdown("**Stations:**  \n🟢 Green = Recommended station  \n🟠 Orange = Other stations")
            
        except Exception as e:
            st.warning(f"Could not display map: {e}")


if __name__ == "__main__":
    main()
