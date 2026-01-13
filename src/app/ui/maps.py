# src/app/ui/maps.py
from __future__ import annotations

import base64
import inspect
import math
from typing import Any, Dict, List, Optional, Tuple

import pydeck as pdk

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

from ui.formatting import (
    _station_uuid,
    _safe_text,
    _fmt_price,
    _fmt_eur,
    _fmt_km,
    _fmt_min,
)


def calculate_zoom_for_bounds(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    padding_percent: float = 0.10,
    map_width_px: int = 700,
    map_height_px: int = 500,
) -> float:
    """
    Calculate an approximate Web Mercator zoom level to fit given bounds with padding.

    Returns a zoom clamped to [1, 15]. Falls back to 7.5 on error.
    """
    try:
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        # Point-like case
        if lon_range < 0.0001 and lat_range < 0.0001:
            return 15.0

        if lon_range < 0.0001:
            lon_range = 0.1
        if lat_range < 0.0001:
            lat_range = 0.1

        lon_min -= lon_range * padding_percent / 2
        lon_max += lon_range * padding_percent / 2
        lat_min -= lat_range * padding_percent / 2
        lat_max += lat_range * padding_percent / 2

        # Clamp to valid Web Mercator latitude range
        lat_min = max(-85.05, min(85.05, lat_min))
        lat_max = max(-85.05, min(85.05, lat_max))

        lon_delta = lon_max - lon_min
        if lon_delta <= 0:
            lon_delta = 0.1
        zoom_lon = math.log2(360 * map_width_px / (256 * lon_delta))

        lat_min_rad = math.radians(lat_min)
        lat_max_rad = math.radians(lat_max)

        y_min = math.log(math.tan(math.pi / 4 + lat_max_rad / 2))
        y_max = math.log(math.tan(math.pi / 4 + lat_min_rad / 2))

        y_delta = y_max - y_min
        if abs(y_delta) < 0.0001:
            y_delta = 0.1

        zoom_lat = math.log2(math.pi * map_height_px / (256 * abs(y_delta)))

        zoom = min(zoom_lon, zoom_lat)
        return max(1, min(zoom, 15))
    except Exception:
        return 7.5


def supports_pydeck_selections() -> bool:
    """
    Return True if the installed Streamlit supports pydeck selection events.
    """
    if st is None:
        return False
    try:
        sig = inspect.signature(st.pydeck_chart)
        return "on_select" in sig.parameters and "selection_mode" in sig.parameters
    except Exception:
        return False


def _pin_icon_data_url(fill_hex: str, stroke_hex: str) -> str:
    """
    SVG pin icon as data URL for deck.gl IconLayer.
    """
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
        f"<path d='M32 2C21 2 12 11 12 22c0 17 20 40 20 40s20-23 20-40C52 11 43 2 32 2z' "
        f"fill='{fill_hex}' stroke='{stroke_hex}' stroke-width='2'/>"
        "<circle cx='32' cy='22' r='7' fill='white'/>"
        "</svg>"
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _bounds_from_points(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    """
    points: list of (lon, lat)
    returns: (lon_min, lon_max, lat_min, lat_max)
    """
    if not points:
        return None
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return (min(lons), max(lons), min(lats), max(lats))


def create_map_visualization(
    route_coords: List[List[float]],
    stations: List[Dict[str, Any]],
    best_station_uuid: Optional[str] = None,
    via_full_coords: Optional[List[List[float]]] = None,
    zoom_level: float = 7.5,
    *,
    fuel_code: Optional[str] = None,
    selected_station_uuid: Optional[str] = None,
    map_provider: str = "carto",
    map_style: Optional[str] = "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    show_station_labels: bool = True,  # kept for compatibility; currently not rendered as TextLayer
    enable_sanity_check_warning: bool = True,
) -> pdk.Deck:
    """
    Canonical pydeck map:
      - baseline route (and optional via-station route),
      - start/end markers,
      - all stations (best highlighted, selected outlined),
      - tooltip payload for hover.

    Explorer support (Step 7):
      - If route_coords is empty but stations exist, auto-center/zoom to station bounds.

    The layer id for stations is "stations" so Streamlit selection extraction works.
    """

    # --- Marker sizing (pixel-based, scaled by zoom) ---
    base_size_other = 2.0
    base_size_best = 7.0
    base_size_selected = 9.0

    zoom_factor = max(0.6, min(1.6, zoom_level / 10.0))
    radius_other = base_size_other * zoom_factor
    radius_best = base_size_best * zoom_factor
    radius_selected = base_size_selected * zoom_factor

    pred_key = f"pred_price_{fuel_code}" if fuel_code else None
    curr_key = f"price_current_{fuel_code}" if fuel_code else None

    station_data: List[Dict[str, Any]] = []

    for s in stations:
        lat_raw, lon_raw = s.get("lat"), s.get("lon")
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except (TypeError, ValueError):
            continue

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        if not math.isfinite(lat) or not math.isfinite(lon):
            continue

        station_uuid = _station_uuid(s)
        is_best = bool(best_station_uuid and station_uuid and station_uuid == best_station_uuid)
        is_selected = bool(selected_station_uuid and station_uuid and station_uuid == selected_station_uuid)

        name = _safe_text(s.get("tk_name") or s.get("osm_name") or "Station")
        brand = _safe_text(s.get("brand") or "")

        current_price = _fmt_price(s.get(curr_key)) if curr_key else "—"
        predicted_price = _fmt_price(s.get(pred_key)) if pred_key else "—"

        detour_km = _fmt_km(s.get("detour_distance_km") or s.get("detour_km"))
        detour_min = _fmt_min(s.get("detour_duration_min") or s.get("detour_min"))

        econ_net_key = f"econ_net_saving_eur_{fuel_code}" if fuel_code else None
        raw_net = s.get(econ_net_key) if econ_net_key else None
        if raw_net is None:
            raw_net = s.get("econ_net_saving_eur")
        net_saving = _fmt_eur(raw_net)

        # Visual hierarchy
        fill_color = [0, 200, 0] if is_best else [255, 165, 0]
        if is_selected:
            line_color = [255, 255, 255]
            line_width = 3
            radius = float(radius_selected)
        elif is_best:
            line_color = [0, 0, 0]
            line_width = 2
            radius = float(radius_best)
        else:
            line_color = [0, 0, 0]
            line_width = 1
            radius = float(radius_other)

        fill_hex = "#00C800" if is_best else "#FFA500"
        stroke_hex = "#FFFFFF" if is_selected else "#000000"
        icon_url = _pin_icon_data_url(fill_hex, stroke_hex)
        icon_size = 45 if is_selected else (39 if is_best else 30)

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
                "fill_color": fill_color,
                "line_color": line_color,
                "line_width": int(line_width),
                "radius": radius,
                "icon": {"url": icon_url, "width": 64, "height": 64, "anchorY": 64},
                "icon_size": icon_size,
            }
        )

    # Optional sanity check warning (kept to preserve your current behavior)
    if enable_sanity_check_warning and st is not None and station_data:
        try:
            uniq = {(round(d["lon"], 5), round(d["lat"], 5)) for d in station_data}
            if len(uniq) <= max(1, int(0.2 * len(station_data))):
                st.warning(
                    f"Map sanity check: {len(uniq)} unique coordinates for {len(station_data)} stations. "
                    "Many markers overlap; check upstream lat/lon generation."
                )
        except Exception:
            pass

    # --- Layers ---
    layers: List[pdk.Layer] = []

    # Route layers only if we have coordinates
    if route_coords:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": route_coords}],
                get_path="path",
                get_width=4,
                get_color=[30, 144, 255, 255],
                width_min_pixels=2,
                pickable=False,
            )
        )

    if via_full_coords:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": via_full_coords}],
                get_path="path",
                get_width=4,
                get_color=[148, 0, 211, 255],
                width_min_pixels=3,
                pickable=False,
            )
        )

    # Stations
    layers.append(
        pdk.Layer(
            "IconLayer",
            id="stations",
            data=station_data,
            get_position=["lon", "lat"],
            get_icon="icon",
            get_size="icon_size",
            size_units="pixels",
            size_scale=1,
            size_min_pixels=30,
            size_max_pixels=50,
            pickable=True,
            auto_highlight=True,
        )
    )

    # Start / end markers (only if route exists)
    start_end_data: List[Dict[str, Any]] = []
    if route_coords and len(route_coords) >= 2:
        start_end_data.append(
            {"name": "Start", "lon": route_coords[0][0], "lat": route_coords[0][1], "color": [0, 128, 0, 255]}
        )
        start_end_data.append(
            {"name": "Destination", "lon": route_coords[-1][0], "lat": route_coords[-1][1], "color": [200, 0, 0, 255]}
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=start_end_data,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_line_color=[255, 255, 255],
            get_line_width=2,
            stroked=True,
            filled=True,
            get_radius=10,
            radius_units="meters",
            pickable=False,
        )
    )

    # --- View state ---
    if route_coords:
        lons = [c[0] for c in route_coords]
        lats = [c[1] for c in route_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        zoom = calculate_zoom_for_bounds(min(lons), max(lons), min(lats), max(lats))
    else:
        # Step 7: Explorer mode – fit to stations if available
        station_points = [(d["lon"], d["lat"]) for d in station_data]
        bounds = _bounds_from_points(station_points)

        if bounds:
            lon_min, lon_max, lat_min, lat_max = bounds
            center_lon = (lon_min + lon_max) / 2
            center_lat = (lat_min + lat_max) / 2
            zoom = calculate_zoom_for_bounds(lon_min, lon_max, lat_min, lat_max)
        else:
            # Fallback (Germany)
            center_lat, center_lon, zoom = 51.1657, 10.4515, 6

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
        controller={
            # Prevent the map from capturing page scroll / trackpad scroll
            "scrollZoom": False,

            # Keep panning with mouse/touch drag
            "dragPan": True,

            # Keep pinch-to-zoom on touch devices
            "touchZoom": True,

            # Optional: reduce accidental rotations
            "touchRotate": False,
            "dragRotate": False,
        },
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

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider=map_provider,
        map_style=map_style,
    )


# --- Backwards-compatible names (minimize edits in pages) -------------------
_calculate_zoom_for_bounds = calculate_zoom_for_bounds
_supports_pydeck_selections = supports_pydeck_selections
_create_map_visualization = create_map_visualization
