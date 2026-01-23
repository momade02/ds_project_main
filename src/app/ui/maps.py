"""
MODULE: Maps — Map Rendering Utilities for Route and Station Visualizations
-------------------------------------------------------------------------

Purpose
- Encapsulates all map-generation logic so pages can render consistent route/station maps without
  duplicating HTML/JS details or coordinate/viewport calculations.

What this module does
- Viewport/zoom:
  - Provides `calculate_zoom_for_bounds(...)` to estimate a Web Mercator zoom level that fits a given
    bounding box (with padding and clamps).
- Map rendering (primary):
  - Builds a self-contained Mapbox GL JS iframe HTML payload via `create_mapbox_gl_html(...)` for use
    with `streamlit.components.html(...)`.
  - Implements “cooperative gestures”:
    - Desktop: Ctrl/⌘ + scroll to zoom
    - Mobile: two-finger gestures to pan/zoom
- Layers and semantics:
  - Renders the main route polyline (baseline) and optional via/alternative polyline.
  - Renders station markers with stable UUIDs, and supports “best”, “selected”, and category-based pins.
  - Builds best-effort popup content (brand/name, address, current/predicted price, detour metrics, and
    optional economic savings vs worst on-route baseline).
- Token handling:
  - Reads Mapbox token from environment/secrets (`MAPBOX_ACCESS_TOKEN` preferred), and returns a
    user-friendly error HTML block when missing.

Inputs and contracts
- Expects route coordinates as [lon, lat] pairs and stations as dicts containing at least `lat`, `lon`
  plus optional pipeline fields for prices, detours, and economics.
- Uses UI formatting helpers (price, EUR, km, min) to ensure consistent display formatting.

Design constraints
- Must not trigger external API calls; it only renders from provided payloads.
- Must be resilient to partial/malformed station records (best-effort rendering, skip invalid points).
- Must preserve backwards compatibility (underscore alias `_calculate_zoom_for_bounds`).
"""

from __future__ import annotations

import base64
import inspect
import math
from typing import Any, Dict, List, Optional, Tuple

import json
import os

import pydeck as pdk

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

from ui.formatting import (
    _station_uuid,
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


# --- Backwards-compatible names (minimize edits in pages) -------------------
_calculate_zoom_for_bounds = calculate_zoom_for_bounds


# --- Mapbox GL JS (cooperative gestures) -------------------------------
def create_mapbox_gl_html(
    *,
    route_coords: list[list[float]],
    stations: list[dict[str, Any]],
    best_station_uuid: str | None,
    via_full_coords: list[list[float]] | None,
    use_satellite: bool,
    selected_station_uuid: str | None,
    marker_category_by_uuid: dict[str, str] | None = None,
    height_px: int = 560,

    # --- New (optional) ---
    fuel_code: str | None = None,
    litres_to_refuel: float | None = None,
    onroute_worst_price: float | None = None,
    value_of_time_eur_per_hour: float | None = None,
    popup_variant: str = "route",  # "route" (Page 01) vs "explorer" (Page 04)
    **_ignored_kwargs: Any,
) -> str:

    """
    Mapbox GL JS map (iframe HTML) with cooperative gestures:
      - Desktop: Ctrl/⌘ + scroll to zoom
      - Mobile: two-finger touch to pan
    Includes: main route (blue), alt/via route (purple dashed), station pins (SVG markers).
    """

    # Token (client-side; will be visible in browser)
    token = (
        os.environ.get("MAPBOX_ACCESS_TOKEN")
        or os.environ.get("MAPBOX_TOKEN")
        or (st.secrets.get("MAPBOX_ACCESS_TOKEN") if st is not None else None)  # type: ignore[attr-defined]
    )
    if not token:
        return (
            "<div style='padding:12px;border:1px solid #ddd;border-radius:8px;'>"
            "<b>Mapbox token missing.</b><br/>"
            "Set <code>MAPBOX_ACCESS_TOKEN</code> (preferred) or <code>MAPBOX_TOKEN</code> in your environment."
            "</div>"
        )

    style_url = (
        "mapbox://styles/mapbox/satellite-streets-v12"
        if use_satellite
        else "mapbox://styles/mapbox/streets-v12"
    )

    # Price keys (optional)
    pred_key = f"pred_price_{fuel_code}" if fuel_code else None
    curr_key = f"price_current_{fuel_code}" if fuel_code else None

    econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}" if fuel_code else None
    econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}" if fuel_code else None

    def _build_address_line(s: dict[str, Any]) -> str:
        """
        Best-effort address line from commonly used fields (Tankerkönig / OSM / your pipeline).
        No external geocoding is used here.
        """
        # Prefer already-formatted fields if present
        for k in ("formatted_address", "address", "full_address", "tk_address", "osm_address"):
            v = s.get(k)
            if v and str(v).strip():
                return str(v).strip()

        street = s.get("street") or s.get("tk_street") or s.get("osm_street")
        house = (
            s.get("houseNumber") or s.get("house_number") or s.get("housenumber")
            or s.get("tk_house_number") or s.get("osm_house_number")
        )
        postcode = (
            s.get("post_code") or s.get("postcode") or s.get("zip") or s.get("postal_code")
            or s.get("tk_postcode") or s.get("osm_postcode")
        )
        city = s.get("place") or s.get("city") or s.get("town") or s.get("village") or s.get("tk_city") or s.get("osm_city")

        line1 = " ".join([str(x).strip() for x in [street, house] if x and str(x).strip()]).strip()
        line2 = " ".join([str(x).strip() for x in [postcode, city] if x and str(x).strip()]).strip()

        if line1 and line2:
            return f"{line1}, {line2}"
        return line1 or line2 or ""

    # ---- Build station features for JS ----
    features: list[dict[str, Any]] = []
    for s in stations or []:
        try:
            lat = float(s.get("lat"))
            lon = float(s.get("lon"))
        except (TypeError, ValueError):
            continue

        suuid = _station_uuid(s) or ""
        is_best = bool(best_station_uuid and suuid == best_station_uuid)
        is_selected = bool(selected_station_uuid and suuid == selected_station_uuid)

        name = s.get("tk_name") or s.get("osm_name") or s.get("name") or "Station"
        brand = s.get("brand") or ""

        props_category = None
        if marker_category_by_uuid and suuid:
            props_category = marker_category_by_uuid.get(suuid)

        # Headline: brand preferred; fallback to station name
        headline = str(brand).strip() if (brand and str(brand).strip()) else str(name)

        # Address (best-effort)
        address_line = _build_address_line(s)

        # Prices (formatted)
        current_price_s = _fmt_price(s.get(curr_key)) if curr_key else "—"
        predicted_price_s = _fmt_price(s.get(pred_key)) if pred_key else "—"
        distance_km_s = _fmt_km(s.get("distance_km"))

        # Detour (formatted)
        detour_km_s = _fmt_km(s.get("detour_distance_km") or s.get("detour_km"))
        detour_min_s = _fmt_min(s.get("detour_duration_min") or s.get("detour_min"))

        # Saving vs worst on-route (net if detour costs exist; else gross)
        save_vs_worst_s = "—"
        if onroute_worst_price is not None and litres_to_refuel and pred_key and s.get(pred_key) is not None:
            try:
                p_station = float(s.get(pred_key))
                litres_f = float(litres_to_refuel)
                worst_f = float(onroute_worst_price)

                gross = (worst_f - p_station) * litres_f

                detour_fuel_cost = s.get(econ_detour_fuel_cost_key) if econ_detour_fuel_cost_key else None
                time_cost = s.get(econ_time_cost_key) if econ_time_cost_key else None

                try:
                    detour_fuel_cost_f = float(detour_fuel_cost) if detour_fuel_cost is not None else 0.0
                except (TypeError, ValueError):
                    detour_fuel_cost_f = 0.0

                # Time cost: prefer precomputed econ_time_cost; else compute from detour minutes + value-of-time (if provided)
                time_cost_f = 0.0
                if time_cost is not None:
                    try:
                        time_cost_f = float(time_cost)
                    except (TypeError, ValueError):
                        time_cost_f = 0.0
                elif value_of_time_eur_per_hour is not None:
                    # Best-effort detour minutes lookup
                    dm = s.get("detour_duration_min") or s.get("detour_min")
                    try:
                        dm_f = float(dm) if dm is not None else 0.0
                        vot_f = float(value_of_time_eur_per_hour)
                        time_cost_f = max(0.0, dm_f) / 60.0 * max(0.0, vot_f)
                    except Exception:
                        time_cost_f = 0.0


                net = gross - detour_fuel_cost_f - time_cost_f
                save_vs_worst_s = _fmt_eur(net)
            except Exception:
                save_vs_worst_s = "—"

        # Add distance_along_m to the properties for each station
        distance_along_m = s.get("distance_along_m", 0)
        distance_along_km = distance_along_m / 1000.0 if distance_along_m else None

        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "station_uuid": suuid,
                    "title": str(name),
                    "brand": str(brand),
                    "headline": headline,
                    "address": str(address_line),
                    "current_price": str(current_price_s),
                    "predicted_price": str(predicted_price_s),
                    "detour_km": str(detour_km_s),
                    "detour_min": str(detour_min_s),
                    "save_vs_worst": str(save_vs_worst_s),

                    "is_best": is_best,
                    "is_selected": is_selected,
                    "marker_category": str(props_category or ("best" if is_best else "better")),
                    "distance_km": str(distance_km_s),
                    # Preserve raw meters and a formatted km string so JS can show/format it reliably
                    "distance_along_m": distance_along_m,
                    "distance_along_km": (f"{distance_along_km:.1f}" if distance_along_km is not None else "—"),
                },
            }
        )

    stations_geojson = {"type": "FeatureCollection", "features": features}

    # ---- Route GeoJSON (lng/lat) ----
    route_geojson = None
    if route_coords:
        route_geojson = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": route_coords},
            "properties": {},
        }

    via_geojson = None
    if via_full_coords:
        via_geojson = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": via_full_coords},
            "properties": {},
        }

    # ---- Bounds: route first, then stations fallback ----
    bounds_points: list[list[float]] = []
    if route_coords:
        bounds_points = route_coords
    elif features:
        bounds_points = [f["geometry"]["coordinates"] for f in features]

    if not bounds_points:
        # Fallback (Germany-ish)
        bounds_points = [[10.4515, 51.1657], [10.4515, 51.1657]]

    lons = [p[0] for p in bounds_points]
    lats = [p[1] for p in bounds_points]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    # JSON payloads for JS
    js_route = json.dumps(route_geojson) if route_geojson else "null"
    js_via = json.dumps(via_geojson) if via_geojson else "null"
    js_stations = json.dumps(stations_geojson)

    # Unique container
    div_id = f"map_{abs(hash((min_lon, min_lat, max_lon, max_lat, len(features))))}"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no"/>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.17.0/mapbox-gl.css" rel="stylesheet"/>
  <script src="https://api.mapbox.com/mapbox-gl-js/v3.17.0/mapbox-gl.js"></script>
  <style>
    html, body {{ margin:0; padding:0; }}
    #{div_id} {{ width:100%; height:{height_px}px; border-radius:12px; overflow:hidden; }}
  </style>
</head>
<body>
  <div id="{div_id}"></div>

  <script>
    mapboxgl.accessToken = {json.dumps(token)};

    const routeGeo = {js_route};
    const viaGeo = {js_via};
    const stationsGeo = {js_stations};

    const initialBounds = [[{min_lon}, {min_lat}], [{max_lon}, {max_lat}]];

    const map = new mapboxgl.Map({{
      container: "{div_id}",
      style: {json.dumps(style_url)},
      cooperativeGestures: true,
      bounds: initialBounds,
      fitBoundsOptions: {{ padding: 40, duration: 0 }}
    }});

    map.addControl(new mapboxgl.NavigationControl(), "top-right");

    function pinSVG(fill, stroke) {{
      return `
        <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 64 64">
          <path d="M32 2C21 2 12 11 12 22c0 17 20 40 20 40s20-23 20-40C52 11 43 2 32 2z"
                fill="${{fill}}" stroke="${{stroke}}" stroke-width="2"/>
          <circle cx="32" cy="22" r="7" fill="white"/>
        </svg>
      `;
    }}

    map.on("load", () => {{

      // Main route (blue)
      if (routeGeo) {{
        map.addSource("route", {{ type: "geojson", data: routeGeo }});
        map.addLayer({{
          id: "route-line",
          type: "line",
          source: "route",
          layout: {{ "line-join": "round", "line-cap": "round" }},
          paint: {{
            "line-width": 4,
            "line-color": "rgba(30, 144, 255, 1.0)",
            "line-opacity": 0.95
          }}
        }});
      }}

      // Alternative/via route (purple dashed)
      if (viaGeo) {{
        map.addSource("via", {{ type: "geojson", data: viaGeo }});
        map.addLayer({{
          id: "via-line",
          type: "line",
          source: "via",
          layout: {{ "line-join": "round", "line-cap": "round" }},
          paint: {{
            "line-width": 4,
            "line-color": "rgba(148, 0, 211, 1.0)",
            "line-opacity": 0.95,
            "line-dasharray": [1.5, 1.5]
          }}
        }});
      }}

            // Stations as SVG pin markers
      const popup = new mapboxgl.Popup({{ closeButton: false, closeOnClick: false }});

      // ---- Ensure layering: draw non-best first, then best, then selected last ----
      const features = (stationsGeo.features || []).slice();

      function _truthy(v) {{
        return (v === true || v === "true");
      }}

      // Sort so that: other/better first, best later, selected last (highest priority)
      features.sort((a, b) => {{
        const pa = (a && a.properties) ? a.properties : {{}};
        const pb = (b && b.properties) ? b.properties : {{}};

        const aBest = _truthy(pa.is_best);
        const bBest = _truthy(pb.is_best);

        const aSel  = _truthy(pa.is_selected);
        const bSel  = _truthy(pb.is_selected);

        // Weight: selected=3, best=2, else=0
        const wa = (aSel ? 3 : 0) + (aBest ? 2 : 0);
        const wb = (bSel ? 3 : 0) + (bBest ? 2 : 0);

        return wa - wb;
      }});

      features.forEach((f) => {{
        const coords = f.geometry && f.geometry.coordinates;
        if (!coords || coords.length !== 2) return;

        const props = f.properties || {{}};
        const isBest = (props.is_best === true || props.is_best === "true");
        const isSelected = (props.is_selected === true || props.is_selected === "true");

        // Color logic (three categories + selection override)
        const cat = (props.marker_category || "").toLowerCase();

        let fill = "#FFA500";   // default = better (orange)
        let stroke = "#000000"; // default black

        if (cat === "other") {{
          fill = "#D32F2F";     // other = red
        }}
        if (cat === "best" || isBest) {{
          fill = "#00C800";     // best = green
          stroke = "#000000";
        }}

        const el = document.createElement("div");

        // ---- Marker sizing (use width/height; do NOT use transform) ----
        const baseSize = 28;

        let scale = 0.9;
        if (isBest) scale = 1.2;
        if (isSelected) scale = 1.4;

        const sizePx = Math.round(baseSize * scale);

        // Use double braces ${{...}} so Python outputs ${...} for JS
        el.style.width = `${{sizePx}}px`;
        el.style.height = `${{sizePx}}px`;

        // Keep the SVG responsive inside the container
        el.innerHTML = pinSVG(fill, stroke);

        // --- 1. Helper: Minimal HTML escaping (defensive) ---
            function esc(x) {{
                return String(x ?? "").replace(/[&<>"']/g, (c) => ({{
                    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
                }}[c]));
            }}

            // --- 2. Prepare Data for Popup ---
            const popupVariant = {json.dumps(popup_variant)};

            const brand = (props.brand || "").trim();
            const title = (props.title || "Station").trim();
            const line1 = brand ? brand : title;

            const address = (props.address || "").trim();

            const curP = props.current_price || "-";
            const dist = props.distance_km || "-";

            // Route-mode fields (Page 01)
            const predP = props.predicted_price || "-";
            const detKm = props.detour_km || "-";
            const detMin = props.detour_min || "-";
            const save = props.save_vs_worst || "-";

            el.addEventListener("mouseenter", () => {{
                // Fix: Extract address logic to avoid nested template literals in Python f-string
                const addrHtml = address ? `<div style="opacity:.85; margin-bottom:6px;">${{esc(address)}}</div>` : "";
                let html = "";

                if (popupVariant === "explorer") {{
                    // Page 04: Brand -> full address -> current price + distance
                    html = `
                        <div style="font-size:12px; line-height:1.25;">
                            <div style="font-weight:700; margin-bottom:4px;">${{esc(line1)}}</div>
                            ${{addrHtml}}
                            <div><b>Current</b>: ${{esc(curP)}}</div>
                            <div><b>Distance (air-line)</b>: ${{esc(dist)}}</div>
                        </div>
                    `;
                }} else {{
                    // Default (Page 01): keep existing richer content
                    const headline = props.headline || props.brand || props.title || "Station";
                    html = `
                        <div style="font-size:12px; line-height:1.25;">
                            <div style="font-weight:700; margin-bottom:4px;">${{esc(headline)}}</div>
                            ${{addrHtml}}
                            <div><b>Current</b>: ${{esc(curP)}} &nbsp; <b>Pred</b>: ${{esc(predP)}}</div>
                            <div><b>Detour</b>: ${{esc(detKm)}} &nbsp; (${{esc(detMin)}})</div>
                            <div><b>Distance to station</b>: ${{esc(props.distance_along_km || "-")}} km</div>
                            <div><b>Safe up to</b>: ${{esc(save)}}</div>
                        </div>`;
                }}

                popup
                    .setLngLat(coords)
                    .setHTML(html)
                    .addTo(map);

                const pe = popup.getElement();
                if (pe) pe.style.zIndex = "9999";
            }});

        el.addEventListener("mouseleave", () => {{
          popup.remove();
        }});

        // ---- Create marker and force z-order ----
        const marker = new mapboxgl.Marker({{ element: el, anchor: "bottom" }})
          .setLngLat(coords)
          .addTo(map);

        // z-index: selected highest, best next, others lowest
        const z = isSelected ? 30 : (isBest ? 20 : 10);
        marker.getElement().style.zIndex = String(z);
      }});

    }});
  </script>
</body>
</html>
"""
    return html
