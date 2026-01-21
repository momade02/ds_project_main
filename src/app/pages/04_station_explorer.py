# src/app/pages/04_station_explorer.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui.styles import apply_app_css

from ui.sidebar import render_sidebar_shell, _render_help_explorer

from config.settings import load_env_once

load_env_once()

from config.settings import ensure_persisted_state_defaults
from services.session_store import init_session_context, restore_persisted_state, maybe_persist_state

import json
import hashlib

# ---------------------------------------------------------------------
# Import bootstrap (same pattern as your other pages)
# ---------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]       # .../src/app
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app_errors import AppError

from src.app.services.station_explorer import (
    StationExplorerInputs,
    search_stations_nearby_list_api,
)

import ui.maps as maps_mod
import inspect

import streamlit.components.v1 as components

from ui.formatting import (
    _station_uuid,
    _safe_text,
    _format_price,
)

# Fixed brand options (same canonical list as Trip Planner / Page 01)
BRAND_OPTIONS = ["ARAL", "AVIA", "AGIP ENI", "Shell", "Total", "ESSO", "JET", "ORLEN", "HEM", "OMV"]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fuel_label_to_code(label: str) -> str:
    m = {"E5": "e5", "E10": "e10", "Diesel": "diesel"}
    return m.get(label, "e5")


def _build_results_table(stations: List[Dict[str, Any]], fuel_code: str) -> pd.DataFrame:
    """
    Explorer results table (Page 04).
    Columns (left -> right):
      Brand, Name, City, Full Address, Distance (air-line in km), Current Price <fuel>, Open (Yes/No)
    """
    price_key = f"price_current_{fuel_code}"

    def _full_address(s: Dict[str, Any]) -> str:
        def _first(*keys: str) -> str:
            for k in keys:
                v = (s or {}).get(k)
                if v is None:
                    continue
                t = str(v).strip()
                if t:
                    return t
            return ""

        street = _first("street", "addr_street", "addr:street")
        house = _first("houseNumber", "house_number", "housenumber", "addr_housenumber", "addr:housenumber")
        postcode = _first("postCode", "postcode", "zip", "postal_code", "postalCode", "addr_postcode", "addr:postcode")
        city = _first("city", "place", "town", "village")

        line1 = " ".join([p for p in [street, house] if p]).strip()
        line2 = " ".join([p for p in [postcode, city] if p]).strip()

        if line1 and line2:
            return f"{line1}, {line2}"
        if line1:
            return line1
        if line2:
            return line2
        return city or ""

    current_price_col = f"Current Price {fuel_code.upper()}"

    rows: List[Dict[str, Any]] = []
    for s in stations or []:
        is_open = (s or {}).get("is_open")
        open_label = "Yes" if is_open is True else "No"

        name = (s or {}).get("tk_name") or (s or {}).get("osm_name") or (s or {}).get("name") or "Unknown"
        brand = (s or {}).get("brand") or ""
        city = (s or {}).get("city") or (s or {}).get("place") or ""

        rows.append(
            {
                "Brand": brand,
                "Name": name,
                "City": city,
                "Full Address": _full_address(s),
                "Distance (air-line in km)": round(float((s or {}).get("distance_km") or 0.0), 2),
                current_price_col: _format_price((s or {}).get(price_key)),
                "Open (Yes/No)": open_label,
            }
        )

    return pd.DataFrame(rows)


def _try_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        # handle "1,599" style strings defensively
        if isinstance(v, str):
            v = v.replace(",", ".").strip()
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick_cheapest_station(stations: List[Dict[str, Any]], fuel_code: str) -> Optional[Dict[str, Any]]:
    """
    Returns the station with the lowest current price for the selected fuel.
    Tie-break: if multiple stations share the same best price, pick the closer one (distance_km).
    Ignores stations with missing/non-numeric prices.
    """
    price_key = f"price_current_{fuel_code}"

    best_s: Optional[Dict[str, Any]] = None
    best_key: Optional[tuple[float, float]] = None  # (price, distance_km)

    for s in stations or []:
        p = _try_float((s or {}).get(price_key))
        if p is None:
            continue

        d = _try_float((s or {}).get("distance_km"))
        d_f = float(d) if d is not None else 1e9

        k = (float(p), d_f)
        if best_key is None or k < best_key:
            best_key = k
            best_s = s

    return best_s


def _render_cheapest_station_header(station: Dict[str, Any], fuel_code: str) -> None:
    """
    Reuse Page 01 'station-header' styling for Explorer.
    No reverse-geocoding; use payload fields only.
    """
    if not station:
        return

    station_name = station.get("tk_name") or station.get("osm_name") or station.get("name") or "Unknown"
    brand = station.get("brand")
    city = station.get("city") or station.get("place") or ""

    # Line 1: brand preferred; fall back to station name
    base_name = brand if (brand and str(brand).strip()) else station_name
    city_clean = str(city).strip() if city else ""
    title_line = f"{base_name} in {city_clean}" if (base_name and city_clean) else base_name

    price_key = f"price_current_{fuel_code}"
    price_line = f"Current {fuel_code.upper()}: {_format_price(station.get(price_key))}"

    subtitle_line = price_line

    label_html = _safe_text("Cheapest & closest station:")
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
# MAIN
# ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Station Explorer", layout="wide")

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
        st.session_state["map_style_mode"] = _preserve_map_style_mode

    # ------------------------------------------------------------------
    # Page 04 hard overrides (do not expose as UI on this page)
    # ------------------------------------------------------------------
    st.session_state["debug_mode"] = True
    st.session_state["explorer_use_realtime"] = True

    apply_app_css()
    st.title("Station Explorer")
    st.caption("##### Browse fuel stations near a location.")

    NAV_TARGETS = {
        "Home": "streamlit_app.py",
        "Analytics": "pages/02_route_analytics.py",
        "Station": "pages/03_station_details.py",
        "Explorer": "pages/04_station_explorer.py",
    }
    CURRENT = "Explorer"

    if st.session_state.get("_active_page") != CURRENT:
        st.session_state["_active_page"] = CURRENT
        st.session_state["top_nav"] = CURRENT

    selected = st.segmented_control(
        label="",
        options=list(NAV_TARGETS.keys()),
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="top_nav",
    )

    target = NAV_TARGETS.get(selected, NAV_TARGETS[CURRENT])
    if target != NAV_TARGETS[CURRENT]:
        # Persist before navigation so a reconnect / next page sees the most recent state.
        try:
            maybe_persist_state(force=True)
        except Exception:
            pass
        st.switch_page(target)

    # Persistent state
    if "explorer_results" not in st.session_state:
        st.session_state["explorer_results"] = []
    if "explorer_center" not in st.session_state:
        st.session_state["explorer_center"] = None


    # ------------------------------------------------------------
    # Sidebar (standard tabs) — Explorer controls live in Action tab
    # ------------------------------------------------------------
    run_clicked = {"value": False}

    # IMPORTANT:
    # Sidebar widgets are rendered only in the sidebar "Action" tab. When the user switches
    # away from Action, Streamlit may not render those widgets, so we decouple:
    #   - widget keys use: w_<name>
    #   - canonical persisted keys use: <name>
    def _w(k: str) -> str:
        return f"w_{k}"

    def _canonical(k: str, default: Any) -> Any:
        return st.session_state.get(k, default)

    def _sync(k: str) -> None:
        wk = _w(k)
        if wk in st.session_state:
            st.session_state[k] = st.session_state[wk]

    def _action_tab():
        # Button at the very top (same primary/green styling approach as Page 01)
        run_clicked["value"] = st.sidebar.button(
            "Search Stations",
            type="primary",
            use_container_width=True,
            key="explorer_search_btn",
            help="Runs a new search with the current settings and refreshes the map and results.",
        )

        st.sidebar.header("Where?")

        st.sidebar.text_input(
            "City / ZIP / Address",
            value=str(_canonical("explorer_location_query", "Tübingen")),
            key=_w("explorer_location_query"),
            help="Center location for the radius search (e.g., city, postal code, or full address).",
        )

        st.sidebar.slider(
            "Search radius (km)",
            min_value=1.0,
            max_value=25.0,
            value=float(_canonical("explorer_radius_km", 10.0)),
            step=0.5,
            key=_w("explorer_radius_km"),
            help="Search radius around the center point. Max 25 km (API limit).",
        )

        st.sidebar.header("What?")

        st.sidebar.selectbox(
            "Fuel type",
            options=["E5", "E10", "Diesel"],
            index=["E5", "E10", "Diesel"].index(str(_canonical("explorer_fuel_label", "E5"))),
            key=_w("explorer_fuel_label"),
            help="If a station has no price for the selected fuel, it will be treated as ‘no current price’ for this view.",
        )

        # Brand filter — fixed list (same as Trip Planner / Page 01)
        persisted = list(_canonical("brand_filter_selected", []))
        default_selected = [b for b in persisted if b in BRAND_OPTIONS]

        st.sidebar.multiselect(
            "Station brand",
            options=BRAND_OPTIONS,
            default=default_selected,
            key=_w("brand_filter_selected"),
            help=(
                "Fixed list of common brands in Germany (same as Trip Planner). "
                "If selected, only stations matching these brands are included. "
                "Brand matching includes common sub-names/aliases (e.g., AVIA XPress, TotalEnergies). "
                "When active, stations with unknown/missing brand are excluded."
            ),
        )

        st.sidebar.header("Advanced Settings")

        st.sidebar.slider(
            "Max number stations",
            min_value=10,
            max_value=200,
            value=int(_canonical("explorer_limit", 50)),
            step=10,
            key=_w("explorer_limit"),
            help="Limits the number of stations returned after filtering and ranking (cheapest first, then closest).",
        )

        st.sidebar.checkbox(
            "Only open stations & realtime prices",
            value=bool(_canonical("explorer_only_open", False)),
            key=_w("explorer_only_open"),
            help="Only stations that are currently open and have a current price for the selected fuel are shown.",
        )

        # Sync widget values back into canonical persisted keys
        for k in (
            "explorer_location_query",
            "explorer_fuel_label",
            "explorer_radius_km",
            "brand_filter_selected",
            "explorer_limit",
            "explorer_only_open",
        ):
            _sync(k)

    sidebar_view = render_sidebar_shell(action_renderer=_action_tab, help_renderer = _render_help_explorer)

    # Read values from session_state so they remain available even on non-Action tabs
    location_query = str(
        st.session_state.get(
            "explorer_location_query",
            st.session_state.get("explorer_last_query", "Tübingen"),
        )
    )
    fuel_label = str(st.session_state.get("explorer_fuel_label", "E5"))
    fuel_code = _fuel_label_to_code(fuel_label)
    radius_km = float(st.session_state.get("explorer_radius_km", 10.0))
    only_open = bool(st.session_state.get("explorer_only_open", False))
    use_realtime = bool(st.session_state.get("explorer_use_realtime", True))
    limit = int(st.session_state.get("explorer_limit", 50))
    brand_filter_selected = list(st.session_state.get("brand_filter_selected", []))

    run = bool(run_clicked["value"]) and (sidebar_view == "Action")

    # -----------------------------
    # Parameters hash (controls recompute warnings)
    # -----------------------------
    params = {
        "explorer_location_query": location_query,
        "fuel_label": fuel_label,
        "fuel_code": fuel_code,
        "radius_km": float(radius_km),
        "only_open": bool(only_open),
        "limit": int(limit),
        "use_realtime": bool(use_realtime),  # forced True on this page, but keep in hash for correctness
        "brand_filter_selected": sorted(brand_filter_selected),
    }
    params_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    have_cached = st.session_state.get("explorer_center") is not None
    cached_is_stale = have_cached and (st.session_state.get("explorer_params_hash") != params_hash)

    # If user changed settings, keep showing cached results but warn
    if not run and cached_is_stale:
        st.warning("Inputs changed. Showing previous results.")

    # Execute search
    if run:
        st.session_state["explorer_last_query"] = location_query

        try:
            payload = search_stations_nearby_list_api(
                StationExplorerInputs(
                    location_query=location_query,
                    fuel_code=fuel_code,
                    radius_km=float(radius_km),
                    limit=int(limit),
                    use_realtime=bool(use_realtime),
                    only_open=bool(only_open),
                    brand_filter_selected=brand_filter_selected,
                )
            )
            st.session_state["explorer_center"] = payload["center"]
            st.session_state["explorer_results"] = payload["stations"]
            st.session_state["explorer_params_hash"] = params_hash

            # Optional: set a "best" selection to reuse your Station Details flow
            cheapest_station = _pick_cheapest_station(payload["stations"], fuel_code=fuel_code)
            cheapest_uuid = _station_uuid(cheapest_station) if cheapest_station else None
            if cheapest_uuid:
                st.session_state["selected_station_uuid"] = cheapest_uuid

        except AppError as exc:
            st.error(exc.user_message)
            if exc.remediation:
                st.info(exc.remediation)
        except Exception as exc:
            st.error("Unexpected error. Please try again. If it persists, check logs.")
            st.caption(str(exc))

    center = st.session_state.get("explorer_center")
    stations: List[Dict[str, Any]] = st.session_state.get("explorer_results") or []

    if not center:
        st.markdown("### Welcome to the Station Explorer")

        st.markdown(
            """
    Use this page to **browse fuel stations around any location** in Germany and quickly compare **realtime prices**.

    ##### What you can do
    - Search stations around a **city, ZIP, or full address**
    - Adjust the **search radius**, choose the **fuel type**, and optionally filter by **brand** / **open stations**
    - Inspect results on an interactive **map** and in a **sortable table**

    ##### How it works (high level)
    - Your location input is **geocoded** to a center point.
    - The app queries for stations within the selected radius (including **current prices** and **open/closed**).
    - Filters (e.g., *Only open stations*) and sorting (price-first, then distance) are applied before rendering.

    ##### What you will see after running a search
    - A map with all stations in the result set
    - A highlighted “cheapest & closest” station (based on the selected fuel)
    - A results table with address, distance, current price, and open status
            """
        )

        st.info(
            "**Get started:** Open the sidebar, set your search location and filters -> click **Search Stations**."
        )
        return

    if not stations:
        st.warning("No stations found for this area/filter. Try increasing the radius or disabling 'Only open stations'.")
        return
    
    # Cheapest station (Explorer)
    cheapest_station = _pick_cheapest_station(stations, fuel_code=fuel_code)
    cheapest_uuid = _station_uuid(cheapest_station) if cheapest_station else None

    if cheapest_station:
        _render_cheapest_station_header(cheapest_station, fuel_code=fuel_code)
    else:
        st.info("No valid current prices found for the selected fuel.")

    # Determine best station for highlighting
    best_uuid = cheapest_uuid  # highlight cheapest station


    # ------------------------------------------------------------------
    # Map (Explorer) — Mapbox GL JS, same UX as Page 01 (toggle + legend)
    # ------------------------------------------------------------------
    if "map_style_mode" not in st.session_state:
        st.session_state["map_style_mode"] = "Standard"

    # ------------------------------------------------------------
    # Explorer metric cards (Page 04) — same visual design as Page 01
    # ------------------------------------------------------------
    price_key = f"price_current_{fuel_code}"

    def _has_current_price(s: Dict[str, Any]) -> bool:
        return _try_float((s or {}).get(price_key)) is not None

    stations_found = int(len(stations or []))
    stations_open = int(sum(1 for s in (stations or []) if (s or {}).get("is_open") is True))
    stations_with_price = int(sum(1 for s in (stations or []) if _has_current_price(s)))

    # Build HTML WITHOUT leading indentation/newline, otherwise Markdown may treat it as code.
    cards = [
        "<div class='metric-card'>"
        f"<div class='metric-label'>{_safe_text('Stations found')}</div>"
        f"<div class='metric-value'>{_safe_text(str(stations_found))}</div>"
        "</div>",
        "<div class='metric-card'>"
        f"<div class='metric-label'>{_safe_text('Stations open')}</div>"
        f"<div class='metric-value'>{_safe_text(str(stations_open))}</div>"
        "</div>",
        "<div class='metric-card'>"
        f"<div class='metric-label'>{_safe_text('Stations with current price')}</div>"
        f"<div class='metric-value'>{_safe_text(str(stations_with_price))}</div>"
        "</div>",
    ]

    html_block = (
        "<div class='explorer-metric-wrap'>"
        "<div class='metric-grid'>"
        + "".join(cards) +
        "</div></div>"
    )
    st.markdown(html_block, unsafe_allow_html=True)

    h_left, h_right = st.columns([0.70, 0.30], vertical_alignment="center")
    with h_left:
        st.subheader(
            "Stations Map",
            help="See legend below map.",
            anchor=False,
        )
        st.caption("Hover a station marker to show details.")

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

    # Marker semantics (Explorer):
    # - best: cheapest station (green)
    # - better: open + has current price for selected fuel (orange)
    # - other: closed OR missing current price (red)
    price_key = f"price_current_{fuel_code}"

    def _has_price(s: Dict[str, Any]) -> bool:
        return _try_float((s or {}).get(price_key)) is not None

    stations_for_map: List[Dict[str, Any]] = list(stations)

    # Determine ALL "best" stations for highlighting:
    # all stations that share the lowest current price for the selected fuel.
    best_uuids: set[str] = set()
    best_price: Optional[float] = None

    # We compare prices with a tiny epsilon to avoid float quirks.
    EPS = 1e-9

    for s in stations_for_map:
        u = _station_uuid(s)
        if not u:
            continue

        p = _try_float((s or {}).get(price_key))
        if p is None:
            continue

        if best_price is None or p < (best_price - EPS):
            best_price = p
            best_uuids = {u}
        elif abs(p - best_price) <= EPS:
            best_uuids.add(u)

    # Keep best_uuid as a single representative (closest to center) for code paths
    # that still expect exactly one best_station_uuid.
    if best_uuids:
        def _dist_km(uuid_: str) -> float:
            for ss in stations_for_map:
                if _station_uuid(ss) == uuid_:
                    d = _try_float((ss or {}).get("distance_km"))
                    return float(d) if d is not None else 0.0
            return 0.0

        best_uuid = min(best_uuids, key=_dist_km)

    # Assign marker categories (this is the only place where `continue` is used)
    marker_category_by_uuid: Dict[str, str] = {}
    for s in stations_for_map:
        u = _station_uuid(s)
        if not u:
            continue

        if u in best_uuids:
            marker_category_by_uuid[u] = "best"
            continue

        is_open = (s.get("is_open") is True)
        if is_open and _has_price(s):
            marker_category_by_uuid[u] = "better"
        else:
            marker_category_by_uuid[u] = "other"


    # Build kwargs (and add popup_variant only if the installed maps.py supports it)
    map_kwargs = dict(
        route_coords=[],                 # Explorer mode: no route
        stations=stations_for_map,
        best_station_uuid=best_uuid,
        via_full_coords=None,            # Explorer mode: no via route
        use_satellite=use_satellite,
        selected_station_uuid=st.session_state.get("selected_station_uuid"),
        marker_category_by_uuid=marker_category_by_uuid,
        height_px=560,
        fuel_code=fuel_code,             # enables "current price" in hover popup
    )

    sig = inspect.signature(maps_mod.create_mapbox_gl_html)
    if "popup_variant" in sig.parameters:
        map_kwargs["popup_variant"] = "explorer"

    map_html = maps_mod.create_mapbox_gl_html(**map_kwargs)

    components.html(map_html, height=560, scrolling=False)

    # Legend (Explorer-specific)
    if bool(only_open):
        st.markdown(
            """
            <div class="explorer-map-legend-wrap">
            <div class="map-legend">
                <div class="legend-item"><span class="legend-dot best"></span><span>Cheapest station(s)</span></div>
                <div class="legend-item"><span class="legend-dot better"></span><span>Other open stations (with current price)</span></div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="explorer-map-legend-wrap">
            <div class="map-legend">
                <div class="legend-item"><span class="legend-dot best"></span><span>Cheapest station(s)</span></div>
                <div class="legend-item"><span class="legend-dot better"></span><span>Open stations (with current price)</span></div>
                <div class="legend-item"><span class="legend-dot other"></span><span>Closed or missing current price</span></div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ------------------------------------------------------------------
    # Stations dropdown (ranked) — placed between Cheapest block and Map
    # ------------------------------------------------------------------
    st.markdown(f"#### Stations Selector")
    st.markdown(
        "<div class='explorer-selector-subtitle'>Find out more in Station Explorer.</div>",
        unsafe_allow_html=True,
    )

    def _station_primary_name(s: Dict[str, Any]) -> str:
        """Brand preferred; fallback to station name."""
        station_name = (s.get("tk_name") or s.get("osm_name") or s.get("name") or "Unknown")
        brand = s.get("brand")
        return str(brand).strip() if (brand and str(brand).strip()) else str(station_name).strip()

    def _station_address_line(s: Dict[str, Any]) -> str:
        """
        Build a compact address string. If structured fields are missing, fall back to a
        'postcode city' style string (or city only as last resort).
        """
        def _first(*keys: str) -> str:
            for k in keys:
                v = s.get(k)
                if v is None:
                    continue
                t = str(v).strip()
                if t:
                    return t
            return ""

        street = _first("street", "addr_street", "addr:street")
        house = _first("houseNumber", "house_number", "housenumber", "addr_housenumber", "addr:housenumber")
        postcode = _first("postCode", "postcode", "zip", "postal_code", "postalCode", "addr_postcode", "addr:postcode")
        city = _first("city", "place", "town", "village")

        line1 = " ".join([p for p in [street, house] if p]).strip()
        line2 = " ".join([p for p in [postcode, city] if p]).strip()

        # prefer full street line; otherwise show postcode+city; otherwise city
        if line1 and line2:
            return f"{line1}, {line2}"
        if line1:
            return line1
        if line2:
            return line2
        return city or ""

    def _station_price_value(s: Dict[str, Any]) -> float:
        v = _try_float(s.get(f"price_current_{fuel_code}"))
        return v if v is not None else float("inf")

    # Rank: cheapest first (then closer as tie-break)
    ranked = sorted(
        stations,
        key=lambda s: (_station_price_value(s), float(s.get("distance_km", 1e9))),
    )

    # Build dropdown options
    options: List[Dict[str, Any]] = []
    labels: List[str] = []
    for i, s in enumerate(ranked, start=1):
        u = _station_uuid(s)
        if not u:
            continue
        primary = _station_primary_name(s)
        addr = _station_address_line(s)
        labels.append(f"{i}. {primary} — {addr}" if addr else f"{i}. {primary}")
        options.append(s)

    if not options:
        st.info("No selectable stations available.")
    else:
        # Pick default: currently selected uuid (if present), otherwise the first ranked station
        selected_uuid = st.session_state.get("selected_station_uuid")
        default_index = 0
        if selected_uuid:
            for idx, s in enumerate(options):
                if _station_uuid(s) == selected_uuid:
                    default_index = idx
                    break

        chosen_label = st.selectbox(
            "Station",
            options=labels,
            index=default_index,
            key="explorer_station_dropdown",
            label_visibility="collapsed",
            help="Ranked by current price for the selected fuel (tie-break: closer to center).",
        )

        chosen_idx = labels.index(chosen_label)
        chosen_station = options[chosen_idx]
        chosen_uuid = _station_uuid(chosen_station)

        # Optional: show a compact line under the dropdown (remove if you want it cleaner)
        st.markdown(
            f"<div class='explorer-selector-price'>Current {fuel_code.upper()}: "
            f"{_safe_text(_format_price(chosen_station.get(f'price_current_{fuel_code}')))}</div>",
            unsafe_allow_html=True,
        )

        if st.button("Open in Station Details", 
                     key="explorer_open_station_details_btn", 
                     use_container_width=True, 
                     help="Opens the selected station in Page 03 with the station preselected.",):
            st.session_state["selected_station_uuid"] = chosen_uuid
            st.session_state["selected_station_data"] = chosen_station
            try:
                maybe_persist_state(force=True)
            except Exception:
                pass
            st.switch_page("pages/03_station_details.py")

    # ------------------------------------------------------------------
    # Results table (final section on this page)
    # ------------------------------------------------------------------
    st.markdown("#### Results")
    st.markdown(
        "<div class='explorer-selector-subtitle'>All stations returned by the current search (after filters and result limit).</div>",
        unsafe_allow_html=True,
    )

    df = _build_results_table(stations, fuel_code=fuel_code)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Persist state (best-effort) and stop rendering here as requested
    maybe_persist_state()
    return


if __name__ == "__main__":
    main()
