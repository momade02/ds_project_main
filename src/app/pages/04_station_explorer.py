# src/app/pages/04_station_explorer.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui.styles import apply_app_css

from ui.sidebar import render_sidebar_shell

from config.settings import load_env_once

load_env_once()

from config.settings import ensure_persisted_state_defaults
from services.session_store import init_session_context, restore_persisted_state, maybe_persist_state

# ---------------------------------------------------------------------
# Import bootstrap (same pattern as your other pages)
# ---------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]       # .../src/app
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app_errors import AppError, ConfigError, ExternalServiceError, DataAccessError

from src.app.services.station_explorer import (
    StationExplorerInputs,
    search_stations_nearby,
)

from src.app.ui.maps import (
    _supports_pydeck_selections,
    _create_map_visualization,
)

from ui.formatting import (
    _station_uuid,
    _safe_text,
    _format_price,
)


def _fuel_label_to_code(label: str) -> str:
    m = {"E5": "e5", "E10": "e10", "Diesel": "diesel"}
    return m.get(label, "e5")


def _build_results_table(stations: List[Dict[str, Any]], fuel_code: str) -> pd.DataFrame:
    price_key = f"price_current_{fuel_code}"
    rows = []
    for s in stations:
        rows.append(
            {
                "Station": s.get("tk_name") or s.get("osm_name") or "Unknown",
                "Brand": s.get("brand") or "",
                "City": s.get("city") or "",
                "Distance (km)": round(float(s.get("distance_km", 0.0)), 2),
                f"Current {fuel_code.upper()}": _format_price(s.get(price_key)),
                "Open": "Yes" if s.get("is_open") is True else ("No" if s.get("is_open") is False else "—"),
                "UUID": _station_uuid(s) or "",
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
    Ignores stations with missing/non-numeric prices.
    """
    price_key = f"price_current_{fuel_code}"

    best_s: Optional[Dict[str, Any]] = None
    best_p: Optional[float] = None

    for s in stations or []:
        p = _try_float((s or {}).get(price_key))
        if p is None:
            continue
        if best_p is None or p < best_p:
            best_p = p
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

    label_html = _safe_text("Cheapest station found:")
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
        st.session_state["map_style_mode"] = _preserve_map_style_mode  # <-- ADD THIS

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

    def _action_tab() -> None:
        st.sidebar.header("Explorer Settings")

        # Hard-force settings for this page (no sidebar toggles here)
        st.session_state["explorer_use_realtime"] = True
        st.session_state["debug_mode"] = True

        # IMPORTANT: These widgets live only in the sidebar "Action" tab. When the user switches
        # the sidebar segmented-control away from Action, Streamlit may garbage-collect widget state
        # for keys whose widgets are not rendered. To keep the latest values stable, we decouple:
        #   - widgets use keys: w_<name>
        #   - canonical persisted values use keys: <name>
        def _w(k: str) -> str:
            return f"w_{k}"

        def _canonical(k: str, default: Any) -> Any:
            return st.session_state.get(k, default)

        def _sync(k: str) -> None:
            wk = _w(k)
            if wk in st.session_state:
                st.session_state[k] = st.session_state[wk]

        st.sidebar.text_input(
            "City / ZIP / Address",
            value=str(_canonical("explorer_location_query", _canonical("explorer_last_query", "Tübingen"))),
            help="Examples: 'Tübingen', '72074 Tübingen', 'Wilhelmstraße 7, Tübingen'",
            key=_w("explorer_location_query"),
        )

        fuel_default = str(_canonical("explorer_fuel_label", "E5"))
        st.sidebar.selectbox(
            "Fuel type",
            options=["E5", "E10", "Diesel"],
            index=["E5", "E10", "Diesel"].index(fuel_default) if fuel_default in ["E5", "E10", "Diesel"] else 0,
            key=_w("explorer_fuel_label"),
        )

        st.sidebar.slider(
            "Search radius (km)",
            min_value=1.0,
            max_value=25.0,
            value=float(_canonical("explorer_radius_km", 10.0)),
            step=1.0,
            key=_w("explorer_radius_km"),
        )

        st.sidebar.checkbox(
            "Only show open stations (requires realtime)",
            value=bool(_canonical("explorer_only_open", False)),
            key=_w("explorer_only_open"),
        )

        st.sidebar.slider(
            "Max stations to load (performance)",
            min_value=50,
            max_value=400,
            value=int(_canonical("explorer_limit", 200)),
            step=50,
            key=_w("explorer_limit"),
        )

        # Sync widget values back into canonical persisted keys
        for k in (
            "explorer_location_query",
            "explorer_fuel_label",
            "explorer_radius_km",
            "explorer_only_open",
            "explorer_limit",
        ):
            _sync(k)

        run_clicked["value"] = st.sidebar.button(
            "Search stations",
            use_container_width=True,
            key="explorer_search_btn",
        )

    sidebar_view = render_sidebar_shell(action_renderer=_action_tab)

    # Read values from session_state so they remain available even on non-Action tabs
    use_realtime = True
    debug_mode = True
    location_query = str(st.session_state.get("explorer_location_query", st.session_state.get("explorer_last_query", "Tübingen")))
    fuel_label = str(st.session_state.get("explorer_fuel_label", "E5"))
    fuel_code = _fuel_label_to_code(fuel_label)
    radius_km = float(st.session_state.get("explorer_radius_km", 10.0))
    only_open = bool(st.session_state.get("explorer_only_open", False))
    use_realtime = bool(st.session_state.get("explorer_use_realtime", True))
    limit = int(st.session_state.get("explorer_limit", 200))
    debug_mode = bool(st.session_state.get("debug_mode", True))

    run = bool(run_clicked["value"]) and (sidebar_view == "Action")

    # Execute search
    if run:
        st.session_state["explorer_last_query"] = location_query

        try:
            payload = search_stations_nearby(
                StationExplorerInputs(
                    location_query=location_query,
                    fuel_code=fuel_code,
                    radius_km=float(radius_km),
                    limit=int(limit),
                    use_realtime=bool(use_realtime),
                    only_open=bool(only_open),
                )
            )
            st.session_state["explorer_center"] = payload["center"]
            st.session_state["explorer_results"] = payload["stations"]

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
        st.info("Enter a location and click **Search stations**.")
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
    # Stations dropdown (ranked) — placed between Cheapest block and Map
    # ------------------------------------------------------------------
    st.markdown(f"### Stations in {int(round(radius_km))} km Radius")
    st.caption("Choose station and find out more in Station Explorer.")

    def _station_display_name(s: Dict[str, Any]) -> str:
        return str(s.get("tk_name") or s.get("osm_name") or s.get("name") or "Unknown").strip()

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
        labels.append(f"{i}. {_station_display_name(s)}")
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
        )

        chosen_idx = labels.index(chosen_label)
        chosen_station = options[chosen_idx]
        chosen_uuid = _station_uuid(chosen_station)

        # Optional: show a compact line under the dropdown (remove if you want it cleaner)
        st.caption(f"Current {fuel_code.upper()}: {_format_price(chosen_station.get(f'price_current_{fuel_code}'))}")

        if st.button("Open in Station Details", key=f"open_station_details_{chosen_uuid}", use_container_width=True):
            st.session_state["selected_station_uuid"] = chosen_uuid
            st.session_state["selected_station_data"] = chosen_station
            try:
                maybe_persist_state(force=True)
            except Exception:
                pass
            st.switch_page("pages/03_station_details.py")

    # Map (Explorer mode: empty route coords, map auto-fits to station bounds)
    st.markdown("### Map")
    deck = _create_map_visualization(
        route_coords=[],
        stations=stations,
        best_station_uuid=best_uuid,
        via_full_coords=None,
        zoom_level=7.5,
        fuel_code=fuel_code,
        selected_station_uuid=st.session_state.get("selected_station_uuid"),
    )

    st.caption("Hover for quick info. Click a marker to select a station.")

    selected_uuid_from_event: Optional[str] = None
    if _supports_pydeck_selections():
        event = st.pydeck_chart(
            deck,
            on_select="rerun",
            selection_mode="single-object",
            key="explorer_map",
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
        st.caption("Note: Your Streamlit version does not expose pydeck click selections.")

    if selected_uuid_from_event:
        st.session_state["selected_station_uuid"] = selected_uuid_from_event

    # Results table + selection UX
    st.markdown("---")
    st.markdown("### Results")

    df = _build_results_table(stations, fuel_code=fuel_code)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Select station by dropdown (reliable across Streamlit versions)
    uuid_list = df["UUID"].tolist()
    label_list = [
        f"{row['Station']} ({row['Brand']}) · {row[f'Current {fuel_code.upper()}']} · {row['Distance (km)']} km"
        for _, row in df.iterrows()
    ]

    # Default selection: clicked UUID, else best UUID, else first
    default_uuid = st.session_state.get("selected_station_uuid") or best_uuid or (uuid_list[0] if uuid_list else "")
    try:
        default_idx = uuid_list.index(default_uuid) if default_uuid in uuid_list else 0
    except Exception:
        default_idx = 0

    chosen_label = st.selectbox("Select a station", options=label_list, index=default_idx)
    chosen_idx = label_list.index(chosen_label)
    chosen_uuid = uuid_list[chosen_idx]

    # Resolve chosen station dict
    uuid_to_station: Dict[str, Dict[str, Any]] = {}
    for s in stations:
        u = _station_uuid(s)
        if u:
            uuid_to_station[u] = s
    chosen_station = uuid_to_station.get(chosen_uuid)

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        if st.button("Set selection", use_container_width=True):
            st.session_state["selected_station_uuid"] = chosen_uuid
            st.session_state["selected_station_data"] = chosen_station
            st.success("Selected station saved for Station Details.")
    with col_b:
        if st.button("Open Station Details", use_container_width=True):
            st.session_state["selected_station_uuid"] = chosen_uuid
            st.session_state["selected_station_data"] = chosen_station
            st.switch_page("pages/03_station_details.py")
    with col_c:
        st.caption("Station Details provides the full deep dive (history, hourly pattern, comparisons).")

    # Lightweight selected summary
    st.markdown("---")
    st.markdown("### Selected station")
    if chosen_station:
        price_key = f"price_current_{fuel_code}"
        st.write(
            f"**{chosen_station.get('tk_name') or chosen_station.get('osm_name') or 'Unknown'}**"
            + (f" ({chosen_station.get('brand')})" if chosen_station.get("brand") else "")
        )
        st.write(
            {
                "UUID": chosen_uuid,
                "Current price": _format_price(chosen_station.get(price_key)),
                "Distance (km)": round(float(chosen_station.get("distance_km", 0.0)), 2),
                "Open": chosen_station.get("is_open"),
            }
        )
    else:
        st.info("No station object resolved. Try selecting another station from the list.")

    if debug_mode:
        st.markdown("---")
        st.markdown("DEBUG: explorer_center / counts")
        st.json({"center": center, "stations_returned": len(stations), "best_uuid": best_uuid})

    maybe_persist_state()


if __name__ == "__main__":
    main()
