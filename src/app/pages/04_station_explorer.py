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
    pick_best_station_uuid,
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


def main() -> None:
    st.set_page_config(page_title="Station Explorer", layout="wide")
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

        st.sidebar.text_input(
            "City / ZIP / Address",
            value=st.session_state.get("explorer_location_query", st.session_state.get("explorer_last_query", "Tübingen")),
            help="Examples: 'Tübingen', '72074 Tübingen', 'Wilhelmstraße 7, Tübingen'",
            key="explorer_location_query",
        )

        st.sidebar.selectbox(
            "Fuel type",
            options=["E5", "E10", "Diesel"],
            index=["E5", "E10", "Diesel"].index(st.session_state.get("explorer_fuel_label", "E5"))
            if st.session_state.get("explorer_fuel_label", "E5") in ["E5", "E10", "Diesel"] else 0,
            key="explorer_fuel_label",
        )

        st.sidebar.slider(
            "Search radius (km)",
            min_value=1.0,
            max_value=25.0,
            value=float(st.session_state.get("explorer_radius_km", 10.0)),
            step=1.0,
            key="explorer_radius_km",
        )

        st.sidebar.checkbox(
            "Only show open stations (requires realtime)",
            value=bool(st.session_state.get("explorer_only_open", False)),
            key="explorer_only_open",
        )

        st.sidebar.checkbox(
            "Use realtime prices (Tankerkönig)",
            value=bool(st.session_state.get("explorer_use_realtime", True)),
            key="explorer_use_realtime",
        )

        st.sidebar.slider(
            "Max stations to load (performance)",
            min_value=50,
            max_value=400,
            value=int(st.session_state.get("explorer_limit", 200)),
            step=50,
            key="explorer_limit",
        )

        st.sidebar.checkbox(
            "Debug mode",
            value=bool(st.session_state.get("debug_mode", False)),
            key="debug_mode",  # shared key across pages
        )

        run_clicked["value"] = st.sidebar.button(
            "Search stations",
            use_container_width=True,
            key="explorer_search_btn",
        )

    sidebar_view = render_sidebar_shell(action_renderer=_action_tab)

    # Read values from session_state so they remain available even on non-Action tabs
    location_query = str(st.session_state.get("explorer_location_query", st.session_state.get("explorer_last_query", "Tübingen")))
    fuel_label = str(st.session_state.get("explorer_fuel_label", "E5"))
    fuel_code = _fuel_label_to_code(fuel_label)
    radius_km = float(st.session_state.get("explorer_radius_km", 10.0))
    only_open = bool(st.session_state.get("explorer_only_open", False))
    use_realtime = bool(st.session_state.get("explorer_use_realtime", True))
    limit = int(st.session_state.get("explorer_limit", 200))
    debug_mode = bool(st.session_state.get("debug_mode", False))

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
            best_uuid = pick_best_station_uuid(payload["stations"], fuel_code=fuel_code)
            if best_uuid:
                st.session_state["selected_station_uuid"] = best_uuid

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

    st.markdown(
        f"**Search center:** {_safe_text(center.get('label', ''))}  \n"
        f"Radius: **{center.get('radius_km', 0)} km** · Fuel: **{fuel_code.upper()}**"
    )

    if not stations:
        st.warning("No stations found for this area/filter. Try increasing the radius or disabling 'Only open stations'.")
        return

    # Determine best station for highlighting
    best_uuid = pick_best_station_uuid(stations, fuel_code=fuel_code)

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


if __name__ == "__main__":
    main()
