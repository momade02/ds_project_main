# src/app/ui/sidebar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from ui.formatting import _fuel_label_to_code, _station_uuid, _safe_text


# =============================================================================
# Existing state model (Trip Planner sidebar)
# =============================================================================

@dataclass(frozen=True)
class SidebarState:
    # Which sidebar tab is selected
    view: str

    # Action view
    run_clicked: bool

    # Route inputs
    start_locality: str
    end_locality: str
    start_address: str
    end_address: str

    # Fuel inputs
    fuel_label: str
    fuel_code: str

    # Economics inputs
    use_economics: bool
    litres_to_refuel: float
    consumption_l_per_100km: float
    value_of_time_eur_per_hour: float
    max_detour_km: float
    max_detour_min: float
    min_net_saving_eur: float

    # Diagnostics
    debug_mode: bool

    # Advanced Settings
    filter_closed_at_eta: bool

    # Brand whitelist (canonical brand labels)
    brand_filter_selected: list[str]


def _ss(key: str, default: Any) -> Any:
    """Read session_state with a default; used to keep behavior stable across tabs."""
    return st.session_state.get(key, default)


def _coerce_sidebar_view(value: Any) -> str:
    """
    Ensure sidebar_view is always one of: Action/Help/Settings/Profile.
    Backward compatibility:
      - "Status" (old) -> "Help"
      - "Info"   (old) -> "Help"
    """
    allowed = {"Action", "Help", "Settings", "Profile"}
    if value in ("Status", "Info"):
        return "Help"
    if value in allowed:
        return value
    return "Action"


def _read_cached_values_for_non_action() -> SidebarState:
    """
    When not in Action tab, we still read all values from session_state so the rest of the
    app (hashing/cached display) behaves exactly like before.
    """
    fuel_label = str(_ss("fuel_label", "E5"))
    if fuel_label not in {"E5", "E10", "Diesel"}:
        fuel_label = "E5"
    fuel_code = _fuel_label_to_code(fuel_label)

    return SidebarState(
        view=str(_ss("sidebar_view", "Action")),
        run_clicked=False,
        start_locality=str(_ss("start_locality", "Tübingen")),
        end_locality=str(_ss("end_locality", "Sindelfingen")),
        start_address="",
        end_address="",
        fuel_label=fuel_label,
        fuel_code=fuel_code,
        use_economics=bool(_ss("use_economics", True)),
        litres_to_refuel=float(_ss("litres_to_refuel", 40.0)),
        consumption_l_per_100km=float(_ss("consumption_l_per_100km", 7.0)),
        value_of_time_eur_per_hour=float(_ss("value_of_time_eur_per_hour", 0.0)),
        max_detour_km=float(_ss("max_detour_km", 5.0)),
        max_detour_min=float(_ss("max_detour_min", 10.0)),
        min_net_saving_eur=float(_ss("min_net_saving_eur", 0.0)),
        debug_mode=bool(_ss("debug_mode", True)),
        filter_closed_at_eta=bool(_ss("filter_closed_at_eta", True)),
        brand_filter_selected=list(_ss("brand_filter_selected", [])),
    )


def _render_trip_planner_action() -> SidebarState:
    """
    Default Action tab renderer for the Trip Planner (streamlit_app.py).
    This is intentionally a near-copy of your current widgets/keys so CSS + layout remain unchanged.
    """
    run_clicked = st.sidebar.button(
        "Run recommender",
        use_container_width=True,
        key="run_recommender_btn",
    )

    st.sidebar.markdown("### Route")

    start_locality = st.sidebar.text_input(
        "Start",
        value=str(_ss("start_locality", "Tübingen")),
        key="start_locality",
        label_visibility="collapsed",
        placeholder="Start: city or full address",
    )

    # Right-side connector dots between the two inputs (visual only)
    st.sidebar.markdown(
        """
        <div class="route-dots-right" aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    end_locality = st.sidebar.text_input(
        "Destination",
        value=str(_ss("end_locality", "Sindelfingen")),
        key="end_locality",
        label_visibility="collapsed",
        placeholder="Destination: city or full address",
    )

    # (You currently keep these empty; preserve behavior)
    start_address = ""
    end_address = ""

    st.sidebar.subheader("Fuel type")

    fuel_options = ["E5", "E10", "Diesel"]
    fuel_label_default = str(_ss("fuel_label", "E5"))
    if fuel_label_default not in fuel_options:
        fuel_label_default = "E5"

    fuel_label = st.sidebar.selectbox(
        "Fuel type",
        options=fuel_options,
        index=fuel_options.index(fuel_label_default),
        label_visibility="collapsed",
        key="fuel_label",
    )
    fuel_code = _fuel_label_to_code(fuel_label)

    st.sidebar.markdown(
        "### Detour Economics",
        help=(
            "Decide how detours are evaluated. The recommender combines your detour limits "
            "(extra distance/time) with detour costs (extra fuel and optional value of time) "
            "and the expected price advantage to compute a net saving for each candidate station."
        ),
    )

    use_economics = st.sidebar.checkbox(
        "Economics-based decision",
        value=bool(_ss("use_economics", True)),
        key="use_economics",
    )

    litres_to_refuel = st.sidebar.number_input(
        "Litres to refuel",
        min_value=1.0,
        max_value=1000.0,
        value=float(_ss("litres_to_refuel", 40.0)),
        step=1.0,
        key="litres_to_refuel",
    )

    consumption_l_per_100km = st.sidebar.number_input(
        "Car consumption (L/100 km)",
        min_value=0.0,
        max_value=30.0,
        value=float(_ss("consumption_l_per_100km", 7.0)),
        step=0.5,
        key="consumption_l_per_100km",
    )

    value_of_time_eur_per_hour = st.sidebar.number_input(
        "Value of time (€/hour)",
        min_value=0.0,
        max_value=200.0,
        value=float(_ss("value_of_time_eur_per_hour", 0.0)),
        step=5.0,
        key="value_of_time_eur_per_hour",
    )

    max_detour_km = st.sidebar.number_input(
        "Maximum extra distance (km)",
        min_value=0.5,
        max_value=200.0,
        value=float(_ss("max_detour_km", 5.0)),
        step=0.5,
        key="max_detour_km",
    )

    max_detour_min = st.sidebar.number_input(
        "Maximum extra time (min)",
        min_value=1.0,
        max_value=240.0,
        value=float(_ss("max_detour_min", 10.0)),
        step=1.0,
        key="max_detour_min",
        help=(
            "The maximum additional travel time you are willing to accept for a detour compared to the baseline route. "
            "Stations requiring more extra time are excluded (hard constraint)."
        ),
    )

    min_net_saving_eur = st.sidebar.number_input(
        "Min net saving (€)",
        min_value=0.0,
        max_value=100.0,
        value=float(_ss("min_net_saving_eur", 0.0)),
        step=0.5,
        key="min_net_saving_eur",
        help=(
            "Minimum required net benefit for accepting a detour. Net saving = fuel price saving − "
            "detour fuel cost − optional time cost. Set to 0 to allow any positive or zero net saving."
        ),
    )

    st.sidebar.markdown(
        "### Advanced Settings",
        help=(
            "Make sure that only stations are considered that are open at ETA. "
            "If you have a fuel card of a common brand, filter by brand to only see relevant stations."
        ),
    )

    filter_closed_at_eta = st.sidebar.checkbox(
        "Stations open at ETA",
        value=bool(_ss("filter_closed_at_eta", True)),
        key="filter_closed_at_eta",
        help=(
            "If enabled, stations are filtered out when Google indicates they are closed at the "
            "estimated time of arrival (ETA) at the station. If opening hours are unavailable, the station "
            "is kept."
        ),
    )

    brand_options = ["ARAL", "AVIA", "AGIP ENI", "Shell", "Total", "ESSO", "JET", "ORLEN", "HEM", "OMV"]

    brand_filter_selected = st.sidebar.multiselect(
        "Filter by brand",
        options=brand_options,
        default=list(_ss("brand_filter_selected", [])),
        key="brand_filter_selected",
        help=(
            "10 most common brands in Germany. "
            "Only stations of selected brands are considered. "
            "When active, stations with unknown/missing brand are excluded."
        ),
    )

    return SidebarState(
        view="Action",
        run_clicked=bool(run_clicked),
        start_locality=str(start_locality),
        end_locality=str(end_locality),
        start_address=start_address,
        end_address=end_address,
        fuel_label=str(fuel_label),
        fuel_code=str(fuel_code),
        use_economics=bool(use_economics),
        litres_to_refuel=float(litres_to_refuel),
        consumption_l_per_100km=float(consumption_l_per_100km),
        value_of_time_eur_per_hour=float(value_of_time_eur_per_hour),
        max_detour_km=float(max_detour_km),
        max_detour_min=float(max_detour_min),
        min_net_saving_eur=float(min_net_saving_eur),
        debug_mode=bool(_ss("debug_mode", True)),
        filter_closed_at_eta=bool(filter_closed_at_eta),
        brand_filter_selected=list(brand_filter_selected),
    )


def render_sidebar(
    *,
    action_renderer: Optional[Callable[[], SidebarState]] = None,
    help_placeholder: str = "Placeholder: Help (content will be added later).",
    settings_placeholder: str = "Placeholder: Settings (content will be added later).",
    profile_placeholder: str = "Placeholder: Profile (content will be added later).",
) -> SidebarState:
    """
    Standardized sidebar with 4 sections:
      - Action
      - Help
      - Settings
      - Profile

    For now, Help/Settings/Profile are placeholders.
    Later, each page can pass its own action_renderer (and optionally its own placeholders).
    """
    if "sidebar_view" not in st.session_state:
        st.session_state["sidebar_view"] = "Action"
    st.session_state["sidebar_view"] = _coerce_sidebar_view(st.session_state.get("sidebar_view"))

    view = st.sidebar.segmented_control(
        label="Sidebar sections",
        options=["Action", "Help", "Settings", "Profile"],
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="sidebar_view",
    )

    # ACTION
    if view == "Action":
        renderer = action_renderer or _render_trip_planner_action
        return renderer()

    # NON-ACTION: keep downstream behavior stable by reading cached values
    state = _read_cached_values_for_non_action()

    if view == "Help":
        st.sidebar.info(help_placeholder)
        return SidebarState(**{**state.__dict__, "view": "Help"})
    if view == "Settings":
        st.sidebar.info(settings_placeholder)
        return SidebarState(**{**state.__dict__, "view": "Settings"})

    st.sidebar.info(profile_placeholder)
    return SidebarState(**{**state.__dict__, "view": "Profile"})


# =============================================================================
# Shared sidebar shell (Pages 2–4)
# =============================================================================

def render_sidebar_shell(
    *,
    action_renderer: Optional[Callable[[], None]] = None,
    action_placeholder: str = "Placeholder: Action",
    help_placeholder: str = "Placeholder: Help (will be added later).",
    settings_placeholder: str = "Placeholder: Settings (will be added later).",
    profile_placeholder: str = "Placeholder: Profile (will be added later).",
) -> str:
    """
    Shared 4-tab sidebar shell for Pages 2–4 (and optionally others):
      - Action
      - Help
      - Settings
      - Profile

    Action tab renders page-specific controls via action_renderer.
    Other tabs are placeholders for now.
    """
    if "sidebar_view" not in st.session_state:
        st.session_state["sidebar_view"] = "Action"
    st.session_state["sidebar_view"] = _coerce_sidebar_view(st.session_state.get("sidebar_view"))

    view = st.sidebar.segmented_control(
        label="Sidebar sections",
        options=["Action", "Help", "Settings", "Profile"],
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="sidebar_view",
    )

    if view == "Action":
        if action_renderer is not None:
            action_renderer()
        else:
            st.sidebar.info(action_placeholder)
        return "Action"

    if view == "Help":
        st.sidebar.info(help_placeholder)
        return "Help"

    if view == "Settings":
        st.sidebar.info(settings_placeholder)
        return "Settings"

    st.sidebar.info(profile_placeholder)
    return "Profile"


# =============================================================================
# NEW: Station Details sidebar helpers (Page 03 control plane)
# =============================================================================

@dataclass(frozen=True)
class StationSelection:
    """
    Result object returned by render_station_selector.

    This is intentionally minimal and backwards compatible with your current
    session_state-based navigation (selected_station_uuid / selected_station_data).
    """
    station_uuid: str
    station: Optional[Dict[str, Any]]
    source: str  # "route" | "explorer"
    label: str


def _station_display_name(station: Dict[str, Any]) -> str:
    name = station.get("tk_name") or station.get("osm_name") or station.get("name") or "Unknown"
    return _safe_text(str(name))


def _station_display_city(station: Dict[str, Any]) -> str:
    city = (station.get("city") or "").strip()
    return _safe_text(str(city))


def _station_display_brand(station: Dict[str, Any]) -> str:
    brand = (station.get("brand") or "").strip()
    return _safe_text(str(brand))


def _build_station_label(
    station: Dict[str, Any],
    *,
    tag: str,
    rank_index: Optional[int] = None,
    include_city: bool = True,
    include_brand: bool = True,
) -> str:
    """
    Human-friendly label used in the station selector. Keep this stable; it becomes muscle memory.
    """
    name = _station_display_name(station)
    brand = _station_display_brand(station)
    city = _station_display_city(station)

    prefix = f"#{rank_index} " if rank_index is not None else ""
    parts: List[str] = [f"{prefix}{name}"]

    if include_brand and brand:
        parts.append(f"({brand})")

    if include_city and city:
        parts.append(f"· {city.upper()}")

    parts.append(f"[{tag}]")
    return " ".join(parts).strip()


def _resolve_uuid_to_station_map(
    ranked: List[Dict[str, Any]],
    stations: List[Dict[str, Any]],
    explorer_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build a uuid -> station dict mapping across all sources, with a preference order:
      ranked > stations > explorer_results
    so that the selector returns the richest station context where possible.
    """
    m: Dict[str, Dict[str, Any]] = {}

    for s in explorer_results or []:
        u = _station_uuid(s)
        if u:
            m[u] = s

    for s in stations or []:
        u = _station_uuid(s)
        if u:
            m[u] = s

    for s in ranked or []:
        u = _station_uuid(s)
        if u:
            m[u] = s

    return m


def render_station_selector(
    *,
    last_run: Optional[Dict[str, Any]] = None,
    explorer_results: Optional[List[Dict[str, Any]]] = None,
    # session-state keys (kept as defaults to preserve your cross-page wiring)
    selected_uuid_key: str = "selected_station_uuid",
    selected_data_key: str = "selected_station_data",
    selected_source_key: str = "selected_station_source",
    # widget keys (namespaced to avoid collisions)
    widget_key_prefix: str = "station_details",
    # sizing (conservative defaults)
    max_ranked: int = 20,
    max_excluded: int = 50,
    max_explorer: int = 200,
) -> StationSelection:
    """
    Sidebar station selection control for Page 03 ("Station Details & Analysis").

    What it does:
      - Lets the user choose a station source: latest route run vs station explorer
      - Presents a single selector (selectbox) for stations from the chosen source
      - Writes the selection into session_state:
          selected_station_uuid
          selected_station_data (when resolvable)
          selected_station_source (route|explorer)

    This is additive and safe:
      - If last_run/explorer_results are missing, it degrades gracefully.
      - It never assumes fields beyond what your pages already use (_station_uuid + tk_name/osm_name/city/brand).

    Returns:
      StationSelection (uuid, station dict, source, label)
    """
    last_run = last_run or {}
    ranked: List[Dict[str, Any]] = list(last_run.get("ranked") or [])
    stations: List[Dict[str, Any]] = list(last_run.get("stations") or [])
    explorer_results = list(explorer_results or [])

    # Determine which sources are actually available
    has_route = bool(ranked or stations)
    has_explorer = bool(explorer_results)

    available_sources: List[Tuple[str, str]] = []
    if has_route:
        available_sources.append(("route", "From latest route run"))
    if has_explorer:
        available_sources.append(("explorer", "From station explorer"))

    # Current selection (from session_state)
    current_uuid = str(st.session_state.get(selected_uuid_key) or "")
    current_station = st.session_state.get(selected_data_key)
    current_source = str(st.session_state.get(selected_source_key) or "")

    # If no explicit source, infer a default once (non-brittle; does not rely on detour keys)
    if current_source not in {"route", "explorer"}:
        inferred = None
        if current_uuid and has_explorer:
            ex_uuids = {_station_uuid(s) for s in explorer_results if _station_uuid(s)}
            if current_uuid in ex_uuids:
                inferred = "explorer"
        if inferred is None and has_route:
            inferred = "route"
        if inferred is None and has_explorer:
            inferred = "explorer"
        current_source = inferred or "route"

    # Render source selector (only if both exist)
    if len(available_sources) >= 2:
        # index based on inferred / stored source
        src_ids = [sid for sid, _ in available_sources]
        default_idx = src_ids.index(current_source) if current_source in src_ids else 0
        chosen_source_label = st.sidebar.radio(
            "Station source",
            options=[lbl for _, lbl in available_sources],
            index=default_idx,
            key=f"{widget_key_prefix}_source_radio",
        )
        source = next((sid for sid, lbl in available_sources if lbl == chosen_source_label), src_ids[0])
    elif len(available_sources) == 1:
        source = available_sources[0][0]
        # Keep the UI compact; still show where selection comes from.
    else:
        # No data at all: return an empty selection object (do not mutate state)
        st.sidebar.info("No stations available yet. Run a route or use Station Explorer first.")
        return StationSelection(station_uuid="", station=None, source="route", label="")

    # Build option lists by source
    uuid_to_station = _resolve_uuid_to_station_map(ranked, stations, explorer_results)

    options: List[Tuple[str, str]] = []  # (uuid, label)

    if source == "route":
        ranked_uuids = {_station_uuid(s) for s in ranked if _station_uuid(s)}
        excluded = [s for s in stations if (_station_uuid(s) and _station_uuid(s) not in ranked_uuids)]

        for i, s in enumerate(ranked[:max_ranked], start=1):
            uid = _station_uuid(s)
            if uid:
                options.append((uid, _build_station_label(s, tag="ranked", rank_index=i)))

        for s in excluded[:max_excluded]:
            uid = _station_uuid(s)
            if uid:
                options.append((uid, _build_station_label(s, tag="excluded", rank_index=None)))

    else:  # explorer
        for s in explorer_results[:max_explorer]:
            uid = _station_uuid(s)
            if uid:
                # include distance in label if present
                base = _build_station_label(s, tag="explorer", rank_index=None)
                dist = s.get("distance_km")
                try:
                    if dist is not None:
                        base = f"{base} · {float(dist):.1f} km"
                except (TypeError, ValueError):
                    pass
                options.append((uid, base))

    # Ensure the currently selected UUID is present (even if not in top-N lists)
    if current_uuid and current_uuid not in {u for u, _ in options}:
        st_obj = None
        if isinstance(current_station, dict):
            st_obj = current_station
        elif current_uuid in uuid_to_station:
            st_obj = uuid_to_station.get(current_uuid)

        if isinstance(st_obj, dict):
            options.insert(0, (current_uuid, _build_station_label(st_obj, tag="current", rank_index=None)))
        else:
            options.insert(0, (current_uuid, f"{current_uuid} [current]"))

    if not options:
        st.sidebar.info("No stations available in this source. Try the other source or run a search/route.")
        return StationSelection(station_uuid="", station=None, source=source, label="")

    # Create label mapping for format_func
    uuid_to_label: Dict[str, str] = {u: lbl for u, lbl in options}
    option_uuids: List[str] = [u for u, _ in options]

    # Default selection:
    # - Use current_uuid if it exists in options
    # - Else: first option (rank-1 station for route, first explorer station for explorer)
    default_uuid = current_uuid if current_uuid in uuid_to_label else option_uuids[0]
    try:
        default_idx = option_uuids.index(default_uuid)
    except ValueError:
        default_idx = 0

    chosen_uuid = st.sidebar.selectbox(
        "Select station",
        options=option_uuids,
        index=default_idx,
        format_func=lambda u: uuid_to_label.get(u, u),
        key=f"{widget_key_prefix}_station_select",
    )

    chosen_station = uuid_to_station.get(chosen_uuid)
    chosen_label = uuid_to_label.get(chosen_uuid, chosen_uuid)

    # Persist selection for cross-page navigation
    st.session_state[selected_uuid_key] = chosen_uuid
    st.session_state[selected_source_key] = source
    if isinstance(chosen_station, dict):
        st.session_state[selected_data_key] = chosen_station
    else:
        # Do not overwrite selected_station_data with None if we can't resolve.
        # Keep whatever is already there (safer for downstream pages).
        pass

    return StationSelection(
        station_uuid=chosen_uuid,
        station=chosen_station if isinstance(chosen_station, dict) else None,
        source=source,
        label=chosen_label,
    )
