# src/app/ui/sidebar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import streamlit as st

from ui.formatting import _fuel_label_to_code


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
        "### Detour economics",
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

    debug_mode = st.sidebar.checkbox(
        "Debug mode",
        value=bool(_ss("debug_mode", True)),
        key="debug_mode",
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
        debug_mode=bool(debug_mode),
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
        label="",
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


# Helper: simplified sidebar renderer for Pages 2–4
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
        label="",
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
