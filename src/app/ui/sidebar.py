# src/app/ui/sidebar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from ui.formatting import _fuel_label_to_code, _station_uuid, _safe_text

import random
import time


# ---------------------------------------------------------------------------
# Profile placeholder (shown in Sidebar -> Profile tab)
# ---------------------------------------------------------------------------

PROFILE_PLACEHOLDER_MD = """
### Profile (Coming soon)

Welcome to the Profile section.

Personal profiles (sign-up / sign-in with email + password) will be added once this project moves beyond the academic stage.

**Why this is not available yet**
- A secure profile system requires a dedicated CIAM setup (Microsoft Entra External ID / Azure AD B2C) plus a persistent database for user data.
- In the current student/academic Azure hosting setup, the required identity provisioning and always-on database setup are not feasible within the project constraints (availability/licensing/cost).

**What you will get in the future**
- Saved routes and one-click runs with predefined settings
- Recommendation history
- Favorite stations and personal defaults

**Temporary workaround (Redis cache)**\n
This app uses a Redis-backed session cache. After running a recommendation, you can copy the full URL from your browser (including the `sid=...` parameter) and reopen it later to continue your session.
""".strip()


def _render_profile_placeholder(profile_md: str) -> None:
    """Render Profile tab content as markdown (sidebar-friendly, not too long)."""
    st.sidebar.markdown(profile_md)


# =============================================================================
# Settings tab: Quick Routes (test environment presets)
# =============================================================================

def _render_settings_quick_routes() -> None:
    """Render Settings tab: one-click quick route presets.

    - Writes canonical session_state keys (compatible with Redis persistence).
    - Uses an ephemeral one-shot flag to trigger a run on Home after navigation.
    - Forces top navigation to "Home" so the user does not bounce back to the previous page.
    - Applies mild randomized deviations (close to defaults) to keep presets varied.
    """

    st.sidebar.markdown(
    """
### Example Presets!

**You can try out preset routes to the home locations of the app creators.**

Each preset will automatically:

**1.** Switch to the Home page

**2.** Set a predefined route and fuel type

**3.** Apply random settings

**4.** Run the recommender immediately

**How to use?**
-> Just pick a person!

        """.strip()
    )

    # --------------------------------------------------------------
    # Helper: apply config + trigger run (best-effort Redis persist)
    # --------------------------------------------------------------
    def _apply_and_run(config: dict) -> None:
        # Apply route/settings immediately (canonical keys)
        for k, v in config.items():
            st.session_state[k] = v

        # One-shot run trigger (do NOT persist)
        st.session_state["_quick_run_now"] = True

        # Re-apply after Home restores Redis state (avoids overwrite race)
        if config:
            st.session_state["_pending_quick_route_config"] = dict(config)

        # IMPORTANT: force the top navigation to Home on Page 01,
        # otherwise Page 01 may immediately auto-switch back to Analytics/Station/Explorer.
        st.session_state["nav_request_top_nav"] = "Home"

        # Best-effort persistence before navigation
        try:
            from services.session_store import maybe_persist_state  # type: ignore
            maybe_persist_state(force=True)
        except Exception:
            pass

        # Navigate to Home. If already on Home, switch_page may fail -> rerun.
        try:
            st.switch_page("streamlit_app.py")
        except Exception:
            st.rerun()

    def _build_preset(*, seed_key: str, destination: str, fuel_label: str, mode: str) -> dict:
        """Create a preset config with mild randomized deviations from defaults."""
        rng = random.Random(int(time.time() * 1000) ^ hash(seed_key))

        # Mild, bounded variations
        litres = float(rng.choice([35.0, 40.0, 45.0]))
        consumption = float(rng.choice([6.5, 7.0, 7.5]))

        max_detour_km = float(rng.choice([5.0, 6.0, 7.0, 8.0]))
        max_detour_min = float(rng.choice([10.0, 12.0, 15.0]))

        min_net_saving = float(rng.choice([0.0, 0.0, 0.5, 1.0]))

        # Defaults
        value_of_time = 0.0
        use_economics = True

        # Mode tweaks (kept close to defaults)
        if mode == "advanced":
            value_of_time = float(rng.choice([5.0, 10.0, 15.0]))
            min_net_saving = float(rng.choice([0.0, 0.5, 1.0]))
        elif mode == "economical":
            value_of_time = 0.0
            min_net_saving = float(rng.choice([0.0, 0.0, 0.5]))
        elif mode == "no_econ":
            use_economics = False
            value_of_time = 0.0

        return {
            # Route
            "start_locality": "Universität Tübingen",
            "end_locality": destination,

            # Fuel & economics mode
            "fuel_label": fuel_label,
            "use_economics": use_economics,

            # Economics inputs
            "litres_to_refuel": litres,
            "consumption_l_per_100km": consumption,
            "value_of_time_eur_per_hour": value_of_time,

            # Constraints
            "max_detour_km": max_detour_km,
            "max_detour_min": max_detour_min,
            "min_net_saving_eur": min_net_saving,

            # Keep behavior consistent with your usual run defaults
            "filter_closed_at_eta": True,
        }

    presets = [
        {
            "title": "Drive home to Axel!",
            "destination": "Stühlingen, Baden-Württemberg",
            "fuel_label": "E5",
            "mode": "economical",
        },
        {
            "title": "Drive home to Moritz!",
            "destination": "Salmdorf, Bayern",
            "fuel_label": "Diesel",
            "mode": "advanced",
        },
        {
            "title": "Drive home to Celine!",
            "destination": "Weida, Thüringen",
            "fuel_label": "E10",
            "mode": "economical",
        },
    ]

    for i, p in enumerate(presets, start=1):
        with st.sidebar.container(border=True):
            st.markdown(f"**{p['title']}**")
            st.markdown("Start: Universität Tübingen")
            st.markdown(f"Destination: {p['destination']}")
            st.markdown(f"Fuel type: {p['fuel_label']}")

            if st.button(
                "Run this preset now",
                use_container_width=True,
                key=f"quickroute_{i}",
            ):
                cfg = _build_preset(
                    seed_key=p["title"],
                    destination=p["destination"],
                    fuel_label=p["fuel_label"],
                    mode=p["mode"],
                )
                _apply_and_run(cfg)

    st.sidebar.markdown(
    """
**Go to the "Action" tab afterwards to see the settings applied.**

        """.strip()
    )

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
    # Distance along-route filters (km)
    min_distance_km: float
    max_distance_km: float
    min_net_saving_eur: float

    # Diagnostics
    debug_mode: bool

    # Advanced Settings
    filter_closed_at_eta: bool

    # Brand whitelist (canonical brand labels)
    brand_filter_selected: list[str]

    # Distance to station filters (km)
    min_distance_km: float
    max_distance_km: float


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
        min_distance_km=float(_ss("min_distance_km", 0.0)),
        max_distance_km=float(_ss("max_distance_km", 750.0)),
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

    # ------------------------------------------------------------------
    # IMPORTANT: Stable, cross-tab state
    # ------------------------------------------------------------------
    # Streamlit may garbage-collect widget state for keys whose widgets are
    # not rendered in a given run. This happens here when the user switches
    # the sidebar segmented-control away from "Action" (we stop rendering the
    # Action widgets). If the Action widgets use the *same* keys as your
    # persisted contract keys (e.g., "litres_to_refuel"), those keys can
    # disappear and later re-initialize from defaults.
    #
    # Solution: decouple widget keys from persisted keys.
    # - Widgets use internal keys:  w_<name>
    # - Persisted/canonical values live under the original keys: <name>
    # - After rendering widgets we sync: <name> = w_<name>
    #
    # This keeps values stable across:
    # - switching sidebar tabs (Action/Help/Settings/Profile)
    # - switching pages via st.switch_page
    # - Redis restore / persist cycles
    def _w(key: str) -> str:
        return f"w_{key}"

    def _canonical(key: str, default: Any) -> Any:
        return st.session_state.get(key, default)

    def _sync(key: str) -> None:
        wk = _w(key)
        if wk in st.session_state:
            st.session_state[key] = st.session_state[wk]

    # Ensure the connector dots render above Streamlit input widgets (stacking fix only)
    st.sidebar.markdown(
        """
        <style>
          /* The dots are injected via HTML; make sure they sit above the text inputs */
          .route-dots-right{
            z-index: 9999 !important;
            pointer-events: none !important;
          }
          .route-dots-right span{
            z-index: 10000 !important;
            pointer-events: none !important;
          }

          /* Anchor pseudo-elements without moving the dots column */
            .route-dots-right{
            position: relative;              /* does NOT move it; only anchors ::before/::after */
            z-index: 9999 !important;
            pointer-events: none !important;
            }

            /* Make the three connector dots black */
            .route-dots-right span{
            background: rgba(0,0,0,0.90) !important;
            border: none !important;
            box-shadow: none !important;
            }
            .route-dots-right span{
            width: 5px !important;
            height: 5px !important;
            border-radius: 999px !important;
            }

            /* Top: black/grey outlined start circle */
            .route-dots-right::before{
            content: "";
            position: absolute;
            left: 50%;
            top: -22px;                      /* sits above the first dot */
            transform: translateX(-50%);
            width: 14px;
            height: 14px;
            border: 3px solid rgba(0,0,0,.90);
            border-radius: 999px;
            background: transparent;
            z-index: 10000;
            pointer-events: none;
            }

            /* Bottom: red destination pin (inline SVG) */
            .route-dots-right::after{
            content: "";
            position: absolute;
            left: 50%;
            bottom: -26px;                   /* sits below the last dot */
            transform: translateX(-50%);
            width: 22px;
            height: 22px;
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
            background-image: url("data:image/svg+xml;utf8,\
            <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>\
            <path fill='%23d32f2f' d='M12 2c-3.86 0-7 3.14-7 7 0 5.25 7 13 7 13s7-7.75 7-13c0-3.86-3.14-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5S10.62 6.5 12 6.5s2.5 1.12 2.5 2.5S13.38 11.5 12 11.5z'/>\
            </svg>");
            z-index: 10000;
            pointer-events: none;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "### Route",
        help=(
            "Specify your trip's starting point and destination. "
            "You can enter either a city name or a full address. "
            "If you use a foreign address, please add the country name for better results."
        ),
    )

    start_locality = st.sidebar.text_input(
        "Start",
        value=str(_canonical("start_locality", "Tübingen")),
        key=_w("start_locality"),
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
        value=str(_canonical("end_locality", "Sindelfingen")),
        key=_w("end_locality"),
        label_visibility="collapsed",
        placeholder="Destination: city or full address"
    )

    # (You currently keep these empty; preserve behavior)
    start_address = ""
    end_address = ""

    st.sidebar.markdown(
        "### Fuel type",
        help=(
            "Select the type of fuel your vehicle uses. E5, E10 and Diesel are supported."
        ),
    )

    fuel_options = ["E5", "E10", "Diesel"]
    fuel_label_default = str(_canonical("fuel_label", "E5"))
    if fuel_label_default not in fuel_options:
        fuel_label_default = "E5"

    fuel_label = st.sidebar.selectbox(
        "Fuel type",
        options=fuel_options,
        index=fuel_options.index(fuel_label_default),
        label_visibility="collapsed",
        key=_w("fuel_label"),
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
        value=bool(_canonical("use_economics", True)),
        key=_w("use_economics"),
    )

    litres_to_refuel = st.sidebar.number_input(
        "Litres to refuel",
        min_value=1.0,
        max_value=200.0,
        step=1.0,
        value=float(_canonical("litres_to_refuel", 40.0)),
        key=_w("litres_to_refuel"),
        help=(
            "Amount of fuel (in litres) you plan to refuel at the selected station. "
            "The more you refuel, the higher the potential savings from price differences."
            ),
    )

    consumption_l_per_100km = st.sidebar.number_input(
        "Car consumption (L/100 km)",
        min_value=0.0,
        max_value=30.0,
        step=0.5,
        value=float(_canonical("consumption_l_per_100km", 7.0)),
        key=_w("consumption_l_per_100km"),
        help=(
            "Your car's average fuel consumption in litres per 100 km. "
            "Used to estimate extra fuel costs for detours."
        ),
    )

    value_of_time_eur_per_hour = st.sidebar.number_input(
        "Value of time (€/hour)",
        min_value=0.0,
        max_value=50.0,
        step=1.0,
        value=float(_canonical("value_of_time_eur_per_hour", 0.0)),
        key=_w("value_of_time_eur_per_hour"),
        help=(
            "Monetary value you assign to your time (in euros per hour). "
            "Used to estimate the cost of extra travel time for detours. "
            ),
    )

    max_detour_km = st.sidebar.number_input(
        "Maximum extra distance (km)",
        min_value=0.5,
        max_value=200.0,
        step=0.5,
        value=float(_canonical("max_detour_km", 5.0)),
        key=_w("max_detour_km"),
        help=(
            "The maximum additional distance you are willing to drive for a detour compared to the baseline route. "
            "Stations requiring more extra distance are excluded (hard constraint)."
        ),
    )

    max_detour_min = st.sidebar.number_input(
        "Maximum extra time (min)",
        min_value=1.0,
        max_value=240.0,
        step=1.0,
        value=float(_canonical("max_detour_min", 10.0)),
        key=_w("max_detour_min"),
        help=(
            "The maximum additional travel time you are willing to accept for a detour compared to the baseline route. "
            "Stations requiring more extra time are excluded (hard constraint)."
        ),
    )

    min_net_saving_eur = st.sidebar.number_input(
        "Min net saving (€)",
        min_value=0.0,
        max_value=100.0,
        step=0.5,
        value=float(_canonical("min_net_saving_eur", 0.0)),
        key=_w("min_net_saving_eur"),
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
        value=bool(_canonical("filter_closed_at_eta", True)),
        key=_w("filter_closed_at_eta"),
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
        default=list(_canonical("brand_filter_selected", [])),
        key=_w("brand_filter_selected"),
        help=(
            "10 most common brands in Germany. "
            "Only stations of selected brands are considered. "
            "When active, stations with unknown/missing brand are excluded."
        ),
    )

    max_distance_km = st.sidebar.number_input(
        "Maximum distance to station (km)",
        min_value=1.0,
        max_value=1000.0,
        step=1.0,
        value=float(_canonical("max_distance_km", 750.0)),
        key=_w("max_distance_km"),
        help=(
            "The maximum distance from the starting point to a station. Stations farther than this distance are excluded ." \
            "Useful to set for long trips if your fuel tank is low and you want to refill before running out."
        ),
    )

    min_distance_km = st.sidebar.number_input(
        "Minimum distance to station (km)",
        min_value=0.0,
        max_value=1000.0,
        step=1.0,
        value=float(_canonical("min_distance_km", 0.0)),
        key=_w("min_distance_km"),
        help=(
            "The minimum distance from the starting point to a station. Stations closer than this distance are excluded ." \
            "Useful to set if your fuel tank is full and you want to refill after driving some distance."
        ),
    )
    # Validation: min_distance_km <= max_distance_km
    if min_distance_km > max_distance_km:
        st.sidebar.error(
            "Minimum distance cannot be greater than maximum distance. "
            "Please adjust the values."
        )
        
    # Sync widget values back into canonical persisted keys
    for k in (
        "start_locality",
        "end_locality",
        "fuel_label",
        "use_economics",
        "litres_to_refuel",
        "consumption_l_per_100km",
        "value_of_time_eur_per_hour",
        "max_detour_km",
        "max_detour_min",
        "min_net_saving_eur",
        "min_distance_km",
        "max_distance_km",
        "filter_closed_at_eta",
        "brand_filter_selected",
    ):
        _sync(k)

    # Use canonical values from session_state for downstream consistency
    start_locality = str(_canonical("start_locality", start_locality))
    end_locality = str(_canonical("end_locality", end_locality))
    fuel_label = str(_canonical("fuel_label", fuel_label))
    use_economics = bool(_canonical("use_economics", use_economics))
    litres_to_refuel = float(_canonical("litres_to_refuel", litres_to_refuel))
    consumption_l_per_100km = float(_canonical("consumption_l_per_100km", consumption_l_per_100km))
    value_of_time_eur_per_hour = float(_canonical("value_of_time_eur_per_hour", value_of_time_eur_per_hour))
    max_detour_km = float(_canonical("max_detour_km", max_detour_km))
    max_detour_min = float(_canonical("max_detour_min", max_detour_min))
    min_distance_km = float(_canonical("min_distance_km", min_distance_km))
    max_distance_km = float(_canonical("max_distance_km", max_distance_km))
    min_net_saving_eur = float(_canonical("min_net_saving_eur", min_net_saving_eur))
    filter_closed_at_eta = bool(_canonical("filter_closed_at_eta", filter_closed_at_eta))
    brand_filter_selected = list(_canonical("brand_filter_selected", brand_filter_selected))

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
        min_distance_km=float(min_distance_km),
        max_distance_km=float(max_distance_km),
        min_net_saving_eur=float(min_net_saving_eur),
        debug_mode=bool(_ss("debug_mode", True)),
        filter_closed_at_eta=bool(filter_closed_at_eta),
        brand_filter_selected=list(brand_filter_selected),
    )


def _render_help_action() -> SidebarState:
    """

    """
    state = _read_cached_values_for_non_action()

    st.sidebar.markdown("### Detailed Information")
    st.sidebar.markdown("Click on the topic headings below to get more information.")

    with st.sidebar.popover("Data Sources"):
        st.markdown("**Google Maps APIs**: " \
            "\n- Geocoding: convert addresses into latitude/longitude coordinates" \
            "\n- Directions: find best driving route" \
            "\n- Places Text Search Enterprise: find fuel stations along the route" \
            "\n\n**Tankerkönig**:" \
            "\n- API for current fuel prices of 14000+ stations in Germany" \
            "\n- Historical price data for model training.")

    with st.sidebar.popover("Route finding process"):
        st.markdown("First the user’s start and destination input is converted into exact latitude/longitude coordinates using the **Google Maps Geocoding API**. " \
            "\n\n Next, the best driving route is calculated with the **Google Maps Directions API**. Besides total distance and travel time, we also extract the detailed route geometry, which is  later used to find stations along the route.")

    with st.sidebar.popover("Finding Stations along the route"):
        st.markdown("Fuel stations are retrieved along the calculated route using the **Places API Text Search Enterprise**. " \
            "\n\nFor each returned station, the detour time and distance caused by stopping there is calculated from the API’s routing summary output. Further an estimated arrival time is computed. Opening-hours information is then evaluated to determine whether the station is likely open at that ETA.")

    with st.sidebar.popover("Detour Economics Explained"):
        st.markdown("The Detour Economics logic evaluates the cost-effectiveness of taking a detour to refuel. It combines the additional fuel costs (based on distance and car consumption) and time costs (using the user's value of time) with the potential savings from lower fuel prices at the detour station. " \
            "The goal is to calculate the net savings and ensure that the detour is economically beneficial." \
            "Formula: \n\n$$Net Saving = Fuel Price Saving − Detour Fuel Cost − Time Cost$$" \
            )
        
    with st.sidebar.popover("Price prediction model"):
        st.markdown("The price prediction is based on an **ARDL model (Autoregressive Distributed Lag)** that uses past fuel prices to estimate future prices. " \
            "Separate models are trained for each fuel type (E5, E10, Diesel) and for different prediction horizons (from “now” up to about two hours ahead). " \
            "\n\nTo generate a prediction, it is first determined how far in the future the arrival at a station lies. If the arrival is very soon (within a few minutes) and a current price is available, the current price is used directly and no model is applied. " \
            "Otherwise, the appropriate horizon model is selected: short-term horizons (h1–h4) are used for near-future arrivals, while a daily-only model (h0) is used when the arrival lies further ahead or when no current price is available. " \
            "The model then receives a **feature vector** consisting of **daily lagged prices** (prices from previous days) and, for short horizons, an additional **intraday price feature**. " \
            "Based on these inputs, the ARDL model predicts the expected fuel price at the estimated arrival time. ")

    # Return a SidebarState compatible object (keep cached values, only change view)
    return SidebarState(**{**state.__dict__, "view": "Help"})


def _render_help_station():
    """
    Render help content for Station Details page (Page 03).
    Explains the key concepts and features available.
    """
    # Keep cached values for downstream behaviour and return a SidebarState
    state = _read_cached_values_for_non_action()

    st.sidebar.markdown("### Detailed Information")
    st.sidebar.markdown("Click on the topic headings below to get more information.")
    
    # Savings Calculator / Price Comparison
    with st.sidebar.expander("Savings / Price Comparison"):
        st.markdown("""
            **Trip Planner only: Savings Calculator**

            We compare this station against the **worst on-route** price (most expensive station directly on your route, no detour needed).

            - **Gross Savings** = Price difference × Liters
            - **Net Savings** = Gross Savings − Detour Costs
            - **Detour fuel cost**: extra km × consumption × price
            - **Time cost**: extra minutes × your hourly rate

            **Station Explorer: Price Comparison**

            Shows this station's price vs the cheapest and most expensive in your search results. No calculations - just a quick overview to see where this station stands.
            """)
    
    # Best Time to Refuel
    with st.sidebar.expander("Best Time to Refuel"):
        st.markdown("""
            **Hourly Price Patterns**

            Based on 14 days of historical data, we show when prices are typically:
            - 🟢 Cheapest (often early morning or late evening)
            - 🟡 Average
            - 🔴 Most expensive (often mid-day)

            **Heatmap:** Shows price patterns by day and hour. White cells mean no data was recorded for that time slot.

            **When is data "not enough"?**

            We need at least 20 price changes across 4 different hours in the last 14 days. In our experience, stations with fewer updates rarely change prices, making hourly patterns unreliable. A typical station updates prices 3-5 times per day.
            """)
    
    # Station Selection
    with st.sidebar.expander("Station Selection"):
        st.markdown("""
            **Station Source**

            Use the radio button to switch between:
            - **From latest route run:** Stations from your Trip Planner route
            - **From station explorer:** Stations from your proximity search

            **Ranked** 

            - In **Trip Planner** mode, stations are ranked by the net savings (desc).
            - In **Station Explorer** mode, stations are ranked by current price and distance (desc).

            **Comparison:** 
            - Select stations from the sidebar to compare historical prices.
                    """)
    # Return a SidebarState compatible object (keep cached values, only change view)
    return SidebarState(**{**state.__dict__, "view": "Help"})

def _render_help_explorer():
    """
    Render help content for Station Explorer page (Page 04).
    Explains the key concepts and features available.
    """
    # Keep cached values for downstream behaviour and return a SidebarState
    state = _read_cached_values_for_non_action()

    st.sidebar.markdown("### Additional Information")
    st.sidebar.markdown("Click on the topic headings below to get more information.")
    
    # Station Explorer Overview
    with st.sidebar.expander("Difference in number of stations open vs. with current price"):
        st.markdown("""
            The data for prices and opening times is pulled from Tankerkönig. The difference is already in the raw data.
            Stations which have at least one information missing (either price or opening times) will be labeled as red on the map.
            """)
        
    # Station Explorer Overview
    with st.sidebar.expander("Why is the station marked as closed / no price when Google shows it is open?"):
        st.markdown("""
            All data as explained before is pulled from Tankerkönig. If a station is marked as closed / no price, 
            it means that Tankerkönig does not have this information for the specific station.
            """)
        
    # Station Explorer Overview
    with st.sidebar.expander("Why are there more than one green markers on the map?"):
        st.markdown("""
            It can happen that multiple stations have the same "lowest" price. In this case, all stations with the 
            lowest price are marked green. The closest one to the center of the search is shown as selected by default 
            and has a larger marker.
            """)

    # Return a SidebarState compatible object (keep cached values, only change view)
    return SidebarState(**{**state.__dict__, "view": "Help"})
def render_sidebar(
    *,
    action_renderer: Optional[Callable[[], SidebarState]] = None,
    help_renderer: Optional[Callable[[], SidebarState]] = None,
    settings_renderer: Optional[Callable[[], None]] = None,
    help_placeholder: str = "Placeholder: Help (content will be added later).",
    settings_placeholder: str = "Placeholder: Settings (content will be added later).",
    profile_placeholder: str = PROFILE_PLACEHOLDER_MD,
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
        renderer = help_renderer or _render_help_action
        return renderer()
    if view == "Settings":
        renderer = settings_renderer or _render_settings_quick_routes
        renderer()
        return SidebarState(**{**state.__dict__, "view": "Settings"})

    _render_profile_placeholder(profile_placeholder)
    return SidebarState(**{**state.__dict__, "view": "Profile"})

# =============================================================================
# Shared sidebar shell (Pages 2—4)
# =============================================================================

def render_sidebar_shell(
    *,
    top_renderer: Optional[Callable[[], None]] = None,
    action_renderer: Optional[Callable[[], None]] = None,
    action_placeholder: str = "Placeholder: Action",
    help_renderer: Optional[Callable[[], None]] = None,
    settings_renderer: Optional[Callable[[], None]] = None,
    settings_placeholder: str = "Placeholder: Settings (will be added later).",
    profile_placeholder: str = PROFILE_PLACEHOLDER_MD,
) -> str:
    """
    Shared 4-tab sidebar shell for Pages 2—4 (and optionally others):
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

    # Optional page-specific controls rendered directly below the sidebar toggle
    # (e.g., quick back-navigation buttons on some pages).
    # IMPORTANT: Only show these on the Action tab.
    if view == "Action" and top_renderer is not None:
        try:
            top_renderer()
        except Exception:
            # Best-effort: sidebar chrome should never break the page.
            pass

    
    # Render content based on selected tab
    # Only Action tab renders the (potentially heavy) action_renderer
    with st.sidebar.container():
        if view == "Action":
            if action_renderer is not None:
                action_renderer()
            else:
                st.info(action_placeholder)
        elif view == "Help":
            if help_renderer is not None:
                help_renderer()
            else:
                st.info("Placeholder: Help (content will be added later).")
        elif view == "Settings":
            renderer = settings_renderer or _render_settings_quick_routes
            renderer()
        else:
            _render_profile_placeholder(profile_placeholder)

    return st.session_state.get("sidebar_view", "Action")


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
        parts.append(f"· {city.title()}")

    parts.append(f"[{tag}]")
    return " ".join(parts).strip()


def _extract_net_saving_eur(station: Dict[str, Any], fuel_code: str) -> Optional[float]:
    """
    Best-effort extraction of net saving (EUR) from a station dict.

    Primary key is fuel-specific:
      econ_net_saving_eur_<fuel_code>  (e.g., econ_net_saving_eur_e5)

    Fallbacks are supported for robustness across older cached runs.
    """
    if not isinstance(station, dict):
        return None

    fc = (fuel_code or "").strip().lower()
    candidate_keys = []
    if fc:
        candidate_keys.append(f"econ_net_saving_eur_{fc}")

    # Fall back to older / generic keys (defensive)
    candidate_keys += [
        "econ_net_saving_eur",
        "net_saving_eur",
        "net_saving",
    ]

    for k in candidate_keys:
        if k in station:
            try:
                v = station.get(k)
                if v is None:
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None

    return None


def _build_compact_route_label(rank_index: int, station: Dict[str, Any]) -> str:
    """
    Label format for the route selector:
      "<i>. <BRAND> — <Address>"

    Address is built from Tankerkönig fields if available:
      street + house_number + post_code + city
    Fallback: city only, else station name/uuid placeholder.
    """
    # BRAND (preferred) with safe fallback
    brand = _safe_text((station or {}).get("brand", "")).strip()
    if not brand:
        name = station.get("tk_name") or station.get("osm_name") or station.get("name")
        brand = _safe_text(str(name) if name else "Unknown")

    # Address from TK fields (best for ALL stations)
    street = (station or {}).get("street")
    house_number = (station or {}).get("house_number")
    post_code = (station or {}).get("post_code")
    city = (station or {}).get("city")

    address = ""
    if street and city:
        street_title = str(street).title().strip()
        city_title = str(city).title().strip()

        street_part = f"{street_title} {str(house_number).strip()}" if house_number else street_title
        city_part = f"{str(post_code).strip()} {city_title}".strip() if post_code else city_title

        address = f"{street_part}, {city_part}".strip()
    elif city:
        address = str(city).title().strip()

    if not address:
        # Very defensive fallback (should be rare for TK stations)
        address = _safe_text(
            str(station.get("city") or station.get("osm_name") or station.get("tk_name") or "")
        ).strip()

    if not address:
        address = "Address unavailable"

    return f"{rank_index}. {brand} — {address}"


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
        src_ids = [sid for sid, _ in available_sources]
        radio_key = f"{widget_key_prefix}_source_radio"
        
        # Determine default index:
        # - If radio widget already has a value (user interacted), use that
        # - Otherwise, use the session state source
        if radio_key in st.session_state:
            # User has already interacted with radio - find index from their selection
            existing_label = st.session_state.get(radio_key)
            existing_source = next((sid for sid, lbl in available_sources if lbl == existing_label), None)
            if existing_source in src_ids:
                default_idx = src_ids.index(existing_source)
            else:
                default_idx = src_ids.index(current_source) if current_source in src_ids else 0
        else:
            # First render - use session state source
            default_idx = src_ids.index(current_source) if current_source in src_ids else 0
        
        chosen_source_label = st.sidebar.radio(
            "Station source",
            options=[lbl for _, lbl in available_sources],
            index=default_idx,
            key=radio_key,
            help="Switch between stations from your Trip Planner route or from your Station Explorer search."
        )
        source = next((sid for sid, lbl in available_sources if lbl == chosen_source_label), src_ids[0])
        
        # Update session state to match user's radio selection
        # This ensures the header ("Explorer Mode" vs "Trip Planner Settings") stays in sync
        if source != current_source:
            st.session_state[selected_source_key] = source
            # RESET selection when switching sources - avoids "ghost" stations from other source
            st.session_state[selected_uuid_key] = ""
            st.session_state[selected_data_key] = None
            current_uuid = ""  # Also reset local variable
            current_station = None
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
        # NEW BEHAVIOR (route source only):
        # - Show ALL found route stations (no ranked/not-ranked split)
        # - Sort descending by net saving (missing net saving goes to bottom)
        # - Label: "1. BRAND — Address"
        fuel_code = str(last_run.get("fuel_code") or "").strip().lower()

        # Prefer the full station universe; fall back to ranked if stations missing for any reason.
        base = list(stations or ranked or [])

        # Deduplicate by UUID while keeping the best (highest) net saving for that UUID.
        # Missing net saving is treated as -inf so it never wins over a real value.
        best_by_uuid: Dict[str, Dict[str, Any]] = {}
        best_score_by_uuid: Dict[str, float] = {}

        for s in base:
            uid = _station_uuid(s)
            if not uid:
                continue

            net = _extract_net_saving_eur(s, fuel_code)
            score = float(net) if net is not None else float("-inf")

            if uid not in best_by_uuid or score > best_score_by_uuid.get(uid, float("-inf")):
                best_by_uuid[uid] = s
                best_score_by_uuid[uid] = score

        all_stations = list(best_by_uuid.values())

        def _sort_key(s: Dict[str, Any]) -> Tuple[int, float, float]:
            net = _extract_net_saving_eur(s, fuel_code)
            missing = 1 if net is None else 0
            # missing=0 first, then descending net (use -net), then stable tie-breakers
            frac = s.get("fraction_of_route", float("inf"))
            dist = s.get("distance_along_m", float("inf"))
            try:
                frac_f = float(frac) if frac is not None else float("inf")
            except (TypeError, ValueError):
                frac_f = float("inf")
            try:
                dist_f = float(dist) if dist is not None else float("inf")
            except (TypeError, ValueError):
                dist_f = float("inf")

            return (missing, -(float(net) if net is not None else 0.0), frac_f + 0.0 + dist_f * 0.0)

        # Note: the tie-breakers above are intentionally conservative; primary is net saving.
        all_stations_sorted = sorted(all_stations, key=_sort_key)

        for i, s in enumerate(all_stations_sorted, start=1):
            uid = _station_uuid(s)
            if uid:
                options.append((uid, _build_compact_route_label(i, s)))


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

# =============================================================================
# Comparison Station Selector (for Station Details page)
# =============================================================================

def _station_name_for_comparison(station: Dict[str, Any]) -> str:
    """Extract station name for comparison selector."""
    name = station.get("tk_name") or station.get("osm_name") or station.get("name")
    brand = station.get("brand")
    
    if name:
        return _safe_text(name)
    elif brand:
        return _safe_text(brand)
    else:
        return "Unknown Station"


def _station_brand_for_comparison(station: Dict[str, Any]) -> str:
    """Extract station brand."""
    return _safe_text(station.get("brand", ""))


def render_comparison_selector(
    ranked: List[Dict[str, Any]],
    stations: List[Dict[str, Any]],
    current_station_uuid: str,
    max_ranked: int = 10,
    max_excluded: int = 50,
    explorer_results: Optional[List[Dict[str, Any]]] = None,
    max_explorer: int = 50,
    active_source: Optional[str] = None,
) -> Optional[str]:
    """
    Render a comparison station selector in the sidebar.
    Uses the same list structure as render_station_selector for consistency.
    
    Args:
        ranked: Ranked stations list from last_run
        stations: Full stations list from last_run
        current_station_uuid: UUID of the currently selected station (to exclude)
        max_ranked: Maximum number of ranked stations to show
        max_excluded: Maximum number of excluded stations to show
        explorer_results: Optional list of stations from Explorer
        max_explorer: Maximum number of explorer stations to show
        active_source: "route" or "explorer" - determines which stations to show
    
    Returns:
        UUID of the selected comparison station, or None if none selected
    """
    st.sidebar.markdown("### Compare stations")
    
    explorer_results = explorer_results or []
    
    # Determine active source from radio widget if not explicitly provided
    if active_source is None:
        radio_key = "station_details_source_radio"
        if radio_key in st.session_state:
            radio_label = st.session_state.get(radio_key, "")
            if "explorer" in radio_label.lower():
                active_source = "explorer"
            elif "route" in radio_label.lower():
                active_source = "route"
        if active_source is None:
            active_source = st.session_state.get("selected_station_source", "route")
    
    if not ranked and not stations and not explorer_results:
        st.sidebar.caption("Run Trip Planner or Explorer to enable comparison.")
        return None
    
    # Build UUID to label mapping based on active source
    uuid_labels: Dict[str, str] = {}
    
    if active_source == "explorer":
        # EXPLORER MODE: Show only explorer stations
        for s in explorer_results[:max_explorer]:
            u = _station_uuid(s)
            if not u or u in uuid_labels:
                continue
            nm = _station_name_for_comparison(s)
            br = _station_brand_for_comparison(s)
            city = _safe_text(s.get("city", ""))
            
            label = f"{nm}"
            if br and br != nm:
                label += f" ({br})"
            if city:
                label += f" · {city}"
            
            # Add distance if available
            dist = s.get("distance_km")
            try:
                if dist is not None:
                    label += f" · {float(dist):.1f} km"
            except (TypeError, ValueError):
                pass
            
            label += " [explorer]"
            uuid_labels[u] = label
        
        if not uuid_labels:
            st.sidebar.caption("No explorer stations available for comparison.")
            return None
    
    else:
        # ROUTE MODE: Show ranked and excluded stations
        # First: ranked stations with #1, #2, etc.
        ranked_uuids = set()
        for i, s in enumerate(ranked[:max_ranked], start=1):
            u = _station_uuid(s)
            if not u:
                continue
            ranked_uuids.add(u)

            # UI only: compact label format (do not change selection logic)
            uuid_labels[u] = _build_compact_route_label(i, s)

        # Second: excluded stations (in stations but not in ranked)
        excluded = [
            s for s in stations
            if (_station_uuid(s) and _station_uuid(s) not in ranked_uuids)
        ]

        # Continue numbering after the ranked list (UI only)
        next_index = len(uuid_labels) + 1

        for j, s in enumerate(excluded[:max_excluded], start=next_index):
            u = _station_uuid(s)
            if not u or u in uuid_labels:
                continue

            # UI only: compact label format (do not change selection logic)
            uuid_labels[u] = _build_compact_route_label(j, s)

        
        if not uuid_labels:
            st.sidebar.caption("No route stations available for comparison.")
            return None
    
    # Filter out the current station
    candidates = [u for u in uuid_labels.keys() if u and u != current_station_uuid]
    
    if not candidates:
        st.sidebar.caption("No other stations to compare.")
        return None
    
    # Add "None" option at the beginning
    options_with_none = [""] + candidates
    labels_with_none = {"": "- Select a station -"}
    labels_with_none.update(uuid_labels)
    
    # Get current selection from session state
    current_comparison = st.session_state.get("comparison_station_uuid", "")
    
    # If current comparison is the same as selected station, reset it
    if current_comparison == current_station_uuid:
        current_comparison = ""
    
    # Ensure current comparison is valid
    if current_comparison and current_comparison not in candidates:
        current_comparison = ""
    
    # Find index for default
    try:
        default_idx = options_with_none.index(current_comparison) if current_comparison in options_with_none else 0
    except ValueError:
        default_idx = 0
    
    chosen = st.sidebar.selectbox(
        "Compare against",
        options=options_with_none,
        index=default_idx,
        format_func=lambda u: labels_with_none.get(u, uuid_labels.get(u, u)),
        key=f"comparison_station_selectbox_{active_source}",
        help=f"Select a station to compare historical prices ({len(candidates)} available).",
    )
    
    # Store selection in session state
    if chosen:
        st.session_state["comparison_station_uuids"] = [chosen]
        st.session_state["comparison_station_uuid"] = chosen
        return chosen
    else:
        st.session_state["comparison_station_uuids"] = []
        st.session_state["comparison_station_uuid"] = ""
        return None
        return None