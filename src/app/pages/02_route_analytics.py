"""
Route Analytics (Page 2)

Purpose:
- Provide auditability and comparability for a single route run.
- Explain why a station was recommended and why others were excluded.
- Offer drill-down navigation into Station Details.

Data source:
- st.session_state["last_run"] written by src/app/streamlit_app.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None

# ---------------------------------------------------------------------
# Minimum plausible €/L (guards against 0.00 or corrupted values in cached runs)
MIN_VALID_PRICE_EUR_L: float = 0.50
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Path setup (compatible with running from /src/app/pages via Streamlit)
# ---------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]        # .../src/app
PROJECT_ROOT = Path(__file__).resolve().parents[3]    # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import load_env_once

load_env_once()

from ui.formatting import (
    _station_uuid,
    _safe_text,
    _fmt_price,
    _fmt_eur,
    _fmt_km,
    _fmt_min,
)
from ui.styles import apply_app_css
from ui.sidebar import render_sidebar_shell, _render_help_action, _render_settings_quick_routes

from config.settings import ensure_persisted_state_defaults
from services.session_store import init_session_context, restore_persisted_state, maybe_persist_state


# -----------------------------
# Helpers
# -----------------------------

# ---------------------------------------------------------------------
# Compatibility helpers (used by Page 02 constraint/summary blocks)
# Keep these thin wrappers so the section code stays readable.
# ---------------------------------------------------------------------
def _format_km(x: Any) -> str:
    return _fmt_km(x)


def _format_min(x: Any) -> str:
    return _fmt_min(x)


def _format_eur(x: Any) -> str:
    return _fmt_eur(x)


def _format_liters(x: Any) -> str:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return "—"
        return f"{float(x):.2f} L"
    except (TypeError, ValueError):
        return "—"


def render_card_grid(items: List[Tuple[str, str]], cols: int = 2) -> None:
    """
    Responsive card grid (uses the same visual card language as the Start/Destination blocks).
    - Desktop: `cols` columns
    - Mobile: Streamlit stacks columns automatically
    """
    if not items:
        return

    cols = max(1, int(cols))

    # CSS is injected once per run (safe); class names are page-local
    st.markdown(
        """
        <style>
          .p02-card {
            width: 100%;
            padding: 0.5rem 1.0rem;
            border: 2px solid rgba(49, 51, 63, 0.3);
            border-radius: 0.9rem;
            text-align: center;
            margin-bottom: 0.25rem;
            margin-top: 0rem;
          }
          .p02-card-label {
            font-size: 0.9rem;
            opacity: 0.78;
            margin: 0 0 0.25rem 0;
            line-height: 1.15;
          }
          .p02-card-value {
            font-size: 1.3rem;
            font-weight: 750;
            margin: 0;
            line-height: 1.15;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    row = st.columns(cols)
    for i, (label, value) in enumerate(items):
        with row[i % cols]:
            st.markdown(
                f"""
                <div class="p02-card">
                  <div class="p02-card-label">{label}</div>
                  <div class="p02-card-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if (i % cols) == (cols - 1) and i != (len(items) - 1):
            row = st.columns(cols)


def render_key_visualization(
    *,
    route_info: Dict[str, Any],
    best_station: Optional[Dict[str, Any]],
) -> None:
    """
    Page 02 / Recommended Route:
    Two proportional route bars (baseline vs alternative via best station),
    with a station marker on the alternative bar.
    """

    # --- baseline route metrics (best-effort key fallbacks) ---
    baseline_km = _as_float(route_info.get("route_km") or route_info.get("distance_km") or route_info.get("km"))
    baseline_min = _as_float(route_info.get("route_min") or route_info.get("duration_min") or route_info.get("min"))

    # --- alternative route metrics (baseline + detour deltas) ---
    detour_km = None
    detour_min = None
    if isinstance(best_station, dict) and best_station:
        dk, dm = _detour_metrics(best_station)
        detour_km = dk
        detour_min = dm

    alt_km = (baseline_km + detour_km) if (baseline_km is not None and detour_km is not None) else None
    alt_min = (baseline_min + detour_min) if (baseline_min is not None and detour_min is not None) else None

    # --- station marker position on alternative line (prefer fraction_of_route) ---
    station_frac = None
    if isinstance(best_station, dict) and best_station:
        station_frac = _as_float(best_station.get("fraction_of_route"))

        # fallback: derive from distance_along_m if total route length is known
        if station_frac is None:
            dist_along_m = _as_float(best_station.get("distance_along_m"))
            route_dist_m = _as_float(
                route_info.get("route_distance_m")
                or route_info.get("distance_m")
                or route_info.get("route_m")
            )
            if dist_along_m is not None and route_dist_m not in (None, 0.0):
                station_frac = dist_along_m / route_dist_m

    # clamp to [0, 1]
    if station_frac is not None:
        station_frac = max(0.0, min(1.0, float(station_frac)))

    # --- proportional widths: longest bar spans full container width ---
    lengths = [x for x in [baseline_km, alt_km] if x is not None and x > 0]
    denom = max(lengths) if lengths else 1.0

    baseline_w = (baseline_km / denom) if (baseline_km is not None and baseline_km > 0) else 0.0
    alt_w = (alt_km / denom) if (alt_km is not None and alt_km > 0) else 0.0

    # if we do not have a best station, we still show the baseline bar and a placeholder alternative bar
    has_alt = (alt_km is not None and alt_w > 0.0)

    # --- display strings ---
    baseline_text = f"Distance: {_format_km(baseline_km)} · Time: {_format_min(baseline_min)}"
    alt_line1: str
    alt_line2: Optional[str] = None

    if has_alt:
        alt_line1 = f"Distance: {_format_km(alt_km)} · Time: {_format_min(alt_min)}"

        # Detour as % of baseline distance
        detour_pct = None
        if baseline_km not in (None, 0.0) and detour_km is not None:
            detour_pct = (float(detour_km) / float(baseline_km)) * 100.0

        if detour_pct is not None:
            alt_line2 = f"Detour: {_format_km(detour_km)} / {detour_pct:.2f}% of baseline"
        else:
            alt_line2 = f"Detour: {_format_km(detour_km)} / —% of baseline"
    else:
        alt_line1 = "Alternative route is unavailable (no recommended station cached)."
        alt_line2 = None


    # --- render (page-local CSS; uses your map colors) ---
    st.markdown(
        """
        <style>
          .p02-kv { width: 100%; margin-top: -2rem; }
          .p02-kv-row { margin: 0.65rem 0 1.05rem 0; }
          .p02-kv-subtitle { font-weight: 700; margin: 0 0 0.35rem 0; }
          .p02-kv-track { width: 100%; }
          .p02-kv-line {
            position: relative;
            height: 0;
            border-top-width: 6px;
            border-top-style: solid;
            border-radius: 999px;
            opacity: 0.95;
          }
          .p02-kv-line.baseline { border-top-color: rgba(30, 144, 255, 1.0); }
          .p02-kv-line.alt { border-top-color: rgba(148, 0, 211, 1.0); border-top-style: dashed; }

            .p02-kv-dot {
            position: absolute;
            top: -12px;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            background: rgba(220, 20, 60, 1.0);  /* red dot */
            border: 2px solid rgba(255, 255, 255, 0.96);
            box-shadow: 0 2px 10px rgba(0,0,0,0.20);
            transform: translateX(-50%);
            }

            .p02-kv-detail { margin: 0.40rem 0 0 0; opacity: 0.85; line-height: 1.25; }
            .p02-kv-detail.detour { opacity: 0.78; margin-top: 0.20rem; }

          /* Mobile: slightly thicker separation, keep readability */
          @media (max-width: 600px) {
            .p02-kv-row { margin-bottom: 1.20rem; }
            .p02-kv-line { border-top-width: 7px; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build HTML without indentation to avoid Markdown "code block" behavior
    baseline_bar = (
        f"<div class='p02-kv-line baseline' style='width:{baseline_w*100:.1f}%;'></div>"
        if baseline_w > 0
        else "<div class='p02-kv-line baseline' style='width:0%;'></div>"
    )

    if has_alt:
        dot_html = ""
        if station_frac is not None:
            dot_html = f"<span class='p02-kv-dot' style='left:{station_frac*100:.1f}%;'></span>"
        alt_bar = f"<div class='p02-kv-line alt' style='width:{alt_w*100:.1f}%;'>{dot_html}</div>"
    else:
        alt_bar = "<div class='p02-kv-line alt' style='width:0%;'></div>"

    html_block = (
        "<div class='p02-kv'>" +
        "<div class='p02-kv-row'>" +
        "<div class='p02-kv-subtitle'>Baseline Google route:</div>" +
        f"<div class='p02-kv-track'>{baseline_bar}</div>" +
        f"<div class='p02-kv-detail'>{_safe_text(baseline_text)}</div>" +
        "</div>" +
        "<div class='p02-kv-row'>" +
        "<div class='p02-kv-subtitle'>Alternative route with best station:</div>" +
        f"<div class='p02-kv-track'>{alt_bar}</div>" +
        f"<div class='p02-kv-detail'>{_safe_text(alt_line1)}</div>" +
            (
                f"<div class='p02-kv-detail detour'>{_safe_text(alt_line2)}</div>"
                if alt_line2 else
                ""
            ) +
        "</div>" +
        "</div>"
    )

    st.markdown(html_block, unsafe_allow_html=True)


def _get_last_run() -> Optional[Dict[str, Any]]:
    cached = st.session_state.get("last_run")
    if not isinstance(cached, dict):
        return None
    if "stations" not in cached:
        return None
    return cached


def _detour_metrics(station: Dict[str, Any]) -> Tuple[float, float]:
    """Return (detour_km, detour_min), clamped to >= 0."""
    d_km = station.get("detour_distance_km")
    if d_km is None:
        d_km = station.get("detour_km")

    d_min = station.get("detour_duration_min")
    if d_min is None:
        d_min = station.get("detour_min")

    def _safe_float(v: Any) -> float:
        try:
            return max(0.0, float(v)) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    return _safe_float(d_km), _safe_float(d_min)


def _distance_along_km(station: Dict[str, Any]) -> float:
    """
    Distance along the route in km (used for min/max distance hard constraints).

    Matches decision-layer semantics in recommender.py:
    - Primary source: distance_along_m (meters) -> km
    - Best-effort fallbacks to other common fields
    - Missing/invalid distance returns 0.0 (so min_distance_km can exclude it, as in recommender.py)
    """
    if not isinstance(station, dict):
        return 0.0

    # (key, multiplier_to_km)
    candidates = [
        ("distance_along_m", 1.0 / 1000.0),
        ("distance_to_station_m", 1.0 / 1000.0),
        ("distance_m", 1.0 / 1000.0),
        ("dist_m", 1.0 / 1000.0),
        ("distance_along_km", 1.0),
        ("distance_to_station_km", 1.0),
        ("distance_km", 1.0),
    ]

    for key, mul in candidates:
        v = station.get(key)
        if v is None:
            continue
        try:
            km = float(v) * float(mul)
        except (TypeError, ValueError):
            continue
        if km != km:  # NaN
            continue
        return max(0.0, km)

    return 0.0


def _is_missing_number(x: Any) -> bool:
    if x is None:
        return True
    try:
        # pandas/np NaN handling without importing numpy explicitly
        return bool(pd.isna(x))
    except Exception:
        return False


def _as_float(x: Any) -> Optional[float]:
    """Best-effort numeric conversion used for robust display computations."""
    if _is_missing_number(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _format_eta_local_for_display(eta_any: Any, tz_name: str = "Europe/Berlin") -> str:
    """
    Convert an ETA value (typically debug_*_eta_utc ISO string or station["eta"]) into
    Europe/Berlin local time for display. Returns a safe string; never throws.

    - If input is already a local ISO string with offset, this will keep it as local (or re-normalize).
    - If input is UTC (+00:00), this will convert to +01:00/+02:00 depending on DST.
    """
    if eta_any is None:
        return "—"

    # If it is already a datetime, normalize; otherwise parse from string
    dt: datetime
    try:
        if isinstance(eta_any, datetime):
            dt = eta_any
        else:
            s = str(eta_any).strip()
            if not s:
                return "—"
            # Handle "Z" suffix if present
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
    except Exception:
        # Fall back to raw value if parsing fails
        return _safe_text(eta_any)

    # If naive, assume UTC only as a last resort (debug values should be tz-aware)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to Europe/Berlin if available
    if ZoneInfo is not None:
        try:
            local_tz = ZoneInfo(tz_name)
            dt_local = dt.astimezone(local_tz)
            return dt_local.isoformat()
        except Exception:
            pass

    # Fallback: use system local timezone if ZoneInfo is unavailable
    try:
        dt_local = dt.astimezone()
        return dt_local.isoformat()
    except Exception:
        return _safe_text(eta_any)


def _parse_any_dt_to_berlin(dt_any: Any, tz_name: str = "Europe/Berlin") -> Optional[datetime]:
    """
    Parse a datetime-like value (ISO string / datetime / common formats) and convert to Europe/Berlin.
    Returns None on failure.
    """
    if dt_any is None:
        return None

    dt: Optional[datetime] = None

    try:
        if isinstance(dt_any, datetime):
            dt = dt_any
        else:
            s = str(dt_any).strip()
            if not s:
                return None
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            # ISO first
            try:
                dt = datetime.fromisoformat(s)
            except Exception:
                # common "computed_at" format from Page 01: "YYYY-mm-dd HH:MM:SS"
                try:
                    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    return None
    except Exception:
        return None

    # If naive, assume UTC (safer for containers); then convert to Berlin below.
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    if dt is None:
        return None

    # Convert to Europe/Berlin if possible
    if ZoneInfo is not None:
        try:
            return dt.astimezone(ZoneInfo(tz_name))
        except Exception:
            return dt
    return dt


def _time_cell_48(dt_local: Optional[datetime]) -> Optional[int]:
    """
    Map a local datetime to a 48-cell day grid (30-minute buckets):
      cell = hour*2 + (minute >= 30)
    Returns None if dt_local is None.
    """
    if dt_local is None:
        return None
    try:
        return int(dt_local.hour) * 2 + (1 if int(dt_local.minute) >= 30 else 0)
    except Exception:
        return None


def _station_google_address_best_effort(station: Dict[str, Any]) -> str:
    """
    Best-effort 'Address' WITHOUT calling external APIs on Page 02.
    If Page 01 already filled st.session_state['reverse_geocode_cache'], reuse it.
    Otherwise fall back to station fields (street/houseNumber/postCode/place/address).
    """
    try:
        cache = st.session_state.get("reverse_geocode_cache") or {}
    except Exception:
        cache = {}

    # Try to match Page-01 cache keying
    u = _station_uuid(station) or f"{station.get('lat')}_{station.get('lon')}"
    payload = cache.get(u) if isinstance(cache, dict) else None

    if isinstance(payload, dict):
        formatted = payload.get("formatted_address")
        if formatted:
            return str(formatted).strip()

    # Fallback: build from station data
    address = station.get("address")
    if address:
        return str(address).strip()

    street = station.get("street")
    house = station.get("houseNumber") or station.get("house_number")
    postcode = station.get("postCode") or station.get("postcode") or station.get("zip")
    place = station.get("place") or station.get("city")

    parts = []
    if street:
        parts.append(str(street).strip())
    if house:
        parts[-1] = f"{parts[-1]} {str(house).strip()}" if parts else str(house).strip()
    if postcode or place:
        parts.append(" ".join([str(x).strip() for x in [postcode, place] if x]))

    return ", ".join([p for p in parts if p]) if parts else "—"


def _coerce_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and not _is_missing_number(x):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
    return None


def _open_at_eta_flag(station: Dict[str, Any]) -> Optional[bool]:
    """Return whether the station is open at the Google ETA, if the pipeline provided the flag."""
    for k in ("is_open_at_eta", "open_at_eta", "google_open_at_eta", "is_open_google_eta"):
        v = _coerce_bool(station.get(k))
        if v is not None:
            return v
    return None


def _is_closed_at_eta(station: Dict[str, Any]) -> bool:
    """True if we have an explicit open-at-ETA flag and it is False."""
    v = _open_at_eta_flag(station)
    return v is False


def _render_page02_sidebar_action() -> None:
    """
    Page 02 (Route Analytics) Action tab:
    - Back to Trip Planner button (primary styling)
    - Analysis type selector (buttons)
    """

    # Top button: same "primary" style pattern as Page 04 ("Search Stations")
    if st.sidebar.button(
        "Back to Trip Planner",
        type="primary",
        use_container_width=True,
        key="p02_back_to_trip_planner",
        help="Return to the Home page.",
    ):
        # Do NOT set st.session_state["top_nav"] here (top_nav widget already exists on this page).
        # Instead, request Home navigation and let streamlit_app.py apply it before rendering its widget.
        st.session_state["nav_request_top_nav"] = "Home"

        try:
            maybe_persist_state(force=True)
        except Exception:
            pass

        st.switch_page("streamlit_app.py")

    with st.sidebar.container(key="p02_analysis_type_block"):
        # Page-02-only spacing fix (scoped to this container)
        st.markdown(
            """
            <style>
            /* Only affects content inside this specific sidebar container */
            section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stVerticalBlock"] div.p02-analysis-type-block) h3 {
                margin-bottom: 0.65rem !important;
            }
            section[data-testid="stSidebar"] div.p02-analysis-selected {
                margin-top: 0rem !important;
                margin-bottom: 1.2rem !important;
                font-size: 1rem;
                opacity: 0.78;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Add a marker div so the selector can scope reliably
        st.markdown("<div class='p02-analysis-type-block'></div>", unsafe_allow_html=True)

        st.markdown(
            "### Chose Analysis Type",
            help=(
                "Select which analytics view to display on this page. "
            ),
        )

        # Show current selection (or none) without using st.caption (your global CSS affects captions)
        current_view = str(st.session_state.get("route_analytics_view") or "").strip()
        selected_label = current_view if current_view else "None selected"
        st.markdown(
            f"<div class='p02-analysis-selected'>Selected: {selected_label}</div>",
            unsafe_allow_html=True,
        )

    def _apply_view_and_rerun(view: str) -> None:
        st.session_state["route_analytics_view"] = view
        try:
            maybe_persist_state(force=True)
        except Exception:
            pass

        # Streamlit version compatibility
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    st.sidebar.markdown(
        """
        <style>
        /* Page 02 sidebar: left-align ONLY the 4 analysis buttons (scoped by keys) */
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_route button,
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_stations button,
        section[data-testid="stSidebar"] .st-key-p02_view_prediction_algorithm button,
        section[data-testid="stSidebar"] .st-key-p02_view_full_result_tables button {
            width: 100%;
            justify-content: flex-start !important;
            text-align: left !important;
        }

        /* Streamlit typically wraps the label inside a button > div wrapper; reset that */
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_route button > div,
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_stations button > div,
        section[data-testid="stSidebar"] .st-key-p02_view_prediction_algorithm button > div,
        section[data-testid="stSidebar"] .st-key-p02_view_full_result_tables button > div {
            width: 100% !important;
            margin: 0 !important;
            justify-content: flex-start !important;
        }

        /* Force every possible label container to full-width + left aligned */
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_route button * ,
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_stations button * ,
        section[data-testid="stSidebar"] .st-key-p02_view_prediction_algorithm button * ,
        section[data-testid="stSidebar"] .st-key-p02_view_full_result_tables button * {
            text-align: left !important;
            justify-content: flex-start !important;
        }

        /* Optional: small left padding so it reads as “left-bound” */
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_route button,
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_stations button,
        section[data-testid="stSidebar"] .st-key-p02_view_prediction_algorithm button,
        section[data-testid="stSidebar"] .st-key-p02_view_full_result_tables button {
            padding-left: 1.0rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # View buttons (stacked, full width)
    if st.sidebar.button("**1. Recommended Route**", use_container_width=True, key="p02_view_recommended_route"):
        _apply_view_and_rerun("Recommended Route")

    if st.sidebar.button("**2. Recommended Stations**", use_container_width=True, key="p02_view_recommended_stations"):
        _apply_view_and_rerun("Recommended Stations")

    if st.sidebar.button("**3. Prediction Algorithm**", use_container_width=True, key="p02_view_prediction_algorithm"):
        _apply_view_and_rerun("Prediction Algorithm")

    if st.sidebar.button("**4. Full Result Tables**", use_container_width=True, key="p02_view_full_result_tables"):
        _apply_view_and_rerun("Full Result Tables")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Route Analytics", layout="wide")

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

    apply_app_css()

    # Header (consistent across pages)
    st.title("Route Analytics")
    st.caption("##### Understand station and route recommendations.")

    NAV_TARGETS = {
        "Home": "streamlit_app.py",
        "Analytics": "pages/02_route_analytics.py",
        "Station": "pages/03_station_details.py",
        "Explorer": "pages/04_station_explorer.py",
    }
    CURRENT = "Analytics"

    # Ensure the correct tab is selected when landing on this page,
    # but do not overwrite user interaction during the same run.
    if st.session_state.get("_active_page") != CURRENT:
        st.session_state["_active_page"] = CURRENT
        st.session_state["top_nav"] = CURRENT
        # Default behavior for Page 02: no analysis section selected on entry
        st.session_state["route_analytics_view"] = ""

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


    # Debug mode forced ON on Page 02
    st.session_state["debug_mode"] = True

    # Render standardized shell (Action/Help/Settings/Profile) with Page-02 Action content
    render_sidebar_shell(
        action_renderer=_render_page02_sidebar_action,
        help_renderer=_render_help_action,
        settings_renderer=_render_settings_quick_routes,
    )

    cached = _get_last_run()
    if not cached:
        st.info("No cached run found. Run a route recommendation first on the home page.")
        return


    fuel_code: str = str(cached.get("fuel_code") or "e5").lower()
    run_summary: Dict[str, Any] = cached.get("run_summary") or {}
    use_economics: bool = bool(run_summary.get("use_economics") or cached.get("use_economics", False))


    # Advanced Settings (persisted by Page 01)
    # Source of truth: Page 01 writes these into cached["advanced_settings"].
    # (Older runs may store parts in run_summary or inside run_summary["constraints"].)
    adv = (cached.get("advanced_settings") or run_summary.get("advanced_settings") or {})
    constraints = run_summary.get("constraints") or {}

    filter_closed_at_eta_enabled: bool = bool(
        (adv.get("filter_closed_at_eta") if isinstance(adv, dict) else False)
        or run_summary.get("filter_closed_at_eta")
        or cached.get("filter_closed_at_eta", False)
    )
    closed_at_eta_filtered_n = None
    if isinstance(adv, dict) and adv.get("closed_at_eta_filtered_n") is not None:
        closed_at_eta_filtered_n = adv.get("closed_at_eta_filtered_n")
    if closed_at_eta_filtered_n is None:
        closed_at_eta_filtered_n = run_summary.get("closed_at_eta_filtered_n", 0)

    # Brand filter (whitelist) configuration
    brand_filter_selected = []
    if isinstance(adv, dict) and adv.get("brand_filter_selected") is not None:
        brand_filter_selected = adv.get("brand_filter_selected") or []
    elif isinstance(constraints, dict) and constraints.get("brand_filter_selected") is not None:
        brand_filter_selected = constraints.get("brand_filter_selected") or []
    else:
        brand_filter_selected = run_summary.get("brand_filter_selected") or []

    if isinstance(brand_filter_selected, str):
        brand_filter_selected = [brand_filter_selected]
    brand_filter_selected = [str(x).strip() for x in brand_filter_selected if str(x).strip()]

    brand_filtered_out_n = None
    if isinstance(adv, dict) and adv.get("brand_filtered_out_n") is not None:
        brand_filtered_out_n = adv.get("brand_filtered_out_n")
    elif isinstance(constraints, dict) and constraints.get("brand_filtered_out_n") is not None:
        brand_filtered_out_n = constraints.get("brand_filtered_out_n")
    else:
        brand_filtered_out_n = run_summary.get("brand_filtered_out_n", 0)

    try:
        closed_at_eta_filtered_n = int(closed_at_eta_filtered_n or 0)
    except (TypeError, ValueError):
        closed_at_eta_filtered_n = 0

    try:
        brand_filtered_out_n = int(brand_filtered_out_n or 0)
    except (TypeError, ValueError):
        brand_filtered_out_n = 0


    litres_to_refuel = cached.get("litres_to_refuel")
    debug_mode: bool = bool(st.session_state.get("debug_mode", False))

    # Keep cache consistent so other pages (and this page on rerun) see the toggle.
    cached["debug_mode"] = debug_mode
    st.session_state["last_run"] = cached

    stations: List[Dict[str, Any]] = cached.get("stations") or []

    analysis_view = str(st.session_state.get("route_analytics_view") or "").strip()

    # Default landing: no selection => show welcome content below the top nav
    if not analysis_view:
        st.markdown(
            """
    ### Welcome to Route Analytics

    This page is the **analysis layer** for your last recommender run. It lets you inspect the route result end-to-end:
    from the chosen route, to the ranked stations, to the price prediction logic that drives the ranking.

    #### Page structure (sections)
    - **Recommended Route**  
    Visualizes the final route and key route settings used for the run.

    - **Recommended Stations**  
    Summarizes the top recommendations and key KPIs (including gross/net logic when economics is enabled).

    - **Prediction Algorithm**  
    Station-level audit trail for the price prediction pipeline (Spot vs Forecast decision, horizon selection, model inputs, and explainability).

    - **Full Result Tables**  
    Complete result tables for all stations (best for validating the final output and filtering/sorting by metrics).
    """
        )

        st.info("Get started: open the sidebar and select one of the analysis views to display its section.")

        maybe_persist_state()
        return

    # ------------------------------------------------------------
    # View routing (Page 02 subviews)
    # ------------------------------------------------------------

    if analysis_view == "Recommended Route":
        # Headline slightly smaller than the main title
        st.markdown("### **1. Recommended Route**")

        # [CHANGE 1] Decrease space above "Route settings:"
        # used negative margin-top to pull it up, and small bottom margin
        st.markdown(
            """
            <h4 style="margin-top: -1.0rem; margin-bottom: 0.25rem;">
                Route settings:
            </h4>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Route (Start -> Destination)
        # -----------------------------
        def _fmt_coord(pt: object) -> str:
            """Format a coordinate point robustly across common shapes."""
            if pt is None:
                return "—"
            try:
                # tuple/list: (lat, lon)
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    lat, lon = float(pt[0]), float(pt[1])
                    return f"{lat:.5f}, {lon:.5f}"
                # dict: {"lat":..,"lng":..} or {"latitude":..,"longitude":..}
                if isinstance(pt, dict):
                    if "lat" in pt and ("lng" in pt or "lon" in pt):
                        lat = float(pt["lat"])
                        lon = float(pt.get("lng", pt.get("lon")))
                        return f"{lat:.5f}, {lon:.5f}"
                    if "latitude" in pt and "longitude" in pt:
                        lat = float(pt["latitude"])
                        lon = float(pt["longitude"])
                        return f"{lat:.5f}, {lon:.5f}"
            except Exception:
                return "—"
            return "—"


        # Prefer the cached run inputs (stable even if user changed sidebar afterward)
        params = cached.get("params") or {}
        start_label = str(params.get("start_locality") or st.session_state.get("start_locality") or "").strip()
        end_label = str(params.get("end_locality") or st.session_state.get("end_locality") or "").strip()
        start_label = start_label if start_label else "—"
        end_label = end_label if end_label else "—"

        # Coordinates: use route endpoints from the cached route polyline
        route_coords = None
        # Always read route_info/route_coords from the cached last_run on Page 02.
        route_info = cached.get("route_info") or {}

        # 1) Preferred: exact geocoding coords used for routing (lat/lon)
        start_coord = route_info.get("start_coord")
        end_coord = route_info.get("end_coord")

        # 2) Fallback: derive from route polyline endpoints (often [lon, lat])
        route_coords = route_info.get("route_coords") or cached.get("route_coords") or []

        def _coerce_latlon_from_route_point(pt: object):
            """
            Route polyline points in this project are typically [lon, lat].
            Convert to (lat, lon) for display when we can infer it.
            """
            try:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    a, b = float(pt[0]), float(pt[1])

                    # Heuristic:
                    # - latitude must be within [-90, 90]
                    # - longitude within [-180, 180]
                    # If (a,b) fits (lon,lat), swap to (lat,lon)
                    if abs(a) <= 180 and abs(b) <= 90:
                        return (b, a)  # (lat, lon)
                    if abs(a) <= 90 and abs(b) <= 180:
                        return (a, b)  # already (lat, lon)
            except Exception:
                pass
            return None

        if not start_coord or not end_coord:
            if isinstance(route_coords, list) and len(route_coords) >= 1:
                start_fallback = _coerce_latlon_from_route_point(route_coords[0])
                end_fallback = _coerce_latlon_from_route_point(route_coords[-1])
                start_coord = start_coord or start_fallback
                end_coord = end_coord or end_fallback

        # Page-02-only styling for these cards (scoped via unique class)
        st.markdown(
            """
            <style>
            .p02-route-cards {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.45rem;
                margin-top: -1.4rem;
                margin-bottom: 0.5rem;
            }
            .p02-route-card {
                width: min(720px, 100%);
                padding: 0.85rem 1.0rem;
                border: 3px solid rgba(49, 51, 63, 0.3);
                border-radius: 0.9rem;
                text-align: center;
            }
            .p02-route-title {
                font-size: 1.3rem;
                font-weight: 750;
                margin: 0 0 0.25rem 0;
                line-height: 1.15;
            }
            .p02-route-sub {
                font-size: 0.92rem;
                opacity: 0.78;
                margin: 0;
                line-height: 1.2;
            }
            .p02-route-arrow {
                font-size: 1.35rem;
                font-weight: 800;
                opacity: 0.85;
                line-height: 1;
                margin: 0.05rem 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="p02-route-cards">
            <div class="p02-route-card">
                <div class="p02-route-title">{start_label}</div>
                <div class="p02-route-sub">Google coordinates: {_fmt_coord(start_coord)}</div>
            </div>

            <div class="p02-route-arrow" style="font-size: 40px; font-weight: bold;">↓</div>

            <div class="p02-route-card">
                <div class="p02-route-title">{end_label}</div>
                <div class="p02-route-sub">Google coordinates: {_fmt_coord(end_coord)}</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # [CHANGE 2] Decrease space below "Recommender constraints:"
        # Set margin-bottom to a very small value (0.1rem) so cards sit tight against it
        st.markdown(
            """
            <h4 style="margin-top: 0.3rem; margin-bottom: -1.3rem;">
                Recommender constraints:
            </h4>
            """,
            unsafe_allow_html=True
        )

        # Read constraints exactly as used on Page 01 (cached in last_run["run_summary"])
        constraints = run_summary.get("constraints") or {}

        max_detour_km = constraints.get("max_detour_km")
        max_detour_min = constraints.get("max_detour_min")

        litres_to_refuel = constraints.get("litres_to_refuel")
        consumption_l_per_100km = constraints.get("consumption_l_per_100km")
        value_of_time_eur_per_hour = constraints.get("value_of_time_eur_per_hour")
        min_net_saving_eur = constraints.get("min_net_saving_eur") if use_economics else None

        # Advanced Settings (persisted by Page 01)
        adv = (run_summary.get("advanced_settings") or cached.get("advanced_settings") or {})
        open_at_eta = bool(
            adv.get("filter_closed_at_eta")
            or run_summary.get("filter_closed_at_eta")
            or cached.get("filter_closed_at_eta")
        )

        brand_filter_selected = adv.get("brand_filter_selected") or []
        if isinstance(brand_filter_selected, str):
            brand_filter_selected = [brand_filter_selected]
        brand_filter_selected = [str(x).strip() for x in brand_filter_selected if str(x).strip()]
        brand_filtered_out_n = adv.get("brand_filtered_out_n")
        try:
            brand_filtered_out_n = int(brand_filtered_out_n) if brand_filtered_out_n is not None else 0
        except (TypeError, ValueError):
            brand_filtered_out_n = 0

        # UI summary (responsive via your existing metric-grid CSS)
        items = [
            ("Fuel type", fuel_code.upper()),
            ("Decision mode", "Economics-based" if use_economics else "Price-only"),
            ("Max extra distance", _format_km(max_detour_km)),
            ("Max extra time", _format_min(max_detour_min)),
        ]

        if use_economics:
            items.extend(
                [
                    ("Litres to refuel", _format_liters(litres_to_refuel)),
                    ("Consumption", f"{float(consumption_l_per_100km):.2f} L/100 km" if consumption_l_per_100km is not None else "—"),
                    ("Value of time", _format_eur(value_of_time_eur_per_hour) + "/h" if value_of_time_eur_per_hour is not None else "—"),
                    ("Min net saving", _format_eur(min_net_saving_eur)),
                ]
            )

        items.append(("Stations open at ETA", "On" if open_at_eta else "Off"))
        items.append(("Brand filter", "Any" if not brand_filter_selected else f"{len(brand_filter_selected)} selected"))

        render_card_grid(items, cols=2)


        # [CHANGE 3] Increase space above + Add Help Tooltip
        # We use a manual spacer div to create the margin, then use standard
        # st.markdown with help="..." to generate the "?" hover button.
        st.markdown("<div style='margin-top: -1px;'></div>", unsafe_allow_html=True)
        
        st.markdown(
            "#### Key route visualization:",
            help=(
                "Visual comparison of the route options:\n\n"
                "• **Blue line**: Your original Google Maps route (Baseline).\n\n"
                "• **Purple line**: The alternative route that includes the detour to the station.\n\n"
                "• **Red dot**: The recommended station's exact location along the trip.\n\n"
                "The bar lengths are proportional to the total trip distance."
            )
        )

        best_station: Optional[Dict[str, Any]] = cached.get("best_station")
        route_info: Dict[str, Any] = cached.get("route_info") or {}
        render_key_visualization(route_info=route_info, best_station=best_station)

        maybe_persist_state()
        return


    # ------------------------------------------------------------------
    # 2. Recommended Stations
    # ------------------------------------------------------------------

    if analysis_view == "Recommended Stations":
        st.markdown(
            """
            <h3 style='margin-top: 0px; margin-bottom: -15px;'>
                <b>2. Recommended Stations</b>
            </h3>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(
            "#### Stations funnel",
            help="""This section reports the station selection pipeline in two stages:

1) Hard-feasible: stations that remain after non-negotiable constraints are applied (e.g., upstream filters like brand and closed-at-ETA, valid predicted price, deduplication, detour caps).

2) Economically selected: among hard-feasible stations, a station is considered economically viable if its net saving is at least as high as the worst on-route station (benchmark).

If no on-route stations exist, the benchmark is not applicable and the economics stage is skipped (all hard-feasible stations are counted as economically selected).""",
        )

        # Force cards to sit higher, counteracting st.columns padding
        st.markdown(
            """
            <style>
            /* Page 2 specific override for cards */
            .p02-card {
                margin-top: -1rem !important; 
                margin-bottom: 1.3rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Per-element spacing controls (REM)
        # Adjust these values to tune spacing between specific elements.
        # -----------------------------
        P02_S = {
            # Heading -> cards
            "after_h_selected": 0,
            "after_cards_selected": 0,

            # Chart 1 block
            "before_chart_selected": 0,
            "after_chart_selected": 0,

            # Heading 2
            "before_h_discarded": 0,
            "after_h_discarded": 0,

            # Chart 2 block
            "before_chart_discarded": 0,
            "after_chart_discarded": 0.5,
        }

        def _gap(rem: float) -> None:
            """Deterministic vertical spacing. Supports negative values to pull elements up."""
            try:
                r = float(rem)
            except Exception:
                return
            
            if r == 0:
                return
            
            if r > 0:
                # Positive gap: creates empty vertical space
                st.markdown(f"<div style='height:{r}rem;'></div>", unsafe_allow_html=True)
            else:
                # Negative gap: pulls the next element up using negative margin
                st.markdown(f"<div style='margin-top:{r}rem;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <style>
            .tsf-chart-block {
                margin-top: 0rem;     /* space ABOVE the chart block */
                margin-bottom: -1rem;  /* space BELOW the chart block */
                padding-left: 0.1rem;   /* left "margin" via padding */
                padding-right: 0.00rem;  /* right "margin" via padding */
            }

            /* Optional: reduce plot container padding (Streamlit default is sometimes roomy) */
            .tsf-chart-block [data-testid="stVegaLiteChart"] {
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # -----------------------------
        # Local helpers)
        # -----------------------------
        def _as_int(x: Any) -> int:
            try:
                return int(x or 0)
            except Exception:
                return 0

        def _donut_chart(
            title: str,
            counts: Dict[str, int],
            *,
            center_text: Optional[str] = None,
            height: int = 320,
            min_label_share: float = 0.0,
            space_top: float = 0.0,
            space_bottom: float = 0.0,
            pad_top: int = 10,
            pad_bottom: int = 0,
        ) -> None:
            """
            Responsive donut chart with:
            - Legend always below the chart (no overlap on mobile)
            - Legend includes ALL categories, even when Count == 0
            - Slice labels placed inside the ring to avoid boundary artifacts
            """
            # Preserve input order for stable legend/category mapping
            domain = list(counts.keys())

            df_all = pd.DataFrame(
                [{"Category": k, "Count": int(max(0, v))} for k, v in counts.items()]
            )

            # Stable ordering: map the input key order to a numeric index
            order_map = {k: i for i, k in enumerate(domain)}
            df_all["sort_idx"] = df_all["Category"].map(order_map).astype(int)

            total = int(df_all["Count"].sum())
            if total <= 0:
                st.caption("No data available for this chart.")
                return

            df_all["Share"] = df_all["Count"] / float(total)

            # Slice label: absolute counts (keep min_label_share as the visibility threshold)
            df_all["Label"] = df_all.apply(
                lambda r: (
                    f"{int(r['Count']):,}"
                    if (float(r["Share"]) >= float(min_label_share) and int(r["Count"]) > 0)
                    else ""
                ),
                axis=1,
            )

            # Separate view for arcs (Count > 0), but keep a hidden layer for legend completeness
            df_pos = df_all[df_all["Count"] > 0].copy()

            df_pos["sort_idx"] = df_pos["Category"].map(order_map).astype(int)

            legend = alt.Legend(
                orient="bottom",
                direction="vertical",
                title=None,
                labelLimit=240,
                symbolSize=120,
            )

            color_enc = alt.Color(
                "Category:N",
                legend=legend,
                scale=alt.Scale(domain=domain),  # stable category -> color assignment
            )

            # Hidden layer that "forces" the legend to include ALL categories (including zeros)
            legend_layer = (
                alt.Chart(df_all)
                .mark_point(opacity=0)
                .encode(color=color_enc)
            )

            arcs = (
                alt.Chart(df_pos)
                .mark_arc(innerRadius=70, outerRadius=125)
                .encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    color=color_enc,
                    tooltip=[
                        alt.Tooltip("Category:N", title="Category"),
                        alt.Tooltip("Count:Q", title="Count", format=",.0f"),
                        alt.Tooltip("Share:Q", title="Share", format=".0%"),
                    ],
                    # Explicit ordering prevents rare "flip" effects across reruns
                    order=alt.Order("sort_idx:Q"),
                )
            )

            # Place percentage labels inside the ring to avoid boundary placement on small screens
            labels = (
                alt.Chart(df_pos)
                .mark_text(radius=98, size=12, fontWeight="bold")
                .encode(
                    theta=alt.Theta("Count:Q", stack=True),
                    text="Label:N",
                    order=alt.Order("sort_idx:Q"),
                )
            )

            center = (
                alt.Chart(pd.DataFrame({"text": [center_text or f"{total}"]}))
                .mark_text(size=26, fontWeight="bold")
                .encode(text="text:N")
            )

            chart = (
                (legend_layer + arcs + labels + center)
                .properties(
                    height=height,
                    title=title,
                    padding={"left": 10, "right": 10, "top": pad_top, "bottom": pad_bottom},
                )
                .configure_view(stroke=None)
                .configure_title(offset=12)
                .configure_legend(
                    orient="bottom",
                    direction="vertical",
                    titleFontSize=12,
                    labelFontSize=12,
                    padding=0,
                    offset=20,
                    labelColor="black",
                    titleColor="black",
                )
            )

            _gap(space_top)

            st.altair_chart(chart, use_container_width=True)

            _gap(space_bottom)


        # -----------------------------
        # Pull required cached data
        # -----------------------------
        ranked: List[Dict[str, Any]] = cached.get("ranked") or []
        constraints = run_summary.get("constraints") or {}
        cap_km = constraints.get("max_detour_km")
        cap_min = constraints.get("max_detour_min")

        min_distance_km = constraints.get("min_distance_km")
        max_distance_km = constraints.get("max_distance_km")

        pred_key = f"pred_price_{fuel_code}"
        econ_net_key = f"econ_net_saving_eur_{fuel_code}"

        # Upstream filters (already parsed above this view)
        eta_upstream_n = _as_int(closed_at_eta_filtered_n) if filter_closed_at_eta_enabled else 0

        # Page 01 applies the brand filter upstream (before caching) but keeps the pre-brand station
        # universe in cached["stations_for_map_all"]. Use it to keep counts/tables audit-correct.
        stations_universe = cached.get("stations_for_map_all")
        if not isinstance(stations_universe, list) or (stations_universe and not isinstance(stations_universe[0], dict)):
            stations_universe = stations

        # "stations" in cache are AFTER upstream removals (brand + open-at-ETA if enabled).
        candidates_in_cache_n = len(stations)

        # Brand-filtered-out count: prefer the explicit audit number, but validate against the stored universe.
        if isinstance(brand_filter_selected, list) and len(brand_filter_selected) > 0:
            inferred_brand_n = max(0, len(stations_universe) - candidates_in_cache_n)
            brand_upstream_n = max(_as_int(brand_filtered_out_n), int(inferred_brand_n))
        else:
            brand_upstream_n = 0

        # Total stations found (includes upstream removals). Note:
        # - stations_universe already includes brand-filtered-out stations (if brand filter active)
        # - eta_upstream_n must still be added because those stations were removed before stations_universe existed
        total_found_n = max(0, len(stations_universe) + eta_upstream_n)


        # -----------------------------
        # SECTION: Stations funnel (Found → Hard-feasible → Economically selected)
        # -----------------------------

        # --- Reconstruct "hard-feasible" universe (mirrors recommender hard filters) ---
        # Hard-feasible means:
        #   - upstream filters already applied (closed-at-ETA if enabled, brand filter if enabled)
        #   - valid predicted price (>= MIN_VALID_PRICE_EUR_L)
        #   - deduplicated by station_uuid (keep lowest predicted)
        #   - detour caps applied when relevant (economics mode OR user provided caps)
        def _is_valid_price_local(v: Any, *, min_price: float = MIN_VALID_PRICE_EUR_L) -> bool:
            if _is_missing_number(v):
                return False
            try:
                f = float(v)
            except (TypeError, ValueError):
                return False
            if f != f:  # NaN
                return False
            return f >= float(min_price)

        def _dedupe_key(s: Dict[str, Any]) -> Optional[str]:
            u = (s or {}).get("station_uuid")
            if u is None:
                return None
            try:
                return str(u)
            except Exception:
                return None

        # 1) Valid-price candidates (in-cache, after upstream filters)
        valid_candidates: List[Dict[str, Any]] = []
        invalid_or_missing_pred_n = 0
        for s in stations:
            p = (s or {}).get(pred_key)
            if _is_valid_price_local(p):
                valid_candidates.append(s)
            else:
                invalid_or_missing_pred_n += 1

        # 2) Dedupe by station_uuid (keep lowest predicted)
        deduped: List[Dict[str, Any]] = []
        duplicates_removed_n = 0
        by_uuid: Dict[str, Dict[str, Any]] = {}
        dup_counts: Dict[str, int] = {}

        for s in valid_candidates:
            k = _dedupe_key(s)
            if not k:
                # No UUID -> treat as unique (do not dedupe)
                deduped.append(s)
                continue

            dup_counts[k] = dup_counts.get(k, 0) + 1
            existing = by_uuid.get(k)
            if existing is None:
                by_uuid[k] = s
            else:
                # keep lowest predicted price variant
                try:
                    if float(s.get(pred_key)) < float(existing.get(pred_key)):
                        by_uuid[k] = s
                except Exception:
                    pass

        for k, n in dup_counts.items():
            if n > 1:
                duplicates_removed_n += (n - 1)

        deduped.extend(list(by_uuid.values()))

        # 3) Apply detour caps (same high-level semantics as recommender)
        # In economics mode, caps are mandatory and default to the recommender defaults if unset.
        cap_km_eff = _as_float(cap_km)
        cap_min_eff = _as_float(cap_min)
        if bool(use_economics):
            if cap_km_eff is None:
                cap_km_eff = 10.0
            if cap_min_eff is None:
                cap_min_eff = 10.0

        should_cap = (cap_km_eff is not None) or (cap_min_eff is not None)
        cap_failed_n = 0

        cap_pass: List[Dict[str, Any]] = []
        if should_cap:
            for s in deduped:
                km, mins = _detour_metrics(s)
                cap_fail = False
                try:
                    if cap_km_eff is not None and float(km) > float(cap_km_eff):
                        cap_fail = True
                except Exception:
                    pass
                try:
                    if cap_min_eff is not None and float(mins) > float(cap_min_eff):
                        cap_fail = True
                except Exception:
                    pass

                if cap_fail:
                    cap_failed_n += 1
                else:
                    cap_pass.append(s)
        else:
            cap_pass = list(deduped)

        # 4) Apply distance window (min/max distance to station) — NEW hard constraints
        min_distance_km_eff = _as_float(min_distance_km)
        max_distance_km_eff = _as_float(max_distance_km)

        below_min_failed_n = 0
        above_max_failed_n = 0

        hard_feasible: List[Dict[str, Any]] = []
        if (min_distance_km_eff is not None) or (max_distance_km_eff is not None):
            for s in cap_pass:
                dist_km = _distance_along_km(s)

                if min_distance_km_eff is not None and dist_km < float(min_distance_km_eff):
                    below_min_failed_n += 1
                    continue

                if max_distance_km_eff is not None and dist_km > float(max_distance_km_eff):
                    above_max_failed_n += 1
                    continue

                hard_feasible.append(s)
        else:
            hard_feasible = list(cap_pass)

        hard_feasible_n = len(hard_feasible)

        # Hard-discarded = total found - hard feasible
        hard_discarded_n = max(0, total_found_n - hard_feasible_n)

        # --- Economic selection benchmark: "worst on-route" net saving ---
        econ_benchmark_available = False
        worst_onroute_net: Optional[float] = None

        if use_economics and hard_feasible:
            onroute_nets: List[float] = []
            for s in hard_feasible:
                km, mins = _detour_metrics(s)
                if km <= 1.0 and mins <= 5.0:
                    net = _as_float((s or {}).get(econ_net_key))
                    if net is not None:
                        onroute_nets.append(float(net))

            if onroute_nets:
                econ_benchmark_available = True
                worst_onroute_net = float(min(onroute_nets))

        # Economic selection rule:
        # - If benchmark exists: selected iff net_saving >= worst_onroute_net
        # - If no on-route exists: benchmark is N/A -> pass-through all hard-feasible
        econ_selected: List[Dict[str, Any]] = []
        econ_rejected: List[Dict[str, Any]] = []

        if use_economics and hard_feasible and econ_benchmark_available and worst_onroute_net is not None:
            for s in hard_feasible:
                net = _as_float((s or {}).get(econ_net_key))
                if net is None:
                    # Unexpected in proper economics runs; keep it visible as rejected for auditability.
                    econ_rejected.append(s)
                    continue
                try:
                    if float(net) >= float(worst_onroute_net):
                        econ_selected.append(s)
                    else:
                        econ_rejected.append(s)
                except Exception:
                    econ_rejected.append(s)
        else:
            # Case 1 (and also "economics off"): benchmark not applicable.
            econ_selected = list(hard_feasible)
            econ_rejected = []

        econ_selected_n = len(econ_selected)
        econ_rejected_n = len(econ_rejected)

        # Final discarded definition for this view:
        # discarded = hard discarded + economic rejected
        discarded_total_n = max(0, total_found_n - econ_selected_n)

        # -----------------------------
        # Headline + cards
        # -----------------------------

        _gap(P02_S["after_h_selected"])

        econ_selected_display = f"{econ_selected_n:,}"
        if not econ_benchmark_available:
            econ_selected_display = f"{econ_selected_n:,} (N/A)"

        render_card_grid(
            [
                ("Stations found", f"{total_found_n:,}"),
                ("Hard-feasible", f"{hard_feasible_n:,}"),
                ("Economically selected", econ_selected_display),
            ],
            cols=3,
        )

        _gap(P02_S["after_cards_selected"])

        # -----------------------------
        # Chart 1: Funnel segments (3 slices)
        # -----------------------------
        _donut_chart(
            "",
            {
                "Hard discarded": hard_discarded_n,
                "Economically rejected": econ_rejected_n,
                "Economically selected": econ_selected_n,
            },
            center_text=f"{total_found_n}",
            height=340,
            space_top=P02_S["before_chart_selected"],
            space_bottom=P02_S["after_chart_selected"],
        )

        # -----------------------------
        # SECTION: Discarded reasons (hard vs economics)
        # -----------------------------
        _gap(P02_S["before_h_discarded"])

        st.markdown(
            "#### Discarded reasons",
            help="""Discarded stations are explained using the same two-stage pipeline:

- Hard discards: stations removed by upstream filters or hard constraints (brand filter, closed at ETA, invalid/missing predicted price, detour caps, duplicate removal).

- Economic rejection: stations that are hard-feasible but do not meet the benchmark (net saving below the worst on-route station).

Counts reconcile to: Discarded = Found − Economically selected.""",
        )

        _gap(P02_S["after_h_discarded"])

        reasons: Dict[str, int] = {
            "Brand filter": 0,
            "Closed at ETA": 0,
            "Missing prediction": 0,
            "Below minimum distance": 0,
            "Above maximum distance": 0,
            "Duplicate station removed": 0,
            "Failed detour caps (time/distance)": 0,
            "Economically rejected": 0,
            "Other / unclassified": 0,
        }

        # 1) Upstream removals
        reasons["Brand filter"] += max(0, brand_upstream_n)
        reasons["Closed at ETA"] += max(0, eta_upstream_n)

        # 2) In-cache hard discards (derived)
        reasons["Missing prediction"] += max(0, int(invalid_or_missing_pred_n))
        reasons["Duplicate station removed"] += max(0, int(duplicates_removed_n))
        reasons["Failed detour caps (time/distance)"] += max(0, int(cap_failed_n))
        reasons["Below minimum distance"] += max(0, int(below_min_failed_n))
        reasons["Above maximum distance"] += max(0, int(above_max_failed_n))

        # 3) Economics rejections (benchmark-based)
        if econ_benchmark_available:
            reasons["Economically rejected"] += max(0, int(econ_rejected_n))

        # Reconcile totals (important for auditability)
        known_sum = sum(int(v) for v in reasons.values())
        if known_sum != discarded_total_n:
            delta = int(discarded_total_n - known_sum)
            reasons["Other / unclassified"] = max(0, int(reasons["Other / unclassified"]) + delta)

        _donut_chart(
            "",
            reasons,
            center_text=f"{discarded_total_n}",
            height=380,
            space_top=P02_S["before_chart_discarded"],
            space_bottom=P02_S["after_chart_discarded"],
            pad_top=40,
            pad_bottom=40,
        )

    # -----------------------------
    # SECTION: Important results
    # -----------------------------
        st.markdown(
            "#### Important results",
            help=(
                "**What these cards show (top block):**\n\n"
                "**Prices (€/L)**\n"
                "- **Best (pred):** lowest *predicted* arrival price among the stations considered for ranking.\n"
                "- **Best (current):** lowest *current realtime* price among the same candidate set.\n\n"
                "**Detour & time costs (€)**\n"
                "- **Detour fuel:** estimated fuel cost caused by the detour distance/time.\n"
                "- **Time cost:** monetized time penalty for the detour (if time valuation is enabled; otherwise 0).\n\n"
                "**Net savings (€)**\n"
                "- **Net vs median:** net savings compared to the median station (savings after detour/time costs).\n"
                "- **Net vs worst:** net savings compared to the most expensive station (after detour/time costs).\n\n"
                "**General:** Net savings = (reference cost - chosen station cost) - detour costs - time cost."
            )
        )

        # --- Pull best-station + benchmark prices (display-only computations) ---
        best_station: Optional[Dict[str, Any]] = cached.get("best_station")
        pred_key = f"pred_price_{fuel_code}"
        curr_key = f"price_current_{fuel_code}"
        detour_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
        time_cost_key = f"econ_time_cost_eur_{fuel_code}"

        def _fmt_price_or_dash(x: Any) -> str:
            v = _as_float(x)
            if v is None or v < MIN_VALID_PRICE_EUR_L:
                return "—"
            return _fmt_price(v)

        def _fmt_eur_or_dash(x: Any) -> str:
            v = _as_float(x)
            if v is None:
                return "—"
            return _fmt_eur(v)

        def _onroute_price_stats(
            universe: List[Dict[str, Any]],
        ) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
            """
            Return (best, median, worst, n_onroute) predicted price among on-route stations
            within the provided universe.

            IMPORTANT: We intentionally compute benchmarks from the hard-feasible universe
            (post hard-constraints), not from `ranked`, because `ranked` may be economically
            filtered in a way that removes all on-route stations.
            """
            vals: List[float] = []
            n_onroute = 0

            for s in universe:
                p = _as_float((s or {}).get(pred_key))
                if p is None or p < MIN_VALID_PRICE_EUR_L:
                    continue

                km, mins = _detour_metrics(s)
                if km <= 1.0 and mins <= 5.0:
                    n_onroute += 1
                    vals.append(float(p))

            if not vals:
                return None, None, None, n_onroute

            ser = pd.Series(vals, dtype="float")
            best = float(ser.min())
            median = float(ser.median())
            worst = float(ser.max())
            return best, median, worst, n_onroute

        # Benchmark universe = hard-feasible (stable, post hard constraints)
        on_best, on_median, on_worst, onroute_n = _onroute_price_stats(hard_feasible)

        onroute_benchmark_available = (
            on_best is not None and on_median is not None and on_worst is not None
        )

        if not onroute_benchmark_available:
            st.info(
                "On-route benchmark unavailable for this run: no on-route station (≤ 1.0 km detour and ≤ 5 min detour) "
                "with a valid predicted price remained after hard constraints. Benchmark-driven KPIs are shown as N/A."
            )

        chosen_pred = _as_float(best_station.get(pred_key)) if isinstance(best_station, dict) else None
        chosen_curr = _as_float(best_station.get(curr_key)) if isinstance(best_station, dict) else None

        detour_fuel_cost = _as_float(best_station.get(detour_cost_key)) if isinstance(best_station, dict) else None
        time_cost = _as_float(best_station.get(time_cost_key)) if isinstance(best_station, dict) else None

        litres = _as_float(litres_to_refuel)

        def _gross_saving(ref_price: Optional[float]) -> Optional[float]:
            if ref_price is None or chosen_pred is None or litres is None:
                return None
            return (float(ref_price) - float(chosen_pred)) * float(litres)

        def _net_saving(ref_price: Optional[float]) -> Optional[float]:
            g = _gross_saving(ref_price)
            if g is None or detour_fuel_cost is None or time_cost is None:
                return None
            return float(g) - float(detour_fuel_cost) - float(time_cost)

        # Primary KPIs
        net_vs_median_str = (
            "N/A" if not onroute_benchmark_available else _fmt_eur_or_dash(_net_saving(on_median))
        )
        net_vs_worst_str = (
            "N/A" if not onroute_benchmark_available else _fmt_eur_or_dash(_net_saving(on_worst))
        )

        primary_cards: List[Tuple[str, str]] = [
            ("Best (pred)", _fmt_price_or_dash(chosen_pred)),
            ("Best (current)", _fmt_price_or_dash(chosen_curr)),
            ("Detour fuel", _fmt_eur_or_dash(detour_fuel_cost)),
            ("Time cost", _fmt_eur_or_dash(time_cost)),
            ("Net vs median", net_vs_median_str),
            ("Net vs worst", net_vs_worst_str),
        ]
        render_card_grid(primary_cards, cols=3)

        # Benchmarks
        st.markdown(
            """
            #### Benchmarks (on-route, predicted):
            """,
            help=(
                "**What these benchmark prices mean:**\n\n"
                "These are *predicted arrival prices* for stations that are **on-route** (minimal/no detour), used as a reference distribution.\n\n"
                "- **On-route best:** lowest predicted price among on-route stations.\n"
                "- **On-route median:** median predicted price among on-route stations.\n"
                "- **On-route worst:** highest predicted price among on-route stations.\n\n"
                "**How to use them:**\n"
                "- They provide context for whether the recommended station is meaningfully cheaper than typical on-route options.\n"
                "- They are *price-only* benchmarks; detour/time costs are handled separately in the net savings cards."
            ),
            unsafe_allow_html=True,

        )
        benchmark_cards: List[Tuple[str, str]] = [
            ("On-route best", "N/A" if not onroute_benchmark_available else _fmt_price_or_dash(on_best)),
            ("On-route median", "N/A" if not onroute_benchmark_available else _fmt_price_or_dash(on_median)),
            ("On-route worst", "N/A" if not onroute_benchmark_available else _fmt_price_or_dash(on_worst)),
        ]
        render_card_grid(benchmark_cards, cols=3)

        # Savings breakdown (collapsed)
        with st.expander("**Savings breakdown**", expanded=False):

            gross_cards: List[Tuple[str, str]] = [
                ("Gross vs best", "N/A" if not onroute_benchmark_available else _fmt_eur_or_dash(_gross_saving(on_best))),
                ("Gross vs median", "N/A" if not onroute_benchmark_available else _fmt_eur_or_dash(_gross_saving(on_median))),
                ("Gross vs worst", "N/A" if not onroute_benchmark_available else _fmt_eur_or_dash(_gross_saving(on_worst))),
            ]
            net_cards: List[Tuple[str, str]] = [
                ("Net vs best", "N/A" if not onroute_benchmark_available else _fmt_eur_or_dash(_net_saving(on_best))),
                ("Net vs median", "N/A" if not onroute_benchmark_available else _fmt_eur_or_dash(_net_saving(on_median))),
                ("Net vs worst", "N/A" if not onroute_benchmark_available else _fmt_eur_or_dash(_net_saving(on_worst))),
            ]

            render_card_grid(gross_cards, cols=3)
            render_card_grid(net_cards, cols=3)




    # ------------------------------------------------------------------
    # 3. Prediction Algorithm 
    # ------------------------------------------------------------------

    if analysis_view == "Prediction Algorithm":
        # Headline in the same style as the other sections
        st.markdown("### **3. Prediction Algorithm**")

        st.markdown(
            """
            For **every station** found along your route, the **system predicts** a price that is used on the map and in the ranking.
            This happens in a **consistent, station-level pipeline**:
            """
        )

        with st.expander("High-level pipeline"):
            st.markdown(
                """
            **What happens for each station on the route;**  
            The system runs a consistent, station-level prediction pipeline. The goal is to produce one **expected price at arrival** per station.


            #### 1) ETA is computed per station
            - The system combines:
                - the **base route travel time**, and
                - the station’s **detour time**
            - -> **ETA (arrival time)** for every station.
            - The ETA determines **whether** a forecast is needed and **which forecast horizon** applies.


            #### 2) Spot vs Forecast decision
            - If **ETA ≤ 10 minutes** → **Spot mode**
                - **No model** is used.
                - **Predicted price = current realtime price**.
            - If **ETA > 10 minutes** → **Forecast mode**
                - System uses a trained model to estimate the **expected price at arrival**.

            This prevents “fake forecasting” for stations that will be reached almost immediately.

            
            #### 3) Time-cell mapping (Forecast mode only)
            To keep forecasting consistent across times of day, the system maps time to fixed buckets:
            - A day is split into **48 time cells** (each cell = **30 minutes**).
            - The system derives:
                - **Now cell** (current local time bucket),
                - **ETA cell** (arrival time bucket),
                - **Cells ahead** = how many 30-minute buckets lie between Now and ETA.

            This turns an ETA timestamp into a discrete horizon choice.

            
            #### 4) Horizon selection and which model is used
            **Forecast mode selects one of the trained horizon models** based on how far the station lies in the future:

            - Short horizons (**intraday**):  
                - **h1 / h2 / h3 / h4** correspond to roughly **30 / 60 / 90 / 120 minutes** ahead  
                - Used when the ETA is “close enough” that intraday structure matters
            - Longer horizons (**daily**):  
                - **h0_daily** is used when the ETA is farther out (beyond the short intraday horizons)

            Result:
            - Every station in Forecast mode is paired with exactly one model:
            - `fuel_price_model_ARDL_<fuel>_h{1..4}_<...>.joblib` or  
            - `fuel_price_model_ARDL_<fuel>_h0_daily.joblib`

            
            #### 5) What the model uses as inputs

            **Future price at arrival ≈ historical price at comparable times + an intraday anchor (for short horizons)**

            Inputs at runtime match the training feature set:

            - **Historical daily lags** (same time-of-day cell on prior days):
                - `price_lag_1d`, `price_lag_2d`, `price_lag_3d`, `price_lag_7d`
            - **Intraday anchor** (only for h1–h4):
                - a horizon-specific intraday feature (e.g., `price_lag_1cell`, `price_lag_2cell`, …)

            Connection between training and runtime:
            - During training:
                - fitting of **separate models per fuel type and horizon** so each model can learn a stable mapping for that lookahead.
            - During prediction:
                - reproduction of the same feature schema and then selection of the corresponding horizon model based on the station’s ETA.
            - This is why there are **15 models total** (3 fuels × 5 horizons):  
                -> each one is specialized for its forecast horizon and fuel type.


            #### 6) Output: one expected arrival price 
            For each station, the pipeline yields exactly one value:
            - **Spot mode** → expected price = current realtime price
            - **Forecast mode** → expected price = model forecast at ETA
        
            """
            )

        # -----------------------------
        # A–D: User-facing explanation (keep lightweight; no ranking logic changes)
        # -----------------------------

        # Prefer the broadest available station universe for auditability:
        # - if a brand filter is active, Page 01 keeps the pre-brand list in "stations_for_map_all"
        # - otherwise fall back to "stations", then to ranked variants
        stations_for_explain: List[Dict[str, Any]] = []

        v_all = (cached or {}).get("stations_for_map_all")
        if isinstance(v_all, list) and v_all and isinstance(v_all[0], dict):
            stations_for_explain = v_all
        else:
            v_all2 = (cached or {}).get("stations")
            if isinstance(v_all2, list) and v_all2 and isinstance(v_all2[0], dict):
                stations_for_explain = v_all2
            else:
                v_ranked_all = (cached or {}).get("ranked_for_map_all")
                if isinstance(v_ranked_all, list) and v_ranked_all and isinstance(v_ranked_all[0], dict):
                    stations_for_explain = v_ranked_all
                else:
                    v_ranked = (cached or {}).get("ranked")
                    if isinstance(v_ranked, list) and v_ranked and isinstance(v_ranked[0], dict):
                        stations_for_explain = v_ranked


        best_station = (cached or {}).get("best_station")
        best_uuid = _station_uuid(best_station) if isinstance(best_station, dict) else ""

        def _station_label(s: Dict[str, Any]) -> str:
            brand = s.get("brand") or s.get("tk_name") or s.get("station_name") or s.get("name") or "Station"
            addr = _station_google_address_best_effort(s)
            short_addr = addr.replace(", Deutschland", "").strip() if isinstance(addr, str) else "—"
            return f"{_safe_text(brand)} — {_safe_text(short_addr)}"

        explained_station: Optional[Dict[str, Any]] = None
        if stations_for_explain:
            default_idx = 0
            if best_uuid:
                for i, s in enumerate(stations_for_explain):
                    if _station_uuid(s) == best_uuid:
                        default_idx = i
                        break

            explained_station = st.selectbox(
                "Use the **selector below** to choose a station and see how its price was predicted:",
                options=stations_for_explain,
                index=default_idx,
                format_func=_station_label,
            )

        # --- Compute station-level explanation fields (best-effort, never raise) ---
        ft = str((cached or {}).get("fuel_code") or (cached or {}).get("fuel_type") or fuel_code or "diesel").lower().strip()
        ft_label = ft.upper() if ft != "diesel" else "Diesel"

        # "Now" anchor: prefer cached timestamp (this page is an audit view of a past run)
        now_local = _parse_any_dt_to_berlin((cached or {}).get("computed_at") or (cached or {}).get("ts") or (cached or {}).get("timestamp"))
        now_str = now_local.strftime("%Y-%m-%d %H:%M:%S") if now_local else "—"
        cell_now = _time_cell_48(now_local)

        def _read_station_explain_payload(s: Dict[str, Any]) -> Dict[str, Any]:
            # ETA (multiple possible representations depending on run/version)
            eta_raw = s.get("eta") or s.get(f"debug_{ft}_eta_utc") or s.get("eta_utc")
            eta_local_dt = _parse_any_dt_to_berlin(eta_raw)
            eta_str = (
                eta_local_dt.strftime("%Y-%m-%d %H:%M:%S")
                if eta_local_dt
                else _format_eta_local_for_display(eta_raw)
            )
            cell_eta = _time_cell_48(eta_local_dt)

            # Predictor-produced debug values (preferred)
            minutes_to_arrival = s.get(f"debug_{ft}_minutes_to_arrival") or s.get(f"debug_{ft}_minutes_ahead")
            cells_ahead = s.get(f"debug_{ft}_cells_ahead_raw") or s.get(f"debug_{ft}_cells_ahead")

            # Compute cells ahead if missing (handles day wrap)
            if cells_ahead is None and (now_local is not None) and (eta_local_dt is not None) and (cell_now is not None) and (cell_eta is not None):
                try:
                    day_now = now_local.date().toordinal()
                    day_eta = eta_local_dt.date().toordinal()
                    cells_ahead = (day_eta - day_now) * 48 + (cell_eta - cell_now)
                except Exception:
                    cells_ahead = None

            # Forecast-basis flags set by predictor
            used_current = _coerce_bool(s.get(f"debug_{ft}_used_current_price"))
            horizon_used = s.get(f"debug_{ft}_horizon_used")

            # Prices (+ fallbacks)
            curr_key = f"price_current_{ft}"
            pred_key_local = f"pred_price_{ft}"

            current_price = s.get(curr_key)
            if current_price is None:
                current_price = s.get("price") or s.get(f"price_{ft}")

            pred_price = s.get(pred_key_local)

            lag1 = s.get(f"price_lag_1d_{ft}")
            lag2 = s.get(f"price_lag_2d_{ft}")
            lag3 = s.get(f"price_lag_3d_{ft}")
            lag7 = s.get(f"price_lag_7d_{ft}")

            # Human-readable "price basis" string (keep aligned with table)
            if used_current is True:
                price_basis = "Realtime (Tankerkönig) — no forecast"
            elif used_current is False:
                price_basis = f"Forecast (ARDL) — lags: 1d/2d/3d/7d ({ft_label})"
            else:
                price_basis = f"Forecast/Realtime (best-effort) — lags: 1d/2d/3d/7d ({ft_label})"

            # Only show a horizon when a forecast model is actually used.
            if used_current is not True and horizon_used not in (None, "", 0):
                price_basis = f"{price_basis} — horizon={horizon_used}"

            # Mode label
            if used_current is True:
                mode = "Spot (ETA < 10 min)"
            elif used_current is False:
                mode = "Forecast (ETA ≥ 10 min)"
            else:
                mode = "Best-effort (mixed / missing flags)"

            return {
                "eta_str": eta_str,
                "eta_local_dt": eta_local_dt,
                "cell_eta": cell_eta,
                "minutes_to_arrival": minutes_to_arrival,
                "cells_ahead": cells_ahead,
                "used_current": used_current,
                "horizon_used": horizon_used,
                "price_basis": price_basis,
                "mode": mode,
                "current_price": current_price,
                "pred_price": pred_price,
                "lag1": lag1,
                "lag2": lag2,
                "lag3": lag3,
                "lag7": lag7,
            }

        explain = _read_station_explain_payload(explained_station) if isinstance(explained_station, dict) else None

        # -----------------------------
        # A) Spot vs Forecast decision
        # -----------------------------
        st.markdown(
            "#### A) Spot vs Forecast decision",
            help=(
                "Station-level rule from the predictor:\n"
                "- If the station is reachable within 10 minutes and a realtime price exists, the system uses **current price** = **predicted price**.\n"
                "- Otherwise it uses Forecast mode and selects a horizon model.\n"
            ),
        )

        # Cards below (still next to each other)
        # --- A) Spot vs Forecast: tight layout (single HTML block) ---
        if explain:
            _mode = _safe_text(explain.get("mode"))
            _now = _safe_text(now_str)
            _eta = _safe_text(explain.get("eta_str"))
        else:
            _mode, _now, _eta = "—", "—", "—"

        st.markdown(
            f"""
            <style>
            .p02-a-wrap {{
                margin-top: -0.6rem;
                margin-bottom: 0.8rem;
            }}
            .p02-a-wrap ul {{
                margin: 0 0 0.35rem 1.1rem;   /* controls spacing BELOW bullets */
                padding: 0;
            }}
            .p02-a-cards {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.2rem;                /* horizontal/vertical gap between cards */
                margin: 0;                   /* no extra top margin */
            }}
            .p02-a-card {{
                flex: 1 1 260px;             /* responsive: 3 on desktop, stacks on mobile */
                width: 100%;
                padding: 0.5rem 1.0rem;
                border: 2px solid rgba(49, 51, 63, 0.3);
                border-radius: 0.9rem;
                text-align: center;
            }}
            .p02-a-card-label {{
                font-size: 0.9rem;
                opacity: 0.78;
                margin: 0 0 0.25rem 0;
                line-height: 1.15;
            }}
            .p02-a-card-value {{
                font-size: 1.3rem;
                font-weight: 750;
                margin: 0;
                line-height: 1.15;
            }}
            </style>

            <div class="p02-a-wrap">
            <ul>
                <li><b>Spot mode:</b> Predicted price = current realtime price → <b>no forecast</b></li>
                <li><b>Forecast mode:</b> Use ARDL model with an ETA-dependent horizon</li>
            </ul>

            <div class="p02-a-cards">
                <div class="p02-a-card">
                <div class="p02-a-card-label">Now (local)</div>
                <div class="p02-a-card-value">{_now}</div>
                </div>
                <div class="p02-a-card">
                <div class="p02-a-card-label">ETA (local)</div>
                <div class="p02-a-card-value">{_eta}</div>
                </div>
                <div class="p02-a-card">
                <div class="p02-a-card-label">Mode</div>
                <div class="p02-a-card-value">{_mode}</div>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


        # -----------------------------
        # B) Horizon selection & time cells
        # -----------------------------
        st.markdown(
            "#### B) Horizon selection",
            help=(
                "The predictor uses minutes-to-arrival (preferred) and time-cell deltas (fallback) to pick a horizon:\n"
                "- h1..h4 for 30..120 minutes ahead\n"
                "- h0_daily for longer lookaheads or if intraday inputs are unavailable\n"
                "Time cells are 0..47 in 30-minute buckets (local timezone)."
            ),
        )

        # --- B) Horizon selection: tight layout (single HTML block) ---
        if explain:
            used_current = explain.get("used_current")

            # If Spot mode is active, explicitly show that no model was used.
            if used_current is True:
                derived_h_str = "Spot price"
            else:
                derived_h = explain.get("horizon_used")

                # Robust formatting: horizon_used can be int-like (1..4) or a string ("h0_daily")
                if derived_h in (None, "", 0, "0"):
                    derived_h_str = "h0_daily"
                else:
                    # Keep string horizons as-is, otherwise render as "h{n}"
                    if isinstance(derived_h, str):
                        derived_h_str = derived_h.strip()
                    else:
                        try:
                            derived_h_str = f"h{int(derived_h)}"
                        except Exception:
                            derived_h_str = str(derived_h)

            now_cell_str = "—" if cell_now is None else str(cell_now)
            eta_cell = explain.get("cell_eta")
            eta_cell_str = "—" if eta_cell is None else str(eta_cell)

            cells_ahead = explain.get("cells_ahead")
            cells_ahead_str = "—" if cells_ahead is None else str(cells_ahead)

            minutes_ahead = explain.get("minutes_to_arrival")

            if minutes_ahead is None:
                minutes_ahead_str = "—"
            else:
                try:
                    minutes_ahead_str = f"{float(minutes_ahead):.2f}"
                except Exception:
                    minutes_ahead_str = str(minutes_ahead)

            st.markdown(
                f"""
                <style>
                .p02-b-wrap {{
                    margin-top: -0.4rem;
                }}
                .p02-b-cards {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 0.2rem;
                    margin: 0 0 0.35rem 0;   /* spacing below cards */
                }}
                .p02-b-card {{
                    flex: 1 1 210px;         /* responsive */
                    padding: 0.55rem 0.9rem;
                    border: 2px solid rgba(49, 51, 63, 0.3);
                    border-radius: 0.9rem;
                    text-align: center;
                }}
                .p02-b-label {{
                    font-size: 0.9rem;
                    opacity: 0.78;
                    margin: 0 0 0.25rem 0;
                    line-height: 1.1;
                }}
                .p02-b-value {{
                    font-size: 1.25rem;
                    font-weight: 750;
                    margin: 0;
                    line-height: 1.15;
                }}
                details.p02-b-details > summary {{
                    cursor: pointer;
                    font-weight: 650;
                    margin-top: 0.1rem;
                }}
                details.p02-b-details {{
                    margin-top: 0.2rem;
                }}
                .p02-b-rules {{
                    margin: 0.35rem 0 0 1.1rem;
                }}
                .p02-b-rules li {{
                    margin: 0.15rem 0;
                }}
                code.p02-code {{
                    padding: 0.1rem 0.3rem;
                    border-radius: 0.35rem;
                    background: rgba(49, 51, 63, 0.06);
                    border: 1px solid rgba(49, 51, 63, 0.12);
                    font-size: 0.9em;
                }}
                </style>

                <div class="p02-b-wrap">
                <div class="p02-b-cards">
                    <div class="p02-b-card">
                    <div class="p02-b-label">Now cell</div>
                    <div class="p02-b-value">{_safe_text(now_cell_str)}</div>
                    </div>
                    <div class="p02-b-card">
                    <div class="p02-b-label">ETA cell</div>
                    <div class="p02-b-value">{_safe_text(eta_cell_str)}</div>
                    </div>
                    <div class="p02-b-card">
                    <div class="p02-b-label">Cells ahead</div>
                    <div class="p02-b-value">{_safe_text(cells_ahead_str)}</div>
                    </div>
                    <div class="p02-b-card">
                    <div class="p02-b-label">Minutes ahead</div>
                    <div class="p02-b-value">{_safe_text(minutes_ahead_str)}</div>
                    </div>
                    <div class="p02-b-card">
                    <div class="p02-b-label">Selected model</div>
                    <div class="p02-b-value">{_safe_text(derived_h_str)}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True)

            st.markdown(
                "- If **minutes to arrival ≤ 10** and realtime price exists → **Spot mode**\n"
                "- Else, **minutes to arrival / 30 min** gives a raw horizon bucket:\n"
                "  - **1 to 4 → h1..h4**\n"
                "  - **>4 → only daily price inputs (historical)**\n"
                "- If **current prices** are **unavailable**, only **historical** prices are used\n"
            )

        # -----------------------------
        # C) What information goes into the forecast?
        # -----------------------------
        st.markdown(
            "#### C) Inputs used by the model",
            help=(
                "ARDL uses daily lag features (1d/2d/3d/7d at the same time cell). "
                "For intraday horizons h1..h4, the current realtime price acts as the intraday anchor. "
                "The resulting output is the station’s expected price at arrival."
            ),
        )

        if explain:
            # Baselines (simple, user-readable)
            avg_123 = None
            try:
                vals = [float(explain.get("lag1")), float(explain.get("lag2")), float(explain.get("lag3"))]
                avg_123 = sum(vals) / 3.0
            except Exception:
                avg_123 = None

            # Show compact comparison
            compare_rows = [
                {"Signal": "Realtime price", "Value": _fmt_price(explain.get("current_price"))},
                { "Signal": f"Lag 1 day ({ft_label})", "Value": _fmt_price(explain.get("lag1")) },
                { "Signal": f"Lag 2 days ({ft_label})", "Value": _fmt_price(explain.get("lag2")) },
                { "Signal": f"Lag 3 days ({ft_label})", "Value": _fmt_price(explain.get("lag3")) },
                { "Signal": f"Lag 7 days ({ft_label})", "Value": _fmt_price(explain.get("lag7")) },
                { "Signal": "Output: predicted price", "Value": _fmt_price(explain.get("pred_price")) },
            ]
            st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)

        # -----------------------------
        # D) Why did it predict that? (always visible)
        # -----------------------------
        st.markdown(
            "#### D) Why did it predict that?",
            help="This section explains the prediction using contribution tables. Open the expander below for details on the concept and interpretation.",
        )

        with st.expander("How to interpret these tables (concept + guidance)", expanded=False):
            st.markdown(
                """
        This section provides *model-based explainability* for the selected station in **Forecast mode**.

        ##### 1) Concept (what is being explained?)
        The forecast models used here are **linear ARDL regressions**.
        A linear model can be written as:

        **Predicted price = Intercept + Σ (coefficientᵢ × inputᵢ)**

        Because the model is additive, the prediction can be broken down into **per-input contributions** that sum exactly to the predicted €/L.

        ##### 2) Data foundation (why this is reliable)
        For this station, the system loads:
        - the **same horizon model** that was used during prediction (**h1–h4** or **h0_daily**), and
        - the **exact feature vector** that was actually passed into the model, stored by the predictor per station:
        - `debug_<fuel>_feature_cols`
        - `debug_<fuel>_feature_values`

        The coefficients are translated back to the original €/L scale before computing contributions.
        This ensures the numbers are interpretable in the same units as the prices shown on screen.

        ##### 3) “Absolute contributions”
        This table answers: **“How does the model add up to the final predicted price?”**

        - **Intercept**: constant offset of the model (after accounting for scaling).
        - **Each feature row**: additive contribution in **€/L**.
        - **Running sum**: cumulative total; the last row equals the predicted price.

        Practical reading:
        - Large positive contributions indicate inputs that push the prediction upward.
        - Large negative contributions indicate inputs that pull it downward.
        - For intraday horizons (**h1–h4**), the intraday anchor feature (e.g., `price_lag_1cell`) often dominates by design.

        ##### 4) “Baseline-relative contributions” (why above/below a baseline?)
        This table answers: **“Why is the prediction higher/lower than a simple reference?”**

        Baseline used:
        - Prefer **Lag 1 day** (`price_lag_1d`) as baseline level.
        - If lag1d is missing, fallback to **Avg(1d,2d,3d)**.

        The table shows, per feature:
        - **Δ input vs baseline**: how far this input is above/below the baseline level (€/L).
        - **Δ contribution**: how much that difference changes the prediction (€/L).
        - **Running Δ**: sums to the total deviation vs baseline.

        Interpretation:
        - If the baseline-relative Δ is small, the model prediction is close to the baseline.
        - If the Δ is large, you can see *which inputs* are responsible (often the intraday anchor for short horizons).

        ##### 5) Safety / limitations
        - **Spot mode (ETA ≤ 10 min)**: no model is called; the shown price is the realtime price → no decomposition is shown.
        - If the feature-vector snapshot keys are missing, the decomposition cannot be computed reliably.
        - Minor differences between “Δ sum” and “Predicted − baseline” can occur due to rounding/float precision in displayed values.
        """
            )

        if not explain or not isinstance(explained_station, dict):
            st.info("No station selected (or no explain payload available).")
        else:
            # Rule requested: if ETA is <10 minutes, show info instead of a table
            mta = _as_float(explain.get("minutes_to_arrival"))
            if mta is not None and mta <= 10.0:
                st.info(
                    "Spot mode (ETA ≤ 10 min): the shown price is the current realtime price. "
                    "No forecasting model was called, so there is no coefficient decomposition."
                )
            else:
                try:
                    from src.modeling.model import load_model_for_horizon  # type: ignore

                    # Horizon/model used
                    horizon_used = explain.get("horizon_used")
                    h_int = 0
                    try:
                        h_int = int(horizon_used) if horizon_used not in (None, "", 0) else 0
                    except Exception:
                        h_int = 0

                    model_obj = load_model_for_horizon(ft, h_int)

                    # Exact feature vector from predictor (persisted in predict.py)
                    debug_fc_key = f"debug_{ft}_feature_cols"
                    debug_fv_key = f"debug_{ft}_feature_values"

                    cols = explained_station.get(debug_fc_key)
                    vals = explained_station.get(debug_fv_key)

                    if not isinstance(cols, list) or not isinstance(vals, list) or not cols or len(cols) != len(vals):
                        st.info(
                            "No reliable feature-vector snapshot is available for this station. "
                            "To enable exact explainability, ensure predict.py persists:\n"
                            f"- {debug_fc_key}\n"
                            f"- {debug_fv_key}"
                        )
                    else:
                        # Build X exactly as used in prediction
                        X = pd.DataFrame([vals], columns=cols)

                        def _try_linear_decompose(m: Any, X_row: pd.DataFrame) -> Optional[pd.DataFrame]:
                            """
                            Attempt to compute a clean linear contribution breakdown.
                            Works when model is sklearn Pipeline: StandardScaler -> linear estimator (coef_).
                            Returns a small DataFrame or None if not possible.
                            """
                            pipe = m
                            scaler = None
                            estimator = None

                            # Pipeline-like
                            if hasattr(pipe, "named_steps"):
                                for step in pipe.named_steps.values():
                                    if scaler is None and hasattr(step, "mean_") and hasattr(step, "scale_"):
                                        scaler = step
                                    if hasattr(step, "coef_"):
                                        estimator = step
                                if estimator is None:
                                    estimator = list(pipe.named_steps.values())[-1]

                            # Non-pipeline
                            if estimator is None and hasattr(pipe, "coef_"):
                                estimator = pipe

                            if estimator is None or not hasattr(estimator, "coef_"):
                                return None

                            coef = getattr(estimator, "coef_", None)
                            intercept = float(getattr(estimator, "intercept_", 0.0) or 0.0)
                            if coef is None:
                                return None

                            # Flatten coef for single-target regression
                            try:
                                coef_vec = list(coef.ravel())  # type: ignore[attr-defined]
                            except Exception:
                                coef_vec = list(coef) if isinstance(coef, (list, tuple)) else None
                            if not coef_vec:
                                return None

                            cols_local = list(X_row.columns)
                            if len(coef_vec) != len(cols_local):
                                return None

                            # Read X row as floats
                            x_vals = []
                            for c in cols_local:
                                try:
                                    x_vals.append(float(X_row.iloc[0][c]))
                                except Exception:
                                    return None

                            # If we have a scaler, translate contributions back to original feature units
                            if scaler is not None:
                                mean = list(getattr(scaler, "mean_", []))
                                scale = list(getattr(scaler, "scale_", []))
                                if len(mean) == len(cols_local) and len(scale) == len(cols_local):
                                    eff_beta = [(coef_vec[i] / scale[i]) for i in range(len(cols_local))]
                                    eff_intercept = intercept - sum(
                                        (coef_vec[i] * mean[i] / scale[i]) for i in range(len(cols_local))
                                    )
                                    contrib = [eff_beta[i] * x_vals[i] for i in range(len(cols_local))]
                                    base = eff_intercept
                                else:
                                    contrib = [coef_vec[i] * x_vals[i] for i in range(len(cols_local))]
                                    base = intercept
                            else:
                                contrib = [coef_vec[i] * x_vals[i] for i in range(len(cols_local))]
                                base = intercept

                            rows = [{"Component": "Intercept", "Contribution (€/L)": base}]
                            for i, c in enumerate(cols_local):
                                rows.append({"Component": c, "Contribution (€/L)": contrib[i]})

                            df = pd.DataFrame(rows)
                            df["Contribution (€/L)"] = df["Contribution (€/L)"].astype(float)
                            df["Running sum (€/L)"] = df["Contribution (€/L)"].cumsum()
                            return df

                        # ---- Table A: absolute contributions ----
                        st.markdown("##### Absolute contribution breakdown")

                        df_contrib = _try_linear_decompose(model_obj, X)

                        if df_contrib is None:
                            st.info(
                                "This model does not expose a compatible (scaler + linear-coef) structure for decomposition. "
                                "The cached prediction is still valid; only the coefficient breakdown is unavailable."
                            )
                        else:
                            # Force expected columns + order (prevents accidental single-column rendering)
                            expected_cols = ["Component", "Contribution (€/L)", "Running sum (€/L)"]
                            if all(c in df_contrib.columns for c in expected_cols):
                                df_show = df_contrib[expected_cols].copy()
                                for c in ("Contribution (€/L)", "Running sum (€/L)"):
                                    df_show[c] = pd.to_numeric(df_show[c], errors="coerce").round(4)
                            else:
                                df_show = df_contrib.copy()
                                st.warning(f"Unexpected columns in contribution table: {list(df_contrib.columns)}")

                            st.dataframe(df_show, hide_index=True, use_container_width=True)

                            # Sanity check: compare decomposition to cached prediction (belongs under Table A)
                            cached_pred = _as_float(explain.get("pred_price"))
                            approx_pred = _as_float(df_contrib["Running sum (€/L)"].iloc[-1]) if not df_contrib.empty else None
                            if cached_pred is not None and approx_pred is not None:
                                st.caption(
                                    f"Decomposition sum: {approx_pred:.4f} €/L · Cached predicted price: {cached_pred:.4f} €/L"
                                )

                            # ---- Table B: baseline-relative (delta) explanation ----
                            st.markdown("##### Baseline-relative breakdown (why above/below yesterday)")

                            # Determine a baseline level (€/L) to compare against:
                            # Prefer Lag 1d, fallback to Avg(1d,2d,3d)
                            baseline_level = None
                            try:
                                if "price_lag_1d" in X.columns:
                                    baseline_level = float(X.loc[0, "price_lag_1d"])
                            except Exception:
                                baseline_level = None

                            baseline_name = "Lag 1d"
                            if baseline_level is None:
                                try:
                                    cands = []
                                    for c in ("price_lag_1d", "price_lag_2d", "price_lag_3d"):
                                        if c in X.columns:
                                            cands.append(float(X.loc[0, c]))
                                    if len(cands) == 3:
                                        baseline_level = sum(cands) / 3.0
                                        baseline_name = "Avg(1d,2d,3d)"
                                except Exception:
                                    baseline_level = None

                            if baseline_level is None:
                                st.info("Baseline-relative explanation unavailable (missing lag inputs).")
                            else:
                                # Recompute effective betas in original €/L scale so delta contributions are correct.
                                pipe = model_obj
                                scaler = None
                                estimator = None

                                if hasattr(pipe, "named_steps"):
                                    for step in pipe.named_steps.values():
                                        if scaler is None and hasattr(step, "mean_") and hasattr(step, "scale_"):
                                            scaler = step
                                        if hasattr(step, "coef_"):
                                            estimator = step
                                    if estimator is None:
                                        estimator = list(pipe.named_steps.values())[-1]

                                if estimator is None and hasattr(pipe, "coef_"):
                                    estimator = pipe

                                if estimator is None or not hasattr(estimator, "coef_"):
                                    st.info("Baseline-relative explanation unavailable (model has no accessible coefficients).")
                                else:
                                    coef = getattr(estimator, "coef_", None)
                                    if coef is None:
                                        st.info("Baseline-relative explanation unavailable (missing coefficients).")
                                    else:
                                        try:
                                            coef_vec = list(coef.ravel())  # type: ignore[attr-defined]
                                        except Exception:
                                            coef_vec = list(coef) if isinstance(coef, (list, tuple)) else None

                                        if not coef_vec or len(coef_vec) != len(X.columns):
                                            st.info("Baseline-relative explanation unavailable (coefficient shape mismatch).")
                                        else:
                                            cols_local = list(X.columns)

                                            # Effective betas in original units (if scaler exists)
                                            if scaler is not None:
                                                mean = list(getattr(scaler, "mean_", []))
                                                scale = list(getattr(scaler, "scale_", []))
                                                if len(mean) == len(cols_local) and len(scale) == len(cols_local):
                                                    eff_beta = [(coef_vec[i] / scale[i]) for i in range(len(cols_local))]
                                                else:
                                                    eff_beta = list(coef_vec)
                                            else:
                                                eff_beta = list(coef_vec)

                                            rows_delta = []
                                            running = 0.0

                                            for i, c in enumerate(cols_local):
                                                try:
                                                    x_i = float(X.loc[0, c])
                                                except Exception:
                                                    continue

                                                dx = x_i - baseline_level
                                                dcontrib = eff_beta[i] * dx
                                                running += dcontrib

                                                rows_delta.append(
                                                    {
                                                        "Component": c,
                                                        "Δ input vs baseline": dx,
                                                        "Δ contribution (€/L)": dcontrib,
                                                        "Running Δ (€/L)": running,
                                                    }
                                                )

                                            df_delta = pd.DataFrame(rows_delta)

                                            st.markdown(
                                                f"Baseline: **{baseline_level:.3f} €/L** ({baseline_name}). "
                                            )
                                            st.dataframe(df_delta, hide_index=True, use_container_width=True)

                                            cached_pred = _as_float(explain.get("pred_price"))
                                            if cached_pred is not None:
                                                st.caption(
                                                    f"Δ decomposition sum: {running:.4f} €/L · "
                                                    f"Predicted − baseline: {(cached_pred - baseline_level):.4f} €/L"
                                                )

                except Exception as e:
                    st.info(
                        "Explainability is best-effort and can fail safely for some model variants. "
                        f"Details: {type(e).__name__}: {e}"
                    )


        # Fourth headline
        st.markdown(
            "#### E) Full input/output values:",
            help=(
                "This table is meant to make the prediction and economics layer auditable.\n\n"
                "**Which stations are included?**\n"
                "- The table shows **all stations available in the cached run payload** (no hard-constraint filtering, no deduplication).\n"
                "- This includes stations that would normally be excluded by the decision layer (e.g., invalid forecast inputs, detour caps, distance window).\n"
                "- **Limitation:** stations that were filtered upstream *before* the run was cached (e.g., earlier pipeline filters) cannot be shown here.\n\n"
                "**Default ordering**\n"
                "- The table is **sorted by Net savings (descending)**.\n"
                "- Stations without a net-savings value appear at the bottom.\n\n"
                "**How to read missing values (\"—\")**\n"
                "- Missing fields typically mean the station did not have the required inputs at that stage (e.g., missing lag prices, no ETA, no prediction output).\n"
                "- These stations are shown intentionally for transparency; they may not be eligible for recommendation under strict constraints.\n\n"
                "**Interpretation tips**\n"
                "- Use the time fields (\"UTC+1 (now)\", ETA, time cells) to verify timezone conversion and ETA-to-horizon mapping.\n"
                "- Use \"Price basis\" to verify whether the system used realtime price vs. a forecast for the selected fuel.\n"
                "- Use \"Net savings\" to compare the economic outcome across all cached stations, including those that would otherwise be filtered out."
            ),
        )

        # -----------------------------
        # Input/Output table (ALL cached stations; no hard-constraint filtering)
        # -----------------------------
        ft = str(fuel_code or "e5").lower()
        ft_label = ft.upper()

        pred_key = f"pred_price_{ft}"
        net_key = f"econ_net_saving_eur_{ft}"

        # Base candidate set (prefer cached 'stations' universe; fallback to other known keys if needed)
        base_candidates: List[Dict[str, Any]] = []
        for k in ("stations", "stations_all", "all_stations", "candidates", "ranked"):
            v = cached.get(k)
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                base_candidates = v
                break

        if not base_candidates:
            st.info("No stations found in the cached run (no station list available).")
        else:
            # NOTE:
            # - This table intentionally shows ALL stations available in the cached run.
            # - Stations that would fail hard constraints (invalid prediction inputs, detour caps, distance window, etc.)
            #   will still appear, but some fields may be missing ("—").
            # - Stations removed upstream on Page 01 (before caching) cannot be shown here, because they are not in the cached list.

            # ---- Times (best-effort, consistent with cached run) ----
            now_local = (
                _parse_any_dt_to_berlin(cached.get("computed_at"))
                or _parse_any_dt_to_berlin(datetime.now(timezone.utc))
            )
            time_now_str = now_local.strftime("%Y-%m-%d %H:%M:%S") if now_local else "—"
            cell_now = _time_cell_48(now_local)

            def _net_saving_best_effort(s: Dict[str, Any]) -> Optional[float]:
                # Primary key used in Section 4 tables
                v = _as_float((s or {}).get(net_key))
                if v is not None:
                    return v

                # Backward-compatible fallbacks (older variants)
                for k in (
                    "econ_net_saving_eur",
                    f"net_saving_eur_{ft}",
                    "net_saving_eur",
                    "net_saving",
                ):
                    v = _as_float((s or {}).get(k))
                    if v is not None:
                        return v
                return None

            rows: List[Dict[str, Any]] = []

            for s in base_candidates:
                if not isinstance(s, dict):
                    continue

                # Station metadata
                brand = s.get("brand") or s.get("tk_name") or s.get("station_name") or s.get("name") or "—"
                addr_google = _station_google_address_best_effort(s)

                # ETA parsing (multiple possible fields depending on run/version)
                eta_raw = (
                    s.get("eta")
                    or s.get(f"debug_{ft}_eta_utc")
                    or s.get("eta_utc")
                )
                eta_local_dt = _parse_any_dt_to_berlin(eta_raw)
                time_eta_str = (
                    eta_local_dt.strftime("%Y-%m-%d %H:%M:%S")
                    if eta_local_dt
                    else _format_eta_local_for_display(eta_raw)
                )
                cell_eta = _time_cell_48(eta_local_dt)

                # Use pipeline-provided deltas when available; otherwise compute from cells (including day wrap)
                minutes_to_arrival = (
                    s.get(f"debug_{ft}_minutes_to_arrival")
                    or s.get(f"debug_{ft}_minutes_ahead")
                )
                cells_ahead = (
                    s.get(f"debug_{ft}_cells_ahead_raw")
                    or s.get(f"debug_{ft}_cells_ahead")
                )

                if cells_ahead is None and (now_local is not None) and (eta_local_dt is not None) and (cell_now is not None) and (cell_eta is not None):
                    try:
                        day_now = now_local.date().toordinal()
                        day_eta = eta_local_dt.date().toordinal()
                        cells_ahead = (day_eta - day_now) * 48 + (cell_eta - cell_now)
                    except Exception:
                        cells_ahead = None

                # Model/basis
                used_current = _coerce_bool(s.get(f"debug_{ft}_used_current_price"))
                horizon_used = s.get(f"debug_{ft}_horizon_used")

                if used_current is True:
                    price_basis = "Realtime (Tankerkönig) — no forecast"
                elif used_current is False:
                    price_basis = f"Forecast (ARDL) — lags: 1d/2d/3d/7d ({ft_label})"
                else:
                    price_basis = f"Forecast/Realtime (best-effort) — lags: 1d/2d/3d/7d ({ft_label})"

                if horizon_used not in (None, "", 0):
                    price_basis = f"{price_basis} — horizon={horizon_used}"

                # Prices (+ fallbacks for older run variants)
                curr_key = f"price_current_{ft}"
                pred_key_local = f"pred_price_{ft}"

                current_price = s.get(curr_key)
                if current_price is None:
                    current_price = s.get("price") or s.get(f"price_{ft}")

                pred_price = s.get(pred_key_local)

                lag1 = s.get(f"price_lag_1d_{ft}")
                lag2 = s.get(f"price_lag_2d_{ft}")
                lag3 = s.get(f"price_lag_3d_{ft}")
                lag7 = s.get(f"price_lag_7d_{ft}")

                net_saving_val = _net_saving_best_effort(s)

                rows.append({
                    "Brand/Name": _safe_text(brand),
                    "Address": _safe_text(addr_google),
                    "UTC+1 (now)": _safe_text(time_now_str),
                    "Time cell (now)": cell_now if cell_now is not None else "—",
                    "ETA": _safe_text(time_eta_str),
                    "Minutes until arrival": minutes_to_arrival if minutes_to_arrival is not None else "—",
                    "Time cells ahead": cells_ahead if cells_ahead is not None else "—",
                    "Time cell at ETA": cell_eta if cell_eta is not None else "—",
                    "Horizon": horizon_used if horizon_used is not None else "—",
                    "Price basis (which model → lags)": _safe_text(price_basis),
                    "Current price": _fmt_price(current_price),
                    f"Lag1d ({ft_label})": _fmt_price(lag1),
                    f"Lag2d ({ft_label})": _fmt_price(lag2),
                    f"Lag3d ({ft_label})": _fmt_price(lag3),
                    f"Lag7d ({ft_label})": _fmt_price(lag7),
                    "Predicted price": _fmt_price(pred_price),

                    # Sorting helper (numeric) + displayed column (formatted)
                    "__net_saving_sort": net_saving_val,
                    "Net savings": "—" if net_saving_val is None else _fmt_eur(net_saving_val),
                })

            df_io = pd.DataFrame(rows)

            # Default ordering: Net savings DESC (missing values at bottom)
            if "__net_saving_sort" in df_io.columns:
                df_io = df_io.sort_values(by="__net_saving_sort", ascending=False, na_position="last")
                df_io = df_io.drop(columns=["__net_saving_sort"])

            st.dataframe(df_io, use_container_width=True, hide_index=True)

        maybe_persist_state()
        return




    # ------------------------------------------------------------------
    # 4. Full Result Tables
    # ------------------------------------------------------------------

    # Keep other views empty for now (header + top nav remain visible)
    if analysis_view != "Full Result Tables":
        maybe_persist_state()
        return

    # ------------------------------------------------------------------
    # Headlines + selected/discarded full tables
    # ------------------------------------------------------------------

    st.markdown("### **4. Full Table Results**")

    st.markdown("#### What this page shows:")

    st.markdown(
        "This section provides an **audit view** of the station decision pipeline for the cached run, presented as two tables. "
        "It makes the selection logic transparent and shows the underlying price and cost components used for the savings calculations.\n\n"
        "**Decision stages:**\n"
        "1) **Hard constraints (non-negotiable):** stations must have valid prediction inputs and pass feasibility filters such as "
        "open at ETA (if enabled), detour caps, distance window (if configured), and deduplication.\n"
        "2) **Economic benchmark (only when available and economics mode is enabled):** among hard-feasible stations, a station is "
        "economically viable if its **net saving is at least as high as the worst on-route station** (reference benchmark).\n\n"
        "**How to read the tables:**\n"
        "- **Selected stations:** pass hard constraints and, if an on-route benchmark exists, also satisfy the economic rule "
        "(net saving ≥ worst on-route net saving).\n"
        "- **Discarded stations:** fail at least one hard constraint or are economically rejected (hard-feasible but net saving < worst on-route net saving).\n"
        "- **Benchmark columns (right side):** show the on-route best/median/worst predicted price as global references, plus the per-station "
        "gross/net savings versus those references for the configured refuel volume.\n"
        "- **If no on-route benchmark exists:** the economic step is treated as N/A and all hard-feasible stations appear as selected.\n\n"
        "**Important:** this view only reflects stations contained in the cached station list for this run. Stations removed upstream "
        "(e.g., due to corridor search scope) cannot appear here."
    )
    # -----------------------------
    # Config / keys (aligned with Section 2 semantics)
    # -----------------------------
    pred_key__p4 = f"pred_price_{fuel_code}"
    curr_key__p4 = f"price_current_{fuel_code}"

    econ_net_key__p4 = f"econ_net_saving_eur_{fuel_code}"
    econ_gross_key__p4 = f"econ_gross_saving_eur_{fuel_code}"
    econ_detour_cost_key__p4 = f"econ_detour_fuel_cost_eur_{fuel_code}"
    econ_time_cost_key__p4 = f"econ_time_cost_eur_{fuel_code}"

    constraints__p4 = run_summary.get("constraints") or {}

    # -----------------------------
    # Run-level context columns (constant per row)
    # -----------------------------
    def _fuel_label__p4(code: Any) -> str:
        c = str(code or "").strip().lower()
        if c == "e5":
            return "E5"
        if c == "e10":
            return "E10"
        if c in ("diesel", "d"):
            return "Diesel"
        return str(code or "—").upper()

    fuel_type_label__p4 = _fuel_label__p4(fuel_code)

    # Min net saving threshold (from sidebar / constraints). Best-effort key support.
    min_net_saving_eur__p4 = _as_float(
        (constraints__p4.get("min_net_saving_eur") if isinstance(constraints__p4, dict) else None)
        or (constraints__p4.get("min_net_saving") if isinstance(constraints__p4, dict) else None)
    )

    # Detour caps: in economics mode, caps default to 10 km / 10 min if unset.
    cap_km_f__p4 = _as_float(constraints__p4.get("max_detour_km"))
    cap_min_f__p4 = _as_float(constraints__p4.get("max_detour_min"))

    # Distance window (optional)
    min_distance_km__p4 = _as_float(constraints__p4.get("min_distance_km"))
    max_distance_km__p4 = _as_float(constraints__p4.get("max_distance_km"))

    # Inputs for detour liters + fallback cost computation (best-effort)
    consumption_l_per_100km__p4 = _as_float(constraints__p4.get("consumption_l_per_100km"))
    value_of_time_eur_per_hour__p4 = _as_float(constraints__p4.get("value_of_time_eur_per_hour"))

    if bool(use_economics):
        if cap_km_f__p4 is None:
            cap_km_f__p4 = 10.0
        if cap_min_f__p4 is None:
            cap_min_f__p4 = 10.0

    should_cap__p4 = (cap_km_f__p4 is not None) or (cap_min_f__p4 is not None)

    # Soft-constraint thresholds (benchmark definition: "on-route")
    filter_log__p4: Dict[str, Any] = cached.get("filter_log") or {}
    filter_thresholds__p4: Dict[str, Any] = filter_log__p4.get("thresholds") or {}
    onroute_km__p4 = float(filter_thresholds__p4.get("onroute_max_detour_km", 1.0) or 1.0)
    onroute_min__p4 = float(filter_thresholds__p4.get("onroute_max_detour_min", 5.0) or 5.0)

    # -----------------------------
    # Formatting helpers (local to Page-4 tables)
    # -----------------------------
    def _fmt_price_or_dash__p4(x: Any) -> str:
        v = _as_float(x)
        if v is None or v < MIN_VALID_PRICE_EUR_L:
            return "—"
        return _fmt_price(v)

    def _fmt_eur_or_dash__p4(x: Any) -> str:
        v = _as_float(x)
        return "—" if v is None else _fmt_eur(v)

    def _current_price_best_effort__p4(s: Dict[str, Any]) -> Any:
        v = (s or {}).get(curr_key__p4)
        if v is None:
            # Fallbacks for older run variants
            v = (s or {}).get("price") or (s or {}).get(f"price_{fuel_code}")
        return v

    # -----------------------------
    # Build selected vs discarded sets (hard first, then economics)
    # -----------------------------

    # 1) Hard: valid prediction input
    hard_invalid_pred__p4: List[Dict[str, Any]] = []
    candidates__p4: List[Dict[str, Any]] = []
    for s in stations:
        p_f = _as_float((s or {}).get(pred_key__p4))
        if p_f is None or p_f < MIN_VALID_PRICE_EUR_L:
            hard_invalid_pred__p4.append(s)
        else:
            candidates__p4.append(s)

    # 2) Hard: open-at-ETA (if enabled)
    hard_closed_eta__p4: List[Dict[str, Any]] = []
    open_ok__p4: List[Dict[str, Any]] = []
    for s in candidates__p4:
        if filter_closed_at_eta_enabled and _is_closed_at_eta(s):
            hard_closed_eta__p4.append(s)
        else:
            open_ok__p4.append(s)

    # 3) Hard: deduplicate by station_uuid (keep lowest predicted price variant)
    hard_duplicates_removed__p4: List[Dict[str, Any]] = []
    deduped__p4: List[Dict[str, Any]] = []

    groups__p4: Dict[str, List[Dict[str, Any]]] = {}
    no_uuid__p4: List[Dict[str, Any]] = []

    for s in open_ok__p4:
        u = (s or {}).get("station_uuid")
        u = str(u).strip() if u is not None else ""
        if not u:
            no_uuid__p4.append(s)
        else:
            groups__p4.setdefault(u, []).append(s)

    deduped__p4.extend(no_uuid__p4)

    for _, lst in groups__p4.items():
        if not lst:
            continue

        best = lst[0]
        best_p = _as_float((best or {}).get(pred_key__p4))

        for cand in lst[1:]:
            p = _as_float((cand or {}).get(pred_key__p4))
            if best_p is None:
                best, best_p = cand, p
            elif p is not None and p < best_p:
                best, best_p = cand, p

        deduped__p4.append(best)
        for cand in lst:
            if cand is not best:
                hard_duplicates_removed__p4.append(cand)

    # 4) Hard: detour caps
    hard_cap_failed__p4: List[Dict[str, Any]] = []
    if should_cap__p4:
        hard_feasible__p4: List[Dict[str, Any]] = []
        for s in deduped__p4:
            dk, dm = _detour_metrics(s)

            cap_fail = False
            if cap_km_f__p4 is not None and dk is not None:
                try:
                    cap_fail = cap_fail or (float(dk) > float(cap_km_f__p4))
                except Exception:
                    pass
            if cap_min_f__p4 is not None and dm is not None:
                try:
                    cap_fail = cap_fail or (float(dm) > float(cap_min_f__p4))
                except Exception:
                    pass

            if cap_fail:
                hard_cap_failed__p4.append(s)
            else:
                hard_feasible__p4.append(s)
    else:
        hard_feasible__p4 = list(deduped__p4)

    # 5) Hard: distance window (min/max distance)
    hard_distance_failed__p4: List[Dict[str, Any]] = []
    if (min_distance_km__p4 is not None) or (max_distance_km__p4 is not None):
        tmp_ok: List[Dict[str, Any]] = []
        for s in hard_feasible__p4:
            dist_km = _distance_along_km(s)

            if min_distance_km__p4 is not None and dist_km < float(min_distance_km__p4):
                hard_distance_failed__p4.append(s)
                continue
            if max_distance_km__p4 is not None and dist_km > float(max_distance_km__p4):
                hard_distance_failed__p4.append(s)
                continue

            tmp_ok.append(s)

        hard_feasible__p4 = tmp_ok

    # -----------------------------
    # On-route benchmark stats (for per-row "vs best/median/worst" columns)
    # Benchmark universe = hard-feasible (stable, post hard constraints)
    # -----------------------------
    litres__p4 = _as_float((constraints__p4 or {}).get("litres_to_refuel"))

    def _onroute_price_stats__p4(universe: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
        vals: List[float] = []
        n_onroute = 0

        for s in universe:
            p = _as_float((s or {}).get(pred_key__p4))
            if p is None or p < MIN_VALID_PRICE_EUR_L:
                continue

            km, mins = _detour_metrics(s)
            if km is None or mins is None:
                continue

            # Use the same definition as your benchmark UI (defaults to 1 km / 5 min)
            if float(km) <= float(onroute_km__p4) and float(mins) <= float(onroute_min__p4):
                n_onroute += 1
                vals.append(float(p))

        if not vals:
            return None, None, None, n_onroute

        ser = pd.Series(vals, dtype="float")
        return float(ser.min()), float(ser.median()), float(ser.max()), n_onroute

    on_best__p4, on_median__p4, on_worst__p4, onroute_n__p4 = _onroute_price_stats__p4(hard_feasible__p4)
    onroute_benchmark_available__p4 = (
        on_best__p4 is not None and on_median__p4 is not None and on_worst__p4 is not None
    )

    # Economic benchmark: worst on-route net saving (Case 1 handled as N/A)
    econ_benchmark_available__p4 = False
    worst_onroute_net__p4: Optional[float] = None

    if bool(use_economics) and hard_feasible__p4:
        nets: List[float] = []
        for s in hard_feasible__p4:
            dk, dm = _detour_metrics(s)
            if dk is None or dm is None:
                continue
            if float(dk) <= float(onroute_km__p4) and float(dm) <= float(onroute_min__p4):
                v = _as_float((s or {}).get(econ_net_key__p4))
                if v is not None:
                    nets.append(float(v))

        if nets:
            econ_benchmark_available__p4 = True
            worst_onroute_net__p4 = float(min(nets))

    # Selected vs discarded (economics)
    selected__p4: List[Dict[str, Any]] = []
    econ_rejected__p4: List[Dict[str, Any]] = []

    if bool(use_economics) and econ_benchmark_available__p4 and (worst_onroute_net__p4 is not None):
        for s in hard_feasible__p4:
            net = _as_float((s or {}).get(econ_net_key__p4))
            if net is None:
                econ_rejected__p4.append(s)
                continue
            try:
                if float(net) >= float(worst_onroute_net__p4):
                    selected__p4.append(s)
                else:
                    econ_rejected__p4.append(s)
            except Exception:
                econ_rejected__p4.append(s)
    else:
        # Case 1 (no on-route benchmark) or economics off: economics selection is N/A → pass-through
        selected__p4 = list(hard_feasible__p4)

    # Include stations removed purely by the Page-01 brand whitelist filter.
    # Page 01 keeps the pre-brand station universe in cached["stations_for_map_all"].
    brand_filtered_out__p4: List[Dict[str, Any]] = []
    if isinstance(brand_filter_selected, list) and len(brand_filter_selected) > 0:
        universe__p4 = cached.get("stations_for_map_all")
        if isinstance(universe__p4, list) and universe__p4:
            # Object-identity comparison is safe here because Page 01 builds kept_stations via list
            # comprehension from the same dict objects.
            kept_obj_ids = {id(s) for s in (stations or [])}
            brand_filtered_out__p4 = [s for s in universe__p4 if id(s) not in kept_obj_ids]

    discarded__p4: List[Dict[str, Any]] = (
        list(brand_filtered_out__p4)
        + list(hard_invalid_pred__p4)
        + list(hard_closed_eta__p4)
        + list(hard_duplicates_removed__p4)
        + list(hard_cap_failed__p4)
        + list(hard_distance_failed__p4)
        + list(econ_rejected__p4)
    )


    # Default sort (UI): descending by net saving (when available)
    if bool(use_economics):

        def _net_sort_key__p4(sta: Dict[str, Any]) -> Tuple:
            net = _as_float((sta or {}).get(econ_net_key__p4))
            net_f = float(net) if net is not None else float("-inf")  # None goes to bottom
            return (
                -net_f,
                sta.get("fraction_of_route", float("inf")),
                sta.get("distance_along_m", float("inf")),
            )

        selected__p4 = sorted(selected__p4, key=_net_sort_key__p4)
        discarded__p4 = sorted(discarded__p4, key=_net_sort_key__p4)

    # Optional: surface the N/A state explicitly to the user
    if bool(use_economics) and not econ_benchmark_available__p4:
        st.caption(
            "Economic benchmark: N/A (no on-route stations available) → showing all hard-feasible stations as selected."
        )

    # -----------------------------
    # Row builder (shared across both tables)
    # -----------------------------
    def _fmt_fraction__p4(x: Any) -> str:
        v = _as_float(x)
        return "—" if v is None else f"{v:.3f}"

    def _fmt_dist_to_station__p4(s: Dict[str, Any]) -> str:
        d_m = _as_float(s.get("distance_along_m"))
        if d_m is None:
            return "—"
        return _format_km(d_m / 1000.0)

    def _fmt_eta_local__p4(s: Dict[str, Any]) -> str:
        eta_raw = s.get("eta") or s.get(f"debug_{fuel_code}_eta_utc") or s.get("eta_utc")
        dt = _parse_any_dt_to_berlin(eta_raw)
        if dt is not None:
            return dt.strftime("%Y-%m-%d %H:%M")
        return _safe_text(_format_eta_local_for_display(eta_raw, tz_name="Europe/Berlin"))

    def _fmt_minutes_until_arrival__p4(s: Dict[str, Any]) -> Any:
        v = s.get(f"debug_{fuel_code}_minutes_to_arrival") or s.get(f"debug_{fuel_code}_minutes_ahead")
        return v if v is not None else "—"
    
    def _fmt_open_at_eta__p4(s: Dict[str, Any]) -> str:
        """
        Open-at-ETA audit flag.
        Uses the pipeline-provided value via _open_at_eta_flag() (conservative: None => Unknown).
        """
        v = _open_at_eta_flag(s)
        if v is True:
            return "Yes"
        if v is False:
            return "No"
        return "Unknown"

    def _fmt_utc_offset__p4(s: Dict[str, Any]) -> str:
        """
        Display the station-local UTC offset (as provided by Google Places).
        Example: UTC+01:00, UTC-05:30
        """
        raw = (s or {}).get("utc_offset_minutes")
        try:
            if raw is None:
                return "—"
            mins = int(float(raw))
        except Exception:
            return "—"

        sign = "+" if mins >= 0 else "-"
        mins_abs = abs(mins)
        hh = mins_abs // 60
        mm = mins_abs % 60
        return f"UTC{sign}{hh:02d}:{mm:02d}"

    def _fmt_opening_hours_eta_day__p4(s: Dict[str, Any]) -> str:
        """
        Best-effort evidence column:
        - Prefer the weekdayDescription for the ETA weekday (if we have 7 entries).
        - Otherwise fall back to a compact joined string.
        """
        oh = (s or {}).get("opening_hours")
        if not isinstance(oh, list) or not oh:
            return "—"

        # ETA parsing follows the same robust chain used elsewhere on Page 02
        eta_raw = (s or {}).get("eta") or (s or {}).get(f"debug_{fuel_code}_eta_utc") or (s or {}).get("eta_utc")
        dt = _parse_any_dt_to_berlin(eta_raw)

        # Google weekdayDescriptions typically has 7 strings (Mon..Sun). Use ETA weekday when possible.
        if dt is not None and len(oh) >= 7:
            try:
                return str(oh[int(dt.weekday())]).strip() or "—"  # 0=Mon..6=Sun
            except Exception:
                pass

        # Fallback: compact join (kept short to avoid exploding table width)
        items = [str(x).strip() for x in oh if x]
        if not items:
            return "—"
        return "; ".join(items[:3]) + ("; …" if len(items) > 3 else "")

    def _row__p4(s: Dict[str, Any]) -> Dict[str, Any]:
        dk, dm = _detour_metrics(s)
        brand = s.get("brand") or s.get("tk_name") or s.get("station_name") or s.get("name") or "—"
        name = s.get("tk_name") or s.get("osm_name") or s.get("name") or s.get("station_name") or ""
        addr_google = _station_google_address_best_effort(s)

        # Detour fuel (L) (best-effort)
        detour_liters__p4: Optional[float] = None
        if consumption_l_per_100km__p4 is not None and dk is not None:
            try:
                detour_liters__p4 = (float(dk) * float(consumption_l_per_100km__p4)) / 100.0
            except Exception:
                detour_liters__p4 = None

        # Costs: prefer economics outputs; fallback compute if missing
        detour_fuel_cost__p4 = _as_float((s or {}).get(econ_detour_cost_key__p4))
        if detour_fuel_cost__p4 is None and detour_liters__p4 is not None:
            price_for_cost = _as_float((s or {}).get(pred_key__p4))
            if price_for_cost is None:
                price_for_cost = _as_float(_current_price_best_effort__p4(s))
            if price_for_cost is not None and price_for_cost >= MIN_VALID_PRICE_EUR_L:
                detour_fuel_cost__p4 = float(detour_liters__p4) * float(price_for_cost)

        time_cost__p4 = _as_float((s or {}).get(econ_time_cost_key__p4))
        if time_cost__p4 is None and value_of_time_eur_per_hour__p4 is not None and dm is not None:
            try:
                time_cost__p4 = (float(dm) / 60.0) * float(value_of_time_eur_per_hour__p4)
            except Exception:
                time_cost__p4 = None

        # --- Benchmark-based savings vs on-route best/median/worst (same cards as Section 2, but per-row) ---
        station_pred__p4 = _as_float((s or {}).get(pred_key__p4))

        def _gross_vs__p4(ref_price: Optional[float]) -> Optional[float]:
            if ref_price is None or station_pred__p4 is None or litres__p4 is None:
                return None
            return (float(ref_price) - float(station_pred__p4)) * float(litres__p4)

        def _net_vs__p4(ref_price: Optional[float]) -> Optional[float]:
            g = _gross_vs__p4(ref_price)
            if g is None or detour_fuel_cost__p4 is None or time_cost__p4 is None:
                return None
            return float(g) - float(detour_fuel_cost__p4) - float(time_cost__p4)

        def _price_or_na__p4(ref_price: Optional[float]) -> str:
            return "N/A" if not onroute_benchmark_available__p4 else _fmt_price_or_dash__p4(ref_price)

        def _eur_or_na__p4(v: Optional[float]) -> str:
            return "N/A" if not onroute_benchmark_available__p4 else _fmt_eur_or_dash__p4(v)

        gross = (s or {}).get(econ_gross_key__p4)
        net = (s or {}).get(econ_net_key__p4)

        return {
            "Brand": _safe_text(brand),
            "Name": _safe_text(name),
            "Address": _safe_text(addr_google),
            "Distance to station": _fmt_dist_to_station__p4(s),
            "Fraction of route": _fmt_fraction__p4(s.get("fraction_of_route")),
            "ETA": _fmt_eta_local__p4(s),
            "Minutes until arrival": _fmt_minutes_until_arrival__p4(s),

            # Open-at-ETA audit columns (both tables)
            "Open at ETA": _fmt_open_at_eta__p4(s),
            "Opening hours (ETA day)": _fmt_opening_hours_eta_day__p4(s),
            "Station UTC offset": _fmt_utc_offset__p4(s),

            "Detour distance": _format_km(dk),
            "Detour time": _format_min(dm),
            "Detour fuel (L)": _format_liters(detour_liters__p4),
            "Detour fuel cost": _fmt_eur_or_dash__p4(detour_fuel_cost__p4),
            "Time cost": _fmt_eur_or_dash__p4(time_cost__p4),
            "Current price": _fmt_price_or_dash__p4(_current_price_best_effort__p4(s)),
            "Predicted price": _fmt_price_or_dash__p4((s or {}).get(pred_key__p4)),

            # Run context (constant per row; makes the table self-contained)
            "Fuel type": _safe_text(fuel_type_label__p4),
            "Min net saving (threshold)": _fmt_eur_or_dash__p4(min_net_saving_eur__p4),

            "Gross saving": "—" if not bool(use_economics) else _fmt_eur_or_dash__p4(gross),
            "Net saving": "—" if not bool(use_economics) else _fmt_eur_or_dash__p4(net),

            # NEW: On-route benchmarks (same value for all rows)
            "On-route best (pred)": _price_or_na__p4(on_best__p4),
            "On-route median (pred)": _price_or_na__p4(on_median__p4),
            "On-route worst (pred)": _price_or_na__p4(on_worst__p4),

            # NEW: Savings breakdown vs benchmarks (per-row)
            "Gross vs best": _eur_or_na__p4(_gross_vs__p4(on_best__p4)),
            "Gross vs median": _eur_or_na__p4(_gross_vs__p4(on_median__p4)),
            "Gross vs worst": _eur_or_na__p4(_gross_vs__p4(on_worst__p4)),
            "Net vs best": _eur_or_na__p4(_net_vs__p4(on_best__p4)),
            "Net vs median": _eur_or_na__p4(_net_vs__p4(on_median__p4)),
            "Net vs worst": _eur_or_na__p4(_net_vs__p4(on_worst__p4)),
        }

    def _reorder_cols__p4(df: pd.DataFrame, *, is_selected: bool) -> pd.DataFrame:
        base_order = [
            "Brand",
            "Name",
            "Address",
            "Distance to station",
            "Fraction of route",
            "ETA",
            "Minutes until arrival",
            "Detour distance",
            "Detour time",
            "Detour fuel (L)",
            "Detour fuel cost",
            "Time cost",
            "Current price",
            "Predicted price",

            # NEW: run context (keep near the price/saving block)
            "Fuel type",
            "Min net saving (threshold)",

            "Gross saving",
            "Net saving",
            "On-route best (pred)",
            "On-route median (pred)",
            "On-route worst (pred)",
            "Gross vs best",
            "Gross vs median",
            "Gross vs worst",
            "Net vs best",
            "Net vs median",
            "Net vs worst",
            "Open at ETA",
            "Opening hours (ETA day)",
            "Station UTC offset",
        ]
        selected_only = [
            "Baseline: net saving (worst on-route)",
            "Baseline: predicted price (worst on-route)",
        ]
        desired = base_order + (selected_only if is_selected else [])
        desired_present = [c for c in desired if c in df.columns]
        remainder = [c for c in df.columns if c not in desired_present]
        return df[desired_present + remainder]

    # -----------------------------
    # Render: Selected table
    # -----------------------------
    st.markdown(
        "#### Selected stations (hard-feasible + economically viable):",
        help=(
            "Stations shown here are the ones that the recommender would consider eligible candidates for ranking.\n\n"
            "Selection is applied in two stages:\n"
            "1) Hard constraints (non-negotiable):\n"
            "   - Valid prediction inputs (predicted price must be available and plausible)\n"
            "   - Open at ETA filter (only if enabled): stations explicitly closed at ETA are removed\n"
            "   - Within detour caps (km/min)\n"
            "   - Within the configured min/max distance window (if configured)\n"
            "   - Deduplicated by station UUID (only one variant is kept)\n"
            "2) Economic benchmark (only when applicable and economics mode is enabled):\n"
            "   - A station is economically viable if Net saving ≥ the worst on-route net saving (reference benchmark)\n"
            "   - If no on-route benchmark exists, the economic step is treated as N/A and all hard-feasible stations appear here\n\n"
            "Open-at-ETA audit columns:\n"
            "   - Open at ETA: Yes / No / Unknown\n"
            "   - Unknown means opening-hours evidence was missing or not parseable; it is NOT treated as closed\n"
            "   - Opening hours (ETA day) shows the station’s weekday schedule text for the ETA weekday (best-effort)\n"
            "   - Station UTC offset shows the offset used for local-time evaluation (when provided)\n\n"
            "Baseline columns (reference benchmark):\n"
            "   - Baseline net saving = worst on-route net saving used as the economic threshold\n"
            "   - Baseline predicted price = predicted price of that same worst on-route reference station\n\n"
            "Benchmark breakdown (right-side columns):\n"
            "   - On-route best/median/worst are computed from the hard-feasible on-route subset\n"
            "   - Gross vs best/median/worst compares predicted prices for the configured refuel volume\n"
            "   - Net vs best/median/worst subtracts detour fuel cost and time cost from the gross saving\n"
        )
    )

    if selected__p4:
        df_selected__p4 = pd.DataFrame([_row__p4(s) for s in selected__p4])

        # Baseline columns (constant per cell; only for Selected table)
        baseline_net_str__p4 = "—"
        baseline_pred_str__p4 = "—"

        if bool(use_economics) and econ_benchmark_available__p4 and (worst_onroute_net__p4 is not None):
            baseline_net_str__p4 = _fmt_eur_or_dash__p4(worst_onroute_net__p4)

            # Find the station that defines the "worst on-route" net saving (deterministic tie-break)
            best_match: Optional[Dict[str, Any]] = None
            best_key: Optional[Tuple] = None

            for s in hard_feasible__p4:
                dk, dm = _detour_metrics(s)
                if dk is None or dm is None:
                    continue
                if float(dk) > float(onroute_km__p4) or float(dm) > float(onroute_min__p4):
                    continue

                net_v = _as_float((s or {}).get(econ_net_key__p4))
                if net_v is None:
                    continue

                pred_v = _as_float((s or {}).get(pred_key__p4))
                k = (
                    float(net_v),  # smallest net (worst)
                    -(float(pred_v) if pred_v is not None else float("-inf")),  # then highest predicted price
                    _as_float((s or {}).get("fraction_of_route")) or float("inf"),
                    _as_float((s or {}).get("distance_along_m")) or float("inf"),
                )

                if best_key is None or k < best_key:
                    best_key = k
                    best_match = s

            if isinstance(best_match, dict):
                baseline_pred_str__p4 = _fmt_price_or_dash__p4(best_match.get(pred_key__p4))

        df_selected__p4["Baseline: net saving (worst on-route)"] = baseline_net_str__p4
        df_selected__p4["Baseline: predicted price (worst on-route)"] = baseline_pred_str__p4

        df_selected__p4 = _reorder_cols__p4(df_selected__p4, is_selected=True)
        st.dataframe(df_selected__p4, use_container_width=True, hide_index=True)
    else:
        st.info("No selected stations to display for this cached run under the current criteria.")

    # -----------------------------
    # Render: Discarded table
    # -----------------------------
    st.markdown(
        "#### Discarded stations (hard and/or economic):",
        help=(
            "Stations shown here are everything that did not end up in the Selected table.\n\n"
            "A station can appear here for two reasons:\n"
            "1) Hard-discarded (failed feasibility constraints):\n"
            "   - Missing or invalid predicted price (no reliable prediction input/output)\n"
            "   - Closed at ETA (only if the open-at-ETA filter is enabled and the system has explicit evidence)\n"
            "   - Exceeds detour caps (km/min)\n"
            "   - Outside the configured min/max distance window\n"
            "   - Removed during deduplication (duplicate station UUID variants; only one is retained)\n"
            "2) Economic rejection (only when an on-route benchmark exists and economics mode is enabled):\n"
            "   - The station is hard-feasible but Net saving < worst on-route net saving (benchmark threshold)\n\n"
            "Open-at-ETA audit columns:\n"
            "   - Open at ETA is still shown for transparency.\n"
            "   - Unknown indicates missing/unparseable opening-hours evidence and is NOT the reason for exclusion by itself.\n"
            "   - Opening hours (ETA day) and Station UTC offset help validate timezone-correct evaluation.\n\n"
            "Benchmark columns:\n"
            "   - On-route best/median/worst are global reference values for this run.\n"
            "   - Gross/Net vs references are shown even for rejected stations to make the magnitude and direction of savings visible.\n"
        )
    )

    if discarded__p4:
        df_discarded__p4 = pd.DataFrame([_row__p4(s) for s in discarded__p4])
        df_discarded__p4 = _reorder_cols__p4(df_discarded__p4, is_selected=False)
        st.dataframe(df_discarded__p4, use_container_width=True, hide_index=True)
    else:
        st.caption("No discarded stations to display.")

    maybe_persist_state()
    return

if __name__ == "__main__":
    main()
