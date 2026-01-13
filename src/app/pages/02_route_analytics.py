"""
Route Analytics (Page 2)

Purpose:
- Provide auditability and comparability for a single route run.
- Explain why a station was recommended and why others were excluded.
- Offer drill-down navigation into Station Details.

Data source:
- st.session_state["last_run"] written by src/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import altair as alt 

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
from services.presenters import build_ranking_dataframe

from ui.styles import apply_app_css

from ui.sidebar import render_sidebar_shell

# -----------------------------
# Helpers
# -----------------------------
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
    d_min = station.get("detour_duration_min")

    if d_km is None:
        d_km = station.get("detour_km")
    if d_min is None:
        d_min = station.get("detour_min")

    try:
        km = float(d_km) if d_km is not None else 0.0
    except (TypeError, ValueError):
        km = 0.0

    try:
        mins = float(d_min) if d_min is not None else 0.0
    except (TypeError, ValueError):
        mins = 0.0

    return max(0.0, km), max(0.0, mins)


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


def _compute_baseline_price_for_display(
    stations: List[Dict[str, Any]],
    *,
    fuel_code: str,
    onroute_max_detour_km: float = 1.0,
    onroute_max_detour_min: float = 5.0,
) -> Optional[float]:
    """
    Recompute a baseline price similar to the decision layer:
    - cheapest predicted price among "on-route" stations (per onroute thresholds)
    - fallback: global cheapest predicted price
    """
    pred_key = f"pred_price_{fuel_code}"

    onroute_prices: List[float] = []
    all_prices: List[float] = []

    for s in stations or []:
        p = _as_float(s.get(pred_key))
        if p is None:
            continue
        all_prices.append(p)

        km, mins = _detour_metrics(s)
        if km <= float(onroute_max_detour_km) and mins <= float(onroute_max_detour_min):
            onroute_prices.append(p)

    if onroute_prices:
        return min(onroute_prices)
    if all_prices:
        return min(all_prices)
    return None


def _ensure_breakeven_liters_for_display(
    stations: List[Dict[str, Any]],
    *,
    fuel_code: str,
    baseline_price: Optional[float] = None,
    onroute_max_detour_km: float = 1.0,
    onroute_max_detour_min: float = 5.0,
) -> None:
    """
    Backfill missing econ_breakeven_liters_* values in-place so tables/captions do not show "-".

    Break-even litres:
        (detour_fuel_cost + time_cost) / (baseline_price - station_price)
    Only meaningful if station_price < baseline_price.
    """
    if not stations:
        return

    econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
    econ_baseline_key = f"econ_baseline_price_{fuel_code}"
    econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
    econ_detour_fuel_l_key = f"econ_detour_fuel_l_{fuel_code}"
    econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"

    pred_key = f"pred_price_{fuel_code}"
    curr_key = f"price_current_{fuel_code}"

    # 1) Determine baseline price (prefer what decision layer already computed)
    b = _as_float(baseline_price)
    if b is None:
        for s in stations:
            b = _as_float(s.get(econ_baseline_key))
            if b is not None:
                break
    if b is None:
        b = _compute_baseline_price_for_display(
            stations,
            fuel_code=fuel_code,
            onroute_max_detour_km=onroute_max_detour_km,
            onroute_max_detour_min=onroute_max_detour_min,
        )
    if b is None:
        return

    # 2) Backfill per station if missing
    for s in stations:
        existing = s.get(econ_breakeven_key)
        if existing is not None and not _is_missing_number(existing):
            continue

        # station price: prefer prediction, fallback to current
        p_station = _as_float(s.get(pred_key))
        if p_station is None:
            p_station = _as_float(s.get(curr_key))
        if p_station is None:
            continue

        diff = float(b) - float(p_station)
        if diff <= 0:
            # Not cheaper than baseline => no meaningful break-even litres
            continue

        detour_cost = _as_float(s.get(econ_detour_fuel_cost_key))
        if detour_cost is None:
            # fallback if only detour liters exist
            detour_l = _as_float(s.get(econ_detour_fuel_l_key))
            if detour_l is not None:
                detour_cost = detour_l * float(p_station)

        time_cost = _as_float(s.get(econ_time_cost_key))
        if time_cost is None:
            time_cost = 0.0

        if detour_cost is None:
            continue

        s[econ_breakeven_key] = max(0.0, (float(detour_cost) + float(time_cost)) / diff)

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


def _price_basis_row(station: Dict[str, Any], fuel_code: str) -> Dict[str, Any]:
    """
    Best-effort extraction of 'price basis' debug fields. We do not assume every run
    populates these keys; UI should degrade gracefully.
    """
    pred_key = f"pred_price_{fuel_code}"
    curr_key = f"price_current_{fuel_code}"

    used_current = _coerce_bool(station.get(f"debug_{fuel_code}_used_current_price"))
    horizon_used = station.get(f"debug_{fuel_code}_horizon_used")
    eta_utc = station.get(f"debug_{fuel_code}_eta_utc") or station.get("eta_utc") or station.get("eta")

    minutes_to_arrival = (
        station.get(f"debug_{fuel_code}_minutes_to_arrival")
        or station.get(f"debug_{fuel_code}_minutes_ahead")
    )
    cells_ahead = station.get(f"debug_{fuel_code}_cells_ahead_raw") or station.get(f"debug_{fuel_code}_cells_ahead")

    # Robust basis label
    if used_current is True:
        basis = "Current price used (no forecast)"
    elif used_current is False:
        basis = "Forecast used"
    else:
        # Fallback inference if the explicit flag is missing
        pred = station.get(pred_key)
        curr = station.get(curr_key)
        if (not _is_missing_number(pred)) and (not _is_missing_number(curr)):
            try:
                basis = "Current/Forecast (inferred)" if abs(float(pred) - float(curr)) < 1e-6 else "Forecast (inferred)"
            except Exception:
                basis = "Current/Forecast (inferred)"
        else:
            basis = "Unknown (missing debug)"

    if horizon_used not in (None, "", 0):
        basis = f"{basis} — horizon={horizon_used}"

    return {
        "Station": _safe_text(station.get("brand") or station.get("station_name") or station.get("name")),
        "Basis": basis,
        "ETA (UTC)": _safe_text(eta_utc),
        "Minutes to arrival": minutes_to_arrival if minutes_to_arrival is not None else "—",
        "Cells ahead": cells_ahead if cells_ahead is not None else "—",
    }


def _price_basis_table(stations: List[Dict[str, Any]], fuel_code: str, limit: int = 12) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s in stations[: max(0, int(limit))]:
        rows.append(_price_basis_row(s, fuel_code=fuel_code))
    return pd.DataFrame(rows)


def _compute_funnel_counts(
    stations: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    excluded: List[Dict[str, Any]],
    fuel_code: str,
    use_economics: bool,
    cap_detour_km: Optional[float],
    cap_detour_min: Optional[float],
    min_net_saving_eur: Optional[float],
) -> Dict[str, int]:
    """
    High-level funnel counts computed from the station objects *you already cache*.
    This avoids relying on filter_log internals and remains stable if filter_log changes.
    """
    pred_key = f"pred_price_{fuel_code}"
    econ_key = f"econ_net_saving_eur_{fuel_code}"

    total = len(stations)
    ranked_n = len(ranked)

    missing_pred = 0
    cap_failed = 0
    econ_failed = 0
    other = 0

    for s in excluded:
        # 1) missing prediction
        if _is_missing_number(s.get(pred_key)):
            missing_pred += 1
            continue

        # 2) detour caps
        km, mins = _detour_metrics(s)
        cap_fail = False
        if cap_detour_km is not None and km is not None and km > float(cap_detour_km):
            cap_fail = True
        if cap_detour_min is not None and mins is not None and mins > float(cap_detour_min):
            cap_fail = True
        if cap_fail:
            cap_failed += 1
            continue

        # 3) economics threshold
        if use_economics and min_net_saving_eur is not None:
            net = s.get(econ_key)
            if (not _is_missing_number(net)) and (float(net) < float(min_net_saving_eur)):
                econ_failed += 1
                continue

        other += 1

    return {
        "candidates_total": total,
        "ranked": ranked_n,
        "excluded": len(excluded),
        "missing_prediction": missing_pred,
        "failed_detour_caps": cap_failed,
        "failed_economics": econ_failed,
        "excluded_other": other,
    }


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric coercion:
    - handles comma decimals ("1,706")
    - strips currency/unit text ("€/L", " km", etc.)
    - keeps negatives and decimals
    """
    as_str = s.astype(str)
    as_str = as_str.str.replace(",", ".", regex=False)
    as_str = as_str.str.replace(r"[^0-9.\-]+", "", regex=True)
    return pd.to_numeric(as_str, errors="coerce")


def _compute_onroute_worst_price(
    ranked: List[Dict[str, Any]],
    fuel_code: str,
    *,
    onroute_max_detour_km: float = 1.0,
    onroute_max_detour_min: float = 5.0,
) -> Optional[float]:
    """Worst (highest) predicted price among on-route stations."""
    pred_key = f"pred_price_{fuel_code}"
    vals: List[float] = []
    for s in ranked:
        p = s.get(pred_key)
        if p is None:
            continue
        km, mins = _detour_metrics(s)
        if km <= onroute_max_detour_km and mins <= onroute_max_detour_min:
            try:
                vals.append(float(p))
            except (TypeError, ValueError):
                continue
    return max(vals) if vals else None


def _compute_net_vs_onroute_worst(
    best: Dict[str, Any],
    onroute_worst: Optional[float],
    fuel_code: str,
    litres_to_refuel: Optional[float],
) -> Optional[float]:
    """
    (worst_onroute - chosen) * litres - detour_fuel_cost - time_cost.

    This matches the "story" you display on Page 1 (worst on-route baseline).
    Note: Your economics engine uses a different baseline ("cheapest on-route"),
    so we show both on this page.
    """
    if onroute_worst is None or litres_to_refuel is None:
        return None

    pred_key = f"pred_price_{fuel_code}"
    p = best.get(pred_key)
    if p is None:
        return None

    try:
        chosen = float(p)
        litres = float(litres_to_refuel)
    except (TypeError, ValueError):
        return None

    detour_fuel_cost = best.get(f"econ_detour_fuel_cost_eur_{fuel_code}")
    time_cost = best.get(f"econ_time_cost_eur_{fuel_code}")

    try:
        detour_fuel_cost_f = float(detour_fuel_cost) if detour_fuel_cost is not None else 0.0
    except (TypeError, ValueError):
        detour_fuel_cost_f = 0.0

    try:
        time_cost_f = float(time_cost) if time_cost is not None else 0.0
    except (TypeError, ValueError):
        time_cost_f = 0.0

    gross = (float(onroute_worst) - chosen) * litres
    return gross - detour_fuel_cost_f - time_cost_f


def _exclusion_reason(
    s: Dict[str, Any],
    *,
    fuel_code: str,
    use_economics: bool,
    cap_km: Optional[float],
    cap_min: Optional[float],
    min_net_saving: Optional[float],
) -> str:
    """Best-effort reason label for excluded stations."""
    pred_key = f"pred_price_{fuel_code}"
    econ_net_key = f"econ_net_saving_eur_{fuel_code}"

    if s.get(pred_key) is None:
        return "Missing predicted price"

    km, mins = _detour_metrics(s)

    if cap_km is not None and km > cap_km:
        return "Detour distance above cap"
    if cap_min is not None and mins > cap_min:
        return "Detour time above cap"

    if use_economics and (min_net_saving is not None):
        net = s.get(econ_net_key)
        if net is None:
            return "Missing economics metrics"
        try:
            if float(net) < float(min_net_saving):
                return "Below minimum net saving"
        except (TypeError, ValueError):
            return "Invalid economics metrics"

    return "Not ranked (other)"


def _station_label(s: Dict[str, Any], idx: Optional[int] = None, tag: str = "") -> str:
    brand = s.get("brand")
    name = s.get("tk_name") or s.get("osm_name") or s.get("name") or "Station"
    city = s.get("city") or ""
    title = brand if (brand and str(brand).strip()) else name
    prefix = f"#{idx} " if idx is not None else ""
    suffix = f" [{tag}]" if tag else ""
    return _safe_text(f"{prefix}{title} ({city}){suffix}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Route Analytics", layout="wide")

    apply_app_css()

    # Header (consistent across pages)
    st.title("Route Analytics")
    st.caption("##### Understand the recommendation.")

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

    render_sidebar_shell(
        action_placeholder="Placeholder: Action (nothing to configure on Route Analytics yet)."
    )

    # ------------------------------------------------------------------
    # Sidebar (bottom): Debug mode toggle (controls st.session_state used by other pages)
    # ------------------------------------------------------------------
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = True

    with st.sidebar:
        st.markdown("---")
        st.checkbox(
            "Debug mode",
            key="debug_mode",
            help="When enabled, the UI shows additional raw/debug payloads on all pages.",
        )

    cached = _get_last_run()
    if not cached:
        st.info("No cached run found. Run a route recommendation first on the main page.")
        if st.button("Open main page"):
            st.switch_page("streamlit_app.py")
        return


    fuel_code: str = str(cached.get("fuel_code") or "e5").lower()
    run_summary: Dict[str, Any] = cached.get("run_summary") or {}
    use_economics: bool = bool(run_summary.get("use_economics") or cached.get("use_economics", False))
    litres_to_refuel = cached.get("litres_to_refuel")
    debug_mode: bool = bool(st.session_state.get("debug_mode", False))

    # Keep cache consistent so other pages (and this page on rerun) see the toggle.
    cached["debug_mode"] = debug_mode
    st.session_state["last_run"] = cached

    stations: List[Dict[str, Any]] = cached.get("stations") or []
    ranked: List[Dict[str, Any]] = cached.get("ranked") or []
    best_station: Optional[Dict[str, Any]] = cached.get("best_station")
    route_info: Dict[str, Any] = cached.get("route_info") or {}

    # Constraints (from Page 1 run_summary; fall back to None when missing)
    constraints = run_summary.get("constraints") or {}
    cap_km = constraints.get("max_detour_km")
    cap_min = constraints.get("max_detour_min")
    min_net_saving = constraints.get("min_net_saving_eur") if use_economics else None

    # ------------------------------------------------------------------
    # Optional exact filter diagnostics (emitted by decision layer)
    # ------------------------------------------------------------------
    filter_log: Dict[str, Any] = cached.get("filter_log") or {}
    primary_reason_by_uuid: Dict[str, str] = filter_log.get("primary_reason_by_uuid") or {}
    filter_counts: Dict[str, int] = filter_log.get("counts") or {}
    filter_notes: List[str] = filter_log.get("notes") or []
    filter_thresholds: Dict[str, Any] = filter_log.get("thresholds") or {}

    # ------------------------------------------------------------------
    # SECTION A — Your route & recommendation
    # ------------------------------------------------------------------
    st.subheader("A) Your route and the recommended decision")

    # A1. Run summary
    total_n = len(stations)
    ranked_n = len(ranked)
    excluded_n = max(0, total_n - ranked_n)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations (total)", f"{total_n}")
    c2.metric("Stations (ranked)", f"{ranked_n}")
    c3.metric("Stations (excluded)", f"{excluded_n}")
    c4.metric("Economics mode", "On" if use_economics else "Off")

    # Route metadata (best-effort)
    start_lbl = route_info.get("start_label") or ""
    end_lbl = route_info.get("end_label") or ""
    route_km = route_info.get("route_km")
    route_min = route_info.get("route_min")

    meta_left, meta_right = st.columns([2, 1])
    with meta_left:
        if start_lbl or end_lbl:
            st.write(f"**Route:** {_safe_text(start_lbl)} → {_safe_text(end_lbl)}")
        else:
            st.write("**Route:** (labels unavailable in cache)")

        constraints_lines = []
        if cap_km is not None:
            constraints_lines.append(f"Max detour distance: **{float(cap_km):.1f} km**")
        if cap_min is not None:
            constraints_lines.append(f"Max detour time: **{float(cap_min):.0f} min**")
        if use_economics and (min_net_saving is not None):
            constraints_lines.append(f"Minimum net saving: **{float(min_net_saving):.2f} €**")
        if constraints_lines:
            st.write("**Constraints:** " + " · ".join(constraints_lines))

    with meta_right:
        if route_km is not None:
            try:
                st.metric("Route length", f"{float(route_km):.1f} km")
            except (TypeError, ValueError):
                st.metric("Route length", "—")
        else:
            st.metric("Route length", "—")

        if route_min is not None:
            try:
                st.metric("Route duration", f"{float(route_min):.0f} min")
            except (TypeError, ValueError):
                st.metric("Route duration", "—")
        else:
            st.metric("Route duration", "—")

    st.markdown("")

    # A2. Recommended station audit
    if best_station:
        st.markdown("#### Recommended station (audit)")

        onroute_worst = _compute_onroute_worst_price(ranked, fuel_code)
        net_vs_worst = _compute_net_vs_onroute_worst(best_station, onroute_worst, fuel_code, litres_to_refuel)

        pred_key = f"pred_price_{fuel_code}"
        curr_key = f"price_current_{fuel_code}"
        econ_net_key = f"econ_net_saving_eur_{fuel_code}"
        econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
        econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
        econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
        econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
        econ_baseline_key = f"econ_baseline_price_{fuel_code}"

        km, _mins = _detour_metrics(best_station)
        dist_m = best_station.get("distance_along_m")

        # Headline metrics (match your Page 1 set)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Predicted {fuel_code.upper()} price", _fmt_price(best_station.get(pred_key)))
        m2.metric(f"Current {fuel_code.upper()} price", _fmt_price(best_station.get(curr_key)))
        if dist_m is None:
            m3.metric("Distance along route", "—")
        else:
            try:
                m3.metric("Distance along route", f"{float(dist_m)/1000.0:.1f} km")
            except (TypeError, ValueError):
                m3.metric("Distance along route", "—")

        m4, m5, m6 = st.columns(3)
        m4.metric("On-route worst price", "—" if onroute_worst is None else _fmt_price(onroute_worst))
        m5.metric("Net saving vs on-route worst", "—" if net_vs_worst is None else _fmt_eur(net_vs_worst))
        m6.metric("Detour distance", _fmt_km(km))

        if use_economics:
            st.markdown("**Economics decomposition**")
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Baseline price used", _fmt_price(best_station.get(econ_baseline_key)))
            d2.metric("Gross saving", _fmt_eur(best_station.get(econ_gross_key)))
            d3.metric("Detour fuel cost", _fmt_eur(best_station.get(econ_detour_fuel_cost_key)))
            d4.metric("Time cost", _fmt_eur(best_station.get(econ_time_cost_key)))
            d5.metric("Net saving (model)", _fmt_eur(best_station.get(econ_net_key)))

            # Ensure break-even litres is present for display (defensive)
            _ensure_breakeven_liters_for_display(
                [best_station],
                fuel_code=fuel_code,
                onroute_max_detour_km=float(filter_thresholds.get("onroute_max_detour_km", 1.0) or 1.0),
                onroute_max_detour_min=float(filter_thresholds.get("onroute_max_detour_min", 5.0) or 5.0),
            )

            be = _as_float(best_station.get(econ_breakeven_key))
            if be is not None:
                st.caption(f"Break-even litres: {be:.1f} L")

        with st.expander("How the predicted price is chosen (price basis)", expanded=False):
            st.write(
                "The app may use either the station's current price or a forecasted price depending on the "
                "ETA-to-station and the model horizon selected for that ETA. The table below shows the "
                "debug fields used to decide this for the recommended station. If some fields are missing, "
                "the run did not populate them and the UI falls back gracefully."
            )
            basis_df = _price_basis_table([best_station], fuel_code=fuel_code, limit=1)
            st.dataframe(basis_df, hide_index=True, use_container_width=True)

            if ranked:
                st.caption("Top ranked stations — price basis snapshot")
                st.dataframe(_price_basis_table(ranked, fuel_code=fuel_code, limit=10), hide_index=True, use_container_width=True)

            action_col1, action_col2 = st.columns([1, 3])
            with action_col1:
                if st.button("Open Station Details", use_container_width=True):
                    st.session_state["selected_station_uuid"] = _station_uuid(best_station)
                    st.session_state["selected_station_data"] = best_station
                    st.switch_page("pages/03_station_details.py")
            with action_col2:
                st.caption("Use Station Details for the full station profile (prices, prediction basis, debugging).")

    else:
        st.warning("No recommended station available in cache (likely no stations passed filters).")

    st.markdown("---")

    # ------------------------------------------------------------------
    # SECTION B — Station universe & exclusions
    # ------------------------------------------------------------------
    st.subheader("B) All stations and why some were excluded")
    
    ranked_uuids = {_station_uuid(s) for s in ranked if _station_uuid(s)}
    excluded = [s for s in stations if (_station_uuid(s) not in ranked_uuids)]

    funnel = _compute_funnel_counts(
        stations=stations,
        ranked=ranked,
        excluded=excluded,
        fuel_code=fuel_code,
        use_economics=use_economics,
        cap_detour_km=cap_km,
        cap_detour_min=cap_min,
        min_net_saving_eur=min_net_saving,
    )

    st.markdown("#### Filtering funnel (high-level)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Candidates", str(funnel["candidates_total"]))
    c2.metric("Missing prediction", str(funnel["missing_prediction"]))
    c3.metric("Failed detour caps", str(funnel["failed_detour_caps"]))
    c4.metric("Failed economics", str(funnel["failed_economics"]) if use_economics else "—")
    c5.metric("Ranked", str(funnel["ranked"]))

    with st.expander("Funnel details (counts)", expanded=False):
        funnel_df = pd.DataFrame([funnel])
        st.dataframe(funnel_df, hide_index=True, use_container_width=True)

    if filter_thresholds or filter_notes:
        with st.expander("Filter diagnostics (exact)", expanded=False):
            if filter_thresholds:
                st.markdown("**Thresholds used**")
                st.json(filter_thresholds, expanded=False)
            if filter_notes:
                st.markdown("**Notes**")
                for n in filter_notes:
                    st.write(f"- {n}")
            if filter_counts:
                st.markdown("**Counts (as emitted by decision layer)**")
                st.json(filter_counts, expanded=False)

    # B1. Exclusion breakdown
    if excluded:
        # Prefer exact primary reasons from filter_log if available
        if primary_reason_by_uuid:
            reasons = [primary_reason_by_uuid.get(_station_uuid(s), "Not ranked (other)") for s in excluded]
        else:
            reasons = [
                _exclusion_reason(
                    s,
                    fuel_code=fuel_code,
                    use_economics=use_economics,
                    cap_km=float(cap_km) if cap_km is not None else None,
                    cap_min=float(cap_min) if cap_min is not None else None,
                    min_net_saving=float(min_net_saving) if min_net_saving is not None else None,
                )
                for s in excluded
            ]

        reason_counts = pd.Series(reasons).value_counts()

        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Exclusion breakdown**")
            for reason, cnt in reason_counts.items():
                st.write(f"- {reason}: **{int(cnt)}**")
        with right:
            chart_df = reason_counts.reset_index()
            chart_df.columns = ["Reason", "Count"]
            st.bar_chart(chart_df.set_index("Reason"))
    else:
        st.info("No stations were excluded (all candidates appear in the ranked list).")

    st.markdown("")

    # B2a. Value view: stations not worse than the worst on-route option
    # This is a display-only table intended to be more consistent for end users.
    st.markdown("#### Stations better than the worst on-route option (value view)")

    pred_key = f"pred_price_{fuel_code}"

    # Reconstruct the "hard-pass" candidate set (what the decision layer would consider before the min-net-saving soft filter).
    cap_km_f = float(cap_km) if cap_km is not None else None
    cap_min_f = float(cap_min) if cap_min is not None else None

    hard_pass: List[Dict[str, Any]] = []
    for s in stations:
        # Require a usable prediction for the selected fuel.
        p = s.get(pred_key)
        try:
            p_f = float(p) if p is not None else None
        except (TypeError, ValueError):
            p_f = None
        if p_f is None or p_f < MIN_VALID_PRICE_EUR_L:
            continue

        # Require passing detour caps (hard constraints).
        km, mins = _detour_metrics(s)
        if (cap_km_f is not None and km > cap_km_f) or (cap_min_f is not None and mins > cap_min_f):
            continue

        hard_pass.append(s)

    # Compute the on-route worst price within the hard-pass set.
    onroute_km = float(filter_thresholds.get("onroute_max_detour_km", 1.0) or 1.0)
    onroute_min = float(filter_thresholds.get("onroute_max_detour_min", 5.0) or 5.0)

    onroute_worst_price = _compute_onroute_worst_price(
        hard_pass,
        fuel_code=fuel_code,
        onroute_max_detour_km=onroute_km,
        onroute_max_detour_min=onroute_min,
    )

    if onroute_worst_price is None:
        st.info(
            "Could not compute the on-route worst price (no stations qualify as on-route under the current thresholds)."
        )
    else:
        # Keep stations whose predicted price is not worse than the on-route worst predicted price.
        value_view: List[Dict[str, Any]] = []
        for s in hard_pass:
            p = s.get(pred_key)
            try:
                p_f = float(p) if p is not None else None
            except (TypeError, ValueError):
                p_f = None
            if p_f is None:
                continue

            if p_f <= float(onroute_worst_price) + 1e-9:
                value_view.append(s)

        # Backfill break-even liters (mainly for older cached runs or partial station dicts).
        if use_economics and value_view:
            _ensure_breakeven_liters_for_display(value_view, fuel_code=fuel_code)

        if value_view:
            # Sort with the same primary logic as the decision layer (when available).
            if use_economics:
                econ_net_key = f"econ_net_saving_eur_{fuel_code}"

                def _net_sort_key(sta: Dict[str, Any]) -> Tuple:
                    net = sta.get(econ_net_key)
                    try:
                        net_f = float(net) if net is not None else float("-inf")
                    except (TypeError, ValueError):
                        net_f = float("-inf")
                    return (
                        -net_f,
                        sta.get("fraction_of_route", float("inf")),
                        sta.get("distance_along_m", float("inf")),
                    )

                value_view_sorted = sorted(value_view, key=_net_sort_key)
            else:
                value_view_sorted = sorted(
                    value_view,
                    key=lambda x: (
                        x.get(pred_key, float("inf")),
                        x.get("fraction_of_route", float("inf")),
                        x.get("distance_along_m", float("inf")),
                    ),
                )

            value_df = build_ranking_dataframe(value_view_sorted, fuel_code=fuel_code, debug_mode=debug_mode)
            st.dataframe(value_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Filter used: predicted price ≤ on-route worst predicted price ({float(onroute_worst_price):.3f} €/L), "
                "within the same detour caps as the decision."
            )
        else:
            st.info("No stations are better than the on-route worst price under the current detour caps.")

    st.markdown("")

    # B2. Ranked table
    st.markdown("#### Ranked stations (comparison table)")
    if ranked:
        # Backfill break-even litres for display (defensive for older caches / partial station dicts)
        if use_economics:
            _ensure_breakeven_liters_for_display(ranked, fuel_code=fuel_code)

        ranked_df = build_ranking_dataframe(ranked, fuel_code=fuel_code, debug_mode=debug_mode)

        st.dataframe(ranked_df, use_container_width=True, hide_index=True)
    else:
        st.error("No stations were ranked. Use the exclusion section above to see why.")

    # B3. Excluded table
    st.markdown("#### Excluded stations (audit table)")
    if excluded:
        rows = []
        pred_key = f"pred_price_{fuel_code}"
        curr_key = f"price_current_{fuel_code}"
        econ_net_key = f"econ_net_saving_eur_{fuel_code}"

        for s in excluded:
            km, mins = _detour_metrics(s)
            uid = _station_uuid(s)
            reason = primary_reason_by_uuid.get(uid) if primary_reason_by_uuid else None
            if not reason:
                reason = _exclusion_reason(
                    s,
                    fuel_code=fuel_code,
                    use_economics=use_economics,
                    cap_km=float(cap_km) if cap_km is not None else None,
                    cap_min=float(cap_min) if cap_min is not None else None,
                    min_net_saving=float(min_net_saving) if min_net_saving is not None else None,
                )
            rows.append(
                {
                    "Brand": s.get("brand") or "",
                    "Name": s.get("tk_name") or s.get("osm_name") or s.get("name") or "",
                    "City": s.get("city") or "",
                    "Detour distance [km]": km,
                    "Detour time [min]": mins,
                    f"Predicted {fuel_code.upper()} price": s.get(pred_key),
                    f"Current {fuel_code.upper()} price": s.get(curr_key),
                    "Net saving (if available)": s.get(econ_net_key) if use_economics else None,
                    "Reason": reason,
                    "UUID": uid,
                }
            )

        excluded_df = pd.DataFrame(rows)
        st.dataframe(excluded_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No excluded stations to display.")

    st.markdown("---")

    # ------------------------------------------------------------------
    # Drill-down selection & navigation
    # ------------------------------------------------------------------
    st.subheader("Drill-down: select a station")

    options: List[Tuple[str, str, Dict[str, Any]]] = []

    for i, s in enumerate(ranked[:20], start=1):
        uid = _station_uuid(s)
        if uid:
            options.append((uid, _station_label(s, idx=i, tag="ranked"), s))

    for s in excluded[:50]:
        uid = _station_uuid(s)
        if uid:
            options.append((uid, _station_label(s, idx=None, tag="excluded"), s))

    selected_uuid_default = st.session_state.get("selected_station_uuid")
    option_labels = [lbl for _, lbl, _ in options]
    option_uuids = [uid for uid, _, _ in options]

    if options:
        default_index = option_uuids.index(selected_uuid_default) if selected_uuid_default in option_uuids else 0

        chosen_label = st.selectbox(
            "Select a station to inspect (ranked and excluded)",
            options=option_labels,
            index=default_index,
        )
        chosen_uuid = option_uuids[option_labels.index(chosen_label)]
        chosen_station = next((s for uid, _, s in options if uid == chosen_uuid), None)

        if chosen_station:
            st.session_state["selected_station_uuid"] = chosen_uuid
            st.session_state["selected_station_data"] = chosen_station

            pred_key = f"pred_price_{fuel_code}"
            curr_key = f"price_current_{fuel_code}"
            econ_key = f"econ_net_saving_eur_{fuel_code}"

            km, mins = _detour_metrics(chosen_station)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Current", _fmt_price(chosen_station.get(curr_key)))
            s2.metric("Predicted", _fmt_price(chosen_station.get(pred_key)))
            s3.metric("Detour", f"{_fmt_km(km)} / {_fmt_min(mins)}")
            s4.metric("Net saving", _fmt_eur(chosen_station.get(econ_key)) if use_economics else "—")

            with st.expander("Price basis for this station (current vs forecast)", expanded=False):
                st.dataframe(_price_basis_table([chosen_station], fuel_code=fuel_code, limit=1), hide_index=True, use_container_width=True)

            if debug_mode:
                st.caption("Debug keys present on this station (for mapping):")
                debug_keys = sorted([k for k in best_station.keys() if str(k).startswith("debug_")])
                st.write(debug_keys if debug_keys else "No debug_* keys present on best_station.")

            if st.button("Open Station Details for this station", use_container_width=True):
                st.switch_page("pages/03_station_details.py")

            with st.expander("Show raw station data (debug)"):
                st.json(chosen_station, expanded=False)

    else:
        st.info("No stations available for drill-down.")


if __name__ == "__main__":
    main()
