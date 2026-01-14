"""
Station Details & Analysis (Page 03)

Design intent:
- Single-station analysis workspace: one selected station is the central object.
- Sidebar (Action tab) is the control plane: selection, context, comparison set, display density.
- Main page is the analysis canvas: summary → rationale → history/patterns → economics → comparison.
- Diagnostics are available but not visually leading: bottom expander only.

Selection sources supported (cross-page navigation):
1) st.session_state["selected_station_data"] (preferred)
2) st.session_state["selected_station_uuid"] resolved from last_run["ranked"]/["stations"]
3) st.session_state["explorer_results"] (Page 04 → Page 03 handoff)

State keys preserved:
- st.session_state["last_run"]
- st.session_state["selected_station_uuid"]
- st.session_state["selected_station_data"]
- st.session_state["explorer_results"]

Added (optional) disambiguation key:
- st.session_state["selected_station_source"] ∈ {"route", "explorer"}
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Plotly config (mobile-friendly: hide modebar/buttons)
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False, "responsive": True}


try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from ui.sidebar import render_sidebar_shell, render_station_selector

# ---------------------------------------------------------------------
# Import bootstrap (align with your other pages)
# ---------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]       # .../src/app
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import load_env_once
load_env_once()

from src.integration.historical_data import (
    get_station_price_history,
    calculate_hourly_price_stats,
    get_cheapest_and_most_expensive_hours,
)

from ui.formatting import (
    _station_uuid,
    _fmt_price,
    _fmt_eur,
    _fmt_km,
    _fmt_min,
    _safe_text,
)

from ui.styles import apply_app_css


# =============================================================================
# Plotly charts (kept close to your current implementation, with minor UX tweaks)
# =============================================================================

def create_price_trend_chart(df: pd.DataFrame, fuel_type: str, station_name: str) -> go.Figure:
    """Simple 14-day price trend chart."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(title=f"{fuel_type.upper()} Price History - {station_name}", height=380)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["price"],
        mode="lines",
        name="Price",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="%{x|%b %d, %H:%M}<br>€%{y:.3f}/L<extra></extra>",
    ))

    avg_price = float(df["price"].mean())
    min_price = float(df["price"].min())
    max_price = float(df["price"].max())

    fig.update_layout(
        title=f"{fuel_type.upper()} Price History (Last 14 Days)",
        xaxis_title="Date",
        yaxis_title="Price (€/L)",
        hovermode="x unified",
        height=400,
        showlegend=False,
        margin=dict(b=70),
    )

    fig.add_annotation(
        text=f"Average: €{avg_price:.3f} | Min: €{min_price:.3f} | Max: €{max_price:.3f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.18,
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="center",
    )
    return fig


def create_hourly_pattern_chart(hourly_df: pd.DataFrame, fuel_type: str, station_name: str) -> go.Figure:
    """Hourly price pattern bar chart."""
    if hourly_df is None or hourly_df.empty or hourly_df["avg_price"].isna().all():
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for hourly pattern",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(title=f"Best Time to Refuel - {station_name}", height=380)
        return fig

    optimal = get_cheapest_and_most_expensive_hours(hourly_df)
    cheapest_hour = optimal.get("cheapest_hour")
    most_expensive_hour = optimal.get("most_expensive_hour")

    colors: List[str] = []
    price_range = float(hourly_df["avg_price"].max() - hourly_df["avg_price"].min()) if hourly_df["avg_price"].dropna().any() else 0.0

    for _, row in hourly_df.iterrows():
        if pd.isna(row["avg_price"]):
            colors.append("lightgray")
        elif cheapest_hour is not None and int(row["hour"]) == int(cheapest_hour):
            colors.append("green")
        elif most_expensive_hour is not None and int(row["hour"]) == int(most_expensive_hour):
            colors.append("red")
        else:
            if price_range > 0:
                normalized = float((row["avg_price"] - hourly_df["avg_price"].min()) / price_range)
                colors.append(f"rgb({int(255*normalized)}, {int(200*(1-normalized))}, 0)")
            else:
                colors.append("orange")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_df["hour"],
        y=hourly_df["avg_price"],
        marker_color=colors,
        text=hourly_df["avg_price"].apply(lambda x: f"€{x:.3f}" if pd.notna(x) else "—"),
        textposition="outside",
        hovertemplate="Hour %{x}:00<br>€%{y:.3f}/L<extra></extra>",
    ))

    fig.update_layout(
        title=f"Hourly Pattern ({fuel_type.upper()}) - {station_name}",
        xaxis_title="Hour of Day",
        yaxis_title="Average Price (€/L)",
        height=430,
        showlegend=False,
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        margin=dict(t=70),
    )
    return fig


def create_comparison_chart(
    stations_data: Dict[str, pd.DataFrame],
    fuel_type: str,
    current_station_name: Optional[str] = None,
) -> go.Figure:
    """Comparison chart: current station vs selected alternatives."""
    if not stations_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No stations to compare",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(title=f"Station Comparison ({fuel_type.upper()})", height=380)
        return fig

    fig = go.Figure()
    for idx, (station_name, df) in enumerate(stations_data.items()):
        if df is None or df.empty:
            continue

        is_current = ("(Current)" in station_name) or (current_station_name and station_name.startswith(current_station_name))

        if is_current:
            color = "#00C800"
            width = 3
            opacity = 1.0
            dash = "solid"
        else:
            colors_palette = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6C5CE7", "#00B894"]
            color = colors_palette[idx % len(colors_palette)]
            width = 2
            opacity = 0.65
            dash = "dot"

        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["price"],
            mode="lines",
            name=station_name.replace(" (Current)", ""),
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            hovertemplate=f"{station_name}<br>%{{x|%b %d}}<br>€%{{y:.3f}}/L<extra></extra>",
        ))

    fig.update_layout(
        title="Historical Comparison (Last 14 Days)",
        xaxis_title="Date",
        yaxis_title="Price (€/L)",
        hovermode="x unified",
        height=450,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(r=170),
    )

    fig.add_annotation(
        text="Solid = selected station | Dotted = comparison set",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=10, color="gray"),
        xanchor="center",
    )
    return fig


# =============================================================================
# Helpers: station identity, time localization, price basis inference
# =============================================================================

def _station_name(station: Dict[str, Any]) -> str:
    return str(station.get("tk_name") or station.get("osm_name") or station.get("name") or "Unknown").strip()


def _station_brand(station: Dict[str, Any]) -> str:
    return str(station.get("brand") or "").strip()


def _station_city(station: Dict[str, Any]) -> str:
    return str(station.get("city") or "").strip()


def _to_berlin_dt(value: Any) -> Optional[pd.Timestamp]:
    """
    Convert a timestamp-like value to Europe/Berlin.
    Accepts:
      - pandas Timestamp
      - python datetime
      - ISO string
      - unix epoch seconds
    """
    if value is None:
        return None

    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if ts is pd.NaT:
            return None
        # ts is tz-aware UTC here
        if ZoneInfo is not None:
            return ts.tz_convert(ZoneInfo("Europe/Berlin"))
        # fallback: pytz-like string
        return ts.tz_convert("Europe/Berlin")
    except Exception:
        return None


@dataclass(frozen=True)
class PriceBasisInfo:
    label: str
    used_current_price: Optional[bool]
    horizon_used: Optional[Any]
    eta_local: Optional[pd.Timestamp]
    minutes_to_arrival: Optional[float]
    cells_ahead: Optional[Any]
    raw: Dict[str, Any]


def _infer_price_basis(station: Dict[str, Any], fuel_code: str) -> PriceBasisInfo:
    """
    Robust price-basis inference for a station.
    Does not rely on upstream helper behavior and never raises.
    """
    raw: Dict[str, Any] = {}

    # Canonical debug keys used across the project
    used_current = station.get(f"debug_{fuel_code}_used_current_price")
    horizon = station.get(f"debug_{fuel_code}_horizon_used")
    minutes = station.get(f"debug_{fuel_code}_minutes_to_arrival")
    if minutes is None:
        minutes = station.get(f"debug_{fuel_code}_minutes_ahead")
    cells = station.get(f"debug_{fuel_code}_cells_ahead_raw")

    # ETA can be present under different names depending on pipeline stage
    eta_any = station.get("eta")
    if eta_any is None:
        eta_any = station.get(f"debug_{fuel_code}_eta_utc")
    if eta_any is None:
        eta_any = station.get("eta_utc")

    eta_local = _to_berlin_dt(eta_any)

    raw.update({
        "used_current_price": used_current,
        "horizon_used": horizon,
        "minutes_to_arrival": minutes,
        "cells_ahead_raw": cells,
        "eta_input": eta_any,
        "eta_local": str(eta_local) if eta_local is not None else None,
    })

    # Derive label
    if used_current is True:
        label = "Current price (forced/fallback)"
    elif used_current is False:
        label = "Forecast price (ETA-aligned)"
    else:
        # Heuristic fallback: compare values if present
        curr = station.get(f"price_current_{fuel_code}")
        pred = station.get(f"pred_price_{fuel_code}")
        try:
            if curr is not None and pred is not None and float(pred) != float(curr):
                label = "Forecast price (inferred)"
            else:
                label = "Current price (inferred)"
        except Exception:
            label = "Unknown"

    # Normalize minutes
    minutes_f: Optional[float] = None
    try:
        if minutes is not None:
            minutes_f = float(minutes)
    except Exception:
        minutes_f = None

    return PriceBasisInfo(
        label=label,
        used_current_price=used_current if isinstance(used_current, bool) else None,
        horizon_used=horizon,
        eta_local=eta_local,
        minutes_to_arrival=minutes_f,
        cells_ahead=cells,
        raw=raw,
    )


def _resolve_station(
    *,
    selected_station_data: Any,
    selected_uuid: str,
    ranked: List[Dict[str, Any]],
    stations: List[Dict[str, Any]],
    explorer_results: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Return (station_dict, station_uuid) with a conservative resolution strategy.
    """
    if isinstance(selected_station_data, dict):
        u = _station_uuid(selected_station_data) or selected_uuid or ""
        return selected_station_data, u

    if selected_uuid:
        for s in ranked:
            if _station_uuid(s) == selected_uuid:
                return s, selected_uuid
        for s in stations:
            if _station_uuid(s) == selected_uuid:
                return s, selected_uuid
        for s in explorer_results:
            if _station_uuid(s) == selected_uuid:
                return s, selected_uuid

    return None, ""


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


# =============================================================================
# Main page
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Station Details",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_app_css()

    st.title("Station Details & Analysis")
    st.caption("Inspect one station at a time: price basis, timing, patterns, and economics.")

    # -----------------------------
    # Top navigation (consistent)
    # -----------------------------
    NAV_TARGETS = {
        "Home": "streamlit_app.py",
        "Analytics": "pages/02_route_analytics.py",
        "Station": "pages/03_station_details.py",
        "Explorer": "pages/04_station_explorer.py",
    }
    CURRENT = "Station"

    # Ensure correct tab is selected on landing
    if st.session_state.get("_active_page") != CURRENT:
        st.session_state["_active_page"] = CURRENT
        st.session_state["top_nav"] = CURRENT

    selected_nav = st.segmented_control(
        label="",
        options=list(NAV_TARGETS.keys()),
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="top_nav",
    )

    target = NAV_TARGETS.get(selected_nav, NAV_TARGETS[CURRENT])
    if target != NAV_TARGETS[CURRENT]:
        st.switch_page(target)

    # -----------------------------
    # Read state inputs
    # -----------------------------
    cached: Dict[str, Any] = st.session_state.get("last_run") or {}
    ranked: List[Dict[str, Any]] = list(cached.get("ranked") or [])
    stations: List[Dict[str, Any]] = list(cached.get("stations") or [])
    explorer_results: List[Dict[str, Any]] = list(st.session_state.get("explorer_results") or [])

    params: Dict[str, Any] = dict(cached.get("params") or {})
    fuel_code: str = str(cached.get("fuel_code", "e5"))
    litres_to_refuel: float = float(cached.get("litres_to_refuel", params.get("litres_to_refuel", 40.0)) or 40.0)
    use_economics: bool = bool(params.get("use_economics", True))

    # Do not mutate global debug_mode; diagnostics are always available on this page.
    debug_mode = True

    # -----------------------------
    # Sidebar control plane (Action tab)
    # -----------------------------
    def _action_tab() -> None:
        # A) Station selection (source + selectbox)
        st.sidebar.markdown("### Station")
        selection = render_station_selector(
            last_run=cached,
            explorer_results=explorer_results,
            widget_key_prefix="station_details",
        )

        # B) Context / run metadata (read-only)
        st.sidebar.markdown("### Context")
        ctx_rows = [
            ("Fuel", fuel_code.upper()),
            ("Economics enabled", "Yes" if use_economics else "No"),
            ("Refuel litres (assumed)", f"{float(litres_to_refuel):.0f} L"),
            ("Route stations available", str(len(ranked) if ranked else 0)),
            ("Explorer stations available", str(len(explorer_results) if explorer_results else 0)),
            ("Selected source", str(selection.source)),
        ]
        for k, v in ctx_rows:
            st.sidebar.markdown(f"- **{k}:** {v}")


        # C) Comparison set (configure once; view in Comparison section)
        st.sidebar.markdown("### Comparison set")
        if ranked:
            # Candidate comparison options come from the ranked list (stable and meaningful).
            # Store UUIDs only (labels are display-only).
            uuid_labels: Dict[str, str] = {}
            for i, s in enumerate(ranked[:25], start=1):
                u = _station_uuid(s)
                if not u:
                    continue
                nm = _safe_text(_station_name(s))
                br = _safe_text(_station_brand(s))
                city = _safe_text(_station_city(s).upper()) if _station_city(s) else ""
                label = f"#{i} {nm}" + (f" ({br})" if br else "") + (f" · {city}" if city else "")
                uuid_labels[u] = label

            current_uuid = str(st.session_state.get("selected_station_uuid") or "")
            candidates = [u for u in uuid_labels.keys() if u and u != current_uuid]

            # Default: first 2 candidates (if any)
            default_uuids = st.session_state.get("comparison_station_uuids")
            if not isinstance(default_uuids, list):
                default_uuids = candidates[:2]

            chosen = st.sidebar.multiselect(
                "Compare against (up to 2)",
                options=candidates,
                default=default_uuids[:2],
                max_selections=2,
                format_func=lambda u: uuid_labels.get(u, u),
                key="comparison_station_uuids_widget",
                help="Configured here; rendered in the Comparison section (expander).",
            )
            st.session_state["comparison_station_uuids"] = list(chosen)
        else:
            st.sidebar.info("Run the recommender to enable comparison against ranked alternatives.")

        # D) Display options
        st.sidebar.markdown("### Display")
        density = st.sidebar.radio(
            "Density",
            options=["Detailed", "Compact"],
            index=0 if str(st.session_state.get("station_details_density", "Detailed")) == "Detailed" else 1,
            key="station_details_density",
            help="Compact hides some explanatory text while preserving all numbers.",
        )
        st.sidebar.caption("Timezone: Europe/Berlin (fixed)")

        # E) Diagnostics note (no toggle)
        st.sidebar.markdown("### Diagnostics")
        st.sidebar.caption("Diagnostics are available at the bottom of the page.")

    render_sidebar_shell(action_renderer=_action_tab)

    # -----------------------------
    # Resolve station selection (single source of truth)
    # -----------------------------
    selected_uuid = str(st.session_state.get("selected_station_uuid") or "")
    selected_station_data = st.session_state.get("selected_station_data")

    station, station_uuid = _resolve_station(
        selected_station_data=selected_station_data,
        selected_uuid=selected_uuid,
        ranked=ranked,
        stations=stations,
        explorer_results=explorer_results,
    )

    # Persist resolved station for cross-page consistency (safe)
    if station is not None:
        st.session_state["selected_station_data"] = station
        if station_uuid:
            st.session_state["selected_station_uuid"] = station_uuid

    # -----------------------------
    # Empty state guard
    # -----------------------------
    if station is None:
        st.info(
            "No station selected yet. Use the sidebar (Action tab) to select a station. "
            "If you have no stations available, run a route or search via Station Explorer."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Go to Trip Planner", use_container_width=True):
                st.switch_page("streamlit_app.py")
        with c2:
            if st.button("Go to Station Explorer", use_container_width=True):
                st.switch_page("pages/04_station_explorer.py")
        return

    # -----------------------------
    # Header: identity + source badge line
    # -----------------------------
    name = _safe_text(_station_name(station))
    brand = _safe_text(_station_brand(station))
    city = _safe_text(_station_city(station))

    source = str(st.session_state.get("selected_station_source") or "")
    if source not in {"route", "explorer"}:
        # Best-effort inference: if station is in explorer_results UUID set → explorer, else route if ranked/stations exists.
        u = _station_uuid(station) or ""
        ex_uuids = {_station_uuid(s) for s in explorer_results if _station_uuid(s)}
        if u and u in ex_uuids:
            source = "explorer"
        elif ranked or stations:
            source = "route"
        else:
            source = "explorer"
        st.session_state["selected_station_source"] = source

    basis = _infer_price_basis(station, fuel_code=fuel_code)
    source_badge = "Selected from Route Run" if source == "route" else "Selected from Explorer"

    st.markdown(f"## {name}")
    subtitle_parts = []
    if brand:
        subtitle_parts.append(brand)
    if city:
        subtitle_parts.append(city.upper())
    st.caption(" · ".join(subtitle_parts) if subtitle_parts else "—")

    st.caption(f"{source_badge} · Fuel: {fuel_code.upper()} · Price basis: {basis.label}")

    # -----------------------------
    # KPI strip (single source of truth; rendered once)
    # -----------------------------
    curr_key = f"price_current_{fuel_code}"
    pred_key = f"pred_price_{fuel_code}"
    econ_net_key = f"econ_net_saving_eur_{fuel_code}"

    current_price = _safe_float(station.get(curr_key))
    predicted_price = _safe_float(station.get(pred_key))

    # Route-only detour keys (robust handling)
    detour_km = station.get("detour_distance_km")
    if detour_km is None:
        detour_km = station.get("detour_km")
    detour_min = station.get("detour_duration_min")
    if detour_min is None:
        detour_min = station.get("detour_min")

    detour_km_f = _safe_float(detour_km)
    detour_min_f = _safe_float(detour_min)

    if source != "route":
        detour_text = "—"
    else:
        detour_text = f"{_fmt_km(detour_km_f)} / {_fmt_min(detour_min_f)}"

    # ETA KPI: only show if we have it
    eta_text = "—"
    if basis.eta_local is not None:
        # show with timezone abbreviation (+01/+02)
        try:
            eta_text = basis.eta_local.strftime("%a %H:%M (%Z)")
        except Exception:
            eta_text = str(basis.eta_local)

    net_saving_text = "—"
    if use_economics and econ_net_key in station:
        net_saving_text = _fmt_eur(station.get(econ_net_key))

    # Predicted metric delta vs current (if both exist)
    pred_delta = None
    if current_price is not None and predicted_price is not None:
        pred_delta = predicted_price - current_price

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Current", _fmt_price(current_price))
    with c2:
        st.metric("Predicted", _fmt_price(predicted_price), delta=(f"{pred_delta:+.3f} €/L" if pred_delta is not None else None))
    with c3:
        st.metric("ETA (local)", eta_text)
    with c4:
        st.metric("Detour", detour_text)
    with c5:
        st.metric("Net saving", net_saving_text)

    # -----------------------------
    # Sections (single scroll)
    # -----------------------------
    st.caption("All station analytics are shown on one page. Use the sidebar to change station and comparison set.")


    # Collect exceptions for diagnostics
    diagnostics_errors: List[str] = []

    # =========================
    # Overview
    # =========================
    st.markdown("### Overview")

    density = str(st.session_state.get("station_details_density", "Detailed"))
    compact = density == "Compact"

    st.markdown("### Decision rationale")
    rationale_rows: List[Dict[str, Any]] = []

    # Basis summary
    rationale_rows.append({"Field": "Chosen basis", "Value": basis.label})
    if basis.used_current_price is True:
        rationale_rows.append({"Field": "Reason", "Value": "Forecast not used (forced fallback or unavailable)."})
    elif basis.used_current_price is False:
        rationale_rows.append({"Field": "Reason", "Value": "Forecast used (ETA-aligned)."})
    else:
        rationale_rows.append({"Field": "Reason", "Value": "Inferred from available fields."})

    if basis.horizon_used is not None:
        rationale_rows.append({"Field": "Forecast horizon", "Value": str(basis.horizon_used)})
    if basis.minutes_to_arrival is not None:
        rationale_rows.append({"Field": "Minutes to arrival", "Value": f"{basis.minutes_to_arrival:.0f} min"})
    if basis.eta_local is not None:
        try:
            rationale_rows.append({"Field": "ETA (Europe/Berlin)", "Value": basis.eta_local.strftime("%Y-%m-%d %H:%M (%Z)")})
        except Exception:
            rationale_rows.append({"Field": "ETA (Europe/Berlin)", "Value": str(basis.eta_local)})

    # Route context (only if source == route)
    if source == "route":
        ranked_uuids = {_station_uuid(s) for s in ranked if _station_uuid(s)}
        tag = "Ranked" if station_uuid in ranked_uuids else "Excluded"
        rationale_rows.append({"Field": "Route context", "Value": f"{tag} candidate in latest route run"})
    else:
        # Explorer-specific: show distance if present
        dist = station.get("distance_km")
        if dist is not None:
            try:
                rationale_rows.append({"Field": "Explorer distance", "Value": f"{float(dist):.1f} km"})
            except Exception:
                rationale_rows.append({"Field": "Explorer distance", "Value": str(dist)})

    st.dataframe(pd.DataFrame(rationale_rows), hide_index=True, use_container_width=True)

    st.markdown("### Economics summary")
    if not use_economics:
        st.info("Economics are disabled for the latest route run. Enable economics in Trip Planner and run again.")
    else:
        if econ_net_key not in station:
            st.warning("Economic metrics are not available for this station.")
        else:
            net = _safe_float(station.get(econ_net_key))
            gross = _safe_float(station.get(f"econ_gross_saving_eur_{fuel_code}"))
            detour_fuel_cost = _safe_float(station.get(f"econ_detour_fuel_cost_eur_{fuel_code}"))
            time_cost = _safe_float(station.get(f"econ_time_cost_eur_{fuel_code}"))
            breakeven = _safe_float(station.get(f"econ_breakeven_liters_{fuel_code}"))

            if net is not None:
                if net > 0:
                    st.success("This detour is economically worthwhile for the assumed refuel amount.")
                else:
                    st.warning("This detour is not economically worthwhile for the assumed refuel amount.")

            econ_cols = st.columns(3)
            with econ_cols[0]:
                st.metric("Gross saving", _fmt_eur(gross))
            with econ_cols[1]:
                detour_cost_total = (detour_fuel_cost or 0.0) + (time_cost or 0.0)
                st.metric("Detour cost", _fmt_eur(detour_cost_total))
            with econ_cols[2]:
                st.metric("Net saving", _fmt_eur(net))

            if breakeven is not None and breakeven > 0:
                st.caption(f"Break-even: refuel at least {breakeven:.1f} L for a positive net saving.")

            if not compact:
                with st.expander("Formula (how net saving is computed)", expanded=False):
                    st.markdown(
                        f"""
                        Net saving is computed on the assumed refuel amount (**{litres_to_refuel:.1f} L**) as:

                        **Net saving** = **Gross saving** − **Detour fuel cost** − **Time cost**

                        - Gross saving captures the station's price advantage relative to the baseline (cheapest on-route).
                        - Detour fuel cost captures extra fuel consumed by the detour.
                        - Time cost captures the value of extra time (if configured).
                        """
                    )

# =========================
# History & Patterns
# =========================

    st.markdown("---")
    st.markdown("### Price history & patterns")
    if not station_uuid:
        st.warning("Station UUID is missing; cannot load historical data.")
    else:
        with st.spinner("Loading historical price data..."):
            try:
                df_history = get_station_price_history(
                    station_uuid=station_uuid,
                    fuel_type=fuel_code,
                    days=14,
                )
            except Exception as e:
                df_history = None
                diagnostics_errors.append(f"Price history error: {e!r}")

        if df_history is None or df_history.empty:
            st.info("No historical price data available for this station/fuel type.")
        else:
            st.plotly_chart(create_price_trend_chart(df_history, fuel_code, name), use_container_width=True, config=PLOTLY_CONFIG)

            st.markdown("---")
            st.markdown("### Best time to refuel (hourly pattern)")

            try:
                hourly_stats = calculate_hourly_price_stats(df_history)
            except Exception as e:
                hourly_stats = None
                diagnostics_errors.append(f"Hourly stats error: {e!r}")

            if hourly_stats is None or hourly_stats.empty or hourly_stats["avg_price"].dropna().empty:
                st.info("Not enough data to calculate hourly patterns.")
            else:
                st.plotly_chart(create_hourly_pattern_chart(hourly_stats, fuel_code, name), use_container_width=True, config=PLOTLY_CONFIG)

                optimal = get_cheapest_and_most_expensive_hours(hourly_stats) or {}
                cheapest_hour = optimal.get("cheapest_hour")
                most_expensive_hour = optimal.get("most_expensive_hour")
                cheapest_price = optimal.get("cheapest_price")
                most_expensive_price = optimal.get("most_expensive_price")

                if cheapest_hour is not None and most_expensive_hour is not None and cheapest_price is not None and most_expensive_price is not None:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Cheapest hour", f"{int(cheapest_hour):02d}:00", f"€{float(cheapest_price):.3f}/L")
                    with col_b:
                        st.metric("Most expensive hour", f"{int(most_expensive_hour):02d}:00", f"€{float(most_expensive_price):.3f}/L", delta_color="inverse")
                    with col_c:
                        diff = float(most_expensive_price) - float(cheapest_price)
                        if litres_to_refuel:
                            st.metric("Potential savings", f"€{diff * litres_to_refuel:.2f}", f"€{diff:.3f}/L")
                        else:
                            st.metric("Price difference", f"€{diff:.3f}/L", "Best vs worst hour")

# =========================
# Economics (detailed)
# =========================

    st.markdown("---")
    st.markdown("### Economic breakdown")
    st.caption("Detailed decomposition of savings vs detour costs.")

    if not use_economics:
        st.info("Economics are disabled for the latest route run. Enable economics in Trip Planner and run again.")
    else:
        econ_net_key = f"econ_net_saving_eur_{fuel_code}"
        if econ_net_key not in station:
            st.warning("Economic metrics are not available for this station.")
        else:
            econ_rows = [
                {"Metric": "Baseline on-route price", "Value": _fmt_price(station.get(f"econ_baseline_price_{fuel_code}")), "Note": "Cheapest station without detouring"},
                {"Metric": "Gross saving", "Value": _fmt_eur(station.get(f"econ_gross_saving_eur_{fuel_code}")), "Note": "Price advantage before costs"},
                {"Metric": "Detour fuel (litres)", "Value": (f"{_safe_float(station.get(f'econ_detour_fuel_l_{fuel_code}')):.2f} L" if _safe_float(station.get(f"econ_detour_fuel_l_{fuel_code}")) is not None else "—"), "Note": "Extra fuel consumed due to detour"},
                {"Metric": "Detour fuel cost", "Value": _fmt_eur(station.get(f"econ_detour_fuel_cost_eur_{fuel_code}")), "Note": "Cost of extra fuel"},
                {"Metric": "Time cost", "Value": _fmt_eur(station.get(f"econ_time_cost_eur_{fuel_code}")), "Note": "Value of additional time (if configured)"},
                {"Metric": "Net saving", "Value": _fmt_eur(station.get(econ_net_key)), "Note": "Gross saving minus all costs"},
                {"Metric": "Break-even litres", "Value": (f"{_safe_float(station.get(f'econ_breakeven_liters_{fuel_code}')):.1f} L" if _safe_float(station.get(f"econ_breakeven_liters_{fuel_code}")) is not None else "—"), "Note": "Minimum litres for positive net saving"},
            ]
            st.dataframe(pd.DataFrame(econ_rows), hide_index=True, use_container_width=True)

            # What-if slider (display-only): do not mutate last_run / station
            with st.expander("What-if: different refuel amount (display-only)", expanded=False):
                what_if_l = st.slider("Refuel litres", min_value=1.0, max_value=120.0, value=float(min(max(litres_to_refuel, 1.0), 120.0)), step=1.0)
                st.caption("This section does not re-run the model. It provides an intuitive scaling of the gross saving component.")
                baseline_price = _safe_float(station.get(f"econ_baseline_price_{fuel_code}"))
                current_basis_price = _safe_float(station.get(f"price_current_{fuel_code}"))
                # Use chosen basis for intuitive display: if forecast basis, use predicted, else current.
                basis_price = _safe_float(station.get(f"pred_price_{fuel_code}")) if "Forecast" in basis.label else current_basis_price
                if baseline_price is None or basis_price is None:
                    st.info("Not enough data to compute a what-if estimate.")
                else:
                    gross_per_l = max(0.0, baseline_price - basis_price)
                    gross_est = gross_per_l * float(what_if_l)
                    detour_fuel_cost = _safe_float(station.get(f"econ_detour_fuel_cost_eur_{fuel_code}")) or 0.0
                    time_cost = _safe_float(station.get(f"econ_time_cost_eur_{fuel_code}")) or 0.0
                    net_est = gross_est - detour_fuel_cost - time_cost
                    c_a, c_b, c_c = st.columns(3)
                    with c_a:
                        st.metric("Estimated gross saving", _fmt_eur(gross_est))
                    with c_b:
                        st.metric("Detour cost (fixed)", _fmt_eur(detour_fuel_cost + time_cost))
                    with c_c:
                        st.metric("Estimated net saving", _fmt_eur(net_est))

            with st.expander("How to read these numbers", expanded=False):
                st.markdown(
                    """
                    - **Baseline on-route price**: the cheapest station without leaving the baseline route.
                    - **Gross saving**: price advantage for the assumed refuel amount (litres).
                    - **Detour fuel cost**: extra fuel consumed because of the detour, valued at the baseline price.
                    - **Time cost**: optional valuation of extra detour time.
                    - **Net saving**: gross saving minus detour fuel cost minus time cost.
                    - **Break-even litres**: litres required for net saving to become positive (if applicable).
                    """
                )

# =========================
# Comparison (only multi-station view)
# =========================

    st.markdown("---")

    with st.expander("Comparison (optional)", expanded=False):
        st.markdown("### Comparison")
        st.caption("Compare the selected station against the sidebar-configured comparison set.")

        comp_uuids = st.session_state.get("comparison_station_uuids")
        if not isinstance(comp_uuids, list):
            comp_uuids = []

        if not station_uuid:
            st.warning("Station UUID is missing; cannot load comparison data.")
        elif not comp_uuids:
            st.info("No comparison stations selected. Configure the comparison set in the sidebar (Action tab).")
        else:
            with st.spinner("Loading comparison data..."):
                stations_data: Dict[str, pd.DataFrame] = {}
                try:
                    current_df = get_station_price_history(station_uuid, fuel_code, days=14)
                    if current_df is not None and not current_df.empty:
                        stations_data[f"{name} (Current)"] = current_df
                except Exception as e:
                    diagnostics_errors.append(f"Comparison current station history error: {e!r}")

                # Map UUIDs to readable labels (prefer ranked context)
                label_by_uuid: Dict[str, str] = {}
                for i, s in enumerate(ranked[:50], start=1):
                    u = _station_uuid(s)
                    if not u:
                        continue
                    label_by_uuid[u] = f"#{i} {_safe_text(_station_name(s))}"

                for u in comp_uuids[:2]:
                    if not u or u == station_uuid:
                        continue
                    try:
                        df = get_station_price_history(u, fuel_code, days=14)
                        if df is not None and not df.empty:
                            label = label_by_uuid.get(u, u)
                            stations_data[label] = df
                    except Exception as e:
                        diagnostics_errors.append(f"Comparison history error for {u}: {e!r}")

            if len(stations_data) < 2:
                st.info("Not enough historical data to compare these stations.")
            else:
                st.plotly_chart(create_comparison_chart(stations_data, fuel_code, current_station_name=name), use_container_width=True, config=PLOTLY_CONFIG)

                # Quick insight
                try:
                    current_avg = stations_data.get(f"{name} (Current)", pd.DataFrame())["price"].mean()
                    if pd.notna(current_avg):
                        diffs = []
                        for label, df in stations_data.items():
                            if "(Current)" in label or df.empty:
                                continue
                            alt_avg = df["price"].mean()
                            if pd.notna(alt_avg):
                                diffs.append(float(alt_avg) - float(current_avg))

                        if diffs:
                            avg_diff = sum(diffs) / len(diffs)
                            if avg_diff > 0:
                                st.success(f"On average over 14 days, the selected station is cheaper by about €{avg_diff:.3f}/L versus the comparison set.")
                            elif avg_diff < 0:
                                st.warning(f"On average over 14 days, the selected station is more expensive by about €{abs(avg_diff):.3f}/L versus the comparison set.")
                            else:
                                st.info("On average over 14 days, the selected station matches the comparison set.")
                except Exception as e:
                    diagnostics_errors.append(f"Comparison insight error: {e!r}")

    # -----------------------------
    # Diagnostics (bottom, collapsed)
    # -----------------------------
    st.markdown("---")
    with st.expander("Diagnostics (debug)", expanded=False):
        st.caption("This section is intended for debugging and development. It does not affect the recommender logic.")
        debug_keys = sorted([k for k in station.keys() if str(k).startswith("debug_")])
        st.markdown("**debug_* keys present**")
        st.write(debug_keys if debug_keys else "No debug_* keys present on this station.")

        st.markdown("**Price basis fields (normalized)**")
        st.write(basis.raw)

        if diagnostics_errors:
            st.markdown("**Captured errors**")
            for err in diagnostics_errors:
                st.code(err)

        st.markdown("**Raw station payload**")
        st.json(station, expanded=False)

    st.caption("Tip: Use Route Analytics or Station Explorer to change the selection, and Trip Planner to run a new route.")


if __name__ == "__main__":
    main()
