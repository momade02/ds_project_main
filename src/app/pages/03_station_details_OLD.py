"""
Station Details Page

Detailed analysis for selected fuel station with all charts and comparisons.

Step 6 (Robustness):
- Accept selection from:
  1) st.session_state["selected_station_data"] (preferred),
  2) st.session_state["selected_station_uuid"] resolved from last_run["ranked"]/["stations"],
  3) st.session_state["explorer_results"] (future Page 4),
  4) a selectbox fallback if last_run exists but no station is selected.
- Provide consistent navigation across pages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ui.sidebar import render_sidebar_shell

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

# Historical data functions for charts
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
    _fmt_liters,
    _describe_price_basis,
    _safe_text,
)

from ui.styles import apply_app_css

# =====================================================================
# PLOTLY CHART FUNCTIONS
# =====================================================================

def create_price_trend_chart(
    df: pd.DataFrame,
    fuel_type: str,
    station_name: str
) -> go.Figure:
    """Create simple, clean 14-day price trend chart."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title=f"{fuel_type.upper()} Price History - {station_name}",
            height=400,
        )
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

    today = pd.Timestamp.now()
    if df["date"].min() <= today <= df["date"].max():
        fig.add_vline(
            x=today,
            line_dash="dash",
            line_color="rgba(128, 128, 128, 0.3)",
            annotation_text="Today",
            annotation_position="top",
            annotation_font=dict(size=10, color="gray"),
        )

    avg_price = df["price"].mean()
    min_price = df["price"].min()
    max_price = df["price"].max()

    fig.update_layout(
        title=f"📈 {fuel_type.upper()} Price History (Last 14 Days)",
        xaxis_title="Date",
        yaxis_title="Price (€/L)",
        hovermode="x unified",
        height=400,
        showlegend=False,
        margin=dict(b=80),
    )

    fig.add_annotation(
        text=f"Average: €{avg_price:.3f} | Min: €{min_price:.3f} | Max: €{max_price:.3f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="center",
    )

    return fig


def create_hourly_pattern_chart(
    hourly_df: pd.DataFrame,
    fuel_type: str,
    station_name: str
) -> go.Figure:
    """Create hourly price pattern bar chart."""
    if hourly_df.empty or hourly_df["avg_price"].isna().all():
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for hourly pattern",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title=f"🕐 Best Time to Refuel - {station_name}",
            height=400,
        )
        return fig

    optimal = get_cheapest_and_most_expensive_hours(hourly_df)
    cheapest_hour = optimal["cheapest_hour"]
    most_expensive_hour = optimal["most_expensive_hour"]

    colors: List[str] = []
    for _, row in hourly_df.iterrows():
        if pd.isna(row["avg_price"]):
            colors.append("lightgray")
        elif row["hour"] == cheapest_hour:
            colors.append("green")
        elif row["hour"] == most_expensive_hour:
            colors.append("red")
        else:
            price_range = hourly_df["avg_price"].max() - hourly_df["avg_price"].min()
            if price_range > 0:
                normalized = (row["avg_price"] - hourly_df["avg_price"].min()) / price_range
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
        title=f"🕐 Best Time to Refuel ({fuel_type.upper()}) - {station_name}",
        xaxis_title="Hour of Day",
        yaxis_title="Average Price (€/L)",
        height=450,
        showlegend=False,
        xaxis=dict(
            tickmode="linear",
            tick0=0,
            dtick=2,
        ),
    )
    return fig


def create_comparison_chart(
    stations_data: Dict[str, pd.DataFrame],
    fuel_type: str,
    current_station_name: str = None
) -> go.Figure:
    """Create simplified comparison chart: current station vs top alternatives."""
    if not stations_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No stations to compare",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            title=f"Station Comparison ({fuel_type.upper()})",
            height=400,
        )
        return fig

    fig = go.Figure()
    for idx, (station_name, df) in enumerate(stations_data.items()):
        if df.empty:
            continue

        is_current = (current_station_name and station_name.startswith(current_station_name)) or "(Current)" in station_name

        if is_current:
            color = "#00C800"
            width = 3
            opacity = 1.0
            dash = "solid"
        else:
            colors_palette = ["#FF6B6B", "#4ECDC4", "#FFD93D"]
            color = colors_palette[idx % len(colors_palette)]
            width = 2
            opacity = 0.6
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
        title="📊 Is This Station Consistently Cheaper?",
        xaxis_title="Date",
        yaxis_title="Price (€/L)",
        hovermode="x unified",
        height=450,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    fig.add_annotation(
        text="Solid line = Your selection | Dotted lines = Alternatives for comparison",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=10, color="gray"),
        xanchor="center",
    )
    return fig


# =====================================================================
# SELECTION RESOLUTION (Step 6)
# =====================================================================

def _resolve_station_from_uuid(
    selected_uuid: str,
    ranked: List[Dict[str, Any]],
    stations: List[Dict[str, Any]],
    explorer_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not selected_uuid:
        return None

    for s in ranked:
        if _station_uuid(s) == selected_uuid:
            return s
    for s in stations:
        if _station_uuid(s) == selected_uuid:
            return s
    for s in explorer_results:
        if _station_uuid(s) == selected_uuid:
            return s
    return None


def _station_label(s: Dict[str, Any], idx: int) -> str:
    name = s.get("tk_name") or s.get("osm_name") or f"Station {idx+1}"
    brand = s.get("brand") or ""
    return f"#{idx+1}: {name}" + (f" ({brand})" if brand else "")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    st.set_page_config(page_title="Station Details", layout="wide")

    apply_app_css()

    st.title("Station Details & Analysis")

    st.caption("##### Inspect a selected station: price basis, economics, and supporting details.")

    NAV_TARGETS = {
        "Home": "streamlit_app.py",
        "Analytics": "pages/02_route_analytics.py",
        "Station": "pages/03_station_details.py",
        "Explorer": "pages/04_station_explorer.py",
    }
    CURRENT = "Station"

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

    # Sidebar controls (Station Details: debug is permanently enabled on this page)
    cached = st.session_state.get("last_run") or {}

    def _action_tab() -> None:
        # Placeholder text only (no toggle here)
        st.sidebar.info("Debug mode is permanently enabled on this page.")

    render_sidebar_shell(action_renderer=_action_tab)

    # Force debug mode ON for this page (keep downstream debug branches active)
    st.session_state["debug_mode"] = True
    debug_mode = True

    # Gather session state inputs
    selected_uuid = st.session_state.get("selected_station_uuid") or ""
    selected_station_data = st.session_state.get("selected_station_data")

    ranked: List[Dict[str, Any]] = cached.get("ranked") or []
    stations: List[Dict[str, Any]] = cached.get("stations") or []
    explorer_results: List[Dict[str, Any]] = st.session_state.get("explorer_results") or []

    fuel_code = cached.get("fuel_code", "e5")
    litres_to_refuel = cached.get("litres_to_refuel", 40.0)

    params = cached.get("params") or {}
    use_economics = bool(params.get("use_economics", True))

    # ------------------------------------------------------------------
    # Drill-down: select a station (ranked + excluded)  [moved from Page 02]
    # ------------------------------------------------------------------
    st.subheader("Drill-down: select a station")

    # Build excluded list = stations not in ranked (by UUID)
    ranked_uuids = {_station_uuid(s) for s in ranked if _station_uuid(s)}
    excluded = [s for s in stations if (_station_uuid(s) and _station_uuid(s) not in ranked_uuids)]

    def _drill_label(s: Dict[str, Any], idx: Optional[int], tag: str) -> str:
        name = s.get("tk_name") or s.get("osm_name") or s.get("name") or "Unknown"
        city = (s.get("city") or "").strip()
        prefix = f"#{idx} " if idx is not None else ""
        city_part = f" ({city.upper()})" if city else ""
        return f"{prefix}{name}{city_part} [{tag}]"

    options: List[Tuple[str, str, Dict[str, Any]]] = []

    for i, s in enumerate(ranked[:20], start=1):
        uid = _station_uuid(s)
        if uid:
            options.append((uid, _drill_label(s, idx=i, tag="ranked"), s))

    for s in excluded[:50]:
        uid = _station_uuid(s)
        if uid:
            options.append((uid, _drill_label(s, idx=None, tag="excluded"), s))

    if options:
        option_labels = [lbl for _, lbl, _ in options]
        option_uuids = [uid for uid, _, _ in options]

        selected_uuid_default = st.session_state.get("selected_station_uuid") or ""
        default_index = option_uuids.index(selected_uuid_default) if selected_uuid_default in option_uuids else 0

        chosen_label = st.selectbox(
            "Select a station to inspect (ranked and excluded)",
            options=option_labels,
            index=default_index,
            key="station_drilldown_select",
        )

        chosen_uuid = option_uuids[option_labels.index(chosen_label)]
        chosen_station = next((s for uid, _, s in options if uid == chosen_uuid), None)

        if isinstance(chosen_station, dict):
            # Persist selection for this page and for cross-page navigation
            st.session_state["selected_station_uuid"] = chosen_uuid
            st.session_state["selected_station_data"] = chosen_station

            # Keep local variables in sync for the downstream selection-resolution logic
            selected_uuid = chosen_uuid
            selected_station_data = chosen_station

            pred_key = f"pred_price_{fuel_code}"
            curr_key = f"price_current_{fuel_code}"
            econ_key = f"econ_net_saving_eur_{fuel_code}"

            # Detour metrics (robust key handling)
            d_km = chosen_station.get("detour_distance_km")
            d_min = chosen_station.get("detour_duration_min")
            if d_km is None:
                d_km = chosen_station.get("detour_km")
            if d_min is None:
                d_min = chosen_station.get("detour_min")

            try:
                km = max(0.0, float(d_km)) if d_km is not None else 0.0
            except (TypeError, ValueError):
                km = 0.0
            try:
                mins = max(0.0, float(d_min)) if d_min is not None else 0.0
            except (TypeError, ValueError):
                mins = 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current", _fmt_price(chosen_station.get(curr_key)))
            c2.metric("Predicted", _fmt_price(chosen_station.get(pred_key)))
            c3.metric("Detour", f"{_fmt_km(km)} / {_fmt_min(mins)}")
            c4.metric("Net saving", _fmt_eur(chosen_station.get(econ_key)) if use_economics else "—")

            # Price-basis diagnostics (lightweight, station-specific)
            with st.expander("Price basis for this station (current vs forecast)", expanded=False):
                used_current = chosen_station.get(f"debug_{fuel_code}_used_current_price")
                horizon = chosen_station.get(f"debug_{fuel_code}_horizon_used")
                minutes = (
                    chosen_station.get(f"debug_{fuel_code}_minutes_to_arrival")
                    if chosen_station.get(f"debug_{fuel_code}_minutes_to_arrival") is not None
                    else chosen_station.get(f"debug_{fuel_code}_minutes_ahead")
                )
                cells = chosen_station.get(f"debug_{fuel_code}_cells_ahead_raw")

                eta_local = chosen_station.get("eta") or chosen_station.get(f"debug_{fuel_code}_eta_utc")

                if used_current is True:
                    basis = "Current price (forced or fallback)"
                elif used_current is False:
                    basis = f"Forecast (inferred) — horizon={horizon}"
                else:
                    basis = "Unknown"

                st.dataframe(
                    pd.DataFrame(
                        [{
                            "Basis": basis,
                            "ETA (local)": eta_local,
                            "Minutes to arrival": minutes,
                            "Cells ahead": cells,
                        }]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            if debug_mode:
                st.caption("Debug keys present on this station (for mapping):")
                debug_keys = sorted([k for k in chosen_station.keys() if str(k).startswith("debug_")])
                st.write(debug_keys if debug_keys else "No debug_* keys present on this station.")

            with st.expander("Show raw station data (debug)"):
                st.json(chosen_station, expanded=False)

    else:
        st.info("No stations available for drill-down.")

    # Resolve station object robustly
    station: Optional[Dict[str, Any]] = None

    if isinstance(selected_station_data, dict):
        # Accept selected_station_data even if UUID is not set yet
        station = selected_station_data
        if not selected_uuid:
            selected_uuid = _station_uuid(station) or ""
            if selected_uuid:
                st.session_state["selected_station_uuid"] = selected_uuid
    elif selected_uuid:
        station = _resolve_station_from_uuid(selected_uuid, ranked, stations, explorer_results)

    # Persist resolved station into session for consistency
    st.session_state["selected_station_data"] = station
    if selected_uuid:
        st.session_state["selected_station_uuid"] = selected_uuid

    # Station header
    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"

    name = station.get("tk_name") or station.get("osm_name") or "Unknown"
    brand = station.get("brand") or ""

    st.markdown("### Station Details")
    header = f"**{_safe_text(name)}**"
    if brand:
        header += f"  \n{_safe_text(brand)}"
    st.markdown(header)

    station_uuid = _station_uuid(station)
    if station_uuid:
        st.caption(f"Station UUID: {station_uuid}")

    # Small info line: basis (current vs forecast) if available
    try:
        basis_text = _describe_price_basis(station, fuel_code)
        if basis_text:
            st.caption(f"Price basis: {basis_text}")
    except Exception:
        # Never fail the page due to a caption helper
        pass

    # Tabs
    tab_overview, tab_prices, tab_comparison, tab_economics = st.tabs(
        ["Overview", "Prices", "Comparison", "Economics"]
    )

    # ============================================================
    # OVERVIEW TAB
    # ============================================================
    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current price", _fmt_price(station.get(current_key)))
        col2.metric("Predicted price", _fmt_price(station.get(pred_key)))

        detour_km = station.get("detour_distance_km") or station.get("detour_km")
        detour_min = station.get("detour_duration_min") or station.get("detour_min")
        col3.metric("Detour distance", _fmt_km(detour_km))
        col4.metric("Detour time", _fmt_min(detour_min))

        st.markdown("---")
        st.markdown("### Savings Summary")

        econ_net_key = f"econ_net_saving_eur_{fuel_code}"
        econ_baseline_key = f"econ_baseline_price_{fuel_code}"
        econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
        econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
        econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
        econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"

        if econ_net_key in station:
            net_saving = station.get(econ_net_key)
            gross_saving = station.get(econ_gross_key)

            if net_saving is not None and net_saving > 0:
                st.success("This detour is worth it.")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Net Saving", f"€{net_saving:.2f}")
                with col_b:
                    st.metric("Price Saving", _fmt_eur(gross_saving))
                with col_c:
                    total_cost = (station.get(econ_detour_fuel_cost_key) or 0) + (station.get(econ_time_cost_key) or 0)
                    st.metric("Detour Cost", f"€{total_cost:.2f}")

                with st.expander("How is this calculated?"):
                    refuel_amount = litres_to_refuel if litres_to_refuel else 40.0
                    st.markdown(f"""
                    **For {refuel_amount:.0f}L refuel:**
                    - Price advantage: €{(gross_saving or 0):.2f}
                    - Detour fuel cost: €{(station.get(econ_detour_fuel_cost_key) or 0):.2f}
                    - Time cost: €{(station.get(econ_time_cost_key) or 0):.2f}
                    - **Net saving: €{(net_saving or 0):.2f}**
                    
                    Baseline on-route price: {_fmt_price(station.get(econ_baseline_key))}
                    """)
            else:
                st.warning("All options require detours (net saving is not positive).")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Net Cost", f"-€{abs(net_saving):.2f}" if net_saving is not None else "—")
                with col_b:
                    baseline_price = station.get(econ_baseline_key)
                    if baseline_price:
                        st.metric("Baseline (cheapest on-route)", _fmt_price(baseline_price))

                with st.expander("Why does this cost money?"):
                    detour_cost = (station.get(econ_detour_fuel_cost_key) or 0) + (station.get(econ_time_cost_key) or 0)
                    refuel_amount = litres_to_refuel if litres_to_refuel else 40.0
                    st.markdown(f"""
                    **For {refuel_amount:.0f}L refuel:**
                    - Price advantage: €{(gross_saving or 0):.2f}
                    - Detour cost: €{detour_cost:.2f}
                    - **Net result: -€{abs(net_saving):.2f}** if net_saving is not None else "—"
                    """)

            breakeven = station.get(econ_breakeven_key)
            if breakeven is not None and breakeven > 0:
                st.caption(f"Break-even: refuel at least **{breakeven:.1f} L** for this detour to be worthwhile.")
        else:
            if not use_economics:
                st.info("""
                Economic metrics were not calculated (economics toggle disabled on the Trip Planner run).
                
                To enable economic analysis:
                1. Go to Trip Planner
                2. Enable economics-based detour decision
                3. Run again
                """)
            else:
                st.warning("Economic metrics are not available for this station.")

    # ============================================================
    # PRICES TAB
    # ============================================================
    with tab_prices:
        if station_uuid:
            with st.spinner("Loading price history..."):
                try:
                    df_history = get_station_price_history(
                        station_uuid=station_uuid,
                        fuel_type=fuel_code,
                        days=14,
                    )

                    if df_history is not None and not df_history.empty:
                        fig_trend = create_price_trend_chart(df=df_history, fuel_type=fuel_code, station_name=name)
                        st.plotly_chart(fig_trend, use_container_width=True)

                        st.markdown("---")
                        st.markdown("### Best Time to Refuel")
                        st.caption("Based on historical patterns, when is fuel cheapest at this station?")

                        hourly_stats = calculate_hourly_price_stats(df_history)
                        if hourly_stats is not None and not hourly_stats.empty and hourly_stats["avg_price"].dropna().any():
                            fig_hourly = create_hourly_pattern_chart(hourly_df=hourly_stats, fuel_type=fuel_code, station_name=name)
                            st.plotly_chart(fig_hourly, use_container_width=True)

                            optimal = get_cheapest_and_most_expensive_hours(hourly_stats)
                            if optimal.get("cheapest_hour") is not None:
                                col_best, col_worst, col_savings = st.columns(3)

                                with col_best:
                                    st.metric("Cheapest Hour", f"{optimal['cheapest_hour']:02d}:00", f"€{optimal['cheapest_price']:.3f}/L")
                                with col_worst:
                                    st.metric("Most Expensive Hour", f"{optimal['most_expensive_hour']:02d}:00", f"€{optimal['most_expensive_price']:.3f}/L", delta_color="inverse")
                                with col_savings:
                                    savings_per_liter = optimal["most_expensive_price"] - optimal["cheapest_price"]
                                    if litres_to_refuel:
                                        total_savings = savings_per_liter * litres_to_refuel
                                        st.metric("Potential Savings", f"€{total_savings:.2f}", f"€{savings_per_liter:.3f}/L")
                                    else:
                                        st.metric("Price Difference", f"€{savings_per_liter:.3f}/L", "Best vs Worst time")
                        else:
                            st.info("Not enough data to calculate hourly patterns.")
                    else:
                        st.info("No historical price data available for this station/fuel type.")
                except Exception as e:
                    if debug_mode:
                        st.error(f"Error loading price history: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                    else:
                        st.warning("Could not load price history. Enable debug mode for details.")
        else:
            st.warning("Station UUID not available - cannot load price history.")

        if debug_mode:
            st.markdown("---")
            st.markdown("DEBUG: Price Lag Data")
            rows = [
                {"Metric": "Current price", "Value": _fmt_price(station.get(current_key))},
                {"Metric": "Predicted price", "Value": _fmt_price(station.get(pred_key))},
            ]
            for lag in ("1d", "2d", "3d", "7d"):
                rows.append({"Metric": f"Price lag {lag}", "Value": _fmt_price(station.get(f"price_lag_{lag}_{fuel_code}"))})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            debug_h_key = f"debug_{fuel_code}_horizon_used"
            debug_ucp_key = f"debug_{fuel_code}_used_current_price"
            if debug_h_key in station or debug_ucp_key in station:
                st.markdown("#### Forecast basis")
                st.write({
                    "used_current_price": station.get(debug_ucp_key),
                    "horizon_used": station.get(debug_h_key),
                })

    # ============================================================
    # COMPARISON TAB
    # ============================================================
    with tab_comparison:
        st.markdown("### Is This Station Consistently Cheaper?")
        st.caption("Compare your selected station with top alternatives to see if it's consistently the best choice, or just a one-time deal.")

        ranked_stations = cached.get("ranked", [])
        if not ranked_stations or len(ranked_stations) < 2:
            st.info("Need at least 2 stations to show comparison. Run the recommender first.")
        else:
            station_options: Dict[str, Tuple[str, str]] = {}
            for i, s in enumerate(ranked_stations[:10]):
                s_uuid = _station_uuid(s) or ""
                s_name = s.get("tk_name") or s.get("osm_name") or f"Station {i+1}"
                s_brand = s.get("brand") or ""
                label = f"#{i+1}: {s_name}" + (f" ({s_brand})" if s_brand else "")
                station_options[label] = (s_uuid, s_name)

            st.markdown("**Select up to 2 alternatives to compare:**")
            selected_labels = st.multiselect(
                "Choose stations",
                options=list(station_options.keys()),
                default=list(station_options.keys())[:min(2, len(station_options))],
                max_selections=2,
                key="comparison_stations",
            )

            if selected_labels:
                with st.spinner("Loading comparison data..."):
                    try:
                        stations_data: Dict[str, pd.DataFrame] = {}

                        if station_uuid:
                            try:
                                current_df = get_station_price_history(station_uuid, fuel_code, days=14)
                                if current_df is not None and not current_df.empty:
                                    stations_data[f"{name} (Current)"] = current_df
                            except Exception:
                                pass

                        for label in selected_labels:
                            s_uuid, s_name = station_options[label]
                            if s_uuid and s_uuid != station_uuid:
                                try:
                                    comp_df = get_station_price_history(s_uuid, fuel_code, days=14)
                                    if comp_df is not None and not comp_df.empty:
                                        stations_data[s_name] = comp_df
                                except Exception as e:
                                    if debug_mode:
                                        st.warning(f"Could not load data for {s_name}: {e}")

                        if len(stations_data) >= 2:
                            fig_comparison = create_comparison_chart(
                                stations_data=stations_data,
                                fuel_type=fuel_code,
                                current_station_name=name,
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)

                            st.markdown("---")
                            st.markdown("### Quick Insights")

                            current_avg = None
                            if f"{name} (Current)" in stations_data:
                                current_df = stations_data[f"{name} (Current)"]
                                if not current_df.empty:
                                    current_avg = current_df["price"].mean()

                            if current_avg:
                                cheaper_count = 0
                                total_compared = 0
                                avg_diff = 0.0

                                for s_name, s_df in stations_data.items():
                                    if s_name == f"{name} (Current)" or s_df.empty:
                                        continue
                                    total_compared += 1
                                    alt_avg = s_df["price"].mean()
                                    if current_avg < alt_avg:
                                        cheaper_count += 1
                                        avg_diff += (alt_avg - current_avg)

                                if total_compared > 0:
                                    avg_diff = (avg_diff / total_compared) if cheaper_count > 0 else 0.0
                                    savings_40L = avg_diff * 40

                                    full_station_id = _safe_text(station.get("tk_name") or station.get("osm_name") or name)
                                    if station.get("brand"):
                                        full_station_id += f" ({_safe_text(station.get('brand'))})"

                                    if cheaper_count == total_compared and total_compared > 0:
                                        st.success(f"{full_station_id} is consistently cheaper.")
                                        if savings_40L >= 2.00:
                                            st.success("Worth it: meaningful savings on a typical 40L refuel.")
                                        elif savings_40L >= 0.50:
                                            st.info("Marginal savings: consider convenience as well.")
                                        else:
                                            st.warning("Very small savings: convenience likely dominates.")
                                    elif cheaper_count > 0:
                                        st.info(f"{full_station_id} is cheaper than {cheaper_count} out of {total_compared} alternatives.")
                                    else:
                                        st.warning(f"{full_station_id} is more expensive than the selected alternatives.")

                            if debug_mode:
                                st.markdown("---")
                                st.markdown("DEBUG: Detailed Statistics")
                                summary_rows = []
                                for station_name_cmp, df_cmp in stations_data.items():
                                    if not df_cmp.empty:
                                        summary_rows.append({
                                            "Station": station_name_cmp,
                                            "Avg Price": f"€{df_cmp['price'].mean():.3f}",
                                            "Min Price": f"€{df_cmp['price'].min():.3f}",
                                            "Max Price": f"€{df_cmp['price'].max():.3f}",
                                            "Volatility": f"€{df_cmp['price'].std():.4f}",
                                        })
                                if summary_rows:
                                    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
                        elif len(stations_data) == 1:
                            st.info("Only one station has historical data. Select more stations or wait for data to accumulate.")
                        else:
                            st.warning("No historical data available for selected stations.")
                    except Exception as e:
                        if debug_mode:
                            st.error(f"Error loading comparison data: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                        else:
                            st.warning("Could not load comparison data. Enable debug mode for details.")
            else:
                st.info("Select at least one alternative station to see the comparison.")

    # ============================================================
    # ECONOMICS TAB
    # ============================================================
    with tab_economics:
        st.markdown("### Detailed Economic Breakdown")
        st.caption("For users who want to understand the exact calculations.")

        econ_net_key = f"econ_net_saving_eur_{fuel_code}"
        econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
        econ_detour_fuel_key = f"econ_detour_fuel_l_{fuel_code}"
        econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
        econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
        econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
        econ_baseline_key = f"econ_baseline_price_{fuel_code}"

        if econ_net_key not in station:
            if not use_economics:
                st.info("""
                Economic analysis is not available because economics metrics were not calculated on the route run.

                Enable economics on Trip Planner, run again, then revisit this page.
                """)
            else:
                st.warning("Economic metrics are not available for this station.")
        else:
            detour_fuel_val = station.get(econ_detour_fuel_key)
            detour_fuel_formatted = f"{float(detour_fuel_val):.2f} L" if detour_fuel_val is not None else "—"

            breakeven_val = station.get(econ_breakeven_key)
            if breakeven_val is not None and breakeven_val > 0:
                breakeven_formatted = f"{float(breakeven_val):.1f} L"
                breakeven_explanation = f"Refuel ≥{float(breakeven_val):.1f}L to make detour worthwhile"
            else:
                breakeven_formatted = "—"
                breakeven_explanation = "Detour not economical at typical refuel amounts"

            econ_rows = [
                {"Metric": "Baseline on-route price", "Value": _fmt_price(station.get(econ_baseline_key)), "Note": "Cheapest station without detouring"},
                {"Metric": "Gross saving", "Value": _fmt_eur(station.get(econ_gross_key)), "Note": "Price advantage before costs"},
                {"Metric": "Detour fuel consumption", "Value": detour_fuel_formatted, "Note": "Extra fuel for the detour"},
                {"Metric": "Detour fuel cost", "Value": _fmt_eur(station.get(econ_detour_fuel_cost_key)), "Note": "Cost of the extra fuel"},
                {"Metric": "Time cost", "Value": _fmt_eur(station.get(econ_time_cost_key)), "Note": "Value of extra travel time"},
                {"Metric": "Net saving", "Value": _fmt_eur(station.get(econ_net_key)), "Note": "Gross saving minus all costs"},
                {"Metric": "Break-even amount", "Value": breakeven_formatted, "Note": breakeven_explanation},
            ]
            st.dataframe(pd.DataFrame(econ_rows), hide_index=True, use_container_width=True)

            if litres_to_refuel is not None:
                st.caption(f"Economics computed assuming refuel amount: {float(litres_to_refuel):.1f} L")

            with st.expander("How to read these numbers"):
                st.markdown("""
                - Baseline on-route price: cheapest option without detour
                - Gross saving: price advantage for your refuel amount
                - Detour costs: extra fuel + time
                - Net saving: gross saving minus detour costs
                - Break-even: minimum liters needed for positive net saving
                """)

    st.markdown("---")
    st.caption("Use Route Analytics to change selection quickly, or return to Trip Planner to run a new route.")


if __name__ == "__main__":
    main()
