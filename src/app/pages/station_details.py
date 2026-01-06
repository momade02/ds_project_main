"""
Station Details Page

Detailed analysis for selected fuel station with all charts and comparisons.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import historical data functions for charts
from src.integration.historical_data import (
    get_station_price_history,
    calculate_hourly_price_stats,
    get_cheapest_and_most_expensive_hours,
)

from src.decision.recommender import (
    ONROUTE_MAX_DETOUR_KM,
    ONROUTE_MAX_DETOUR_MIN,
)

st.set_page_config(page_title="Station Details", layout="wide")

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def _station_uuid(station: Dict[str, Any]) -> Optional[str]:
    return station.get("tk_uuid") or station.get("station_uuid")

def _fmt_price(value: Any) -> str:
    try:
        return f"{float(value):.3f} €/L" if value is not None else "—"
    except:
        return "—"

def _fmt_eur(value: Any) -> str:
    try:
        return f"{float(value):.2f} €" if value is not None else "—"
    except:
        return "—"

def _fmt_km(value: Any) -> str:
    try:
        return f"{float(value):.2f} km" if value is not None else "—"
    except:
        return "—"

def _fmt_min(value: Any) -> str:
    try:
        return f"{float(value):.1f} min" if value is not None else "—"
    except:
        return "—"

def _fmt_liters(value: Any) -> str:
    try:
        return f"{float(value):.2f} L" if value is not None else "—"
    except:
        return "—"

def _describe_price_basis(station: Dict[str, Any], fuel_code: str) -> str:
    """Explain how the price was determined."""
    used_current = bool(station.get(f"debug_{fuel_code}_used_current_price"))
    horizon = station.get(f"debug_{fuel_code}_horizon_used")

    if used_current:
        return "Current price (arrival in ≤ 10 min)"

    if horizon is None:
        return "No forecast available (fallback)"

    try:
        h_int = int(horizon)
    except:
        h_int = None

    if h_int is None or h_int < 0:
        return "Forecast (ARDL model)"

    approx_min = h_int * 30
    if approx_min == 0:
        return "Forecast (same block, daily lags only)"
    else:
        return f"Forecast (~{approx_min} min ahead, horizon {h_int})"


# ===========================================================================
# PLOTLY CHART FUNCTIONS
# ===========================================================================

def create_price_trend_chart(
    df: pd.DataFrame,
    fuel_type: str,
    station_name: str
) -> go.Figure:
    """
    Create simple, clean 14-day price trend chart.
    
    SIMPLIFIED: Just line chart, no markers, no clutter.
    User-focused for quick insights.
    """
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=f"{fuel_type.upper()} Price History - {station_name}",
            height=400
        )
        return fig
    
    # Create simple line chart
    fig = go.Figure()
    
    # Add clean price line (no markers!)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',  # Just lines, no markers
        name='Price',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='%{x|%b %d, %H:%M}<br>€%{y:.3f}/L<extra></extra>'
    ))
    
    # Add today marker (subtle)
    today = pd.Timestamp.now()
    if df['date'].min() <= today <= df['date'].max():
        fig.add_vline(
            x=today,
            line_dash="dash",
            line_color="rgba(128, 128, 128, 0.3)",
            annotation_text="Today",
            annotation_position="top",
            annotation_font=dict(size=10, color="gray")
        )
    
    # Calculate statistics for caption only
    avg_price = df['price'].mean()
    min_price = df['price'].min()
    max_price = df['price'].max()
    
    # Update layout - clean and simple!
    fig.update_layout(
        title=f"📈 {fuel_type.upper()} Price History (Last 14 Days)",
        xaxis_title="Date",
        yaxis_title="Price (€/L)",
        hovermode='x unified',
        height=400,
        showlegend=False,  # No legend needed for single line
        margin=dict(b=80)  # Space for caption
    )
    
    # Add simple statistics caption
    fig.add_annotation(
        text=f"Average: €{avg_price:.3f} | Min: €{min_price:.3f} | Max: €{max_price:.3f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="center"
    )
    
    return fig


def create_hourly_pattern_chart(
    hourly_df: pd.DataFrame,
    fuel_type: str,
    station_name: str
) -> go.Figure:
    """
    Create hourly price pattern bar chart.
    
    Parameters
    ----------
    hourly_df : pd.DataFrame
        From calculate_hourly_price_stats() with columns ['hour', 'avg_price', ...]
    fuel_type : str
        'e5', 'e10', or 'diesel'
    station_name : str
        Station name for title
    
    Returns
    -------
    plotly.graph_objects.Figure
        Bar chart showing optimal times
    """
    if hourly_df.empty or hourly_df['avg_price'].isna().all():
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for hourly pattern",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=f"🕐 Best Time to Refuel - {station_name}",
            height=400
        )
        return fig
    
    # Get optimal times
    optimal = get_cheapest_and_most_expensive_hours(hourly_df)
    cheapest_hour = optimal['cheapest_hour']
    most_expensive_hour = optimal['most_expensive_hour']
    
    # Create color scale (green for cheap, red for expensive)
    colors = []
    for idx, row in hourly_df.iterrows():
        if pd.isna(row['avg_price']):
            colors.append('lightgray')
        elif row['hour'] == cheapest_hour:
            colors.append('green')
        elif row['hour'] == most_expensive_hour:
            colors.append('red')
        else:
            # Gradient from green to orange to red
            price_range = hourly_df['avg_price'].max() - hourly_df['avg_price'].min()
            if price_range > 0:
                normalized = (row['avg_price'] - hourly_df['avg_price'].min()) / price_range
                colors.append(f'rgb({int(255*normalized)}, {int(200*(1-normalized))}, 0)')
            else:
                colors.append('orange')
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_df['hour'],
        y=hourly_df['avg_price'],
        marker_color=colors,
        text=hourly_df['avg_price'].apply(lambda x: f'€{x:.3f}' if pd.notna(x) else '—'),
        textposition='outside',
        hovertemplate='Hour %{x}:00<br>€%{y:.3f}/L<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"🕐 Best Time to Refuel ({fuel_type.upper()}) - {station_name}",
        xaxis_title="Hour of Day",
        yaxis_title="Average Price (€/L)",
        height=450,
        showlegend=False,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2,  # Show every 2 hours
            tickformat='%H:00'
        )
    )
    
    return fig


def create_comparison_chart(
    stations_data: Dict[str, pd.DataFrame],
    fuel_type: str,
    current_station_name: str = None
) -> go.Figure:
    """
    Create simplified comparison chart: Current station vs top alternatives.
    
    PURPOSE: Show if recommended station is consistently cheaper or just a one-time deal.
    Max 3 stations for readability.
    """
    if not stations_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No stations to compare",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=f"Station Comparison ({fuel_type.upper()})",
            height=400
        )
        return fig
    
    fig = go.Figure()
    
    # Highlight current station, fade others
    for idx, (station_name, df) in enumerate(stations_data.items()):
        if df.empty:
            continue
        
        # Check if this is the current/selected station
        is_current = (current_station_name and station_name.startswith(current_station_name)) or "(Current)" in station_name
        
        if is_current:
            # Current station: BOLD, bright color
            color = '#00C800'  # Green
            width = 3
            opacity = 1.0
            dash = 'solid'
        else:
            # Other stations: Thinner, faded
            colors_palette = ['#FF6B6B', '#4ECDC4', '#FFD93D']  # Red, Teal, Yellow
            color = colors_palette[idx % len(colors_palette)]
            width = 2
            opacity = 0.6
            dash = 'dot'
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['price'],
            mode='lines',
            name=station_name.replace(" (Current)", ""),  # Clean name
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            hovertemplate=f'{station_name}<br>%{{x|%b %d}}<br>€%{{y:.3f}}/L<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"📊 Is This Station Consistently Cheaper?",
        xaxis_title="Date",
        yaxis_title="Price (€/L)",
        hovermode='x unified',
        height=450,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Add helpful caption
    fig.add_annotation(
        text="Solid line = Your selection | Dotted lines = Alternatives for comparison",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=10, color="gray"),
        xanchor="center"
    )
    
    return fig


def _create_map_visualization(
    route_coords: List[List[float]],
    stations: List[Dict[str, Any]],
    best_station_uuid: Optional[str] = None,
    via_full_coords: Optional[List[List[float]]] = None,
    zoom_level: float = 7.5,
    *,
    fuel_code: Optional[str] = None,
    selected_station_uuid: Optional[str] = None,
    map_style: Optional[str] = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    show_station_labels: bool = True,
) -> pdk.Deck:
    """
    Create a pydeck map showing:
      - baseline route (and optional via-station route),
      - start/end markers,
      - all stations (best station highlighted),
      - hover tooltip (compact),
      - selection support via Streamlit's pydeck selection API.

    Notes
    -----
    - Click-selection is handled by Streamlit (st.pydeck_chart on_select=...).
      This function provides a stable layer id ("stations") and station UUIDs.
    """

    # ------------------------------------------------------------------
    # Pin icon (SVG data URL) for IconLayer
    # NOTE: Must be URL-encoded, otherwise deck.gl often fails to load it,
    #       resulting in large placeholder circles.
    # ------------------------------------------------------------------
    def _pin_icon_data_url(fill_hex: str, stroke_hex: str) -> str:
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'>"
            f"<path d='M32 2C21 2 12 11 12 22c0 17 20 40 20 40s20-23 20-40C52 11 43 2 32 2z' "
            f"fill='{fill_hex}' stroke='{stroke_hex}' stroke-width='2'/>"
            "<circle cx='32' cy='22' r='7' fill='white'/>"
            "</svg>"
        )
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"

    # ------------------------------------------------------------------


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    st.title("📊 Station Details & Analysis")

    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Main"):
            st.switch_page("streamlit_app.py")

    # Check session state
    if "selected_station_data" not in st.session_state:
        st.error("⚠️ No station selected. Please go back and select a station.")
        if st.button("Go to Main Page"):
            st.switch_page("streamlit_app.py")
        st.stop()

    # Load data
    station = st.session_state["selected_station_data"]
    cached = st.session_state.get("last_run", {})
    fuel_code = cached.get("fuel_code", "e5")
    litres_to_refuel = cached.get("litres_to_refuel", 40.0)
    ranked = cached.get("ranked", [])
    debug_mode = cached.get("debug_mode", False)  # Get debug mode from session state
    
    # Get params to check if economics was used
    params = cached.get("params", {})
    use_economics = params.get("use_economics", True)

    # Station header
    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"
    
    name = station.get("tk_name") or station.get("osm_name") or "Unknown"
    brand = station.get("brand") or ""

    st.markdown("### Station Details")
    header = f"**{name}**"
    if brand:
        header += f"  \n{brand}"
    st.markdown(header)

    station_uuid = _station_uuid(station)
    if station_uuid:
        st.caption(f"Station UUID: {station_uuid}")

    # Create tabs
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

        # Savings summary
        st.markdown("---")
        st.markdown("### 💰 Savings Summary")

        econ_net_key = f"econ_net_saving_eur_{fuel_code}"
        econ_baseline_key = f"econ_baseline_price_{fuel_code}"
        econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
        econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
        econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
        econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"

        if econ_net_key in station:
            net_saving = station.get(econ_net_key)
            gross_saving = station.get(econ_gross_key)

            if net_saving and net_saving > 0:
                st.success("✅ **This detour is worth it!**")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Net Saving", f"€{net_saving:.2f}", delta="You save money!")
                with col_b:
                    st.metric("Price Saving", _fmt_eur(gross_saving))
                with col_c:
                    total_cost = (station.get(econ_detour_fuel_cost_key) or 0) + (station.get(econ_time_cost_key) or 0)
                    st.metric("Detour Cost", f"€{total_cost:.2f}")

                with st.expander("ℹ️ How is this calculated?"):
                    refuel_amount = litres_to_refuel if litres_to_refuel else 40.0
                    st.markdown(f"""
                    **For {refuel_amount:.0f}L refuel:**
                    - Price advantage: €{gross_saving:.2f}
                    - Detour fuel cost: €{station.get(econ_detour_fuel_cost_key, 0):.2f}
                    - Time cost: €{station.get(econ_time_cost_key, 0):.2f}
                    - **Net saving: €{net_saving:.2f}**
                    
                    Baseline on-route price: {_fmt_price(station.get(econ_baseline_key))}
                    """)
            else:
                st.warning("⚠️ **All options require detours**")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Net Cost", f"-€{abs(net_saving):.2f}" if net_saving else "—")
                with col_b:
                    baseline_price = station.get(econ_baseline_key)
                    if baseline_price:
                        st.metric("Baseline (cheapest on-route)", _fmt_price(baseline_price))

                st.info("💡 **This is the best we found, but costs €{:.2f}.**\n\n**Better option:** Choose the cheapest on-route station instead.".format(abs(net_saving) if net_saving else 0))

                with st.expander("ℹ️ Why does this cost money?"):
                    detour_cost = (station.get(econ_detour_fuel_cost_key, 0) + station.get(econ_time_cost_key, 0))
                    refuel_amount = litres_to_refuel if litres_to_refuel else 40.0
                    
                    st.markdown(f"""
                    **For {refuel_amount:.0f}L refuel:**
                    - Price advantage: €{gross_saving:.2f}
                    - Detour cost: €{detour_cost:.2f}
                    - **Net result: -€{abs(net_saving):.2f}**
                    
                    Even though this is our #1 recommendation, the small detour still costs more than the price advantage.
                    """)

            breakeven = station.get(econ_breakeven_key)
            if breakeven is not None and breakeven > 0:
                st.caption(f"💡 Break-even point: You need to refuel at least **{breakeven:.1f} L** for this detour to be worthwhile.")

        else:
            if not use_economics:
                st.info("""
                📊 **Economic metrics not calculated**
                
                You unchecked "Use economics-based detour decision" in the main app to see all stations.
                
                **To see economic analysis:**
                1. Go back to the main page
                2. Check ☑ "Use economics-based detour decision" in the sidebar
                3. Click "Run Recommender" again
                4. Return to this station's details
                """)
            else:
                st.warning("Economic metrics not available for this station.")


    # ============================================================
    # PRICES TAB (with charts from working version)
    # ============================================================
    with tab_prices:
        # ============================================================
        # Price History Charts (User-Focused)
        # ============================================================
        if station_uuid:
            with st.spinner("Loading price history..."):
                try:
                    # Fetch price history
                    df_history = get_station_price_history(
                        station_uuid=station_uuid,
                        fuel_type=fuel_code,
                        days=14
                    )
                    
                    if df_history is not None and not df_history.empty:
                        # 1. Price Trend Chart (14 days) - SIMPLIFIED!
                        fig_trend = create_price_trend_chart(
                            df=df_history,
                            fuel_type=fuel_code,
                            station_name=name
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                        
                        # 2. Hourly Pattern Chart - User likes this!
                        st.markdown("---")
                        st.markdown("### 🕐 Best Time to Refuel")
                        st.caption("Based on historical patterns, when is fuel cheapest at this station?")
                        
                        # Calculate hourly stats
                        hourly_stats = calculate_hourly_price_stats(df_history)
                        
                        if hourly_stats is not None and not hourly_stats.empty and hourly_stats['avg_price'].dropna().any():
                            fig_hourly = create_hourly_pattern_chart(
                                hourly_df=hourly_stats,
                                fuel_type=fuel_code,
                                station_name=name
                            )
                            st.plotly_chart(fig_hourly, use_container_width=True)
                            
                            # Show optimal times summary
                            optimal = get_cheapest_and_most_expensive_hours(hourly_stats)
                            if optimal['cheapest_hour'] is not None:
                                col_best, col_worst, col_savings = st.columns(3)
                                
                                with col_best:
                                    st.metric(
                                        "💚 Cheapest Hour",
                                        f"{optimal['cheapest_hour']:02d}:00",
                                        f"€{optimal['cheapest_price']:.3f}/L"
                                    )
                                
                                with col_worst:
                                    st.metric(
                                        "💸 Most Expensive Hour",
                                        f"{optimal['most_expensive_hour']:02d}:00",
                                        f"€{optimal['most_expensive_price']:.3f}/L",
                                        delta_color="inverse"  # FIXED: Red arrow down (expensive = bad)
                                    )
                                
                                with col_savings:
                                    savings_per_liter = optimal['most_expensive_price'] - optimal['cheapest_price']
                                    if litres_to_refuel:
                                        total_savings = savings_per_liter * litres_to_refuel
                                        st.metric(
                                            "💰 Potential Savings",
                                            f"€{total_savings:.2f}",
                                            f"€{savings_per_liter:.3f}/L"
                                        )
                                    else:
                                        st.metric(
                                            "💰 Price Difference",
                                            f"€{savings_per_liter:.3f}/L",
                                            "Best vs Worst time"
                                        )
                        else:
                            st.info("Not enough data to calculate hourly patterns. Prices may only change a few times per day.")
                    
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
        
        # ============================================================
        # DEBUG ONLY: Price Lag Table
        # ============================================================
        if debug_mode:
            st.markdown("---")
            st.markdown("### 🔧 DEBUG: Price Lag Data")
            rows = [
                {"Metric": "Current price", "Value": _fmt_price(station.get(current_key))},
                {"Metric": "Predicted price", "Value": _fmt_price(station.get(pred_key))},
            ]
            for lag in ("1d", "2d", "3d", "7d"):
                rows.append(
                    {
                        "Metric": f"Price lag {lag}",
                        "Value": _fmt_price(station.get(f"price_lag_{lag}_{fuel_code}")),
                    }
                )
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)

            debug_h_key = f"debug_{fuel_code}_horizon_used"
            debug_ucp_key = f"debug_{fuel_code}_used_current_price"
            if debug_h_key in station or debug_ucp_key in station:
                st.markdown("#### Forecast basis")
                st.write(
                    {
                        "used_current_price": station.get(debug_ucp_key),
                        "horizon_used": station.get(debug_h_key),
                    }
                )

    with tab_comparison:
        st.markdown("### 📊 Is This Station Consistently Cheaper?")
        st.caption("Compare your selected station with top alternatives to see if it's consistently the best choice, or just a one-time deal.")
        
        # Get the ranked stations from session state
        ranked_stations = st.session_state.get("last_run", {}).get("ranked", [])
        
        if not ranked_stations or len(ranked_stations) < 2:
            st.info("Need at least 2 stations to show comparison. Run the recommender first.")
        else:
            # Let user select stations to compare (max 3 total including current)
            station_options = {}
            for i, s in enumerate(ranked_stations[:10]):  # Top 10 stations
                s_uuid = _station_uuid(s)
                s_name = s.get("tk_name") or s.get("osm_name") or f"Station {i+1}"
                s_brand = s.get("brand") or ""
                label = f"#{i+1}: {s_name}" + (f" ({s_brand})" if s_brand else "")
                station_options[label] = (s_uuid, s_name)
            
            # Multi-select for comparison stations (max 2, so 3 total with current)
            st.markdown("**Select up to 2 alternatives to compare:**")
            selected_labels = st.multiselect(
                "Choose stations",
                options=list(station_options.keys()),
                default=list(station_options.keys())[:min(2, len(station_options))],  # Default: top 2
                max_selections=2,  # Changed from 4 to 2!
                key="comparison_stations"
            )
            
            if selected_labels:
                with st.spinner("Loading comparison data..."):
                    try:
                        # Fetch data for all selected stations
                        stations_data = {}
                        
                        # Always include current station first
                        if station_uuid:
                            try:
                                current_df = get_station_price_history(station_uuid, fuel_code, days=14)
                                if current_df is not None and not current_df.empty:
                                    stations_data[f"{name} (Current)"] = current_df
                            except:
                                pass
                        
                        # Add selected stations
                        for label in selected_labels:
                            s_uuid, s_name = station_options[label]
                            if s_uuid and s_uuid != station_uuid:  # Don't duplicate current station
                                try:
                                    comp_df = get_station_price_history(s_uuid, fuel_code, days=14)
                                    if comp_df is not None and not comp_df.empty:
                                        stations_data[s_name] = comp_df
                                except Exception as e:
                                    if debug_mode:
                                        st.warning(f"Could not load data for {s_name}: {e}")
                        
                        if len(stations_data) >= 2:
                            # Create comparison chart with current station highlighted
                            fig_comparison = create_comparison_chart(
                                stations_data=stations_data,
                                fuel_type=fuel_code,
                                current_station_name=name  # Pass current station name for highlighting
                            )
                            st.plotly_chart(fig_comparison, use_container_width=True)
                            
                            # ============================================
                            # Actionable Insights (NOT just raw stats!)
                            # ============================================
                            st.markdown("---")
                            st.markdown("### 💡 Quick Insights")
                            
                            # Calculate insights
                            current_avg = None
                            if f"{name} (Current)" in stations_data:
                                current_df = stations_data[f"{name} (Current)"]
                                if not current_df.empty:
                                    current_avg = current_df['price'].mean()
                            
                            if current_avg:
                                # Compare with alternatives
                                cheaper_count = 0
                                total_compared = 0
                                avg_diff = 0
                                
                                for s_name, s_df in stations_data.items():
                                    if s_name == f"{name} (Current)" or s_df.empty:
                                        continue
                                    total_compared += 1
                                    alt_avg = s_df['price'].mean()
                                    if current_avg < alt_avg:
                                        cheaper_count += 1
                                        avg_diff += (alt_avg - current_avg)
                                
                                if total_compared > 0:
                                    avg_diff = avg_diff / total_compared if cheaper_count > 0 else 0
                                    
                                    # Calculate total savings at different refuel amounts
                                    refuel_amounts = [20, 40, 60, 80]
                                    savings_examples = [f"€{avg_diff * amt:.2f} on {amt}L" for amt in refuel_amounts]
                                    
                                    # Calculate savings on typical 40L tank for threshold
                                    savings_40L = avg_diff * 40
                                    
                                    # Get full station info for display
                                    station_display_name = station.get("tk_name") or station.get("osm_name") or name
                                    station_brand = station.get("brand")
                                    full_station_id = f"{station_display_name}"
                                    if station_brand:
                                        full_station_id += f" ({station_brand})"
                                    
                                    if cheaper_count == total_compared:
                                        st.success(f"✅ **{full_station_id}** is consistently cheaper!")
                                        st.markdown(f"""
                                        **Average savings:** €{avg_diff:.3f}/L
                                        
                                        **Total savings:**
                                        - {savings_examples[0]} tank
                                        - {savings_examples[1]} tank ← Typical
                                        - {savings_examples[2]} tank
                                        - {savings_examples[3]} tank
                                        """)
                                        
                                        # FIXED: Better thresholds based on 40L tank
                                        if savings_40L >= 2.00:
                                            st.success("💡 **Worth it?** Yes, meaningful savings!")
                                        elif savings_40L >= 0.50:
                                            st.info("💡 **Worth it?** Small savings - consider convenience too.")
                                        else:
                                            st.warning("💡 **Worth it?** Too small to matter - choose based on convenience.")
                                            
                                    elif cheaper_count > 0:
                                        st.info(f"ℹ️ **{full_station_id}** is cheaper than {cheaper_count} out of {total_compared} alternatives.")
                                        st.markdown(f"""
                                        **Average savings:** €{avg_diff:.3f}/L vs alternatives
                                        
                                        **Total savings:**
                                        - {savings_examples[0]} tank
                                        - {savings_examples[1]} tank ← Typical
                                        """)
                                        
                                        # FIXED: Better thresholds
                                        if savings_40L >= 2.00:
                                            st.success("💡 **Worth it?** Yes, decent savings!")
                                        elif savings_40L >= 0.50:
                                            st.info("💡 **Worth it?** Marginal - may not justify extra effort.")
                                        else:
                                            st.warning("💡 **Worth it?** Too small - choose based on convenience.")
                                    else:
                                        # Find cheapest alternative name
                                        cheapest_alt = None
                                        cheapest_avg = float('inf')
                                        for s_name, s_df in stations_data.items():
                                            if s_name != f"{name} (Current)" and not s_df.empty:
                                                alt_avg = s_df['price'].mean()
                                                if alt_avg < cheapest_avg:
                                                    cheapest_avg = alt_avg
                                                    cheapest_alt = s_name
                                        
                                        st.warning(f"⚠️ **{full_station_id}** is more expensive than all selected alternatives.")
                                        if cheapest_alt:
                                            savings_at_cheapest = (current_avg - cheapest_avg) * 40  # 40L example
                                            st.markdown(f"""
                                            **Better option:** {cheapest_alt} (€{cheapest_avg:.3f}/L avg)
                                            
                                            **You'd save:** €{abs(savings_at_cheapest):.2f} on a 40L tank by choosing the cheapest alternative instead.
                                            
                                            """)
                            
                            # DEBUG ONLY: Show detailed stats table
                            if debug_mode:
                                st.markdown("---")
                                st.markdown("#### 🔧 DEBUG: Detailed Statistics")
                                summary_rows = []
                                for station_name_cmp, df_cmp in stations_data.items():
                                    if not df_cmp.empty:
                                        summary_rows.append({
                                            "Station": station_name_cmp,
                                            "Avg Price": f"€{df_cmp['price'].mean():.3f}",
                                            "Min Price": f"€{df_cmp['price'].min():.3f}",
                                            "Max Price": f"€{df_cmp['price'].max():.3f}",
                                            "Volatility": f"€{df_cmp['price'].std():.4f}"
                                        })
                                
                                if summary_rows:
                                    summary_df = pd.DataFrame(summary_rows)
                                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
                                    st.caption("Volatility = Standard deviation of prices (lower = more stable)")
                        
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
        st.markdown("### 💰 Detailed Economic Breakdown")
        st.caption("For power users who want to understand the exact calculations")

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
                📊 **Economic analysis not available**
                
                Economic metrics were not calculated because you unchecked 
                "Use economics-based detour decision" in the main app.
                
                **To enable economic analysis:**
                1. Return to the main page (click "← Back to Main")
                2. In the left sidebar, check ☑ "Use economics-based detour decision"
                3. Click "Run Recommender"
                4. Come back to view this station's economic breakdown
                
                **What you'll see when enabled:**
                - Net saving/cost of detouring to this station
                - Breakdown of price savings vs detour costs
                - Break-even refuel amount
                - Comparison to on-route alternatives
                """)
            else:
                st.warning("Economic metrics not available for this station.")
        else:
            # Format detour fuel with unit
            detour_fuel_val = station.get(econ_detour_fuel_key)
            detour_fuel_formatted = f"{float(detour_fuel_val):.2f} L" if detour_fuel_val is not None else "—"
            
            # Format break-even with explanation
            breakeven_val = station.get(econ_breakeven_key)
            if breakeven_val is not None and breakeven_val > 0:
                breakeven_formatted = f"{float(breakeven_val):.1f} L"
                breakeven_explanation = f"Refuel ≥{float(breakeven_val):.1f}L to make detour worthwhile"
            else:
                breakeven_formatted = "—"
                breakeven_explanation = "Detour not economical at any refuel amount"
            
            econ_rows = [
                {"Metric": "Baseline on-route price", "Value": _fmt_price(station.get(econ_baseline_key)), "Note": "Cheapest station if you don't detour"},
                {"Metric": "Gross saving", "Value": _fmt_eur(station.get(econ_gross_key)), "Note": "Price advantage before costs"},
                {"Metric": "Detour fuel consumption", "Value": detour_fuel_formatted, "Note": "Extra fuel for detour"},
                {"Metric": "Detour fuel cost", "Value": _fmt_eur(station.get(econ_detour_fuel_cost_key)), "Note": "Cost of extra fuel"},
                {"Metric": "Time cost", "Value": _fmt_eur(station.get(econ_time_cost_key)), "Note": "Value of extra travel time"},
                {"Metric": "Net saving", "Value": _fmt_eur(station.get(econ_net_key)), "Note": "Gross saving - all costs"},
                {"Metric": "Break-even amount", "Value": breakeven_formatted, "Note": breakeven_explanation},
            ]
            st.dataframe(pd.DataFrame(econ_rows), hide_index=True, use_container_width=True)

            if litres_to_refuel is not None:
                st.caption(f"ℹ️ Economics computed assuming refuel amount: {litres_to_refuel:.1f} L")
            
            # Explanation expander
            with st.expander("ℹ️ How to read these numbers"):
                st.markdown("""
                **Understanding the Economics:**
                
                1. **Baseline on-route price:** The cheapest station you could use WITHOUT detouring
                2. **Gross saving:** How much you save on fuel price (per liter × refuel amount)
                3. **Detour fuel consumption:** Extra fuel burned during the detour
                4. **Detour fuel cost:** Cost of that extra fuel
                5. **Time cost:** Monetary value of extra travel time (based on your value of time)
                6. **Net saving:** Final benefit = Gross saving - Detour fuel cost - Time cost
                7. **Break-even amount:** Minimum liters to refuel for the detour to be worthwhile
                
                **Example:**
                - If Net saving = €2.40 → **Worth it!** You save money overall
                - If Net saving = -€0.68 → **Not worth it!** Detour costs more than you save
                - If Break-even = 25L → You need to refuel at least 25 liters for positive net saving
                """)

    st.markdown("---")
    st.caption("💡 Click '← Back to Main' to select a different station")


if __name__ == "__main__":
    main()