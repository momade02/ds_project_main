"""
Page: Station Details & Analysis.

Description:
    This Streamlit page provides a deep-dive analysis of a specific fuel station.
    It is triggered when a user selects a station from the main dashboard.

    Key Features:
    1. Overview: Immediate metrics (Current Price vs. Predicted) and Savings Summary.
    2. Prices: Historical trend charts (14-day) and Hour-of-Day optimization heatmaps.
    3. Comparison: Benchmarking the selected station against other top candidates.
    4. Economics: A transparent breakdown of the Cost-Benefit Analysis (Net Saving).

    Data Sources:
    - Session State (`selected_station_data`): Passed from `streamlit_app.py`.
    - Supabase (`historical_data.py`): Fetched on-demand for charts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Final

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Path Setup ---
# Add project root to sys.path to ensure absolute imports work
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Internal Imports ---
from src.integration.historical_data import (
    get_station_price_history,
    calculate_hourly_price_stats,
    get_cheapest_and_most_expensive_hours,
)

# --- Configuration ---
st.set_page_config(page_title="Station Details", layout="wide")


# ==========================================
# Data Structures & Helpers
# ==========================================

@dataclass(frozen=True)
class EconKeys:
    """Helper to generate fuel-specific dictionary keys for economic metrics."""
    fuel_code: str

    @property
    def pred_price(self) -> str: return f"pred_price_{self.fuel_code}"
    @property
    def current_price(self) -> str: return f"price_current_{self.fuel_code}"
    @property
    def net_saving(self) -> str: return f"econ_net_saving_eur_{self.fuel_code}"
    @property
    def gross_saving(self) -> str: return f"econ_gross_saving_eur_{self.fuel_code}"
    @property
    def baseline_price(self) -> str: return f"econ_baseline_price_{self.fuel_code}"
    @property
    def detour_cost(self) -> str: return f"econ_detour_fuel_cost_eur_{self.fuel_code}"
    @property
    def detour_fuel(self) -> str: return f"econ_detour_fuel_l_{self.fuel_code}"
    @property
    def time_cost(self) -> str: return f"econ_time_cost_eur_{self.fuel_code}"
    @property
    def breakeven(self) -> str: return f"econ_breakeven_liters_{self.fuel_code}"
    @property
    def lag_1d(self) -> str: return f"price_lag_1d_{self.fuel_code}"


def _station_uuid(station: Dict[str, Any]) -> Optional[str]:
    """Robustly extract station UUID from varying schema versions."""
    return station.get("tk_uuid") or station.get("station_uuid")


def _safe_float(value: Any) -> Optional[float]:
    """Safely converts value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _fmt_val(value: Any, fmt: str, suffix: str = "") -> str:
    """Generic formatter avoiding try/except overhead."""
    f_val = _safe_float(value)
    if f_val is None:
        return "—"
    return f"{f_val:{fmt}} {suffix}".strip()


# Specialized formatters for readability
def _fmt_price(v: Any) -> str: return _fmt_val(v, ".3f", "€/L")
def _fmt_eur(v: Any) -> str:   return _fmt_val(v, ".2f", "€")
def _fmt_km(v: Any) -> str:    return _fmt_val(v, ".2f", "km")
def _fmt_min(v: Any) -> str:   return _fmt_val(v, ".1f", "min")
def _fmt_liters(v: Any) -> str: return _fmt_val(v, ".2f", "L")


# ==========================================
# Visualization Components (Plotly)
# ==========================================

def create_price_trend_chart(
    df: pd.DataFrame, fuel_type: str, station_name: str
) -> go.Figure:
    """Renders a clean 14-day price history line chart."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No historical data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title=f"{fuel_type.upper()} Price History", height=400)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['price'], mode='lines', name='Price',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='%{x|%b %d, %H:%M}<br>€%{y:.3f}/L<extra></extra>'
    ))

    # Add "Today" marker
    today = pd.Timestamp.now()
    if df['date'].min() <= today <= df['date'].max():
        fig.add_vline(
            x=today, line_dash="dash", line_color="rgba(128,128,128,0.3)",
            annotation_text="Today", annotation_position="top"
        )

    # Simple footer stats
    stats_text = (
        f"Average: €{df['price'].mean():.3f} | "
        f"Min: €{df['price'].min():.3f} | "
        f"Max: €{df['price'].max():.3f}"
    )
    
    fig.update_layout(
        title=f"📈 {fuel_type.upper()} Price History (14 Days)",
        xaxis_title="Date", yaxis_title="Price (€/L)",
        hovermode='x unified', height=400, showlegend=False,
        margin=dict(b=80)
    )
    fig.add_annotation(
        text=stats_text, xref="paper", yref="paper", x=0.5, y=-0.15, showarrow=False
    )
    return fig


def create_hourly_pattern_chart(
    hourly_df: pd.DataFrame, fuel_type: str, station_name: str
) -> go.Figure:
    """Renders a bar chart coloring optimal (green) vs expensive (red) hours."""
    if hourly_df.empty or hourly_df['avg_price'].isna().all():
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    optimal = get_cheapest_and_most_expensive_hours(hourly_df)
    
    # Color logic
    colors = []
    p_min, p_max = hourly_df['avg_price'].min(), hourly_df['avg_price'].max()
    p_range = p_max - p_min

    for _, row in hourly_df.iterrows():
        h, price = row['hour'], row['avg_price']
        if pd.isna(price):
            colors.append('lightgray')
        elif h == optimal['cheapest_hour']:
            colors.append('green')
        elif h == optimal['most_expensive_hour']:
            colors.append('red')
        else:
            # Gradient: Green -> Yellow -> Red
            norm = (price - p_min) / p_range if p_range > 0 else 0.5
            colors.append(f'rgb({int(255 * norm)}, {int(200 * (1 - norm))}, 0)')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_df['hour'], y=hourly_df['avg_price'],
        marker_color=colors,
        text=hourly_df['avg_price'].apply(lambda x: f'€{x:.3f}' if pd.notna(x) else ''),
        textposition='outside',
        hovertemplate='Hour %{x}:00<br>€%{y:.3f}/L<extra></extra>'
    ))

    fig.update_layout(
        title=f"🕐 Hourly Price Patterns ({fuel_type.upper()})",
        xaxis_title="Hour", yaxis_title="Avg Price (€/L)",
        height=450, showlegend=False,
        xaxis=dict(tickmode='linear', dtick=2, tickformat='%H:00')
    )
    return fig


def create_comparison_chart(
    stations_data: Dict[str, pd.DataFrame], fuel_type: str, current_station_name: str
) -> go.Figure:
    """Renders multi-line chart comparing current station vs alternatives."""
    fig = go.Figure()
    
    for idx, (s_name, df) in enumerate(stations_data.items()):
        if df.empty: continue
        
        is_current = (current_station_name and s_name.startswith(current_station_name))
        
        if is_current:
            color, width, dash, opacity = '#00C800', 3, 'solid', 1.0
        else:
            palette = ['#FF6B6B', '#4ECDC4', '#FFD93D']
            color = palette[idx % len(palette)]
            width, dash, opacity = 2, 'dot', 0.6

        fig.add_trace(go.Scatter(
            x=df['date'], y=df['price'], mode='lines',
            name=s_name.replace(" (Current)", ""),
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity
        ))

    fig.update_layout(
        title="📊 Price Comparison",
        hovermode='x unified', height=450,
        legend=dict(orientation="v", y=0.99, x=1.02)
    )
    return fig


# ==========================================
# Tab Renderers
# ==========================================

def render_overview_tab(
    station: Dict[str, Any], keys: EconKeys, litres: float, use_economics: bool
):
    """Renders the 'Overview' tab: Metrics and Savings logic."""
    # 1. Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", _fmt_price(station.get(keys.current_price)))
    c2.metric("Predicted Price", _fmt_price(station.get(keys.pred_price)))
    
    d_km = station.get("detour_distance_km") or station.get("detour_km")
    d_min = station.get("detour_duration_min") or station.get("detour_min")
    c3.metric("Detour Distance", _fmt_km(d_km))
    c4.metric("Detour Time", _fmt_min(d_min))

    st.markdown("---")
    st.markdown("### 💰 Savings Summary")

    # 2. Economics Logic
    net_saving = _safe_float(station.get(keys.net_saving))
    
    if net_saving is not None:
        gross_saving = _safe_float(station.get(keys.gross_saving)) or 0.0
        cost_fuel = _safe_float(station.get(keys.detour_cost)) or 0.0
        cost_time = _safe_float(station.get(keys.time_cost)) or 0.0
        total_cost = cost_fuel + cost_time

        if net_saving > 0:
            st.success("✅ **This detour is worth it!**")
            ac, bc, cc = st.columns(3)
            ac.metric("Net Saving", f"€{net_saving:.2f}", delta="You save money!")
            bc.metric("Price Saving", f"€{gross_saving:.2f}")
            cc.metric("Detour Cost", f"€{total_cost:.2f}")
        else:
            st.warning("⚠️ **All options require detours**")
            ac, bc = st.columns(2)
            ac.metric("Net Cost", f"-€{abs(net_saving):.2f}")
            bc.metric("Baseline Price", _fmt_price(station.get(keys.baseline_price)))
            
            st.info(f"💡 This is the best recommendation, but it effectively costs €{abs(net_saving):.2f} due to the detour.")

        # Break-even info
        be = _safe_float(station.get(keys.breakeven))
        if be and be > 0:
            st.caption(f"💡 Break-even: Refuel at least **{be:.1f} L** to make this worthwhile.")

    elif not use_economics:
        st.info("Economic metrics disabled. Enable 'Use economics-based detour decision' in main app to view.")
    else:
        st.warning("Economic metrics unavailable for this station.")


def render_prices_tab(
    station: Dict[str, Any], uuid: str, name: str, fuel_code: str, litres: float, debug: bool
):
    """Renders the 'Prices' tab: Historical & Hourly charts."""
    if not uuid:
        st.warning("Station UUID missing. Cannot load history.")
        return

    with st.spinner("Loading charts..."):
        try:
            df = get_station_price_history(uuid, fuel_code, 14)
            if df is not None and not df.empty:
                # Trend Chart
                st.plotly_chart(
                    create_price_trend_chart(df, fuel_code, name), 
                    use_container_width=True
                )
                
                # Hourly Chart
                st.markdown("---")
                st.markdown("### 🕐 Best Time to Refuel")
                stats = calculate_hourly_price_stats(df)
                
                if not stats.empty and stats['avg_price'].any():
                    st.plotly_chart(
                        create_hourly_pattern_chart(stats, fuel_code, name),
                        use_container_width=True
                    )
                    
                    # Optimal Times Summary
                    opt = get_cheapest_and_most_expensive_hours(stats)
                    if opt['cheapest_hour'] is not None:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("💚 Best Hour", f"{opt['cheapest_hour']:02d}:00", f"€{opt['cheapest_price']:.3f}/L")
                        c2.metric("💸 Worst Hour", f"{opt['most_expensive_hour']:02d}:00", f"€{opt['most_expensive_price']:.3f}/L", delta_color="inverse")
                        
                        diff = opt['most_expensive_price'] - opt['cheapest_price']
                        total_save = diff * litres if litres else 0
                        c3.metric("💰 Potential Savings", f"€{total_save:.2f}" if litres else f"€{diff:.3f}/L")
                else:
                    st.info("Insufficient hourly data.")
            else:
                st.info("No historical data found.")

        except Exception as e:
            if debug: st.error(f"Error: {e}")
            else: st.warning("Could not load charts.")


def render_comparison_tab(
    station: Dict[str, Any], uuid: str, name: str, fuel_code: str, ranked_list: List[Dict]
):
    """Renders the 'Comparison' tab: Benchmarking against alternatives."""
    st.markdown("### 📊 Is This Station Consistently Cheaper?")
    
    if not ranked_list or len(ranked_list) < 2:
        st.info("Run the recommender to see comparisons.")
        return

    # Prepare options
    options = {}
    for i, s in enumerate(ranked_list[:10]):
        s_id = _station_uuid(s)
        s_lbl = f"#{i+1}: {s.get('tk_name') or s.get('osm_name') or 'Unknown'}"
        if s.get("brand"): s_lbl += f" ({s['brand']})"
        options[s_lbl] = (s_id, s.get('tk_name', 'Unknown'))

    # Multiselect
    selections = st.multiselect(
        "Compare with:", list(options.keys()), 
        default=list(options.keys())[:min(2, len(options))], 
        max_selections=2
    )

    if selections:
        with st.spinner("Comparing..."):
            data_map = {}
            # Load current
            try:
                curr_df = get_station_price_history(uuid, fuel_code, 14)
                if not curr_df.empty: data_map[f"{name} (Current)"] = curr_df
            except: pass

            # Load alternatives
            for lbl in selections:
                alt_uuid, alt_name = options[lbl]
                if alt_uuid != uuid:
                    try:
                        alt_df = get_station_price_history(alt_uuid, fuel_code, 14)
                        if not alt_df.empty: data_map[alt_name] = alt_df
                    except: pass
            
            if len(data_map) >= 2:
                st.plotly_chart(
                    create_comparison_chart(data_map, fuel_code, name), 
                    use_container_width=True
                )
                
                # Insight Logic
                curr_mean = data_map[f"{name} (Current)"]['price'].mean()
                cheaper_than = 0
                for k, v in data_map.items():
                    if k != f"{name} (Current)" and curr_mean < v['price'].mean():
                        cheaper_than += 1
                
                if cheaper_than == len(data_map) - 1:
                    st.success(f"✅ **{name}** is consistently cheaper than selected alternatives!")
                elif cheaper_than > 0:
                    st.info(f"ℹ️ **{name}** is competitive (cheaper than {cheaper_than} alternatives).")
                else:
                    st.warning(f"⚠️ **{name}** is historically more expensive than these alternatives.")
            else:
                st.warning("Insufficient data for comparison.")


def render_economics_tab(station: Dict[str, Any], keys: EconKeys, use_economics: bool):
    """Renders the 'Economics' tab: Raw numbers table."""
    st.markdown("### 💰 Detailed Economic Breakdown")
    
    if not station.get(keys.net_saving) and not use_economics:
        st.info("Economic analysis disabled in main app settings.")
        return

    be_val = _safe_float(station.get(keys.breakeven))
    be_str = f"{be_val:.1f} L" if be_val and be_val > 0 else "—"

    rows = [
        {"Metric": "Baseline Price", "Value": _fmt_price(station.get(keys.baseline_price))},
        {"Metric": "Gross Saving", "Value": _fmt_eur(station.get(keys.gross_saving))},
        {"Metric": "Detour Fuel", "Value": _fmt_liters(station.get(keys.detour_fuel))},
        {"Metric": "Detour Cost", "Value": _fmt_eur(station.get(keys.detour_cost))},
        {"Metric": "Time Cost", "Value": _fmt_eur(station.get(keys.time_cost))},
        {"Metric": "Net Saving", "Value": _fmt_eur(station.get(keys.net_saving))},
        {"Metric": "Break-even Vol", "Value": be_str},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ==========================================
# Main Orchestration
# ==========================================

def main():
    st.title("📊 Station Details & Analysis")

    # 1. Navigation & State Check
    if st.button("← Back to Main"):
        st.switch_page("streamlit_app.py")

    if "selected_station_data" not in st.session_state:
        st.error("⚠️ No station selected.")
        st.stop()

    # 2. Context Loading
    station = st.session_state["selected_station_data"]
    cached = st.session_state.get("last_run", {})
    
    fuel_code = cached.get("fuel_code", "e5")
    litres = cached.get("litres_to_refuel", 40.0)
    ranked = cached.get("ranked", [])
    debug = cached.get("debug_mode", False)
    use_econ = cached.get("params", {}).get("use_economics", True)
    
    keys = EconKeys(fuel_code)
    uuid = _station_uuid(station)
    name = station.get("tk_name") or station.get("osm_name") or "Unknown"

    # 3. Header
    st.markdown(f"### **{name}**")
    if station.get("brand"): st.markdown(station["brand"])
    if uuid and debug: st.caption(f"UUID: {uuid}")

    # 4. Tabs
    t_over, t_price, t_comp, t_econ = st.tabs(["Overview", "Prices", "Comparison", "Economics"])

    with t_over:
        render_overview_tab(station, keys, litres, use_econ)
    
    with t_price:
        if uuid: render_prices_tab(station, uuid, name, fuel_code, litres, debug)
        else: st.warning("Station UUID invalid.")

    with t_comp:
        if uuid: render_comparison_tab(station, uuid, name, fuel_code, ranked)

    with t_econ:
        render_economics_tab(station, keys, use_econ)


if __name__ == "__main__":
    main()