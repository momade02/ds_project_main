"""
Streamlit UI for the route-aware fuel price recommender.

Two environments
----------------
1) Test mode (example route, no Google calls)
   - Uses `run_example()` from `route_tankerkoenig_integration`.

2) Real route (Google route + Supabase + Tankerkönig pipeline)
   - Uses `get_fuel_prices_for_route(...)`.
   - Always uses real-time Tankerkönig prices.

High-level pipeline in both modes
---------------------------------
integration (route → stations → historical + real-time prices)
    → ARDL models with horizon logic (in `src.modeling.predict`)
    → decision layer (ranking & best station in `src.decision.recommender`)

This UI additionally implements an economic detour decision:
- user-specific litres to refuel,
- car consumption (L/100 km),
- optional value of time (€/hour),
- hard caps for max detour distance / time,
- net saving and break-even litres.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Make sure the project root (containing the `src` package) is on sys.path
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.route_tankerkoenig_integration import (
    run_example,
    get_fuel_prices_for_route,
)
from src.decision.recommender import (
    recommend_best_station,
    rank_stations_by_predicted_price,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _fuel_label_to_code(label: str) -> str:
    """
    Map human-readable fuel label to internal fuel code.

    Parameters
    ----------
    label : str
        One of 'E5', 'E10', 'Diesel'.

    Returns
    -------
    str
        'e5', 'e10' or 'diesel'.
    """
    mapping = {"E5": "e5", "E10": "e10", "Diesel": "diesel"}
    code = mapping.get(label)
    if code is None:
        raise ValueError(f"Unsupported fuel label '{label}'.")
    return code


def _format_price(x: Any) -> str:
    """Format a numeric price as a string with 3 decimals (or '-' for missing)."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "-"


def _format_eur(x: Any) -> str:
    """Format a numeric value as 'x.xx €' (or '-' for missing)."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.2f} €"
    except (TypeError, ValueError):
        return "-"


def _format_km(x: Any) -> str:
    """Format kilometres with one decimal."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.1f} km"
    except (TypeError, ValueError):
        return "-"


def _format_min(x: Any) -> str:
    """Format minutes with one decimal."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.1f} min"
    except (TypeError, ValueError):
        return "-"


def _format_liters(x: Any) -> str:
    """Format litres with two decimals (or '-' for missing)."""
    try:
        if x is None:
            return "-"
        return f"{float(x):.2f} L"
    except (TypeError, ValueError):
        return "-"


def _describe_price_basis(
    station: Dict[str, Any],
    fuel_code: str,
) -> str:
    """
    Turn the debug fields for a station into a human-readable explanation.

    Uses:
        debug_<fuel>_used_current_price
        debug_<fuel>_horizon_used
        debug_<fuel>_cells_ahead_raw
    """
    used_current = bool(station.get(f"debug_{fuel_code}_used_current_price"))
    horizon = station.get(f"debug_{fuel_code}_horizon_used")

    if used_current:
        # Now reflects the refined rule with the ETA threshold
        return "Current price (arrival in ≤ 10 min)"

    if horizon is None:
        # No explicit horizon; either we could not model or we only used daily info.
        return "No forecast available (fallback)"

    try:
        h_int = int(horizon)
    except (TypeError, ValueError):
        h_int = None

    if h_int is None or h_int < 0:
        return "Forecast (ARDL model)"

    approx_min = h_int * 30
    if approx_min == 0:
        return "Forecast (same block, daily lags only)"
    else:
        return f"Forecast (~{approx_min} min ahead, horizon {h_int})"


def _build_ranking_dataframe(
    stations: List[Dict[str, Any]],
    fuel_code: str,
    debug_mode: bool = False,
) -> pd.DataFrame:
    """
    Build a DataFrame with the most relevant columns for ranking display.

    Parameters
    ----------
    stations :
        List of station dictionaries (already ranked).
    fuel_code :
        'e5', 'e10' or 'diesel'.

    Returns
    -------
    pandas.DataFrame
    """
    if not stations:
        return pd.DataFrame()

    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"
    lag1_key = f"price_lag_1d_{fuel_code}"
    lag2_key = f"price_lag_2d_{fuel_code}"
    lag3_key = f"price_lag_3d_{fuel_code}"
    lag7_key = f"price_lag_7d_{fuel_code}"

    # Economic keys (may or may not exist, depending on how ranking was called)
    econ_net_key = f"econ_net_saving_eur_{fuel_code}"
    econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
    econ_detour_fuel_key = f"econ_detour_fuel_l_{fuel_code}"
    econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
    econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
    econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
    econ_baseline_key = f"econ_baseline_price_{fuel_code}"

    rows = []
    for s in stations:
        row = {
            "Station name": s.get("tk_name") or s.get("osm_name"),
            "Brand": s.get("brand"),
            "City": s.get("city"),
            "OSM name": s.get("osm_name"),
            "Fraction of route": s.get("fraction_of_route"),
            "Distance along route [m]": s.get("distance_along_m"),
            # Detour geometry
            "Detour distance [km]": s.get("detour_distance_km"),
            "Detour time [min]": s.get("detour_duration_min"),
            # human-readable explanation based on debug_* fields
            "Price basis": _describe_price_basis(s, fuel_code),
            f"Current {fuel_code.upper()} price": s.get(current_key),
            f"Lag 1d {fuel_code.upper()}": s.get(lag1_key),
            f"Lag 2d {fuel_code.upper()}": s.get(lag2_key),
            f"Lag 3d {fuel_code.upper()}": s.get(lag3_key),
            f"Lag 7d {fuel_code.upper()}": s.get(lag7_key),
            f"Predicted {fuel_code.upper()} price": s.get(pred_key),
        }

        # Economic metrics (only added if present)
        if econ_net_key in s:
            row["Baseline on-route price"] = s.get(econ_baseline_key)
            row["Gross saving [€]"] = s.get(econ_gross_key)
            row["Detour fuel [L]"] = s.get(econ_detour_fuel_key)
            row["Detour fuel cost [€]"] = s.get(econ_detour_fuel_cost_key)
            row["Time cost [€]"] = s.get(econ_time_cost_key)
            row["Net saving [€]"] = s.get(econ_net_key)
            row["Break-even litres"] = s.get(econ_breakeven_key)

        if debug_mode:
            # Raw diagnostic fields from the prediction layer
            row[f"DEBUG current_time_cell_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_current_time_cell"
            )
            row[f"DEBUG cells_ahead_raw_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_cells_ahead_raw"
            )
            row[f"DEBUG minutes_to_arrival_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_minutes_to_arrival"
            )
            row[f"DEBUG horizon_used_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_horizon_used"
            )
            row[f"DEBUG eta_utc_{fuel_code}"] = s.get(
                f"debug_{fuel_code}_eta_utc"
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    # Only these columns are numeric prices – do NOT touch "Price basis"
    numeric_price_cols = [
        f"Current {fuel_code.upper()} price",
        f"Lag 1d {fuel_code.upper()}",
        f"Lag 2d {fuel_code.upper()}",
        f"Lag 3d {fuel_code.upper()}",
        f"Lag 7d {fuel_code.upper()}",
        f"Predicted {fuel_code.upper()} price",
        "Baseline on-route price",
    ]

    for col in numeric_price_cols:
        if col in df.columns:
            df[col] = df[col].map(_format_price)

    # Format economic + detour columns if present
    if "Detour distance [km]" in df.columns:
        df["Detour distance [km]"] = df["Detour distance [km]"].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.1f}"
        )
    if "Detour time [min]" in df.columns:
        df["Detour time [min]"] = df["Detour time [min]"].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.1f}"
        )
    if "Gross saving [€]" in df.columns:
        df["Gross saving [€]"] = df["Gross saving [€]"].map(_format_eur)
    if "Detour fuel [L]" in df.columns:
        df["Detour fuel [L]"] = df["Detour fuel [L]"].map(_format_liters)
    if "Detour fuel cost [€]" in df.columns:
        df["Detour fuel cost [€]"] = df["Detour fuel cost [€]"].map(_format_eur)
    if "Time cost [€]" in df.columns:
        df["Time cost [€]"] = df["Time cost [€]"].map(_format_eur)
    if "Net saving [€]" in df.columns:
        df["Net saving [€]"] = df["Net saving [€]"].map(_format_eur)
    if "Break-even litres" in df.columns:
        df["Break-even litres"] = df["Break-even litres"].map(
            lambda v: "-" if v is None or pd.isna(v) else f"{float(v):.2f}"
        )
        
    return df


def _display_best_station(
    best_station: Dict[str, Any],
    fuel_code: str,
    litres_to_refuel: Optional[float] = None,
) -> None:
    """
    Render a panel with information about the recommended station,
    including a short explanation whether current price or a forecast
    was used, and (if available) economic detour metrics.
    """
    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"

    econ_net_key = f"econ_net_saving_eur_{fuel_code}"
    econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
    econ_detour_fuel_key = f"econ_detour_fuel_l_{fuel_code}"
    econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
    econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
    econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
    econ_baseline_key = f"econ_baseline_price_{fuel_code}"

    if not best_station:
        st.info("No station could be recommended (no valid predictions).")
        return

    station_name = best_station.get("tk_name") or best_station.get("osm_name")
    brand = best_station.get("brand") or "-"
    city = best_station.get("city") or "-"
    frac = best_station.get("fraction_of_route")
    dist_m = best_station.get("distance_along_m")
    pred_price = best_station.get(pred_key)
    current_price = best_station.get(current_key)

    detour_km = best_station.get("detour_distance_km")
    detour_min = best_station.get("detour_duration_min")

    frac_str = "-" if frac is None else f"{float(frac):.3f}"
    if dist_m is None:
        dist_str = "-"
    else:
        try:
            dist_km = float(dist_m) / 1000.0
            dist_str = f"{dist_km:.1f} km"
        except (TypeError, ValueError):
            dist_str = "-"

    st.markdown("### Recommended station")

    st.markdown(
        f"**{station_name}**  \n"
        f"{brand}, {city}"
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label=f"Predicted {fuel_code.upper()} price",
            value=_format_price(pred_price),
        )
    with col2:
        st.metric(
            label=f"Current {fuel_code.upper()} price",
            value=_format_price(current_price),
        )
    with col3:
        st.metric("Fraction of route", frac_str)
    with col4:
        st.metric("Distance along route", dist_str)

    # Detour metrics row
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Detour distance", _format_km(detour_km))
    with col6:
        st.metric("Detour time", _format_min(detour_min))

    # Human-readable explanation of what the model actually used
    explanation = _describe_price_basis(best_station, fuel_code)
    st.caption(
        f"How this price was determined: {explanation} "
        "(based on arrival time and available history for this station)."
    )

    # Economic detour metrics (if available)
    if econ_net_key in best_station:
        baseline_price = best_station.get(econ_baseline_key)
        gross_saving = best_station.get(econ_gross_key)
        detour_fuel_l = best_station.get(econ_detour_fuel_key)
        detour_fuel_cost = best_station.get(econ_detour_fuel_cost_key)
        time_cost = best_station.get(econ_time_cost_key)
        net_saving = best_station.get(econ_net_key)
        breakeven_liters = best_station.get(econ_breakeven_key)

        st.markdown("#### Economic impact of this detour")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "Baseline on-route price",
                _format_price(baseline_price),
            )
        with col_b:
            st.metric(
                "Gross saving",
                _format_eur(gross_saving),
            )
        with col_c:
            st.metric(
                "Detour fuel",
                _format_liters(detour_fuel_l),
            )

        col_d, col_e, col_f = st.columns(3)
        with col_d:
            st.metric(
                "Detour fuel cost",
                _format_eur(detour_fuel_cost),
            )
        with col_e:
            st.metric(
                "Time cost",
                _format_eur(time_cost),
            )
        with col_f:
            st.metric(
                "Net saving",
                _format_eur(net_saving),
            )

        if breakeven_liters is not None:
            breakeven_txt = _format_liters(breakeven_liters)
        else:
            breakeven_txt = "Not applicable (not cheaper than baseline)"

        if litres_to_refuel is not None and litres_to_refuel > 0:
            st.caption(
                f"Assuming you refuel **{_format_liters(litres_to_refuel)}**, "
                f"this detour yields a net saving of {_format_eur(net_saving)}. "
                f"You would need at least {breakeven_txt} here for the detour "
                "to break even."
            )
        else:
            st.caption(
                f"Break-even refuelling amount for this detour: {breakeven_txt}."
            )


# ---------------------------------------------------------------------------
# Main Streamlit app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Route-aware Fuel Price Recommender",
        layout="wide",
    )

    st.title("Route-aware Fuel Price Recommender (Prototype)")

    st.markdown(
        """
This UI wraps the existing pipeline:

- **Integration** (route → stations → historical + real-time prices)
- **ARDL prediction models** for E5, E10 and Diesel (15 models total)
- **Decision layer** to rank and recommend stations along the route

The recommendation logic can optionally incorporate your own
refuelling amount, car consumption and value of time to decide
whether a detour is economically worthwhile.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    env_label = st.sidebar.radio(
        "Environment",
        options=["Test mode (example route)", "Real route (Google pipeline)"],
        index=1,
    )

    fuel_label = st.sidebar.selectbox(
        "Fuel type",
        options=["E5", "E10", "Diesel"],
        index=0,
    )
    fuel_code = _fuel_label_to_code(fuel_label)

    # Route settings (only used in real mode)
    st.sidebar.markdown("### Route settings (real mode)")

    start_locality = st.sidebar.text_input(
        "Start locality (city/town)", value="Tübingen"
    )
    start_address = st.sidebar.text_input(
        "Start address (optional)", value="Wilhelmstraße 32"
    )

    end_locality = st.sidebar.text_input(
        "End locality (city/town)", value="Reutlingen"
    )
    end_address = st.sidebar.text_input(
        "End address (optional)", value="Charlottenstraße 45"
    )

    # Detour economics
    st.sidebar.markdown("### Detour economics")

    litres_to_refuel = st.sidebar.number_input(
        "Litres to refuel",
        min_value=1.0,
        max_value=200.0,
        value=40.0,
        step=1.0,
    )
    consumption_l_per_100km = st.sidebar.number_input(
        "Car consumption (L/100 km)",
        min_value=2.0,
        max_value=30.0,
        value=7.0,
        step=0.5,
    )
    value_of_time_per_hour = st.sidebar.number_input(
        "Value of time (€/hour)",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
    )
    max_detour_km = st.sidebar.number_input(
        "Maximum extra distance (km)",
        min_value=0.5,
        max_value=200.0,
        value=10.0,
        step=0.5,
    )
    max_detour_min = st.sidebar.number_input(
        "Maximum extra time (min)",
        min_value=1.0,
        max_value=240.0,
        value=10.0,
        step=1.0,
    )
    min_net_saving_eur = st.sidebar.number_input(
        "Minimum net saving to accept detour (€, 0 = no threshold)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,
    )

    # Optional diagnostics
    debug_mode = st.sidebar.checkbox(
        "Debug mode (show pipeline diagnostics)", value=False
    )

    run_clicked = st.sidebar.button("Run recommender")

    if not run_clicked:
        st.info("Configure the settings on the left and click **Run recommender**.")
        return

    # ----------------------------------------------------------------------
    # Get data from integration layer
    # ----------------------------------------------------------------------
    if env_label.startswith("Test"):
        st.subheader("Mode: Test route (example data)")
        try:
            stations = run_example()
        except Exception as exc:
            st.error(f"Error while running example integration: {exc}")
            return
    else:
        st.subheader("Mode: Real route (Google pipeline with real-time prices)")

        if not start_locality or not end_locality:
            st.error("Please provide at least start and end localities (cities/towns).")
            return

        try:
            stations = get_fuel_prices_for_route(
                start_locality=start_locality,
                end_locality=end_locality,
                start_address=start_address,
                end_address=end_address,
                use_realtime=True,  # always use current Tankerkönig prices
            )
        except Exception as exc:
            st.error(f"Error while running real route pipeline: {exc}")
            return

    if not stations:
        st.warning("No stations returned by the integration pipeline.")
        return

    st.markdown(
        f"**Total stations with complete price data:** {len(stations)}"
    )

    # ----------------------------------------------------------------------
    # Recommendation and ranking
    # ----------------------------------------------------------------------
    ranked = rank_stations_by_predicted_price(
        stations,
        fuel_code,
        litres_to_refuel=litres_to_refuel,
        consumption_l_per_100km=consumption_l_per_100km,
        value_of_time_per_hour=value_of_time_per_hour,
        max_detour_km=max_detour_km,
        max_detour_min=max_detour_min,
        min_net_saving_eur=min_net_saving_eur,
    )
    if not ranked:
        st.warning(
            "No stations with valid predictions for the selected fuel "
            "and detour constraints."
        )
        return

    best_station = recommend_best_station(
        stations,
        fuel_code,
        litres_to_refuel=litres_to_refuel,
        consumption_l_per_100km=consumption_l_per_100km,
        value_of_time_per_hour=value_of_time_per_hour,
        max_detour_km=max_detour_km,
        max_detour_min=max_detour_min,
        min_net_saving_eur=min_net_saving_eur,
    )
    _display_best_station(best_station, fuel_code, litres_to_refuel=litres_to_refuel)

    # ----------------------------------------------------------------------
    # Full ranking table
    # ----------------------------------------------------------------------
    st.markdown("### Ranking of stations (highest net saving → lowest)")
    st.caption(
        "Stations are ordered by **net economic benefit** of the detour "
        "(gross saving minus detour fuel and time cost), subject to your "
        "detour distance/time caps and the minimum net saving threshold. "
        "The **Price basis** column shows whether the recommendation uses "
        "the observed current price or a model forecast."
    )

    df_ranked = _build_ranking_dataframe(ranked, fuel_code, debug_mode=debug_mode)
    if df_ranked.empty:
        st.info("No stations with valid predictions to display.")
    else:
        st.dataframe(df_ranked.reset_index(drop=True))


if __name__ == "__main__":
    main()
