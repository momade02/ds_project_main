"""
Streamlit UI for the route-aware fuel price recommender.

Two environments
----------------
1) Test mode (example route, no ORS calls)
   - Uses `run_example()` from `route_tankerkoenig_integration`.

2) Real route (full ORS + Supabase + TankerkÃ¶nig pipeline)
   - Uses `get_fuel_prices_for_route(...)`.
   - Always uses real-time TankerkÃ¶nig prices.

High-level pipeline in both modes
---------------------------------
integration (route + station prices)
    -> ARDL models with horizon logic (in `src.modeling.predict`)
    -> decision layer (ranking & best station in `src.decision.recommender`)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Make sure the project root (containing the `src` package) is on sys.path
# ---------------------------------------------------------------------------
import sys
from pathlib import Path
from typing import List, Dict, Any

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
    cells_ahead = station.get(f"debug_{fuel_code}_cells_ahead_raw")

    if used_current:
        # Now reflects the refined rule with the ETA threshold
        return "Current price (arrival in â‰² 10 min)"

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

    rows = []
    for s in stations:
        rows.append(
            {
                "Station name": s.get("tk_name") or s.get("osm_name"),
                "Brand": s.get("brand"),
                "City": s.get("city"),
                "OSM name": s.get("osm_name"),
                "Fraction of route": s.get("fraction_of_route"),
                "Distance along route [m]": s.get("distance_along_m"),
                # human-readable explanation based on debug_* fields
                "Price basis": _describe_price_basis(s, fuel_code),
                f"Current {fuel_code.upper()} price": s.get(current_key),
                f"Lag 1d {fuel_code.upper()}": s.get(lag1_key),
                f"Lag 2d {fuel_code.upper()}": s.get(lag2_key),
                f"Lag 3d {fuel_code.upper()}": s.get(lag3_key),
                f"Lag 7d {fuel_code.upper()}": s.get(lag7_key),
                f"Predicted {fuel_code.upper()} price": s.get(pred_key),
            }
        )

    df = pd.DataFrame(rows)

    # Only these columns are numeric prices â€“ do NOT touch "Price basis"
    numeric_price_cols = [
        f"Current {fuel_code.upper()} price",
        f"Lag 1d {fuel_code.upper()}",
        f"Lag 2d {fuel_code.upper()}",
        f"Lag 3d {fuel_code.upper()}",
        f"Lag 7d {fuel_code.upper()}",
        f"Predicted {fuel_code.upper()} price",
    ]

    for col in numeric_price_cols:
        if col in df.columns:
            df[col] = df[col].map(_format_price)

    return df


def _display_best_station(best_station: Dict[str, Any], fuel_code: str) -> None:
    """
    Render a panel with information about the recommended station,
    including a short explanation whether current price or a forecast
    was used.
    """
    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"

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

    # Human-readable explanation of what the model actually used
    explanation = _describe_price_basis(best_station, fuel_code)
    st.caption(
        f"How this price was determined: {explanation} "
        "(based on arrival time and available history for this station)."
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

- **Integration** (route â†’ stations â†’ historical + real-time prices)
- **ARDL prediction models** for E5, E10 and Diesel (15 models total)
- **Decision layer** to rank and recommend stations along the route
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    env_label = st.sidebar.radio(
        "Environment",
        options=["Test mode (example route)", "Real route (ORS pipeline)"],
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
        "Start locality (city/town)", value="TÃ¼bingen"
    )
    start_address = st.sidebar.text_input(
        "Start address (optional)", value="WilhelmstraÃŸe 32"
    )

    end_locality = st.sidebar.text_input(
        "End locality (city/town)", value="Reutlingen"
    )
    end_address = st.sidebar.text_input(
        "End address (optional)", value="CharlottenstraÃŸe 45"
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
        st.subheader("Mode: Real route (ORS pipeline with real-time prices)")

        if not start_locality or not end_locality:
            st.error("Please provide at least start and end localities (cities/towns).")
            return

        try:
            stations = get_fuel_prices_for_route(
                start_locality=start_locality,
                end_locality=end_locality,
                start_address=start_address,
                end_address=end_address,
                use_realtime=True,  # always use current TankerkÃ¶nig prices
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
    ranked = rank_stations_by_predicted_price(stations, fuel_code)
    if not ranked:
        st.warning("No stations with valid predictions for the selected fuel.")
        return

    best_station = recommend_best_station(stations, fuel_code)
    _display_best_station(best_station, fuel_code)

    # ----------------------------------------------------------------------
    # Full ranking table
    # ----------------------------------------------------------------------
    st.markdown("### Ranking of stations (cheapest â†’ most expensive)")
    st.caption(
        "The **Price basis** column shows whether the recommendation uses the "
        "observed current price (arrival still in this 30-minute block) or a "
        "model forecast and how far ahead that forecast looks."
    )

    df_ranked = _build_ranking_dataframe(ranked, fuel_code)
    if df_ranked.empty:
        st.info("No stations with valid predictions to display.")
    else:
        st.dataframe(df_ranked.reset_index(drop=True))


if __name__ == "__main__":
    main()