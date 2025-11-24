"""
Simple Streamlit UI for the route-aware refuelling recommender.

Two modes:
    1) TEST MODE (example data, no ORS calls)
       - Uses `run_example()` from route_tankerkoenig_integration.py
       - Good for development and when ORS is down.

    2) REAL ROUTE MODE (full pipeline)
       - Uses `get_fuel_prices_for_route(...)`
       - Requires working ORS + Supabase + Tankerkönig data.

In both modes the pipeline is:
    integration (model_input)
      -> ARDL models (predictions)
      -> decision layer (ranking + recommendation)
      -> Streamlit UI (tables and summaries)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Make the project root importable (so `import src.*` works when running
#   `streamlit run src/app/streamlit_app.py` from the project root).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import from the src package
from src.modeling.model import FUEL_TYPES  # ('e5', 'e10', 'diesel')
from src.modeling.predict import predict_for_fuel
from src.decision.recommender import (
    recommend_best_station,
    rank_stations_by_predicted_price,
)
from src.integration.route_tankerkoenig_integration import (
    run_example,
    get_fuel_prices_for_route,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _fuel_label_to_code(label: str) -> str:
    """
    Map human-readable label ('E5', 'E10', 'Diesel') to internal fuel code
    ('e5', 'e10', 'diesel') and validate.
    """
    mapping = {
        "E5": "e5",
        "E10": "e10",
        "Diesel": "diesel",
    }
    code = mapping.get(label)
    if code is None or code not in FUEL_TYPES:
        raise ValueError(f"Unsupported fuel label: {label}")
    return code


def _run_pipeline_test_mode(fuel_code: str) -> List[Dict]:
    """
    Run the full pipeline in TEST mode.

    Uses the fixed example route from `run_example()` (no ORS calls).
    """
    # 1) Get integrated data (stations + historical prices + time cells)
    model_input = run_example()

    # 2) Run ARDL model for selected fuel in-place
    model_input = predict_for_fuel(model_input, fuel_code)

    return model_input


def _run_pipeline_real_mode(
    fuel_code: str,
    start_locality: str,
    end_locality: str,
    start_address: str,
    end_address: str,
    use_realtime: bool,
) -> List[Dict]:
    """
    Run the full pipeline in REAL ROUTE mode.

    Wraps `get_fuel_prices_for_route(...)` and then runs the ARDL model
    for the selected fuel.
    """
    # 1) Get integrated data from full route pipeline
    model_input = get_fuel_prices_for_route(
        start_locality=start_locality,
        end_locality=end_locality,
        start_address=start_address or "",
        end_address=end_address or "",
        use_realtime=use_realtime,
    )

    # 2) Run ARDL model for selected fuel in-place
    model_input = predict_for_fuel(model_input, fuel_code)

    return model_input


def _build_ranking_dataframe(stations: List[Dict], fuel_code: str) -> pd.DataFrame:
    """
    Convert ranked station list into a tidy DataFrame for display.

    The columns focus on what is most relevant for understanding the
    recommendation and inspecting the model inputs.
    """
    if not stations:
        return pd.DataFrame()

    fuel_suffix = fuel_code  # 'e5', 'e10', 'diesel'
    pred_key = f"pred_price_{fuel_suffix}"
    current_key = f"price_current_{fuel_suffix}"
    lag1_key = f"price_lag_1d_{fuel_suffix}"
    lag2_key = f"price_lag_2d_{fuel_suffix}"
    lag3_key = f"price_lag_3d_{fuel_suffix}"
    lag7_key = f"price_lag_7d_{fuel_suffix}"

    df = pd.DataFrame(stations)

    # Select a clean subset of columns (missing columns will just be NaN)
    column_order = [
        "tk_name",
        "brand",
        "city",
        "osm_name",
        "fraction_of_route",
        "distance_along_m",
        "time_cell",
        current_key,
        lag1_key,
        lag2_key,
        lag3_key,
        lag7_key,
        pred_key,
    ]

    # Keep only columns that actually exist
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols].copy()

    # Make column labels nicer for display
    nice_names = {
        "tk_name": "Station name",
        "brand": "Brand",
        "city": "City",
        "osm_name": "OSM name",
        "fraction_of_route": "Fraction of route",
        "distance_along_m": "Distance along route [m]",
        "time_cell": "Time cell (0–47)",
        current_key: f"Current {fuel_suffix.upper()} price",
        lag1_key: f"Lag 1d {fuel_suffix.upper()}",
        lag2_key: f"Lag 2d {fuel_suffix.upper()}",
        lag3_key: f"Lag 3d {fuel_suffix.upper()}",
        lag7_key: f"Lag 7d {fuel_suffix.upper()}",
        pred_key: f"Predicted {fuel_suffix.upper()} price",
    }
    df.rename(columns=nice_names, inplace=True)

    return df


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Route-aware Fuel Price Recommender",
        layout="wide",
    )

    st.title("Route-aware Fuel Price Recommender (Prototype)")

    st.markdown(
        """
        This UI wraps the existing pipeline:

        - **Integration** (route → stations → historical prices)  
        - **ARDL prediction models** for E5, E10 and Diesel  
        - **Decision layer** to rank and recommend stations along the route
        """
    )

    # ----------------------------------------------------------------------
    # Sidebar controls
    # ----------------------------------------------------------------------
    st.sidebar.header("Configuration")

    env_mode = st.sidebar.radio(
        "Environment",
        ["Test mode (example route)", "Real route (ORS pipeline)"],
        index=0,
    )

    fuel_label = st.sidebar.selectbox(
        "Fuel type",
        ["E5", "E10", "Diesel"],
        index=0,
    )
    fuel_code = _fuel_label_to_code(fuel_label)

    if env_mode.startswith("Real route"):
        st.sidebar.markdown("**Route settings**")

        start_locality = st.sidebar.text_input(
            "Start locality (city/town)", value="Tübingen"
        )
        start_address = st.sidebar.text_input(
            "Start address (optional)", value="Wilhelmstraße 7"
        )
        end_locality = st.sidebar.text_input(
            "End locality (city/town)", value="Reutlingen"
        )
        end_address = st.sidebar.text_input(
            "End address (optional)", value="Charlottenstraße 45"
        )

        use_realtime = st.sidebar.checkbox(
            "Use real-time Tankerkönig prices",
            value=False,
            help="If unchecked, yesterday's prices are used as 'current' prices.",
        )
    else:
        # In test mode we ignore route inputs, use fixed example data
        start_locality = end_locality = start_address = end_address = ""
        use_realtime = False

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run recommender")

    # ----------------------------------------------------------------------
    # Main execution
    # ----------------------------------------------------------------------
    if not run_button:
        st.info("Configure the settings in the sidebar and click **Run recommender**.")
        return

    try:
        if env_mode.startswith("Test"):
            st.subheader("Mode: Test (example route)")
            st.caption(
                "Using fixed test route and example stations from "
                "`run_example()` – no ORS calls."
            )
            model_input = _run_pipeline_test_mode(fuel_code)

        else:
            st.subheader("Mode: Real route (ORS pipeline)")
            st.caption(
                "Using `get_fuel_prices_for_route(...)`. Requires working ORS and "
                "Supabase/Tankerkoenig data."
            )

            if not start_locality or not end_locality:
                st.error("Please provide both start and end localities.")
                return

            model_input = _run_pipeline_real_mode(
                fuel_code=fuel_code,
                start_locality=start_locality,
                end_locality=end_locality,
                start_address=start_address,
                end_address=end_address,
                use_realtime=use_realtime,
            )

    except ImportError as e:
        st.error(
            f"Import error while building the route pipeline: {e}\n\n"
            "Most likely `route_stations.py` cannot be imported. "
            "Ensure it is in the project root and that your working "
            "directory is the repository root."
        )
        return
    except Exception as e:
        st.error(f"Unexpected error while running the pipeline:\n\n{e}")
        return

    if not model_input:
        st.warning("The pipeline returned no stations. Check logs / ORS status.")
        return

    num_stations = len(model_input)
    st.markdown(f"**Stations with data:** {num_stations}")

    # Decision layer: ranking + best station
    ranked_stations = rank_stations_by_predicted_price(model_input, fuel_code)
    best_station = recommend_best_station(model_input, fuel_code)

    if best_station is None or not ranked_stations:
        st.warning(
            "No valid predictions for the selected fuel type. "
            "Check whether the required lag price fields are present."
        )
        return

    # ----------------------------------------------------------------------
    # Best station card
    # ----------------------------------------------------------------------
    st.markdown("### Recommended station")

    fuel_suffix = fuel_code.upper()
    pred_key = f"pred_price_{fuel_code}"
    current_key = f"price_current_{fuel_code}"
    lag1_key = f"price_lag_1d_{fuel_code}"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Station",
            f"{best_station.get('tk_name', 'Unknown')} "
            f"({best_station.get('brand', 'N/A')})",
        )
        st.write(best_station.get("city", ""))

    with col2:
        st.metric(
            f"Predicted {fuel_suffix} price",
            f"{best_station.get(pred_key):.3f}"
            if best_station.get(pred_key) is not None
            else "n/a",
        )
        st.metric(
            f"Current {fuel_suffix} price",
            f"{best_station.get(current_key):.3f}"
            if best_station.get(current_key) is not None
            else "n/a",
        )

    with col3:
        st.metric(
            "Fraction of route",
            f"{best_station.get('fraction_of_route', 0.0):.3f}",
        )
        st.metric(
            "Distance along route [m]",
            f"{best_station.get('distance_along_m', 0):,.0f}",
        )

    # ----------------------------------------------------------------------
    # Full ranking table
    # ----------------------------------------------------------------------
    st.markdown("### Ranking of stations (cheapest → most expensive)")

    df_ranked = _build_ranking_dataframe(ranked_stations, fuel_code)
    if df_ranked.empty:
        st.info("No stations with valid predictions to display.")
    else:
        st.dataframe(df_ranked.reset_index(drop=True))


if __name__ == "__main__":
    main()
