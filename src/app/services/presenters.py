# src/app/services/presenters.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from src.app.ui.formatting import (
    _describe_price_basis,
    _format_price,
    _format_eur,
    _format_liters,
)

def build_ranking_dataframe(
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
        # --- 1. PRE-CALCULATE VALUES ---
        
        # Detour geometry
        # The raw route delta can be slightly negative due to routing/rounding artefacts.
        # For user-facing "detour" we display *extra* distance/time (clamped to >= 0),
        # consistent with the economics layer.
        _raw_detour_km = s.get("detour_distance_km")
        _raw_detour_min = s.get("detour_duration_min")

        try:
            _raw_detour_km_f = float(_raw_detour_km) if _raw_detour_km is not None else 0.0
        except (TypeError, ValueError):
            _raw_detour_km_f = 0.0

        try:
            _raw_detour_min_f = float(_raw_detour_min) if _raw_detour_min is not None else 0.0
        except (TypeError, ValueError):
            _raw_detour_min_f = 0.0

        # Clamp for display (extra detour only)
        _detour_km_display = max(_raw_detour_km_f, 0.0)
        _detour_min_display = max(_raw_detour_min_f, 0.0)

        # --- 2. BUILD THE DICTIONARY ---
        row = {
            "Station name": s.get("tk_name") or s.get("osm_name"),
            "Brand": s.get("brand"),
            "City": s.get("city"),
            "OSM name": s.get("osm_name"),
            "Fraction of route": s.get("fraction_of_route"),
            "Distance along route [m]": s.get("distance_along_m"),
            
            # Insert the pre-calculated values here
            "Detour distance [km]": _detour_km_display,
            "Detour time [min]": _detour_min_display,
            
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
            # Raw signed deltas from routing (can be negative)
            row["DEBUG raw detour distance [km]"] = _raw_detour_km_f
            row["DEBUG raw detour time [min]"] = _raw_detour_min_f
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

    # Format raw routing deltas (debug-only)
    if "DEBUG raw detour distance [km]" in df.columns:
        df["DEBUG raw detour distance [km]"] = df["DEBUG raw detour distance [km]"].map(
            lambda v: "-" if pd.isna(v) else f"{float(v):.1f}"
        )
    if "DEBUG raw detour time [min]" in df.columns:
        df["DEBUG raw detour time [min]"] = df["DEBUG raw detour time [min]"].map(
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