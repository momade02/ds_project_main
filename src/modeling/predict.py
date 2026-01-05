"""
Module: Inference Engine.

Description:
    This module orchestrates the prediction pipeline. It iterates over stations,
    determines the appropriate prediction horizon based on ETA, constructs
    feature vectors, and invokes the ARDL models.

    Decision Logic (Minutes-Based):
    1. Arrival < 10 mins: Use Current Price (Assume no change).
    2. No Current Price: Use Horizon 0 (Daily-only features).
    3. Arrival 10-120 mins: Use Horizon 1-4 (Intraday features).
    4. Arrival > 120 mins: Use Horizon 0 (Too far for intraday precision).

Usage:
    Called by `decision/recommender.py` or `integration/route_tankerkoenig_integration.py`.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Final

import pandas as pd

from .model import FUEL_TYPES, load_model_for_horizon

# ==========================================
# Constants & Configuration
# ==========================================

# Threshold (minutes) to consider arrival effectively "Now"
ETA_THRESHOLD_MINUTES: Final[float] = 10.0

# Feature columns for daily-only models
FEATURES_DAILY: Final[List[str]] = [
    "price_lag_1d", "price_lag_2d", "price_lag_3d", "price_lag_7d"
]


# ==========================================
# Helpers: Time & Feature Engineering
# ==========================================

def _normalize_fuel_type(fuel_type: str) -> str:
    """Ensures fuel type is lowercase and valid."""
    ft = fuel_type.lower().strip() if fuel_type else ""
    if ft not in FUEL_TYPES:
        raise ValueError(f"Invalid fuel: {fuel_type}. Valid: {FUEL_TYPES}")
    return ft


def _get_current_utc_time(now: Optional[datetime]) -> datetime:
    """Standardizes 'now' to a timezone-aware UTC datetime."""
    if now is None:
        return datetime.now(timezone.utc)
    
    if now.tzinfo is None:
        # Assume naive input is already UTC
        return now.replace(tzinfo=timezone.utc)
    
    return now.astimezone(timezone.utc)


def _get_time_cell(dt: datetime) -> int:
    """Calculates 30-minute time cell index (0-47)."""
    return dt.hour * 2 + (1 if dt.minute >= 30 else 0)


def _parse_eta_to_utc(eta: Any) -> Optional[datetime]:
    """Robustly parses ETA strings/objects to UTC datetime."""
    if not eta:
        return None
        
    try:
        if isinstance(eta, datetime):
            dt = eta
        else:
            # Handle ISO strings, strip trailing 'Z' if manual parsing needed
            s = str(eta).strip().rstrip("Z")
            dt = datetime.fromisoformat(s)

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _calculate_minutes_to_arrival(eta_utc: Optional[datetime], now_utc: datetime) -> Optional[float]:
    """Returns minutes between now and ETA. Returns None if ETA is missing."""
    if eta_utc is None:
        return None
    
    delta_seconds = (eta_utc - now_utc).total_seconds()
    return max(0.0, delta_seconds / 60.0)


def _has_required_features(station: Dict[str, Any], fuel: str) -> bool:
    """Checks for existence of all daily lag features."""
    required_suffixes = ("1d", "2d", "3d", "7d")
    return all(
        station.get(f"price_lag_{suffix}_{fuel}") is not None 
        for suffix in required_suffixes
    )


# ==========================================
# Core Logic: Horizon Selection
# ==========================================

def _determine_horizon(
    current_price: Optional[float], 
    minutes_to_arrival: Optional[float],
    cells_ahead_fallback: int
) -> Tuple[int, bool]:
    """
    Decides which model horizon to use.
    
    Returns:
        (horizon, use_current_price_directly)
        
    Logic:
        - If current price is missing -> Horizon 0 (Daily).
        - If arrival is 'now' (< 10m) -> Use current price directly (No model).
        - If arrival is 10m-2h -> Horizon 1-4.
        - If arrival is > 2h -> Horizon 0 (Daily).
    """
    # Case 1: Missing Current Price -> Force Daily Model
    if current_price is None:
        return 0, False

    # Case 2: Arrival is "Now" (or very close)
    # If we have ETA, check threshold. If no ETA, check fallback cells.
    is_now = False
    if minutes_to_arrival is not None:
        if minutes_to_arrival <= ETA_THRESHOLD_MINUTES:
            is_now = True
    elif cells_ahead_fallback <= 0:
        is_now = True

    if is_now:
        return 0, True # Flag to skip model and use CP

    # Case 3: Intraday Horizon Calculation
    if minutes_to_arrival is not None:
        # Map minutes to 30-min blocks (1 to 4)
        h_raw = math.ceil(minutes_to_arrival / 30.0)
        
        if 1 <= h_raw <= 4:
            return h_raw, False
        # If > 4 blocks (2 hours), fall back to daily model
        return 0, False
    
    # Case 4: Fallback (Legacy/No ETA)
    h_fallback = max(0, min(cells_ahead_fallback, 4))
    if h_fallback == 0:
        return 0, False # Fallback logic mapped 0 to daily
        
    return h_fallback, False


# ==========================================
# Core Logic: Prediction Loop
# ==========================================

def _predict_single_fuel(
    stations: List[Dict[str, Any]],
    fuel_type: str,
    output_key: Optional[str] = None,
    now: Optional[datetime] = None,
) -> None:
    """
    Predicts prices for one fuel type in-place.
    """
    ft = _normalize_fuel_type(fuel_type)
    pred_key = output_key or f"pred_price_{ft}"
    current_price_key = f"price_current_{ft}"
    
    # Standardize time once per batch
    now_utc = _get_current_utc_time(now)
    current_cell = _get_time_cell(now_utc)

    for s in stations:
        # Initialize output
        if pred_key not in s:
            s[pred_key] = None

        # 1. Validation: Check essential time/feature data
        if s.get("time_cell") is None or not _has_required_features(s, ft):
            continue

        # 2. Extract Context
        current_price = s.get(current_price_key)
        
        # Parse ETA
        eta_utc = _parse_eta_to_utc(s.get("eta"))
        minutes_ahead = _calculate_minutes_to_arrival(eta_utc, now_utc)
        
        # Fallback metric
        cells_ahead = s["time_cell"] - current_cell
        
        # 3. Decision Logic
        horizon, use_cp = _determine_horizon(current_price, minutes_ahead, cells_ahead)
        
        # Debug Metadata
        s[f"debug_{ft}_minutes_ahead"] = minutes_ahead
        s[f"debug_{ft}_horizon"] = horizon
        s[f"debug_{ft}_used_cp"] = use_cp

        # 4. Execution
        if use_cp:
            # Just use current price
            try:
                s[pred_key] = float(current_price) # type: ignore
            except (TypeError, ValueError):
                s[pred_key] = None
            continue

        # 5. Feature Vector Construction
        # Extract daily lags
        features = [
            s[f"price_lag_{suffix}_{ft}"] 
            for suffix in ("1d", "2d", "3d", "7d")
        ]
        
        # Add intraday feature if horizon > 0
        feature_names = list(FEATURES_DAILY)
        if horizon > 0:
            features.append(current_price)
            feature_names.append(f"price_lag_{horizon}cell")

        # 6. Model Inference
        if any(f is None for f in features):
            continue

        try:
            # Wrap in DataFrame with correct column names for sklearn
            X = pd.DataFrame([features], columns=feature_names)
            model = load_model_for_horizon(ft, horizon)
            prediction = model.predict(X)[0]
            s[pred_key] = float(prediction)
        except Exception:
            # Model failures result in None prediction, handled downstream
            s[pred_key] = None


# ==========================================
# Public API
# ==========================================

def predict_for_fuel(
    model_input_list: List[Dict[str, Any]],
    fuel_type: str,
    prediction_key: Optional[str] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Predicts prices for a specific fuel type."""
    _predict_single_fuel(model_input_list, fuel_type, prediction_key, now)
    return model_input_list


def predict_all_fuels(
    model_input_list: List[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Predicts prices for E5, E10, and Diesel."""
    for ft in FUEL_TYPES:
        _predict_single_fuel(model_input_list, ft, now=now)
    return model_input_list