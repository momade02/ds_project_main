"""
Prediction helpers for fuel price models.

This module takes the list of station dictionaries produced by the
integration pipeline and:

  1. For each station and fuel, decides which horizon model (h0..h4) to use.
  2. Builds the feature vector with the **generic** column names the models
     were trained on (price_lag_1d, ..., price_lag_7d, price_lag_kcell).
  3. Calls the corresponding ARDL model.
  4. Writes the prediction back into the station dict.

Conceptually, the decision is now **minutes-based**:

    minutes_ahead = (eta_utc - now_utc) in minutes

* If minutes_ahead <= ETA_THRESHOLD_MIN (default: 10 min) and a current
  price is available, we simply use the current price (no model call).

* If no current price is available, we fall back to the **daily-only**
  model (h0), which uses only daily lags.

* Otherwise (arrival genuinely in the future and current price exists),
  we map minutes_ahead to a discrete horizon h:

      h_raw = ceil(minutes_ahead / 30)
      if 1 <= h_raw <= 4 -> h = h_raw  (intraday horizons)
      if h_raw > 4       -> h = 0      (daily-only; too far ahead)

For historical / demo cases without ETA we fall back to the older
cell-based logic, using:

    cells_ahead = station.time_cell - current_time_cell

clipped to horizons {1,2,3,4} with >4 cells again mapped to h=0.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional

import pandas as pd

from .model import FUEL_TYPES, load_model_for_horizon


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalise_fuel_type(fuel_type: str) -> str:
    """Lower-case and validate fuel type."""
    if fuel_type is None:
        raise ValueError("fuel_type must not be None")

    ft = fuel_type.lower().strip()
    if ft not in FUEL_TYPES:
        valid = ", ".join(FUEL_TYPES)
        raise ValueError(
            f"Unsupported fuel_type '{fuel_type}'. Expected one of: {valid}"
        )
    return ft


def _get_current_time_cell(now: Optional[datetime] = None) -> int:
    """
    Compute the current 30-minute time cell (0..47), **in UTC**.

    If `now` is provided and timezone-aware, it is converted to UTC.
    If `now` is naive (no tzinfo), we treat it as already being in UTC.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    else:
        if now.tzinfo is not None:
            now = now.astimezone(timezone.utc)
        # else: assume `now` is already in UTC

    return now.hour * 2 + (1 if now.minute >= 30 else 0)


def _clip_horizon(cells_ahead: int) -> int:
    """
    Map the number of 30-minute blocks ahead to a supported horizon.

    This is used only as a **fallback** when ETA is not available.

    Rules
    -----
    * cells_ahead <= 0      -> 0  (daily-only model, no intraday feature)
    * 1 <= cells_ahead <= 4 -> that horizon (h1..h4)
    * cells_ahead > 4       -> 0  (fallback to daily-only model)
    """
    if cells_ahead <= 0:
        return 0
    if cells_ahead > 4:
        return 0
    return cells_ahead


def _has_daily_lags(station: Dict[str, Any], fuel: str) -> bool:
    """
    Check whether all required daily lag features for a given fuel are present.

    We require: price_lag_1d_{fuel}, price_lag_2d_{fuel},
                price_lag_3d_{fuel}, price_lag_7d_{fuel}.
    """
    for suffix in ("1d", "2d", "3d", "7d"):
        key = f"price_lag_{suffix}_{fuel}"
        if station.get(key) is None:
            return False
    return True


# Threshold in minutes within which we treat the station as "now"
# instead of running a model in the same 30-min block.
ETA_THRESHOLD_MIN = 10.0


def _minutes_to_arrival(eta_value: Any, now: Optional[datetime]) -> Optional[float]:
    """
    Compute minutes from `now` until `eta_value`.

    `eta_value` may be a datetime or an ISO-8601 string (with or without
    timezone). We normalise both to UTC. On any parsing problem we return
    None and let the caller fall back to a simpler rule.
    """
    if eta_value is None or now is None:
        return None

    # Normalise "now" to UTC (same convention as _get_current_time_cell)
    if now.tzinfo is not None:
        now_utc = now.astimezone(timezone.utc)
    else:
        now_utc = now  # treat naive datetime as already UTC

    try:
        if isinstance(eta_value, datetime):
            eta_dt = eta_value
        else:
            s = str(eta_value).strip()
            # Strip trailing "Z" if present
            if s.endswith("Z"):
                s = s[:-1]
            eta_dt = datetime.fromisoformat(s)

        if eta_dt.tzinfo is not None:
            eta_dt = eta_dt.astimezone(timezone.utc)
        else:
            # Assume same timezone as now_utc (UTC)
            eta_dt = eta_dt.replace(tzinfo=timezone.utc)

        delta_min = (eta_dt - now_utc).total_seconds() / 60.0
        return max(delta_min, 0.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def _predict_single_fuel(
    model_input_list: List[Dict[str, Any]],
    fuel_type: str,
    prediction_key: Optional[str] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Predict prices for one fuel type and write results into the station dicts.

    Additionally, this function annotates each station with debug information:

        debug_<fuel>_current_time_cell   : int (0..47)
        debug_<fuel>_cells_ahead_raw     : int or None
        debug_<fuel>_minutes_ahead       : float or None
        debug_<fuel>_horizon_used        : int in {0..4} or None
        debug_<fuel>_used_current_price  : bool

    Parameters
    ----------
    model_input_list :
        List of station dictionaries produced by the integration pipeline.
    fuel_type :
        'e5', 'e10' or 'diesel' (case-insensitive).
    prediction_key :
        Optional custom key name to store the prediction. If None, a default
        of the form 'pred_price_{fuel}' is used.
    now :
        Optional reference time. Mainly for testing. If None, uses current
        time (UTC convention as in `_get_current_time_cell`).

    Returns
    -------
    list of dict
        The same list, modified in-place with predictions and debug fields.
    """
    if not model_input_list:
        return model_input_list

    ft = _normalise_fuel_type(fuel_type)
    pred_key = prediction_key or f"pred_price_{ft}"

    # Time bookkeeping
    current_time_cell = _get_current_time_cell(now)
    current_price_key = f"price_current_{ft}"

    daily_cols_generic = ["price_lag_1d", "price_lag_2d", "price_lag_3d", "price_lag_7d"]

    # Debug field names
    debug_ct_key = f"debug_{ft}_current_time_cell"
    debug_ca_key = f"debug_{ft}_cells_ahead_raw"
    debug_ma_key = f"debug_{ft}_minutes_ahead"
    debug_h_key = f"debug_{ft}_horizon_used"
    debug_ucp_key = f"debug_{ft}_used_current_price"

    # Attach current time cell for all stations (useful for debugging)
    for station in model_input_list:
        station[debug_ct_key] = current_time_cell

    for station in model_input_list:
        # Ensure prediction key exists, default None
        if pred_key not in station:
            station[pred_key] = None

        time_cell = station.get("time_cell")
        if time_cell is None:
            continue

        # Need all daily lags for this fuel
        if not _has_daily_lags(station, ft):
            continue

        current_price = station.get(current_price_key)
        cells_ahead_raw = time_cell - current_time_cell
        station[debug_ca_key] = cells_ahead_raw

        # Compute minutes to arrival (may be None for historical/demo)
        minutes_to_arrival = _minutes_to_arrival(station.get("eta"), now)
        station[debug_ma_key] = minutes_to_arrival

        # ------------------------------------------------------------------
        # Case 1: arrival very soon -> use current price instead of model
        # ------------------------------------------------------------------
        use_current_price = False
        if current_price is not None:
            if minutes_to_arrival is not None:
                # Refined rule: treat as "now" if arrival is within threshold
                if minutes_to_arrival <= ETA_THRESHOLD_MIN:
                    use_current_price = True
            else:
                # Historical / demo path without ETA: fall back to old rule
                if cells_ahead_raw <= 0:
                    use_current_price = True

        if use_current_price:
            try:
                station[pred_key] = float(current_price)
                station[debug_ucp_key] = True
                station[debug_h_key] = None
            except (TypeError, ValueError):
                station[pred_key] = None
            continue

        # ------------------------------------------------------------------
        # Case 2: need to run a model -> decide horizon + feature columns
        # ------------------------------------------------------------------
        if current_price is None:
            # No current price: daily-only model (h0)
            horizon = 0
            feature_cols = daily_cols_generic
        else:
            # We have a current price and arrival is sufficiently far ahead.
            if minutes_to_arrival is not None:
                # Minutes-based horizon selection.
                # h_raw is the number of 30-min blocks ahead, rounded up.
                h_raw = math.ceil(minutes_to_arrival / 30.0)

                if h_raw <= 0:
                    # Defensive; in practice minutes_to_arrival > ETA_THRESHOLD_MIN.
                    horizon = 1
                elif 1 <= h_raw <= 4:
                    horizon = h_raw
                else:
                    # Too far ahead (> 4 blocks ~ 2h): fall back to daily-only.
                    horizon = 0
            else:
                # Fallback if ETA is missing: use cell-based logic.
                cells_ahead = max(1, cells_ahead_raw)
                horizon = _clip_horizon(cells_ahead)

            if horizon == 0:
                feature_cols = daily_cols_generic
            else:
                intraday_col = f"price_lag_{horizon}cell"
                feature_cols = daily_cols_generic + [intraday_col]

        station[debug_h_key] = horizon

        # Build feature values mapped to generic names
        values: List[float] = []

        for suffix in ("1d", "2d", "3d", "7d"):
            key = f"price_lag_{suffix}_{ft}"  # e.g. price_lag_1d_e5
            values.append(station.get(key))

        if horizon > 0:
            # For horizons 1..4 we use the observed current price as
            # the intraday proxy feature (price_lag_kcell).
            values.append(current_price)

        if any(v is None for v in values):
            # Incomplete feature vector; skip prediction for this station
            continue

        X = pd.DataFrame([values], columns=feature_cols)
        model = load_model_for_horizon(ft, horizon)
        pred = model.predict(X)

        try:
            station[pred_key] = float(pred[0])
        except (TypeError, ValueError, IndexError):
            station[pred_key] = None

    return model_input_list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_for_fuel(
    model_input_list: List[Dict[str, Any]],
    fuel_type: str,
    prediction_key: Optional[str] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Public wrapper to predict prices for a single fuel type.
    """
    return _predict_single_fuel(
        model_input_list=model_input_list,
        fuel_type=fuel_type,
        prediction_key=prediction_key,
        now=now,
    )


def predict_all_fuels(
    model_input_list: List[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Predict prices for all configured fuel types (E5, E10, Diesel).

    Adds the following keys to each station dict:

        pred_price_e5
        pred_price_e10
        pred_price_diesel
    """
    for ft in FUEL_TYPES:
        _predict_single_fuel(
            model_input_list=model_input_list,
            fuel_type=ft,
            prediction_key=None,
            now=now,
        )
    return model_input_list
