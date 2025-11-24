"""
Prediction utilities for fuel price models.

This module connects the integration layer with the trained models:

    - It receives the output from
      `src.integration.route_tankerkoenig_integration.get_fuel_prices_for_route`
      (a list of dictionaries, one per station).

    - It prepares model features for each fuel type (E5, E10, Diesel) by
      selecting and renaming the four lag columns the ARDL models were trained
      on.

    - It calls the corresponding model's `.predict(X)` method and attaches
      the predictions back to the original list of station dicts.

The same functions work for:

    - The *test environment* (example stations coming from `run_example()`)
    - The *real system* (stations coming from `get_fuel_prices_for_route()`
      once ORS POI returns stations again)

No database or API access happens here – everything operates purely on
in-memory Python objects.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from .model import load_model, FUEL_TYPES


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_model_input(model_input_list: List[Dict], fuel_type: str) -> pd.DataFrame:
    """
    Convert integration output to a model-ready DataFrame for a given fuel type.

    This is the programmatic version of the helper previously shown in
    `simple_usage_guide.ipynb`. It:

        1) Converts the list of station dicts into a DataFrame
        2) Selects the four lag columns for the specified fuel
        3) Renames them to generic names that match the training data

    Parameters
    ----------
    model_input_list : list of dict
        Output from `get_fuel_prices_for_route()` or `run_example()`.
    fuel_type : str
        One of 'e5', 'e10', or 'diesel' (case-insensitive).

    Returns
    -------
    pandas.DataFrame
        DataFrame with four columns:

            ['price_lag_1d', 'price_lag_2d', 'price_lag_3d', 'price_lag_7d']

        Rows correspond to the stations in `model_input_list` in the same
        order, including rows with missing values (NaN) if some lags are
        missing. The prediction functions will handle missing rows.
    """
    if not model_input_list:
        # Empty route / no stations – return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=["price_lag_1d", "price_lag_2d", "price_lag_3d", "price_lag_7d"]
        )

    df = pd.DataFrame(model_input_list)

    fuel_type = fuel_type.lower().strip()
    if fuel_type not in FUEL_TYPES:
        valid = ", ".join(FUEL_TYPES)
        raise ValueError(f"Unsupported fuel_type '{fuel_type}'. Expected one of: {valid}")

    cols = [
        f"price_lag_1d_{fuel_type}",
        f"price_lag_2d_{fuel_type}",
        f"price_lag_3d_{fuel_type}",
        f"price_lag_7d_{fuel_type}",
    ]

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing expected columns for fuel_type '{fuel_type}': {missing_cols}. "
            "Check that the integration layer produced all lag columns."
        )

    # Select and rename columns to the feature names used in training
    model_features = df[cols].copy()
    model_features.columns = [
        "price_lag_1d",
        "price_lag_2d",
        "price_lag_3d",
        "price_lag_7d",
    ]

    return model_features


# ---------------------------------------------------------------------------
# Core prediction helpers
# ---------------------------------------------------------------------------

def _predict_single_fuel(
    model_input: List[Dict],
    fuel_type: str,
    prediction_key: Optional[str] = None,
) -> List[Dict]:
    """
    Internal helper: run prediction for a single fuel type and attach results.

    Parameters
    ----------
    model_input : list of dict
        Stations with lag price information.
    fuel_type : str
        One of 'e5', 'e10', 'diesel' (case-insensitive).
    prediction_key : str, optional
        Name of the key under which the prediction shall be stored in each
        station dict. If None, defaults to 'pred_price_{fuel_type}'.

    Returns
    -------
    list of dict
        The same list object that was passed in, but with each station dict
        extended by the prediction key.
    """
    if prediction_key is None:
        prediction_key = f"pred_price_{fuel_type.lower()}"

    if not model_input:
        # Nothing to do – return as is
        return model_input

    # Build DataFrame of features
    features = prepare_model_input(model_input, fuel_type)
    if features.empty:
        # No features → cannot predict
        for station in model_input:
            station[prediction_key] = None
        return model_input

    # Determine which rows have complete data (no NaNs in the four lag columns)
    complete_mask = features.notna().all(axis=1)
    if not complete_mask.any():
        # No station has full lag information – attach None everywhere
        for station in model_input:
            station[prediction_key] = None
        return model_input

    # Load the model for this fuel type
    model = load_model(fuel_type)

    # Predict only for rows with complete data
    X_complete = features[complete_mask]
    preds = model.predict(X_complete)

    # Ensure we have a 1-D iterable of floats
    preds = np.asarray(preds).ravel()

    # Attach predictions back to the original list
    # We iterate over indices so that order is preserved.
    pred_idx = 0
    for i, station in enumerate(model_input):
        if complete_mask.iloc[i]:
            value = float(preds[pred_idx])
            pred_idx += 1
        else:
            # Incomplete feature set for this station – no prediction
            value = None
        station[prediction_key] = value

    return model_input


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------

def predict_for_fuel(
    model_input: List[Dict],
    fuel_type: str,
    prediction_key: Optional[str] = None,
) -> List[Dict]:
    """
    Public wrapper to obtain predictions for a single fuel type.

    Typical usage (after integration):

    >>> from src.integration.route_tankerkoenig_integration import get_fuel_prices_for_route
    >>> from src.modeling.predict import predict_for_fuel
    >>> model_input = get_fuel_prices_for_route(..., use_realtime=False)
    >>> model_input = predict_for_fuel(model_input, "e5")

    Parameters
    ----------
    model_input : list of dict
        Output from `get_fuel_prices_for_route()` or `run_example()`.
    fuel_type : str
        One of 'e5', 'e10', 'diesel' (case-insensitive).
    prediction_key : str, optional
        Custom key name for predictions inside each station dict. If omitted,
        defaults to 'pred_price_{fuel_type}'.

    Returns
    -------
    list of dict
        Same list with an extra field per station containing the prediction.
    """
    return _predict_single_fuel(model_input, fuel_type, prediction_key)


def predict_all_fuels(model_input: List[Dict]) -> List[Dict]:
    """
    Convenience function to predict prices for all supported fuel types.

    This function sequentially calls `predict_for_fuel` for 'e5', 'e10' and
    'diesel', storing predictions in the keys:

        - 'pred_price_e5'
        - 'pred_price_e10'
        - 'pred_price_diesel'

    Parameters
    ----------
    model_input : list of dict
        Output from `get_fuel_prices_for_route()` or `run_example()`.

    Returns
    -------
    list of dict
        The same list, augmented with prediction fields for each fuel type.
    """
    for fuel in FUEL_TYPES:
        predict_for_fuel(model_input, fuel)
    return model_input
