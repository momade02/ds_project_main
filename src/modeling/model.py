"""
Model loading utilities for fuel price prediction.

This module is responsible ONLY for loading and managing the trained models.
It does not know anything about routes, Supabase, or Streamlit – it just
provides model objects for E5, E10 and Diesel.

The trained models are expected to be stored as .joblib files in:

    src/modeling/models/

For each fuel type we now have 5 ARDL models for different horizons (in
30-minute time cells):

    - h0: daily-only model (no intraday price):
        fuel_price_model_ARDL_{fuel}_h0_daily.joblib

    - h1..h4: models that additionally use today's current price as an
      intraday feature, for 1–4 cells ahead:

        fuel_price_model_ARDL_{fuel}_h1_1cell.joblib
        fuel_price_model_ARDL_{fuel}_h2_2cell.joblib
        fuel_price_model_ARDL_{fuel}_h3_3cell.joblib
        fuel_price_model_ARDL_{fuel}_h4_4cell.joblib

Each model must implement a scikit-learn style `.predict(X)` interface, where
`X` is a pandas DataFrame with:

    - horizons h1–h4:    daily lags + current price for that fuel
    - horizon  h0:       daily lags only

The horizon logic (which model to use) is handled in `predict.py`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib

from src.app.app_errors import PredictionError


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Allowed fuel types in the system
FUEL_TYPES = ("e5", "e10", "diesel")

# Base directory of this file (src/modeling/)
_THIS_DIR = Path(__file__).resolve().parent
# Directory where the .joblib files reside
_MODELS_DIR = _THIS_DIR / "models"


def _build_filename(fuel_type: str, horizon: int) -> str:
    """
    Build the expected joblib filename for a given fuel and horizon.

    Horizon is the number of 30-minute cells ahead:

        0 -> daily-only model (no intraday cell):  h0_daily
        1..4 -> models using current price for 1..4 cells ahead: h{h}_{h}cell
    """
    if horizon == 0:
        suffix = "h0_daily"
    else:
        suffix = f"h{horizon}_{horizon}cell"
    return f"fuel_price_model_ARDL_{fuel_type}_{suffix}.joblib"


# Pre-compute the filename mapping for all fuels and horizons (0..4)
MODEL_FILENAMES_HORIZON: Dict[Tuple[str, int], str] = {}
for ft in FUEL_TYPES:
    for h in range(0, 5):
        MODEL_FILENAMES_HORIZON[(ft, h)] = _build_filename(ft, h)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalise_fuel_type(fuel_type: str) -> str:
    """
    Normalise and validate a fuel type string.

    Accepts 'e5', 'E5', 'E10', 'diesel', etc., and returns a lowercase
    canonical form. Raises ValueError for unsupported fuel types.
    """
    if fuel_type is None:
        raise ValueError("fuel_type must not be None")

    ft = fuel_type.lower().strip()
    if ft not in FUEL_TYPES:
        valid = ", ".join(FUEL_TYPES)
        raise ValueError(
            f"Unsupported fuel_type '{fuel_type}'. Expected one of: {valid}"
        )
    return ft


def get_model_path_for_horizon(fuel_type: str, horizon: int) -> Path:
    """
    Return the filesystem path to the joblib file for a fuel + horizon.

    Parameters
    ----------
    fuel_type : str
        One of 'e5', 'e10', 'diesel' (case-insensitive).
    horizon : int
        Number of 30-minute cells ahead. Values outside [0, 4] are
        automatically clipped to this range.

    Returns
    -------
    Path
        Path object pointing to the .joblib file.
    """
    ft = _normalise_fuel_type(fuel_type)
    h = int(horizon)
    if h < 0:
        h = 0
    if h > 4:
        h = 4

    try:
        filename = MODEL_FILENAMES_HORIZON[(ft, h)]
    except KeyError as exc:
        raise ValueError(
            f"No model filename configured for fuel_type='{ft}', horizon={h}"
        ) from exc

    path = _MODELS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Check that the .joblib files are in src/modeling/models/."
        )
    return path


def get_model_path(fuel_type: str) -> Path:
    """
    Backwards-compatible helper: return the *daily-only* model path (horizon=0).

    Existing code that does not know about horizons can still call this.
    """
    return get_model_path_for_horizon(fuel_type, horizon=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=len(FUEL_TYPES) * 5)
def load_model_for_horizon(fuel_type: str, horizon: int) -> Any:
    """
    Load and cache the model for a specific fuel type and horizon.

    The model is loaded only once per process thanks to the LRU cache.

    Parameters
    ----------
    fuel_type : str
        One of 'e5', 'e10', 'diesel' (case-insensitive).
    horizon : int
        Number of 30-minute cells ahead (0..4, will be clipped).

    Returns
    -------
    Any
        The deserialised model object.
    """
    path = get_model_path_for_horizon(fuel_type, horizon)
    try:
        model = joblib.load(path)
    except FileNotFoundError as exc:
        raise PredictionError(
            user_message="Prediction model files are missing.",
            remediation="Ensure the trained .joblib files exist in src/modeling/models/ (or update the configured model path).",
            details=f"Missing model file: {path}",
        ) from exc
    except Exception as exc:
        raise PredictionError(
            user_message="Failed to load the prediction model.",
            remediation="Check model files and Python dependencies (joblib / sklearn versions).",
            details=f"Model load error for {fuel_type=} {horizon=}: {exc}",
        ) from exc
    return model


@lru_cache(maxsize=len(FUEL_TYPES))
def load_model(fuel_type: str) -> Any:
    """
    Backwards-compatible wrapper for the horizon-0 (daily-only) model.

    If you explicitly want horizon-specific models, call
    `load_model_for_horizon(fuel_type, horizon)` instead.
    """
    return load_model_for_horizon(fuel_type, horizon=0)


def load_all_models() -> Dict[str, Any]:
    """
    Convenience helper to load the horizon-0 models for all fuels at once.
    """
    return {ft: load_model(ft) for ft in FUEL_TYPES}
