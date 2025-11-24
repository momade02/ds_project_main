"""
Model loading utilities for fuel price prediction.

This module is responsible ONLY for loading and managing the trained models.
It does not know anything about routes, Supabase, or Streamlit – it just
provides model objects for E5, E10 and Diesel.

The trained models are expected to be stored as .joblib files in:

    src/modeling/models/

    - fuel_price_model_ARDL_e5.joblib
    - fuel_price_model_ARDL_e10.joblib
    - fuel_price_model_ARDL_diesel.joblib

Each model must implement a scikit-learn style `.predict(X)` interface, where
`X` is a pandas DataFrame with the four columns:

    ['price_lag_1d', 'price_lag_2d', 'price_lag_3d', 'price_lag_7d']

This is exactly what `predict.py` will prepare.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import joblib


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Allowed fuel types in the system
FUEL_TYPES = ("e5", "e10", "diesel")

# Mapping from fuel type to joblib filename inside src/modeling/models/
MODEL_FILENAMES: Dict[str, str] = {
    "e5": "fuel_price_model_ARDL_e5.joblib",
    "e10": "fuel_price_model_ARDL_e10.joblib",
    "diesel": "fuel_price_model_ARDL_diesel.joblib",
}

# Base directory of this file (src/modeling/)
_THIS_DIR = Path(__file__).resolve().parent
# Directory where the .joblib files reside
_MODELS_DIR = _THIS_DIR / "models"


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
        raise ValueError(f"Unsupported fuel_type '{fuel_type}'. "
                         f"Expected one of: {valid}")
    return ft


def get_model_path(fuel_type: str) -> Path:
    """
    Return the full filesystem path to the joblib file for a given fuel type.

    Parameters
    ----------
    fuel_type : str
        One of 'e5', 'e10', 'diesel' (case-insensitive).

    Returns
    -------
    Path
        Path object pointing to the .joblib file.

    Raises
    ------
    ValueError
        If the fuel_type is not supported.
    FileNotFoundError
        If the expected joblib file does not exist.
    """
    ft = _normalise_fuel_type(fuel_type)
    try:
        filename = MODEL_FILENAMES[ft]
    except KeyError as exc:
        raise ValueError(f"No model filename configured for fuel_type '{ft}'") from exc

    path = _MODELS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Check that the .joblib files are in src/modeling/models/."
        )
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=len(FUEL_TYPES))
def load_model(fuel_type: str) -> Any:
    """
    Load and cache the model for a specific fuel type.

    The model is loaded only once per process thanks to the LRU cache.
    Subsequent calls with the same fuel_type return the already loaded object.

    Parameters
    ----------
    fuel_type : str
        One of 'e5', 'e10', 'diesel' (case-insensitive).

    Returns
    -------
    Any
        The deserialised model object (e.g. a scikit-learn regressor).
    """
    path = get_model_path(fuel_type)
    # joblib.load will unpickle the model. The environment must have the
    # required libraries installed (e.g. statsmodels/scikit-learn).
    model = joblib.load(path)
    return model


def load_all_models() -> Dict[str, Any]:
    """
    Convenience helper to load all models at once.

    Returns
    -------
    Dict[str, Any]
        Mapping from fuel type ('e5', 'e10', 'diesel') to the loaded model.
    """
    return {ft: load_model(ft) for ft in FUEL_TYPES}
