"""
Module: Prediction Model Loader.

Description:
    This module handles the lifecycle of Machine Learning models used for fuel price prediction.
    It isolates the file system details and serialization logic from the application.

    Supported Models (ARDL - Autoregressive Distributed Lag):
    - Horizon 0 (h0): Daily-only features (used when no intraday price is available).
    - Horizon 1-4 (h1-h4): Intraday models (used for 30min-2hr lookaheads).

    Storage:
    - Path: `src/modeling/models/`
    - Format: `.joblib` serialization (Scikit-Learn).

Usage:
    Called by `predict.py` to acquire model instances.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Tuple, TypeAlias

import joblib

from src.app.app_errors import PredictionError

# ==========================================
# Type Definitions & Constants
# ==========================================

# A Generic Model object (usually sklearn.pipeline.Pipeline)
ModelObject: TypeAlias = Any 

# Allowed Fuel Types
FUEL_TYPES: Final[Tuple[str, ...]] = ("e5", "e10", "diesel")

# File System Paths
_MODULE_DIR: Final[Path] = Path(__file__).resolve().parent
MODELS_DIR: Final[Path] = _MODULE_DIR / "models"


# ==========================================
# Helpers: Path Resolution
# ==========================================

def _normalize_fuel_type(fuel_type: str) -> str:
    """Validates and normalizes fuel string (e.g. 'E5 ' -> 'e5')."""
    if not fuel_type:
        raise ValueError("Fuel type cannot be None or empty.")
    
    ft = fuel_type.lower().strip()
    if ft not in FUEL_TYPES:
        raise ValueError(f"Invalid fuel type '{ft}'. Valid: {FUEL_TYPES}")
    return ft


def _get_model_filename(fuel_type: str, horizon: int) -> str:
    """
    Constructs the standard filename for a specific model configuration.
    
    Naming Convention:
    - Horizon 0: '..._h0_daily.joblib'
    - Horizon N: '..._hN_Ncell.joblib'
    """
    if horizon == 0:
        suffix = "h0_daily"
    else:
        # Horizon matches the cell count in the filename
        suffix = f"h{horizon}_{horizon}cell"
        
    return f"fuel_price_model_ARDL_{fuel_type}_{suffix}.joblib"


def get_model_path_for_horizon(fuel_type: str, horizon: int) -> Path:
    """
    Resolves the absolute path for a requested model.
    Clips horizon to valid range [0, 4] automatically.
    """
    ft = _normalize_fuel_type(fuel_type)
    
    # Clip horizon to supported range
    h = max(0, min(int(horizon), 4))
    
    filename = _get_model_filename(ft, h)
    path = MODELS_DIR / filename
    
    if not path.is_file():
        # Raise generic error caught by app layer, but log specific path
        logging.error(f"Model missing at: {path}")
        raise FileNotFoundError(
            f"Model file not found for {ft} (horizon {h}). Expected at: {path}"
        )
        
    return path


# ==========================================
# Public API: Model Loading
# ==========================================

@lru_cache(maxsize=len(FUEL_TYPES) * 5)
def load_model_for_horizon(fuel_type: str, horizon: int) -> ModelObject:
    """
    Loads a specific horizon model from disk.
    
    Caching Strategy:
    - Uses `lru_cache` to ensure each .joblib file is read from disk only once 
      per process life-cycle.
    
    Raises:
        PredictionError: If the file is missing or corrupt.
    """
    path = get_model_path_for_horizon(fuel_type, horizon)
    
    try:
        model = joblib.load(path)
        return model
        
    except FileNotFoundError as exc:
        raise PredictionError(
            user_message="Prediction model files are missing.",
            remediation="Verify `src/modeling/models/` contains .joblib files.",
            details=f"Missing: {path.name}"
        ) from exc
        
    except Exception as exc:
        raise PredictionError(
            user_message="Failed to load prediction model.",
            remediation="Check library compatibility (scikit-learn/joblib versions).",
            details=f"Load error ({fuel_type}, h={horizon}): {exc}"
        ) from exc


@lru_cache(maxsize=len(FUEL_TYPES))
def load_model(fuel_type: str) -> ModelObject:
    """
    Legacy/Default Loader: Returns the Daily-Only (Horizon 0) model.
    Used when specific horizon logic is not applicable.
    """
    return load_model_for_horizon(fuel_type, horizon=0)