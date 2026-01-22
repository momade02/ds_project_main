"""
Model loading utilities for fuel price forecasting (ARDL joblib artifacts).

Purpose
-------
This module isolates filesystem and serialization concerns for the project’s
forecasting models. It provides a small, stable API to resolve and load the
correct model artifact for a given fuel type and forecast horizon.

Model family
------------
- Fuel types: e5, e10, diesel
- Horizons:
  - h0 (daily-only): used when intraday lookahead is not applicable
  - h1–h4 (intraday): correspond to 30–120 minute lookaheads (in 30-min steps)

Artifact conventions
--------------------
- Storage directory: src/modeling/models/
- File naming:
  - Daily model: fuel_price_model_ARDL_<fuel>_h0_daily.joblib
  - Intraday:     fuel_price_model_ARDL_<fuel>_h<h>_<h>cell.joblib

Implementation details
----------------------
- Horizon values are clipped to the supported range [0, 4].
- Models are cached with `lru_cache` so each artifact is only loaded once per
  process lifetime.
- Errors are wrapped into project-level exceptions to provide actionable UI
  messages (missing or incompatible model artifacts).

Usage
-----
Imported by the prediction helpers (predict.py) to load the proper model for each
station’s horizon decision.
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