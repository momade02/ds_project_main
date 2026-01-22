"""
MODULE: UI Formatting — Stable, Null-Safe Presentation Helpers
--------------------------------------------------------------

Purpose
- Centralizes formatting and labeling logic used across Streamlit pages so that UI output is
  consistent, predictable, and robust to missing fields in station/run payloads.
- Provides backward-compatible helper aliases to avoid breaking existing call sites.

What this module does
- Fuel label mapping:
  - Maps UI-facing labels (e.g., "E5", "E10", "Diesel") to internal fuel codes used by the pipeline.
- Station identifiers:
  - Extracts a stable station UUID from dict-like objects, objects with `.get`, attribute-based objects,
    or raw UUID strings (`station_uuid`).
- Null-safe string handling:
  - Converts arbitrary values to display-safe text with a defined fallback (default "—") (`safe_text`).
- Formatting primitives:
  - Formats numeric values as domain-specific strings:
    - `fmt_price`: 3 decimals + "€/L"
    - `fmt_eur`: 2 decimals + "€"
    - `fmt_km`: 2 decimals + "km"
    - `fmt_min`: 1 decimal + "min"
    - `fmt_liters`: 1 decimal + "L"
- Shared explanatory UI copy:
  - Provides a single point of truth for describing the price basis shown in UI (`describe_price_basis`).
- Page 03 helper:
  - Implements a “smart price alert” heuristic based on hourly historical averages
    (`check_smart_price_alert`).

Design constraints
- Must remain side-effect free (pure helper module) and safe to import everywhere.
- Must preserve compatibility via the exported underscore-prefixed aliases (e.g., `_fmt_price`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# Fuel label mapping (UI label -> internal fuel code)
# -----------------------------------------------------------------------------

FUEL_LABEL_TO_CODE: Dict[str, str] = {
    "Diesel": "diesel",
    "E5": "e5",
    "E10": "e10",
}


def fuel_label_to_code(label: str) -> str:
    # Keep existing fallback behavior stable.
    return FUEL_LABEL_TO_CODE.get(label, "diesel")


# -----------------------------------------------------------------------------
# Safe text / IDs
# -----------------------------------------------------------------------------

def station_uuid(station: Any) -> str:
    """
    Best-effort station UUID extraction supporting:
      - dict-like objects
      - objects with .get
      - raw string UUIDs
    """
    if station is None:
        return ""

    if isinstance(station, str):
        return station.strip()

    try:
        if isinstance(station, dict) or hasattr(station, "get"):
            return str(
                station.get("uuid")
                or station.get("id")
                or station.get("station_uuid")
                or ""
            ).strip()
    except Exception:
        pass

    # Fallback: attribute access
    for attr in ("uuid", "id", "station_uuid"):
        try:
            v = getattr(station, attr)
            if v:
                return str(v).strip()
        except Exception:
            continue

    return ""


def safe_text(value: Any, *, fallback: str = "—") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


# -----------------------------------------------------------------------------
# Formatting primitives
# -----------------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def fmt_price(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.3f} €/L" if v is not None else "—"


def fmt_eur(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.2f} €" if v is not None else "—"


def fmt_km(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.2f} km" if v is not None else "—"


def fmt_min(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.1f} min" if v is not None else "—"


def fmt_liters(value: Any) -> str:
    v = _to_float(value)
    return f"{v:.1f} L" if v is not None else "—"


# -----------------------------------------------------------------------------
# “Format_*” aliases (keep call sites stable)
# -----------------------------------------------------------------------------

def format_price(value: Any) -> str:
    return fmt_price(value)


def format_eur(value: Any) -> str:
    return fmt_eur(value)


def format_km(value: Any) -> str:
    return fmt_km(value)


def format_min(value: Any) -> str:
    return fmt_min(value)


def format_liters(value: Any) -> str:
    return fmt_liters(value)


# -----------------------------------------------------------------------------
# Explanatory text
# -----------------------------------------------------------------------------

def describe_price_basis(use_economics: bool, litres_to_refuel: Optional[float]) -> str:
    """
    Centralized explanation used in both pages.
    Keep semantics identical to your existing descriptions.
    """
    if use_economics and litres_to_refuel is not None:
        return (
            "Prices shown are Tankerkönig real-time prices; predicted prices reflect "
            "your estimated arrival time. Net savings incorporate detour time and "
            f"refuelling volume (~{fmt_liters(litres_to_refuel)})."
        )
    return (
        "Prices shown are Tankerkönig real-time prices; predicted prices reflect "
        "your estimated arrival time. Ranking is based on predicted price only."
    )


# -----------------------------------------------------------------------------
# Page 03: Smart price alert helper (USED by pages/03_station_details.py)
# -----------------------------------------------------------------------------

def check_smart_price_alert(
    hourly_df: pd.DataFrame,
    current_hour: int,
    station_open_until_hour: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Check if there's a significant price drop coming soon.

    Returns dict with alert info if:
      - Price drop >= €0.03/L expected within 3 hours
      - Station is still open at that time (if station_open_until_hour provided)

    Returns None if no alert needed.
    """
    if hourly_df is None or hourly_df.empty:
        return None

    current_price = None
    for _, row in hourly_df.iterrows():
        if int(row.get("hour", -1)) == int(current_hour):
            current_price = row.get("avg_price")
            break

    if current_price is None or pd.isna(current_price):
        return None

    # Check next 3 hours
    for offset in range(1, 4):
        check_hour = (int(current_hour) + offset) % 24

        # Skip if station will be closed (best-effort; depends on how caller defines this)
        if station_open_until_hour is not None:
            try:
                if check_hour > int(station_open_until_hour):
                    continue
            except Exception:
                pass

        for _, row in hourly_df.iterrows():
            if int(row.get("hour", -1)) == check_hour:
                future_price = row.get("avg_price")
                if future_price is None or pd.isna(future_price):
                    continue

                price_drop = float(current_price) - float(future_price)
                if price_drop >= 0.03:
                    return {
                        "hours_to_wait": offset,
                        "drop_hour": check_hour,
                        "price_drop": price_drop,
                    }

    return None


# -----------------------------------------------------------------------------
# Backwards-compatible names (keep current call sites stable)
# -----------------------------------------------------------------------------

_station_uuid = station_uuid
_safe_text = safe_text

_fmt_price = fmt_price
_fmt_eur = fmt_eur
_fmt_km = fmt_km
_fmt_min = fmt_min
_fmt_liters = fmt_liters

_format_price = format_price
_format_eur = format_eur
_format_km = format_km
_format_min = format_min
_format_liters = format_liters

_describe_price_basis = describe_price_basis
_fuel_label_to_code = fuel_label_to_code
