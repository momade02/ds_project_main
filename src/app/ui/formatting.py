# src/app/ui/formatting.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional


# --- Fuel label mapping (UI label -> internal fuel code) ---------------------

FUEL_LABEL_TO_CODE: Dict[str, str] = {
    "Diesel": "diesel",
    "E5": "e5",
    "E10": "e10",
}


def fuel_label_to_code(label: str) -> str:
    return FUEL_LABEL_TO_CODE.get(label, "diesel")


# --- Safe text / IDs --------------------------------------------------------

_UUID_RE = re.compile(r"^[0-9a-fA-F-]{8,}$")


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
        if isinstance(station, dict):
            return str(
                station.get("uuid")
                or station.get("id")
                or station.get("station_uuid")
                or ""
            ).strip()
        # dict-like / pydantic-like
        if hasattr(station, "get"):
            return str(
                station.get("uuid")
                or station.get("id")
                or station.get("station_uuid")
                or ""
            ).strip()
    except Exception:
        pass

    # fallback: attribute access
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


# --- Formatting primitives ---------------------------------------------------

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


# --- “Format_*” aliases (keep current call sites stable) --------------------

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


# --- Explanatory text -------------------------------------------------------

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


# --- Backwards-compatible names (so Step 1 requires minimal edits) ----------

# streamlit_app.py / station_details.py currently use these private names.
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
