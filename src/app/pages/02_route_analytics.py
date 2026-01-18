"""
Route Analytics (Page 2)

Purpose:
- Provide auditability and comparability for a single route run.
- Explain why a station was recommended and why others were excluded.
- Offer drill-down navigation into Station Details.

Data source:
- st.session_state["last_run"] written by src/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import altair as alt 

from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None
    

# ---------------------------------------------------------------------
# Minimum plausible €/L (guards against 0.00 or corrupted values in cached runs)
MIN_VALID_PRICE_EUR_L: float = 0.50
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Path setup (compatible with running from /src/app/pages via Streamlit)
# ---------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parents[1]        # .../src/app
PROJECT_ROOT = Path(__file__).resolve().parents[3]    # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config.settings import load_env_once

load_env_once()

from ui.formatting import (
    _station_uuid,
    _safe_text,
    _fmt_price,
    _fmt_eur,
    _fmt_km,
    _fmt_min,
)
from services.presenters import build_ranking_dataframe

from ui.styles import apply_app_css

from ui.sidebar import render_sidebar_shell

from config.settings import ensure_persisted_state_defaults
from services.session_store import init_session_context, restore_persisted_state, maybe_persist_state


# -----------------------------
# Helpers
# -----------------------------

    # ---------------------------------------------------------------------
    # Compatibility helpers (used by Page 02 constraint/summary blocks)
    # Keep these thin wrappers so the section code stays readable.
    # ---------------------------------------------------------------------
def _format_km(x: Any) -> str:
    return _fmt_km(x)


def _format_min(x: Any) -> str:
    return _fmt_min(x)


def _format_eur(x: Any) -> str:
    return _fmt_eur(x)


def _format_liters(x: Any) -> str:
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return "—"
        return f"{float(x):.2f} L"
    except (TypeError, ValueError):
        return "—"


def render_metric_grid(items: List[Tuple[str, str]], cols: int = 2) -> None:
    """
    Simple responsive metric grid:
    - On desktop, shows `cols` columns.
    - On mobile, Streamlit will naturally stack columns.
    """
    if not items:
        return

    cols = max(1, int(cols))
    row = st.columns(cols)

    for i, (label, value) in enumerate(items):
        row[i % cols].metric(label, value)
        if (i % cols) == (cols - 1) and i != (len(items) - 1):
            row = st.columns(cols)


def render_card_grid(items: List[Tuple[str, str]], cols: int = 2) -> None:
    """
    Responsive card grid (uses the same visual card language as the Start/Destination blocks).
    - Desktop: `cols` columns
    - Mobile: Streamlit stacks columns automatically
    """
    if not items:
        return

    cols = max(1, int(cols))

    # CSS is injected once per run (safe); class names are page-local
    st.markdown(
        """
        <style>
          .p02-card {
            width: 100%;
            padding: 0.5rem 1.0rem;
            border: 2px solid rgba(49, 51, 63, 0.3);
            border-radius: 0.9rem;
            text-align: center;
            margin-bottom: 0.25rem;
            margin-top: 0rem;
          }
          .p02-card-label {
            font-size: 0.9rem;
            opacity: 0.78;
            margin: 0 0 0.25rem 0;
            line-height: 1.15;
          }
          .p02-card-value {
            font-size: 1.3rem;
            font-weight: 750;
            margin: 0;
            line-height: 1.15;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    row = st.columns(cols)
    for i, (label, value) in enumerate(items):
        with row[i % cols]:
            st.markdown(
                f"""
                <div class="p02-card">
                  <div class="p02-card-label">{label}</div>
                  <div class="p02-card-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if (i % cols) == (cols - 1) and i != (len(items) - 1):
            row = st.columns(cols)


def render_key_visualization(
    *,
    route_info: Dict[str, Any],
    best_station: Optional[Dict[str, Any]],
) -> None:
    """
    Page 02 / Recommended Route:
    Two proportional route bars (baseline vs alternative via best station),
    with a station marker on the alternative bar.
    """

    # --- baseline route metrics (best-effort key fallbacks) ---
    baseline_km = _as_float(route_info.get("route_km") or route_info.get("distance_km") or route_info.get("km"))
    baseline_min = _as_float(route_info.get("route_min") or route_info.get("duration_min") or route_info.get("min"))

    # --- alternative route metrics (baseline + detour deltas) ---
    detour_km = None
    detour_min = None
    if isinstance(best_station, dict) and best_station:
        dk, dm = _detour_metrics(best_station)
        detour_km = dk
        detour_min = dm

    alt_km = (baseline_km + detour_km) if (baseline_km is not None and detour_km is not None) else None
    alt_min = (baseline_min + detour_min) if (baseline_min is not None and detour_min is not None) else None

    # --- station marker position on alternative line (prefer fraction_of_route) ---
    station_frac = None
    if isinstance(best_station, dict) and best_station:
        station_frac = _as_float(best_station.get("fraction_of_route"))

        # fallback: derive from distance_along_m if total route length is known
        if station_frac is None:
            dist_along_m = _as_float(best_station.get("distance_along_m"))
            route_dist_m = _as_float(
                route_info.get("route_distance_m")
                or route_info.get("distance_m")
                or route_info.get("route_m")
            )
            if dist_along_m is not None and route_dist_m not in (None, 0.0):
                station_frac = dist_along_m / route_dist_m

    # clamp to [0, 1]
    if station_frac is not None:
        station_frac = max(0.0, min(1.0, float(station_frac)))

    # --- proportional widths: longest bar spans full container width ---
    lengths = [x for x in [baseline_km, alt_km] if x is not None and x > 0]
    denom = max(lengths) if lengths else 1.0

    baseline_w = (baseline_km / denom) if (baseline_km is not None and baseline_km > 0) else 0.0
    alt_w = (alt_km / denom) if (alt_km is not None and alt_km > 0) else 0.0

    # if we do not have a best station, we still show the baseline bar and a placeholder alternative bar
    has_alt = (alt_km is not None and alt_w > 0.0)

    # --- display strings ---
    baseline_text = f"Distance: {_format_km(baseline_km)} · Time: {_format_min(baseline_min)}"
    alt_line1: str
    alt_line2: Optional[str] = None

    if has_alt:
        alt_line1 = f"Distance: {_format_km(alt_km)} · Time: {_format_min(alt_min)}"

        # Detour as % of baseline distance
        detour_pct = None
        if baseline_km not in (None, 0.0) and detour_km is not None:
            detour_pct = (float(detour_km) / float(baseline_km)) * 100.0

        if detour_pct is not None:
            alt_line2 = f"Detour: {_format_km(detour_km)} / {detour_pct:.2f}% of baseline"
        else:
            alt_line2 = f"Detour: {_format_km(detour_km)} / —% of baseline"
    else:
        alt_line1 = "Alternative route is unavailable (no recommended station cached)."
        alt_line2 = None

    # station marker label (optional)
    station_label = ""
    if isinstance(best_station, dict) and best_station:
        name = best_station.get("brand") or best_station.get("station_name") or best_station.get("name") or "Best station"
        station_label = _safe_text(name)

    # --- render (page-local CSS; uses your map colors) ---
    st.markdown(
        """
        <style>
          .p02-kv { width: 100%; margin-top: -2rem; }
          .p02-kv-row { margin: 0.65rem 0 1.05rem 0; }
          .p02-kv-subtitle { font-weight: 700; margin: 0 0 0.35rem 0; }
          .p02-kv-track { width: 100%; }
          .p02-kv-line {
            position: relative;
            height: 0;
            border-top-width: 6px;
            border-top-style: solid;
            border-radius: 999px;
            opacity: 0.95;
          }
          .p02-kv-line.baseline { border-top-color: rgba(30, 144, 255, 1.0); }
          .p02-kv-line.alt { border-top-color: rgba(148, 0, 211, 1.0); border-top-style: dashed; }

            .p02-kv-dot {
            position: absolute;
            top: -12px;
            width: 18px;
            height: 18px;
            border-radius: 999px;
            background: rgba(220, 20, 60, 1.0);  /* red dot */
            border: 2px solid rgba(255, 255, 255, 0.96);
            box-shadow: 0 2px 10px rgba(0,0,0,0.20);
            transform: translateX(-50%);
            }

            .p02-kv-detail { margin: 0.40rem 0 0 0; opacity: 0.85; line-height: 1.25; }
            .p02-kv-detail.detour { opacity: 0.78; margin-top: 0.20rem; }

          /* Mobile: slightly thicker separation, keep readability */
          @media (max-width: 600px) {
            .p02-kv-row { margin-bottom: 1.20rem; }
            .p02-kv-line { border-top-width: 7px; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Build HTML without indentation to avoid Markdown "code block" behavior
    baseline_bar = (
        f"<div class='p02-kv-line baseline' style='width:{baseline_w*100:.1f}%;'></div>"
        if baseline_w > 0
        else "<div class='p02-kv-line baseline' style='width:0%;'></div>"
    )

    if has_alt:
        dot_html = ""
        if station_frac is not None:
            dot_html = f"<span class='p02-kv-dot' style='left:{station_frac*100:.1f}%;'></span>"
        alt_bar = f"<div class='p02-kv-line alt' style='width:{alt_w*100:.1f}%;'>{dot_html}</div>"
    else:
        alt_bar = "<div class='p02-kv-line alt' style='width:0%;'></div>"

    html_block = (
        "<div class='p02-kv'>" +
        "<div class='p02-kv-row'>" +
        "<div class='p02-kv-subtitle'>Baseline Google route:</div>" +
        f"<div class='p02-kv-track'>{baseline_bar}</div>" +
        f"<div class='p02-kv-detail'>{_safe_text(baseline_text)}</div>" +
        "</div>" +
        "<div class='p02-kv-row'>" +
        "<div class='p02-kv-subtitle'>Alternative route with best station:</div>" +
        f"<div class='p02-kv-track'>{alt_bar}</div>" +
        f"<div class='p02-kv-detail'>{_safe_text(alt_line1)}</div>" +
            (
                f"<div class='p02-kv-detail detour'>{_safe_text(alt_line2)}</div>"
                if alt_line2 else
                ""
            ) +
        "</div>" +
        "</div>"
    )

    st.markdown(html_block, unsafe_allow_html=True)


def _get_last_run() -> Optional[Dict[str, Any]]:
    cached = st.session_state.get("last_run")
    if not isinstance(cached, dict):
        return None
    if "stations" not in cached:
        return None
    return cached


def _detour_metrics(station: Dict[str, Any]) -> Tuple[float, float]:
    """Return (detour_km, detour_min), clamped to >= 0."""
    d_km = station.get("detour_distance_km")
    d_min = station.get("detour_duration_min")

    if d_km is None:
        d_km = station.get("detour_km")
    if d_min is None:
        d_min = station.get("detour_min")

    try:
        km = float(d_km) if d_km is not None else 0.0
    except (TypeError, ValueError):
        km = 0.0

    try:
        mins = float(d_min) if d_min is not None else 0.0
    except (TypeError, ValueError):
        mins = 0.0

    return max(0.0, km), max(0.0, mins)


def _is_missing_number(x: Any) -> bool:
    if x is None:
        return True
    try:
        # pandas/np NaN handling without importing numpy explicitly
        return bool(pd.isna(x))
    except Exception:
        return False


def _as_float(x: Any) -> Optional[float]:
    """Best-effort numeric conversion used for robust display computations."""
    if _is_missing_number(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _format_eta_local_for_display(eta_any: Any, tz_name: str = "Europe/Berlin") -> str:
    """
    Convert an ETA value (typically debug_*_eta_utc ISO string or station["eta"]) into
    Europe/Berlin local time for display. Returns a safe string; never throws.

    - If input is already a local ISO string with offset, this will keep it as local (or re-normalize).
    - If input is UTC (+00:00), this will convert to +01:00/+02:00 depending on DST.
    """
    if eta_any is None:
        return "—"

    # If it is already a datetime, normalize; otherwise parse from string
    dt: datetime
    try:
        if isinstance(eta_any, datetime):
            dt = eta_any
        else:
            s = str(eta_any).strip()
            if not s:
                return "—"
            # Handle "Z" suffix if present
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
    except Exception:
        # Fall back to raw value if parsing fails
        return _safe_text(eta_any)

    # If naive, assume UTC only as a last resort (debug values should be tz-aware)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convert to Europe/Berlin if available
    if ZoneInfo is not None:
        try:
            local_tz = ZoneInfo(tz_name)
            dt_local = dt.astimezone(local_tz)
            return dt_local.isoformat()
        except Exception:
            pass

    # Fallback: use system local timezone if ZoneInfo is unavailable
    try:
        dt_local = dt.astimezone()
        return dt_local.isoformat()
    except Exception:
        return _safe_text(eta_any)


def _compute_baseline_price_for_display(
    stations: List[Dict[str, Any]],
    *,
    fuel_code: str,
    onroute_max_detour_km: float = 1.0,
    onroute_max_detour_min: float = 5.0,
) -> Optional[float]:
    """
    Recompute a baseline price similar to the decision layer:
    - cheapest predicted price among "on-route" stations (per onroute thresholds)
    - fallback: global cheapest predicted price
    """
    pred_key = f"pred_price_{fuel_code}"

    onroute_prices: List[float] = []
    all_prices: List[float] = []

    for s in stations or []:
        p = _as_float(s.get(pred_key))
        if p is None:
            continue
        all_prices.append(p)

        km, mins = _detour_metrics(s)
        if km <= float(onroute_max_detour_km) and mins <= float(onroute_max_detour_min):
            onroute_prices.append(p)

    if onroute_prices:
        return min(onroute_prices)
    if all_prices:
        return min(all_prices)
    return None


def _ensure_breakeven_liters_for_display(
    stations: List[Dict[str, Any]],
    *,
    fuel_code: str,
    baseline_price: Optional[float] = None,
    onroute_max_detour_km: float = 1.0,
    onroute_max_detour_min: float = 5.0,
) -> None:
    """
    Backfill missing econ_breakeven_liters_* values in-place so tables/captions do not show "-".

    Break-even litres:
        (detour_fuel_cost + time_cost) / (baseline_price - station_price)
    Only meaningful if station_price < baseline_price.
    """
    if not stations:
        return

    econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
    econ_baseline_key = f"econ_baseline_price_{fuel_code}"
    econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
    econ_detour_fuel_l_key = f"econ_detour_fuel_l_{fuel_code}"
    econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"

    pred_key = f"pred_price_{fuel_code}"
    curr_key = f"price_current_{fuel_code}"

    # 1) Determine baseline price (prefer what decision layer already computed)
    b = _as_float(baseline_price)
    if b is None:
        for s in stations:
            b = _as_float(s.get(econ_baseline_key))
            if b is not None:
                break
    if b is None:
        b = _compute_baseline_price_for_display(
            stations,
            fuel_code=fuel_code,
            onroute_max_detour_km=onroute_max_detour_km,
            onroute_max_detour_min=onroute_max_detour_min,
        )
    if b is None:
        return

    # 2) Backfill per station if missing
    for s in stations:
        existing = s.get(econ_breakeven_key)
        if existing is not None and not _is_missing_number(existing):
            continue

        # station price: prefer prediction, fallback to current
        p_station = _as_float(s.get(pred_key))
        if p_station is None:
            p_station = _as_float(s.get(curr_key))
        if p_station is None:
            continue

        diff = float(b) - float(p_station)
        if diff <= 0:
            # Not cheaper than baseline => no meaningful break-even litres
            continue

        detour_cost = _as_float(s.get(econ_detour_fuel_cost_key))
        if detour_cost is None:
            # fallback if only detour liters exist
            detour_l = _as_float(s.get(econ_detour_fuel_l_key))
            if detour_l is not None:
                detour_cost = detour_l * float(p_station)

        time_cost = _as_float(s.get(econ_time_cost_key))
        if time_cost is None:
            time_cost = 0.0

        if detour_cost is None:
            continue

        s[econ_breakeven_key] = max(0.0, (float(detour_cost) + float(time_cost)) / diff)

def _coerce_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and not _is_missing_number(x):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
    return None



def _open_at_eta_flag(station: Dict[str, Any]) -> Optional[bool]:
    """Return whether the station is open at the Google ETA, if the pipeline provided the flag."""
    for k in ("is_open_at_eta", "open_at_eta", "google_open_at_eta", "is_open_google_eta"):
        v = _coerce_bool(station.get(k))
        if v is not None:
            return v
    return None


def _is_closed_at_eta(station: Dict[str, Any]) -> bool:
    """True if we have an explicit open-at-ETA flag and it is False."""
    v = _open_at_eta_flag(station)
    return v is False


def _price_basis_row(station: Dict[str, Any], fuel_code: str) -> Dict[str, Any]:
    """
    Best-effort extraction of 'price basis' debug fields. We do not assume every run
    populates these keys; UI should degrade gracefully.
    """
    pred_key = f"pred_price_{fuel_code}"
    curr_key = f"price_current_{fuel_code}"

    used_current = _coerce_bool(station.get(f"debug_{fuel_code}_used_current_price"))
    horizon_used = station.get(f"debug_{fuel_code}_horizon_used")
    eta_raw = (
        station.get("eta")  # if your pipeline keeps the local ETA string, prefer it
        or station.get(f"debug_{fuel_code}_eta_utc")  # otherwise convert UTC debug ETA
        or station.get("eta_utc")
    )
    eta_local = _format_eta_local_for_display(eta_raw, tz_name="Europe/Berlin")

    minutes_to_arrival = (
        station.get(f"debug_{fuel_code}_minutes_to_arrival")
        or station.get(f"debug_{fuel_code}_minutes_ahead")
    )
    cells_ahead = station.get(f"debug_{fuel_code}_cells_ahead_raw") or station.get(f"debug_{fuel_code}_cells_ahead")

    # Robust basis label
    if used_current is True:
        basis = "Current price used (no forecast)"
    elif used_current is False:
        basis = "Forecast used"
    else:
        # Fallback inference if the explicit flag is missing
        pred = station.get(pred_key)
        curr = station.get(curr_key)
        if (not _is_missing_number(pred)) and (not _is_missing_number(curr)):
            try:
                basis = "Current/Forecast (inferred)" if abs(float(pred) - float(curr)) < 1e-6 else "Forecast (inferred)"
            except Exception:
                basis = "Current/Forecast (inferred)"
        else:
            basis = "Unknown (missing debug)"

    if horizon_used not in (None, "", 0):
        basis = f"{basis} — horizon={horizon_used}"

    return {
        "Station": _safe_text(station.get("brand") or station.get("station_name") or station.get("name")),
        "Basis": basis,
        "ETA (Europe/Berlin)": _safe_text(eta_local),
        "Minutes to arrival": minutes_to_arrival if minutes_to_arrival is not None else "—",
        "Cells ahead": cells_ahead if cells_ahead is not None else "—",
    }


def _price_basis_table(stations: List[Dict[str, Any]], fuel_code: str, limit: int = 12) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for s in stations[: max(0, int(limit))]:
        rows.append(_price_basis_row(s, fuel_code=fuel_code))
    return pd.DataFrame(rows)


def _compute_funnel_counts(
    stations: List[Dict[str, Any]],
    ranked: List[Dict[str, Any]],
    excluded: List[Dict[str, Any]],
    fuel_code: str,
    use_economics: bool,
    cap_detour_km: Optional[float],
    cap_detour_min: Optional[float],
    min_net_saving_eur: Optional[float],
    *,
    filter_closed_at_eta_enabled: bool,
    closed_at_eta_filtered_n: Optional[int] = None,
) -> Dict[str, int]:
    """
    High-level funnel counts computed from the station objects *you already cache*.

    Notes:
    - "Closed at ETA" can appear either:
      (a) as part of the cached station list (if you keep stations and mark them), or
      (b) only as an upstream count (if you remove them before caching).
    This function supports both patterns.
    """
    pred_key = f"pred_price_{fuel_code}"
    econ_key = f"econ_net_saving_eur_{fuel_code}"

    total = len(stations)
    ranked_n = len(ranked)

    missing_pred = 0
    closed_at_eta = 0
    cap_failed = 0
    econ_failed = 0
    other = 0

    # Count only among excluded stations we can still observe.
    for s in excluded:
        # 1) missing prediction
        if _is_missing_number(s.get(pred_key)):
            missing_pred += 1
            continue

        # 2) closed at ETA (if the pipeline provides the flag)
        if filter_closed_at_eta_enabled and _is_closed_at_eta(s):
            closed_at_eta += 1
            continue

        # 3) detour caps
        km, mins = _detour_metrics(s)
        cap_fail = False
        if cap_detour_km is not None and km is not None and km > float(cap_detour_km):
            cap_fail = True
        if cap_detour_min is not None and mins is not None and mins > float(cap_detour_min):
            cap_fail = True
        if cap_fail:
            cap_failed += 1
            continue

        # 4) economics threshold
        if use_economics and min_net_saving_eur is not None:
            net = s.get(econ_key)
            if (not _is_missing_number(net)) and (float(net) < float(min_net_saving_eur)):
                econ_failed += 1
                continue

        other += 1

    upstream_closed = 0
    try:
        upstream_closed = int(closed_at_eta_filtered_n) if closed_at_eta_filtered_n is not None else 0
    except Exception:
        upstream_closed = 0

    # If upstream filtering removed stations before caching, add that to the observable excluded count.
    closed_at_eta_total = closed_at_eta + max(0, upstream_closed)

    return {
        "candidates_total": total,
        "ranked": ranked_n,
        "excluded": len(excluded),
        "missing_prediction": missing_pred,
        "closed_at_eta": closed_at_eta_total,
        "closed_at_eta_in_cache": closed_at_eta,
        "closed_at_eta_upstream": max(0, upstream_closed),
        "failed_detour_caps": cap_failed,
        "failed_economics": econ_failed,
        "excluded_other": other,
    }


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """
    Robust numeric coercion:
    - handles comma decimals ("1,706")
    - strips currency/unit text ("€/L", " km", etc.)
    - keeps negatives and decimals
    """
    as_str = s.astype(str)
    as_str = as_str.str.replace(",", ".", regex=False)
    as_str = as_str.str.replace(r"[^0-9.\-]+", "", regex=True)
    return pd.to_numeric(as_str, errors="coerce")


def _compute_onroute_worst_price(
    ranked: List[Dict[str, Any]],
    fuel_code: str,
    *,
    onroute_max_detour_km: float = 1.0,
    onroute_max_detour_min: float = 5.0,
) -> Optional[float]:
    """Worst (highest) predicted price among on-route stations."""
    pred_key = f"pred_price_{fuel_code}"
    vals: List[float] = []
    for s in ranked:
        p = s.get(pred_key)
        if p is None:
            continue
        km, mins = _detour_metrics(s)
        if km <= onroute_max_detour_km and mins <= onroute_max_detour_min:
            try:
                vals.append(float(p))
            except (TypeError, ValueError):
                continue
    return max(vals) if vals else None


def _compute_net_vs_onroute_worst(
    best: Dict[str, Any],
    onroute_worst: Optional[float],
    fuel_code: str,
    litres_to_refuel: Optional[float],
) -> Optional[float]:
    """
    (worst_onroute - chosen) * litres - detour_fuel_cost - time_cost.

    This matches the "story" you display on Page 1 (worst on-route baseline).
    Note: Your economics engine uses a different baseline ("cheapest on-route"),
    so we show both on this page.
    """
    if onroute_worst is None or litres_to_refuel is None:
        return None

    pred_key = f"pred_price_{fuel_code}"
    p = best.get(pred_key)
    if p is None:
        return None

    try:
        chosen = float(p)
        litres = float(litres_to_refuel)
    except (TypeError, ValueError):
        return None

    detour_fuel_cost = best.get(f"econ_detour_fuel_cost_eur_{fuel_code}")
    time_cost = best.get(f"econ_time_cost_eur_{fuel_code}")

    try:
        detour_fuel_cost_f = float(detour_fuel_cost) if detour_fuel_cost is not None else 0.0
    except (TypeError, ValueError):
        detour_fuel_cost_f = 0.0

    try:
        time_cost_f = float(time_cost) if time_cost is not None else 0.0
    except (TypeError, ValueError):
        time_cost_f = 0.0

    gross = (float(onroute_worst) - chosen) * litres
    return gross - detour_fuel_cost_f - time_cost_f


def _exclusion_reason(
    s: Dict[str, Any],
    *,
    fuel_code: str,
    use_economics: bool,
    cap_km: Optional[float],
    cap_min: Optional[float],
    min_net_saving: Optional[float],
    filter_closed_at_eta_enabled: bool,
) -> str:
    """Best-effort reason label for excluded stations."""
    pred_key = f"pred_price_{fuel_code}"
    econ_net_key = f"econ_net_saving_eur_{fuel_code}"

    if s.get(pred_key) is None:
        return "Missing predicted price"

    if filter_closed_at_eta_enabled and _is_closed_at_eta(s):
        return "Closed at ETA (Google)"

    km, mins = _detour_metrics(s)

    if cap_km is not None and km > cap_km:
        return "Detour distance above cap"
    if cap_min is not None and mins > cap_min:
        return "Detour time above cap"

    if use_economics and (min_net_saving is not None):
        net = s.get(econ_net_key)
        if net is None:
            return "Missing economics metrics"
        try:
            if float(net) < float(min_net_saving):
                return "Below minimum net saving"
        except (TypeError, ValueError):
            return "Invalid economics metrics"

    return "Not ranked (other)"


def _station_label(s: Dict[str, Any], idx: Optional[int] = None, tag: str = "") -> str:
    brand = s.get("brand")
    name = s.get("tk_name") or s.get("osm_name") or s.get("name") or "Station"
    city = s.get("city") or ""
    title = brand if (brand and str(brand).strip()) else name
    prefix = f"#{idx} " if idx is not None else ""
    suffix = f" [{tag}]" if tag else ""
    return _safe_text(f"{prefix}{title} ({city}){suffix}")


def _render_page02_sidebar_action() -> None:
    """
    Page 02 (Route Analytics) Action tab:
    - Back to Trip Planner button (primary styling)
    - Analysis type selector (buttons)
    """

    # Top button: same "primary" style pattern as Page 04 ("Search Stations")
    if st.sidebar.button(
        "Back to Trip Planner",
        type="primary",
        use_container_width=True,
        key="p02_back_to_trip_planner",
        help="Return to the Home page.",
    ):
        # Do NOT set st.session_state["top_nav"] here (top_nav widget already exists on this page).
        # Instead, request Home navigation and let streamlit_app.py apply it before rendering its widget.
        st.session_state["nav_request_top_nav"] = "Home"

        try:
            maybe_persist_state(force=True)
        except Exception:
            pass

        st.switch_page("streamlit_app.py")

    with st.sidebar.container(key="p02_analysis_type_block"):
        # Page-02-only spacing fix (scoped to this container)
        st.markdown(
            """
            <style>
            /* Only affects content inside this specific sidebar container */
            section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"]:has(div[data-testid="stVerticalBlock"] div.p02-analysis-type-block) h3 {
                margin-bottom: 0.65rem !important;
            }
            section[data-testid="stSidebar"] div.p02-analysis-selected {
                margin-top: 0rem !important;
                margin-bottom: 1.2rem !important;
                font-size: 1rem;
                opacity: 0.78;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Add a marker div so the selector can scope reliably
        st.markdown("<div class='p02-analysis-type-block'></div>", unsafe_allow_html=True)

        st.markdown(
            "### Chose Analysis Type",
            help=(
                "Select which analytics view to display on this page. "
            ),
        )

        # Show current selection (or none) without using st.caption (your global CSS affects captions)
        current_view = str(st.session_state.get("route_analytics_view") or "").strip()
        selected_label = current_view if current_view else "None selected"
        st.markdown(
            f"<div class='p02-analysis-selected'>Selected: {selected_label}</div>",
            unsafe_allow_html=True,
        )

    def _apply_view_and_rerun(view: str) -> None:
        st.session_state["route_analytics_view"] = view
        try:
            maybe_persist_state(force=True)
        except Exception:
            pass

        # Streamlit version compatibility
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    st.sidebar.markdown(
        """
        <style>
        /* Page 02 sidebar: left-align ONLY the 4 analysis buttons (scoped by keys) */
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_route button,
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_stations button,
        section[data-testid="stSidebar"] .st-key-p02_view_prediction_algorithm button,
        section[data-testid="stSidebar"] .st-key-p02_view_full_result_tables button {
            width: 100%;
            justify-content: flex-start !important;
            text-align: left !important;
        }

        /* Streamlit typically wraps the label inside a button > div wrapper; reset that */
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_route button > div,
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_stations button > div,
        section[data-testid="stSidebar"] .st-key-p02_view_prediction_algorithm button > div,
        section[data-testid="stSidebar"] .st-key-p02_view_full_result_tables button > div {
            width: 100% !important;
            margin: 0 !important;
            justify-content: flex-start !important;
        }

        /* Force every possible label container to full-width + left aligned */
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_route button * ,
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_stations button * ,
        section[data-testid="stSidebar"] .st-key-p02_view_prediction_algorithm button * ,
        section[data-testid="stSidebar"] .st-key-p02_view_full_result_tables button * {
            text-align: left !important;
            justify-content: flex-start !important;
        }

        /* Optional: small left padding so it reads as “left-bound” */
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_route button,
        section[data-testid="stSidebar"] .st-key-p02_view_recommended_stations button,
        section[data-testid="stSidebar"] .st-key-p02_view_prediction_algorithm button,
        section[data-testid="stSidebar"] .st-key-p02_view_full_result_tables button {
            padding-left: 1.0rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # View buttons (stacked, full width)
    if st.sidebar.button("**1. Recommended Route**", use_container_width=True, key="p02_view_recommended_route"):
        _apply_view_and_rerun("Recommended Route")

    if st.sidebar.button("**2. Recommended Stations**", use_container_width=True, key="p02_view_recommended_stations"):
        _apply_view_and_rerun("Recommended Stations")

    if st.sidebar.button("**3. Prediction Algorithm**", use_container_width=True, key="p02_view_prediction_algorithm"):
        _apply_view_and_rerun("Prediction Algorithm")

    if st.sidebar.button("**4. Full Result Tables**", use_container_width=True, key="p02_view_full_result_tables"):
        _apply_view_and_rerun("Full Result Tables")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Route Analytics", layout="wide")

    # Redis-backed persistence (best-effort)
    # IMPORTANT: preserve widget-managed keys so Redis restore does not clobber user clicks
    _preserve_top_nav = st.session_state.get("top_nav")
    _preserve_sidebar_view = st.session_state.get("sidebar_view")
    _preserve_map_style_mode = st.session_state.get("map_style_mode")  # <-- ADD THIS

    init_session_context()
    ensure_persisted_state_defaults(st.session_state)

    # Keep refresh persistence working:
    # - overwrite_existing=True restores persisted values on a cold start / hard refresh
    # - then we re-apply widget keys if the user interaction already set them for this rerun
    restore_persisted_state(overwrite_existing=True)

    if _preserve_top_nav is not None:
        st.session_state["top_nav"] = _preserve_top_nav
    if _preserve_sidebar_view is not None:
        st.session_state["sidebar_view"] = _preserve_sidebar_view
    if _preserve_map_style_mode is not None:
        st.session_state["map_style_mode"] = _preserve_map_style_mode  # <-- ADD THIS

    apply_app_css()

    # Header (consistent across pages)
    st.title("Route Analytics")
    st.caption("##### Understand station and route recommendations.")

    NAV_TARGETS = {
        "Home": "streamlit_app.py",
        "Analytics": "pages/02_route_analytics.py",
        "Station": "pages/03_station_details.py",
        "Explorer": "pages/04_station_explorer.py",
    }
    CURRENT = "Analytics"

    # Ensure the correct tab is selected when landing on this page,
    # but do not overwrite user interaction during the same run.
    if st.session_state.get("_active_page") != CURRENT:
        st.session_state["_active_page"] = CURRENT
        st.session_state["top_nav"] = CURRENT
        # Default behavior for Page 02: no analysis section selected on entry
        st.session_state["route_analytics_view"] = ""

    selected = st.segmented_control(
        label="",
        options=list(NAV_TARGETS.keys()),
        selection_mode="single",
        label_visibility="collapsed",
        width="stretch",
        key="top_nav",
    )

    target = NAV_TARGETS.get(selected, NAV_TARGETS[CURRENT])
    if target != NAV_TARGETS[CURRENT]:
        # Persist before navigation so a reconnect / next page sees the most recent state.
        try:
            maybe_persist_state(force=True)
        except Exception:
            pass
        st.switch_page(target)


    # Debug mode forced ON on Page 02
    st.session_state["debug_mode"] = True

    # Render standardized shell (Action/Help/Settings/Profile) with Page-02 Action content
    render_sidebar_shell(action_renderer=_render_page02_sidebar_action)

    cached = _get_last_run()
    if not cached:
        st.info("No cached run found. Run a route recommendation first on the home page.")
        return


    fuel_code: str = str(cached.get("fuel_code") or "e5").lower()
    run_summary: Dict[str, Any] = cached.get("run_summary") or {}
    use_economics: bool = bool(run_summary.get("use_economics") or cached.get("use_economics", False))

    # Advanced Settings (persisted by Page 01)
    filter_closed_at_eta_enabled: bool = bool(run_summary.get("filter_closed_at_eta", False))
    closed_at_eta_filtered_n = run_summary.get("closed_at_eta_filtered_n", 0)

    brand_filter_selected = run_summary.get("brand_filter_selected") or []
    if isinstance(brand_filter_selected, str):
        brand_filter_selected = [brand_filter_selected]
    brand_filter_selected = [str(x) for x in brand_filter_selected]

    brand_filtered_out_n = run_summary.get("brand_filtered_out_n", 0)
    brand_filter_aliases = run_summary.get("brand_filter_aliases") or {}

    try:
        closed_at_eta_filtered_n = int(closed_at_eta_filtered_n or 0)
    except (TypeError, ValueError):
        closed_at_eta_filtered_n = 0

    try:
        brand_filtered_out_n = int(brand_filtered_out_n or 0)
    except (TypeError, ValueError):
        brand_filtered_out_n = 0

    litres_to_refuel = cached.get("litres_to_refuel")
    debug_mode: bool = bool(st.session_state.get("debug_mode", False))

    # Keep cache consistent so other pages (and this page on rerun) see the toggle.
    cached["debug_mode"] = debug_mode
    st.session_state["last_run"] = cached

    stations: List[Dict[str, Any]] = cached.get("stations") or []

    analysis_view = str(st.session_state.get("route_analytics_view") or "").strip()

    # Default landing: no selection => show welcome info below the top nav
    if not analysis_view:
        st.info(
            "Welcome to Route Analytics.\n\n"
            "Use the sidebar to select an analysis view:\n"
            "- Recommended Route\n"
            "- Recommended Stations\n"
            "- Prediction Algorithm\n"
            "- Full Result Tables\n\n"
            "Tip: Start on “Full Result Tables” to review the current end-to-end output."
        )
        maybe_persist_state()
        return

    # ------------------------------------------------------------
    # View routing (Page 02 subviews)
    # ------------------------------------------------------------

    if analysis_view == "Recommended Route":
        # Headline slightly smaller than the main title
        st.markdown("### **1. Recommended Route**")

        # [CHANGE 1] Decrease space above "Route settings:"
        # used negative margin-top to pull it up, and small bottom margin
        st.markdown(
            """
            <h4 style="margin-top: -1.0rem; margin-bottom: 0.25rem;">
                Route settings:
            </h4>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------
        # Route (Start -> Destination)
        # -----------------------------
        def _fmt_coord(pt: object) -> str:
            """Format a coordinate point robustly across common shapes."""
            if pt is None:
                return "—"
            try:
                # tuple/list: (lat, lon)
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    lat, lon = float(pt[0]), float(pt[1])
                    return f"{lat:.5f}, {lon:.5f}"
                # dict: {"lat":..,"lng":..} or {"latitude":..,"longitude":..}
                if isinstance(pt, dict):
                    if "lat" in pt and ("lng" in pt or "lon" in pt):
                        lat = float(pt["lat"])
                        lon = float(pt.get("lng", pt.get("lon")))
                        return f"{lat:.5f}, {lon:.5f}"
                    if "latitude" in pt and "longitude" in pt:
                        lat = float(pt["latitude"])
                        lon = float(pt["longitude"])
                        return f"{lat:.5f}, {lon:.5f}"
            except Exception:
                return "—"
            return "—"


        # Prefer the cached run inputs (stable even if user changed sidebar afterward)
        params = cached.get("params") or {}
        start_label = str(params.get("start_locality") or st.session_state.get("start_locality") or "").strip()
        end_label = str(params.get("end_locality") or st.session_state.get("end_locality") or "").strip()
        start_label = start_label if start_label else "—"
        end_label = end_label if end_label else "—"

        # Coordinates: use route endpoints from the cached route polyline
        route_coords = None
        # Always read route_info/route_coords from the cached last_run on Page 02.
        route_info = cached.get("route_info") or {}

        # 1) Preferred: exact geocoding coords used for routing (lat/lon)
        start_coord = route_info.get("start_coord")
        end_coord = route_info.get("end_coord")

        # 2) Fallback: derive from route polyline endpoints (often [lon, lat])
        route_coords = route_info.get("route_coords") or cached.get("route_coords") or []

        def _coerce_latlon_from_route_point(pt: object):
            """
            Route polyline points in this project are typically [lon, lat].
            Convert to (lat, lon) for display when we can infer it.
            """
            try:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    a, b = float(pt[0]), float(pt[1])

                    # Heuristic:
                    # - latitude must be within [-90, 90]
                    # - longitude within [-180, 180]
                    # If (a,b) fits (lon,lat), swap to (lat,lon)
                    if abs(a) <= 180 and abs(b) <= 90:
                        return (b, a)  # (lat, lon)
                    if abs(a) <= 90 and abs(b) <= 180:
                        return (a, b)  # already (lat, lon)
            except Exception:
                pass
            return None

        if not start_coord or not end_coord:
            if isinstance(route_coords, list) and len(route_coords) >= 1:
                start_fallback = _coerce_latlon_from_route_point(route_coords[0])
                end_fallback = _coerce_latlon_from_route_point(route_coords[-1])
                start_coord = start_coord or start_fallback
                end_coord = end_coord or end_fallback

        # Page-02-only styling for these cards (scoped via unique class)
        st.markdown(
            """
            <style>
            .p02-route-cards {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.45rem;
                margin-top: -1.4rem;
                margin-bottom: 0.5rem;
            }
            .p02-route-card {
                width: min(720px, 100%);
                padding: 0.85rem 1.0rem;
                border: 3px solid rgba(49, 51, 63, 0.3);
                border-radius: 0.9rem;
                text-align: center;
            }
            .p02-route-title {
                font-size: 1.3rem;
                font-weight: 750;
                margin: 0 0 0.25rem 0;
                line-height: 1.15;
            }
            .p02-route-sub {
                font-size: 0.92rem;
                opacity: 0.78;
                margin: 0;
                line-height: 1.2;
            }
            .p02-route-arrow {
                font-size: 1.35rem;
                font-weight: 800;
                opacity: 0.85;
                line-height: 1;
                margin: 0.05rem 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="p02-route-cards">
            <div class="p02-route-card">
                <div class="p02-route-title">{start_label}</div>
                <div class="p02-route-sub">Google coordinates: {_fmt_coord(start_coord)}</div>
            </div>

            <div class="p02-route-arrow" style="font-size: 40px; font-weight: bold;">↓</div>

            <div class="p02-route-card">
                <div class="p02-route-title">{end_label}</div>
                <div class="p02-route-sub">Google coordinates: {_fmt_coord(end_coord)}</div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # [CHANGE 2] Decrease space below "Recommender constraints:"
        # Set margin-bottom to a very small value (0.1rem) so cards sit tight against it
        st.markdown(
            """
            <h4 style="margin-top: 0.3rem; margin-bottom: -1.3rem;">
                Recommender constraints:
            </h4>
            """,
            unsafe_allow_html=True
        )

        # Read constraints exactly as used on Page 01 (cached in last_run["run_summary"])
        constraints = run_summary.get("constraints") or {}

        max_detour_km = constraints.get("max_detour_km")
        max_detour_min = constraints.get("max_detour_min")

        litres_to_refuel = constraints.get("litres_to_refuel")
        consumption_l_per_100km = constraints.get("consumption_l_per_100km")
        value_of_time_eur_per_hour = constraints.get("value_of_time_eur_per_hour")
        min_net_saving_eur = constraints.get("min_net_saving_eur") if use_economics else None

        # Advanced Settings (persisted by Page 01)
        adv = (run_summary.get("advanced_settings") or cached.get("advanced_settings") or {})
        open_at_eta = bool(
            adv.get("filter_closed_at_eta")
            or run_summary.get("filter_closed_at_eta")
            or cached.get("filter_closed_at_eta")
        )

        brand_filter_selected = adv.get("brand_filter_selected") or []
        if isinstance(brand_filter_selected, str):
            brand_filter_selected = [brand_filter_selected]
        brand_filter_selected = [str(x).strip() for x in brand_filter_selected if str(x).strip()]
        brand_filtered_out_n = adv.get("brand_filtered_out_n")
        try:
            brand_filtered_out_n = int(brand_filtered_out_n) if brand_filtered_out_n is not None else 0
        except (TypeError, ValueError):
            brand_filtered_out_n = 0

        # UI summary (responsive via your existing metric-grid CSS)
        items = [
            ("Fuel type", fuel_code.upper()),
            ("Decision mode", "Economics-based" if use_economics else "Price-only"),
            ("Max extra distance", _format_km(max_detour_km)),
            ("Max extra time", _format_min(max_detour_min)),
        ]

        if use_economics:
            items.extend(
                [
                    ("Litres to refuel", _format_liters(litres_to_refuel)),
                    ("Consumption", f"{float(consumption_l_per_100km):.2f} L/100 km" if consumption_l_per_100km is not None else "—"),
                    ("Value of time", _format_eur(value_of_time_eur_per_hour) + "/h" if value_of_time_eur_per_hour is not None else "—"),
                    ("Min net saving", _format_eur(min_net_saving_eur)),
                ]
            )

        items.append(("Stations open at ETA", "On" if open_at_eta else "Off"))
        items.append(("Brand filter", "Any" if not brand_filter_selected else f"{len(brand_filter_selected)} selected"))

        render_card_grid(items, cols=2)


        # [CHANGE 3] Increase space above "Key visualization:"
        # Set margin-top to a larger value (3.0rem)
        st.markdown(
            """
            <h4 style="margin-top: 0.5rem; margin-bottom: 0.2rem;">
                Key Visualization:
            </h4>
            """,
            unsafe_allow_html=True
        )

        best_station: Optional[Dict[str, Any]] = cached.get("best_station")
        route_info: Dict[str, Any] = cached.get("route_info") or {}
        render_key_visualization(route_info=route_info, best_station=best_station)

        maybe_persist_state()
        return

    # Keep other views empty for now (header + top nav remain visible)
    if analysis_view != "Full Result Tables":
        maybe_persist_state()
        return

    ranked: List[Dict[str, Any]] = cached.get("ranked") or []
    best_station: Optional[Dict[str, Any]] = cached.get("best_station")
    route_info: Dict[str, Any] = cached.get("route_info") or {}

    # If Page 01 already computed the "value view" set, keep it available here as well.
    value_view_stations = cached.get("value_view_stations") or []
    value_view_meta = cached.get("value_view_meta") or {}

    # Constraints (from Page 1 run_summary; fall back to None when missing)
    constraints = run_summary.get("constraints") or {}
    cap_km = constraints.get("max_detour_km")
    cap_min = constraints.get("max_detour_min")
    min_net_saving = constraints.get("min_net_saving_eur") if use_economics else None

    # Advanced filter: "open at ETA" (may be absent in older cached runs)
    filter_closed_at_eta_enabled: bool = bool(
        run_summary.get("filter_closed_at_eta")
        or cached.get("filter_closed_at_eta")
        or (run_summary.get("advanced_settings") or {}).get("filter_closed_at_eta")
        or (cached.get("advanced_settings") or {}).get("filter_closed_at_eta")
    )
    closed_at_eta_filtered_n = (
        run_summary.get("closed_at_eta_filtered_n")
        or cached.get("closed_at_eta_filtered_n")
        or (run_summary.get("advanced_settings") or {}).get("closed_at_eta_filtered_n")
        or (cached.get("advanced_settings") or {}).get("closed_at_eta_filtered_n")
    )

    # ------------------------------------------------------------------
    # Optional exact filter diagnostics (emitted by decision layer)
    # ------------------------------------------------------------------
    filter_log: Dict[str, Any] = cached.get("filter_log") or {}
    primary_reason_by_uuid: Dict[str, str] = filter_log.get("primary_reason_by_uuid") or {}
    filter_counts: Dict[str, int] = filter_log.get("counts") or {}
    filter_notes: List[str] = filter_log.get("notes") or []
    filter_thresholds: Dict[str, Any] = filter_log.get("thresholds") or {}

    # ------------------------------------------------------------------
    # SECTION A — Your route & recommendation
    # ------------------------------------------------------------------
    st.subheader("A) Your route and the recommended decision")

    # A1. Run summary
    total_n = len(stations)
    ranked_n = len(ranked)
    excluded_n = max(0, total_n - ranked_n)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations (total)", f"{total_n}")
    c2.metric("Stations (ranked)", f"{ranked_n}")
    c3.metric("Stations (excluded)", f"{excluded_n}")
    c4.metric("Economics mode", "On" if use_economics else "Off")

    # Route metadata (best-effort)
    start_lbl = route_info.get("start_label") or ""
    end_lbl = route_info.get("end_label") or ""
    route_km = route_info.get("route_km")
    route_min = route_info.get("route_min")

    meta_left, meta_right = st.columns([2, 1])
    with meta_left:
        if start_lbl or end_lbl:
            st.write(f"**Route:** {_safe_text(start_lbl)} → {_safe_text(end_lbl)}")
        else:
            st.write("**Route:** (labels unavailable in cache)")

        constraints_lines = []
        if cap_km is not None:
            constraints_lines.append(f"Max detour distance: **{float(cap_km):.1f} km**")
        if cap_min is not None:
            constraints_lines.append(f"Max detour time: **{float(cap_min):.0f} min**")
        if use_economics and (min_net_saving is not None):
            constraints_lines.append(f"Minimum net saving: **{float(min_net_saving):.2f} €**")
        if constraints_lines:
            st.write("**Constraints:** " + " · ".join(constraints_lines))

    with meta_right:
        if route_km is not None:
            try:
                st.metric("Route length", f"{float(route_km):.1f} km")
            except (TypeError, ValueError):
                st.metric("Route length", "—")
        else:
            st.metric("Route length", "—")

        if route_min is not None:
            try:
                st.metric("Route duration", f"{float(route_min):.0f} min")
            except (TypeError, ValueError):
                st.metric("Route duration", "—")
        else:
            st.metric("Route duration", "—")

    st.markdown("")

    # A2. Recommended station audit
    if best_station:
        st.markdown("#### Recommended station (audit)")

        onroute_worst = _compute_onroute_worst_price(ranked, fuel_code)
        net_vs_worst = _compute_net_vs_onroute_worst(best_station, onroute_worst, fuel_code, litres_to_refuel)

        pred_key = f"pred_price_{fuel_code}"
        curr_key = f"price_current_{fuel_code}"
        econ_net_key = f"econ_net_saving_eur_{fuel_code}"
        econ_gross_key = f"econ_gross_saving_eur_{fuel_code}"
        econ_detour_fuel_cost_key = f"econ_detour_fuel_cost_eur_{fuel_code}"
        econ_time_cost_key = f"econ_time_cost_eur_{fuel_code}"
        econ_breakeven_key = f"econ_breakeven_liters_{fuel_code}"
        econ_baseline_key = f"econ_baseline_price_{fuel_code}"

        km, _mins = _detour_metrics(best_station)
        dist_m = best_station.get("distance_along_m")

        # Headline metrics (match your Page 1 set)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Predicted {fuel_code.upper()} price", _fmt_price(best_station.get(pred_key)))
        m2.metric(f"Current {fuel_code.upper()} price", _fmt_price(best_station.get(curr_key)))
        if dist_m is None:
            m3.metric("Distance along route", "—")
        else:
            try:
                m3.metric("Distance along route", f"{float(dist_m)/1000.0:.1f} km")
            except (TypeError, ValueError):
                m3.metric("Distance along route", "—")

        m4, m5, m6 = st.columns(3)
        m4.metric("On-route worst price", "—" if onroute_worst is None else _fmt_price(onroute_worst))
        m5.metric("Net saving vs on-route worst", "—" if net_vs_worst is None else _fmt_eur(net_vs_worst))
        m6.metric("Detour distance", _fmt_km(km))

        if use_economics:
            st.markdown("**Economics decomposition**")
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Baseline price used", _fmt_price(best_station.get(econ_baseline_key)))
            d2.metric("Gross saving", _fmt_eur(best_station.get(econ_gross_key)))
            d3.metric("Detour fuel cost", _fmt_eur(best_station.get(econ_detour_fuel_cost_key)))
            d4.metric("Time cost", _fmt_eur(best_station.get(econ_time_cost_key)))
            d5.metric("Net saving (model)", _fmt_eur(best_station.get(econ_net_key)))

            # Ensure break-even litres is present for display (defensive)
            _ensure_breakeven_liters_for_display(
                [best_station],
                fuel_code=fuel_code,
                onroute_max_detour_km=float(filter_thresholds.get("onroute_max_detour_km", 1.0) or 1.0),
                onroute_max_detour_min=float(filter_thresholds.get("onroute_max_detour_min", 5.0) or 5.0),
            )

            be = _as_float(best_station.get(econ_breakeven_key))
            if be is not None:
                st.caption(f"Break-even litres: {be:.1f} L")

        with st.expander("How the predicted price is chosen (price basis)", expanded=False):
            st.write(
                "The app may use either the station's current price or a forecasted price depending on the "
                "ETA-to-station and the model horizon selected for that ETA. The table below shows the "
                "debug fields used to decide this for the recommended station. If some fields are missing, "
                "the run did not populate them and the UI falls back gracefully."
            )
            basis_df = _price_basis_table([best_station], fuel_code=fuel_code, limit=1)
            st.dataframe(basis_df, hide_index=True, use_container_width=True)

            if ranked:
                st.caption("Top ranked stations — price basis snapshot")
                st.dataframe(_price_basis_table(ranked, fuel_code=fuel_code, limit=10), hide_index=True, use_container_width=True)

            action_col1, action_col2 = st.columns([1, 3])
            with action_col1:
                if st.button("Open Station Details", use_container_width=True):
                    st.session_state["selected_station_uuid"] = _station_uuid(best_station)
                    st.session_state["selected_station_data"] = best_station
                    st.switch_page("pages/03_station_details.py")
            with action_col2:
                st.caption("Use Station Details for the full station profile (prices, prediction basis, debugging).")

    else:
        st.warning("No recommended station available in cache (likely no stations passed filters).")

    st.markdown("---")

    # ------------------------------------------------------------------
    # SECTION B — Station universe & exclusions
    # ------------------------------------------------------------------
    st.subheader("B) All stations and why some were excluded")
    
    ranked_uuids = {_station_uuid(s) for s in ranked if _station_uuid(s)}
    excluded = [s for s in stations if (_station_uuid(s) not in ranked_uuids)]

    funnel = _compute_funnel_counts(
        stations=stations,
        ranked=ranked,
        excluded=excluded,
        fuel_code=fuel_code,
        use_economics=use_economics,
        cap_detour_km=cap_km,
        cap_detour_min=cap_min,
        min_net_saving_eur=min_net_saving,
        filter_closed_at_eta_enabled=filter_closed_at_eta_enabled,
        closed_at_eta_filtered_n=closed_at_eta_filtered_n,
    )

    st.markdown("#### Filtering funnel (high-level)")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Candidates", str(funnel["candidates_total"]))
    c2.metric("Missing prediction", str(funnel["missing_prediction"]))
    c3.metric("Closed at ETA", str(funnel["closed_at_eta"]) if filter_closed_at_eta_enabled else "—")
    c4.metric("Failed detour caps", str(funnel["failed_detour_caps"]))
    c5.metric("Failed economics", str(funnel["failed_economics"]) if use_economics else "—")
    c6.metric("Ranked", str(funnel["ranked"]))
    with st.expander("Funnel details (counts)", expanded=False):
        funnel_df = pd.DataFrame([funnel])
        st.dataframe(funnel_df, hide_index=True, use_container_width=True)

    if filter_thresholds or filter_notes:
        with st.expander("Filter diagnostics (exact)", expanded=False):
            if filter_thresholds:
                st.markdown("**Thresholds used**")
                st.json(filter_thresholds, expanded=False)
            if filter_notes:
                st.markdown("**Notes**")
                for n in filter_notes:
                    st.write(f"- {n}")
            if filter_counts:
                st.markdown("**Counts (as emitted by decision layer)**")
                st.json(filter_counts, expanded=False)

    # B1. Exclusion breakdown
    if excluded:
        # Prefer exact primary reasons from filter_log if available
        if primary_reason_by_uuid:
            reasons = [primary_reason_by_uuid.get(_station_uuid(s), "Not ranked (other)") for s in excluded]
        else:
            reasons = [
                _exclusion_reason(
                    s,
                    fuel_code=fuel_code,
                    use_economics=use_economics,
                    cap_km=float(cap_km) if cap_km is not None else None,
                    cap_min=float(cap_min) if cap_min is not None else None,
                    min_net_saving=float(min_net_saving) if min_net_saving is not None else None,
                    filter_closed_at_eta_enabled=filter_closed_at_eta_enabled,
                )
                for s in excluded
            ]

        reason_counts = pd.Series(reasons).value_counts()

        # Integrate upstream filters (stations removed before caching)
        if filter_closed_at_eta_enabled and closed_at_eta_filtered_n > 0:
            reason_counts.loc["Filtered upstream: Closed at ETA (Google)"] = int(closed_at_eta_filtered_n)

        if brand_filter_selected and brand_filtered_out_n > 0:
            reason_counts.loc["Filtered upstream: Brand filter"] = int(brand_filtered_out_n)

        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Exclusion breakdown**")
            for reason, cnt in reason_counts.items():
                st.write(f"- {reason}: **{int(cnt)}**")
            if brand_filter_selected:
                st.caption(f"Brand whitelist: {', '.join(brand_filter_selected)}")
                # Optional transparency: show aliases used
                if isinstance(brand_filter_aliases, dict) and brand_filter_aliases:
                    st.caption(
                        "Alias matching: "
                        + "; ".join(f"{k}: {', '.join(v)}" for k, v in brand_filter_aliases.items() if isinstance(v, list))
                    )
                st.caption("Note: stations with unknown/missing brand are excluded when the brand filter is active.")
        with right:
            chart_df = reason_counts.reset_index()
            chart_df.columns = ["Reason", "Count"]
            st.bar_chart(chart_df.set_index("Reason"))
    else:
        st.info("No stations were excluded (all candidates appear in the ranked list).")

    st.markdown("")

    # B2a. Value view: stations not worse than the worst on-route option
    # This is a display-only table intended to be more consistent for end users.
    st.markdown("#### Stations better than the worst on-route option (value view)")

    pred_key = f"pred_price_{fuel_code}"

    # Reconstruct the "hard-pass" candidate set (what the decision layer would consider before the min-net-saving soft filter).
    cap_km_f = float(cap_km) if cap_km is not None else None
    cap_min_f = float(cap_min) if cap_min is not None else None

    hard_pass: List[Dict[str, Any]] = []
    for s in stations:
        # Require a usable prediction for the selected fuel.
        p = s.get(pred_key)
        try:
            p_f = float(p) if p is not None else None
        except (TypeError, ValueError):
            p_f = None
        if p_f is None or p_f < MIN_VALID_PRICE_EUR_L:
            continue

        # Require passing detour caps (hard constraints).
        km, mins = _detour_metrics(s)
        if (cap_km_f is not None and km > cap_km_f) or (cap_min_f is not None and mins > cap_min_f):
            continue

        hard_pass.append(s)

    # Compute the on-route worst price within the hard-pass set.
    onroute_km = float(filter_thresholds.get("onroute_max_detour_km", 1.0) or 1.0)
    onroute_min = float(filter_thresholds.get("onroute_max_detour_min", 5.0) or 5.0)

    onroute_worst_price = _compute_onroute_worst_price(
        hard_pass,
        fuel_code=fuel_code,
        onroute_max_detour_km=onroute_km,
        onroute_max_detour_min=onroute_min,
    )

    if onroute_worst_price is None:
        st.info(
            "Could not compute the on-route worst price (no stations qualify as on-route under the current thresholds)."
        )
    else:
        # Keep stations whose predicted price is not worse than the on-route worst predicted price.
        value_view: List[Dict[str, Any]] = []
        for s in hard_pass:
            p = s.get(pred_key)
            try:
                p_f = float(p) if p is not None else None
            except (TypeError, ValueError):
                p_f = None
            if p_f is None:
                continue

            if p_f <= float(onroute_worst_price) + 1e-9:
                value_view.append(s)

        # Backfill break-even liters (mainly for older cached runs or partial station dicts).
        if use_economics and value_view:
            _ensure_breakeven_liters_for_display(value_view, fuel_code=fuel_code)

        if value_view:
            # Sort with the same primary logic as the decision layer (when available).
            if use_economics:
                econ_net_key = f"econ_net_saving_eur_{fuel_code}"

                def _net_sort_key(sta: Dict[str, Any]) -> Tuple:
                    net = sta.get(econ_net_key)
                    try:
                        net_f = float(net) if net is not None else float("-inf")
                    except (TypeError, ValueError):
                        net_f = float("-inf")
                    return (
                        -net_f,
                        sta.get("fraction_of_route", float("inf")),
                        sta.get("distance_along_m", float("inf")),
                    )

                value_view_sorted = sorted(value_view, key=_net_sort_key)
            else:
                value_view_sorted = sorted(
                    value_view,
                    key=lambda x: (
                        x.get(pred_key, float("inf")),
                        x.get("fraction_of_route", float("inf")),
                        x.get("distance_along_m", float("inf")),
                    ),
                )

            value_df = build_ranking_dataframe(value_view_sorted, fuel_code=fuel_code, debug_mode=debug_mode)
            st.dataframe(value_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Filter used: predicted price ≤ on-route worst predicted price ({float(onroute_worst_price):.3f} €/L), "
                "within the same detour caps as the decision."
            )
        else:
            st.info("No stations are better than the on-route worst price under the current detour caps.")

    st.markdown("")

    # B2. Ranked table
    st.markdown("#### Ranked stations (comparison table)")
    if ranked:
        # Backfill break-even litres for display (defensive for older caches / partial station dicts)
        if use_economics:
            _ensure_breakeven_liters_for_display(ranked, fuel_code=fuel_code)

        ranked_df = build_ranking_dataframe(ranked, fuel_code=fuel_code, debug_mode=debug_mode)

        st.dataframe(ranked_df, use_container_width=True, hide_index=True)
    else:
        st.error("No stations were ranked. Use the exclusion section above to see why.")

    # B3. Excluded table
    st.markdown("#### Excluded stations (audit table)")
    if excluded:
        rows = []
        pred_key = f"pred_price_{fuel_code}"
        curr_key = f"price_current_{fuel_code}"
        econ_net_key = f"econ_net_saving_eur_{fuel_code}"

        for s in excluded:
            km, mins = _detour_metrics(s)
            uid = _station_uuid(s)
            reason = primary_reason_by_uuid.get(uid) if primary_reason_by_uuid else None
            if not reason:
                reason = _exclusion_reason(
                    s,
                    fuel_code=fuel_code,
                    use_economics=use_economics,
                    cap_km=float(cap_km) if cap_km is not None else None,
                    cap_min=float(cap_min) if cap_min is not None else None,
                    min_net_saving=float(min_net_saving) if min_net_saving is not None else None,
                    filter_closed_at_eta_enabled=filter_closed_at_eta_enabled,
                )
            rows.append(
                {
                    "Brand": s.get("brand") or "",
                    "Name": s.get("tk_name") or s.get("osm_name") or s.get("name") or "",
                    "City": s.get("city") or "",
                    "Detour distance [km]": km,
                    "Detour time [min]": mins,
                    f"Predicted {fuel_code.upper()} price": s.get(pred_key),
                    f"Current {fuel_code.upper()} price": s.get(curr_key),
                    "Net saving (if available)": s.get(econ_net_key) if use_economics else None,
                    "Open at ETA": _open_at_eta_flag(s),
                    "Reason": reason,
                    "UUID": uid,
                }
            )

        excluded_df = pd.DataFrame(rows)
        st.dataframe(excluded_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No excluded stations to display.")

    maybe_persist_state()
        

if __name__ == "__main__":
    main()
