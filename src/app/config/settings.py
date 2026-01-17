# src/app/config/settings.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore


@lru_cache(maxsize=1)
def load_env_once() -> None:
    """
    Load local .env once for local runs. No-op in deployments if python-dotenv
    is unavailable or no .env is present.

    BWCloud note:
    - In production, prefer OS environment variables provided by systemd/Docker.
    - This function is intentionally best-effort and never raises.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except Exception:
        pass


def get_setting(
    key: str,
    default: Optional[Any] = None,
    *,
    required: bool = False,
) -> Any:
    """
    Resolve a setting in a BWCloud-friendly precedence order:
      1) st.secrets (if available)
      2) OS environment variables
      3) default

    If required=True and nothing is found, raises ValueError.
    """
    # 1) st.secrets
    if st is not None:
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass

    # 2) env
    val = os.getenv(key)
    if val is not None and val != "":
        return val

    # 3) default / required
    if required:
        raise ValueError(f"Missing required setting: {key}")
    return default

# ---------------------------------------------------------------------------
# Session State Contract (for Redis-backed persistence)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, Any, Iterable, MutableMapping


@dataclass(frozen=True)
class SessionStateField:
    """
    Defines a session_state key that we consider part of the persisted UX contract.

    - default: value to apply on first visit (when key absent)
    - persist: if True, included in Redis snapshots
    """
    key: str
    default: Any
    persist: bool = True


# Bump this when you change keys/types in a non-backward-compatible way.
PERSISTED_STATE_VERSION: int = 1

# The canonical list of keys we persist across refresh/reconnect.
PERSISTED_STATE_FIELDS: Iterable[SessionStateField] = (
    # Sidebar: trip inputs & constraints
    SessionStateField("start_locality", "Tübingen"),
    SessionStateField("end_locality", "Sindelfingen"),
    SessionStateField("fuel_label", "E5"),
    SessionStateField("use_economics", True),
    SessionStateField("litres_to_refuel", 40.0),
    SessionStateField("consumption_l_per_100km", 7.0),
    SessionStateField("value_of_time_eur_per_hour", 0.0),
    SessionStateField("max_detour_km", 5.0),
    SessionStateField("max_detour_min", 10.0),
    SessionStateField("min_net_saving_eur", 0.0),
    SessionStateField("filter_closed_at_eta", True),
    SessionStateField("brand_filter_selected", []),
    SessionStateField("sidebar_view", "Action"),

    # Navigation & display prefs
    SessionStateField("top_nav", "Home"),
    SessionStateField("map_style_mode", "Standard"),
    SessionStateField("debug_mode", True),
    SessionStateField("station_details_density", "Detailed"),

    # Cross-page selection
    SessionStateField("selected_station_uuid", None),
    SessionStateField("selected_station_data", None),
    SessionStateField("selected_station_source", ""),
    SessionStateField("comparison_station_uuids", []),

    # Explorer state
    SessionStateField("explorer_location_query", "Tübingen"),
    SessionStateField("explorer_last_query", "Tübingen"),
    SessionStateField("explorer_radius_km", 10.0),
    SessionStateField("explorer_only_open", False),
    SessionStateField("explorer_use_realtime", True),
    SessionStateField("explorer_limit", 50),
    SessionStateField("explorer_fuel_label", "E5"),
    SessionStateField("explorer_center", None),
    SessionStateField("explorer_results", []),
    SessionStateField("explorer_params_hash", None),

    # Cached run results
    SessionStateField("last_run", None),
    SessionStateField("last_params_hash", None),
)

# Convenience structures used by the upcoming Redis snapshot logic.
PERSISTED_STATE_KEYS = {f.key for f in PERSISTED_STATE_FIELDS if f.persist}
PERSISTED_STATE_DEFAULTS: Dict[str, Any] = {f.key: f.default for f in PERSISTED_STATE_FIELDS if f.persist}


def ensure_persisted_state_defaults(session_state: MutableMapping[str, Any]) -> None:
    """
    Ensure that all persisted state keys exist in session_state with safe defaults.

    This is intentionally side-effect-free for existing keys (uses setdefault).
    Deep-copies mutable defaults (lists/dicts) to avoid accidental sharing.
    """
    for key, default in PERSISTED_STATE_DEFAULTS.items():
        if isinstance(default, (dict, list)):
            session_state.setdefault(key, deepcopy(default))
        else:
            session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Azure Cache for Redis settings (session persistence)
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int
    ssl: bool
    db: int
    ttl_seconds: int
    key_prefix: str

    # Auth
    auth_mode: str  # "access_key" | "entra"
    password: str | None = None  # only used for auth_mode="access_key"


def _normalize_redis_auth_mode(val: object, default: str = "access_key") -> str:
    s = str(val or "").strip().lower()
    if s in {"entra", "entraid", "aad", "azuread"}:
        return "entra"
    if s in {"access_key", "accesskey", "key", "password"}:
        return "access_key"
    return default


def _to_bool(val: object, default: bool = False) -> bool:
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_int(val: object, default: int) -> int:
    try:
        return int(str(val).strip())
    except Exception:
        return default


def get_redis_config() -> RedisConfig | None:
    """
    Returns RedisConfig if Redis is configured; otherwise None.

    Supports two auth modes:
      - access_key: uses REDIS_PASSWORD (current behavior)
      - entra: uses Microsoft Entra ID (Managed Identity / SPN). No password required.
    """
    host = get_setting("REDIS_HOST", default=None)
    if not host:
        return None

    auth_mode = _normalize_redis_auth_mode(get_setting("REDIS_AUTH_MODE", default="access_key"))

    password = get_setting("REDIS_PASSWORD", default=None)
    if auth_mode == "access_key":
        # In access_key mode, password is mandatory.
        if not password:
            return None

    port = _to_int(get_setting("REDIS_PORT", default="6380"), default=6380)
    ssl = _to_bool(get_setting("REDIS_SSL", default="true"), default=True)
    db = _to_int(get_setting("REDIS_DB", default="0"), default=0)

    ttl_seconds = _to_int(get_setting("REDIS_TTL_SECONDS", default="43200"), default=43200)
    key_prefix = str(get_setting("REDIS_KEY_PREFIX", default="tsf:session")).strip() or "tsf:session"

    return RedisConfig(
        host=str(host).strip(),
        port=port,
        ssl=ssl,
        db=db,
        ttl_seconds=ttl_seconds,
        key_prefix=key_prefix,
        auth_mode=auth_mode,
        password=str(password).strip() if (password is not None and str(password).strip() != "") else None,
    )
