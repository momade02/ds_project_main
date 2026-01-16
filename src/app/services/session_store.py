# src/app/services/session_store.py
from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from typing import Any, Dict, Optional, Tuple

import streamlit as st

try:
    import certifi
except Exception:  # pragma: no cover
    certifi = None  # type: ignore

try:
    import redis  # redis-py
except Exception:  # pragma: no cover
    redis = None  # type: ignore


# ---------------------------------------------------------------------
# Configuration + contract import (best-effort, to avoid hard coupling)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Contract:
    version: int
    keys: set[str]
    defaults: Dict[str, Any]


def _load_contract() -> _Contract:
    """
    Loads the persisted-state contract from config.settings.

    This is intentionally defensive:
    - if the contract isn't present yet, we fall back to an empty contract
      (no persistence writes, no restore).
    """
    try:
        from config import settings as s  # your src/app/config/settings.py
        version = int(getattr(s, "PERSISTED_STATE_VERSION", 1))
        keys = set(getattr(s, "PERSISTED_STATE_KEYS", set()))
        defaults = dict(getattr(s, "PERSISTED_STATE_DEFAULTS", {}))
        return _Contract(version=version, keys=keys, defaults=defaults)
    except Exception:
        return _Contract(version=1, keys=set(), defaults={})


def _ensure_defaults() -> None:
    """
    Ensure contract defaults exist in st.session_state (idempotent).
    """
    contract = _load_contract()
    # If you have ensure_persisted_state_defaults, use it; otherwise do a minimal setdefault.
    try:
        from config.settings import ensure_persisted_state_defaults  # type: ignore
        ensure_persisted_state_defaults(st.session_state)
        return
    except Exception:
        pass

    for k, v in contract.defaults.items():
        if k not in st.session_state:
            # shallow copy for mutables to reduce accidental shared state
            if isinstance(v, (dict, list)):
                st.session_state[k] = json.loads(json.dumps(v))
            else:
                st.session_state[k] = v


# ---------------------------------------------------------------------
# Query param session id (OK to expose in URL)
# ---------------------------------------------------------------------

_SESSION_ID_KEY = "_persisted_session_id"
_QUERY_PARAM_NAME = "sid"

# ---------------------------------------------------------------------
# Restore guard
# ---------------------------------------------------------------------
#
# Many pages call restore_persisted_state(overwrite_existing=True) at the top
# of main(). That is correct for a cold start / hard refresh, but if we do it
# on every rerun we can effectively "freeze" widgets: the user's interaction
# updates st.session_state, then the next rerun immediately overwrites the
# updated values from Redis again.
#
# We therefore apply Redis restore at most once per Streamlit session (per sid).
_REDIS_RESTORE_DONE_KEY = "_redis_restore_done"
_REDIS_RESTORE_SID_KEY = "_redis_restore_sid"


def _get_query_params() -> Dict[str, Any]:
    # Streamlit 1.52+ supports st.query_params (preferred)
    try:
        return dict(st.query_params)  # type: ignore[attr-defined]
    except Exception:
        # Fallback for older APIs
        try:
            return st.experimental_get_query_params()  # type: ignore[attr-defined]
        except Exception:
            return {}


def _set_query_params(**params: Any) -> None:
    # Avoid unnecessary reruns: only set if something actually changes.
    current = _get_query_params()
    desired = dict(current)
    for k, v in params.items():
        desired[k] = v

    if desired == current:
        return

    try:
        # st.query_params is dict-like in recent Streamlit
        for k, v in params.items():
            st.query_params[k] = v  # type: ignore[attr-defined]
    except Exception:
        try:
            st.experimental_set_query_params(**desired)  # type: ignore[attr-defined]
        except Exception:
            # If we cannot set query params, we still run; we just won't persist across refresh.
            return


def _new_session_id() -> str:
    # No external dependency needed; URL-safe enough for this use case.
    # (You explicitly said it's OK if this is visible in the URL.)
    return sha256(f"{time.time_ns()}::{id(st)}".encode("utf-8")).hexdigest()[:24]


def init_session_context() -> str:
    """
    Ensures a stable session id is present and stored in:
      - st.session_state[_persisted_session_id]
      - URL query param `sid=...` (so refresh keeps the same id)
    """
    if _SESSION_ID_KEY in st.session_state and st.session_state[_SESSION_ID_KEY]:
        sid = str(st.session_state[_SESSION_ID_KEY])
        _set_query_params(**{_QUERY_PARAM_NAME: sid})
        return sid

    qp = _get_query_params()
    raw = qp.get(_QUERY_PARAM_NAME)

    # st.query_params may return list-like values depending on API
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]

    sid = str(raw) if raw else _new_session_id()
    st.session_state[_SESSION_ID_KEY] = sid
    _set_query_params(**{_QUERY_PARAM_NAME: sid})
    return sid


def get_session_id() -> Optional[str]:
    sid = st.session_state.get(_SESSION_ID_KEY)
    return str(sid) if sid else None


# ---------------------------------------------------------------------
# Redis client (Azure Cache for Redis via access key)
# ---------------------------------------------------------------------

def _get_setting(key: str, default: Any = None) -> Any:
    """
    Lightweight, optional override for non-core Redis settings (timeouts, etc.).
    Core Redis config must come from config.settings.get_redis_config().
    """
    try:
        from config.settings import get_setting  # type: ignore
        return get_setting(key, default)
    except Exception:
        import os
        return os.environ.get(key, default)


def _redis_config():
    """
    Single source of truth: config.settings.get_redis_config().
    Returns RedisConfig or None.
    """
    try:
        from config.settings import get_redis_config  # type: ignore
        return get_redis_config()
    except Exception:
        return None


def _normalized_prefix(prefix: str) -> str:
    p = (prefix or "").strip()
    if not p:
        return "tsf:session"
    # Ensure exactly one ":" separator before the sid
    return p[:-1] if p.endswith(":") else p


@lru_cache(maxsize=1)
def _redis_client():
    """
    Creates a cached Redis client if Redis is configured; otherwise returns None.
    Safe for local runs: if REDIS_HOST/REDIS_PASSWORD are missing, config returns None.
    """
    cfg = _redis_config()
    if cfg is None:
        return None

    try:
        import redis  # type: ignore
    except Exception:
        return None

    # Optional tuning knobs (do not belong in RedisConfig)
    socket_timeout = float(_get_setting("REDIS_SOCKET_TIMEOUT", 3.0))
    connect_timeout = float(_get_setting("REDIS_CONNECT_TIMEOUT", 3.0))

    # TLS verification (recommended for Azure). Can be explicitly disabled if needed.
    skip_verify = str(_get_setting("REDIS_SKIP_TLS_VERIFY", "false")).strip().lower() in {"1", "true", "yes", "on"}
    ssl_ca_certs = certifi.where() if (certifi is not None) else None

    try:
        return redis.Redis(
            host=str(cfg.host),
            port=int(cfg.port),
            password=str(cfg.password),
            ssl=bool(cfg.ssl),
            db=int(cfg.db),
            ssl_ca_certs=None if skip_verify else ssl_ca_certs,
            ssl_cert_reqs=None if skip_verify else "required",
            socket_timeout=socket_timeout,
            socket_connect_timeout=connect_timeout,
            decode_responses=False,
        )
    except Exception:
        return None


def _ttl_seconds() -> int:
    cfg = _redis_config()
    if cfg is None:
        return 0
    try:
        return int(cfg.ttl_seconds)
    except Exception:
        return 43200


def _redis_key(session_id: str) -> str:
    cfg = _redis_config()
    prefix = _normalized_prefix(getattr(cfg, "key_prefix", "tsf:session") if cfg is not None else "tsf:session")
    return f"{prefix}:{session_id}"


# ---------------------------------------------------------------------
# Snapshot format + persistence
# ---------------------------------------------------------------------

def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")


def _encode_payload(payload: Dict[str, Any]) -> Tuple[bytes, bool]:
    raw = _json_dumps(payload)
    # Compress only if meaningful
    if len(raw) < 50_000:
        return raw, False
    return zlib.compress(raw, level=6), True


def _decode_payload(data: bytes, compressed: bool) -> Dict[str, Any]:
    if compressed:
        data = zlib.decompress(data)
    return json.loads(data.decode("utf-8"))


def restore_persisted_state(*, overwrite_existing: bool = True) -> bool:
    """
    Loads the persisted session snapshot from Redis and applies it to st.session_state.

    overwrite_existing=True:
      - snapshot values overwrite existing values for persisted keys
      - this is what you want for "refresh should keep my stuff"
    """
    _ensure_defaults()

    sid = get_session_id() or init_session_context()

    # Guard: if we already restored for this sid in this Streamlit session,
    # do not restore again (prevents clobbering widget-updated values).
    if (
        st.session_state.get(_REDIS_RESTORE_DONE_KEY) is True
        and str(st.session_state.get(_REDIS_RESTORE_SID_KEY) or "") == str(sid)
    ):
        return False
    r = _redis_client()
    contract = _load_contract()

    # If the contract is empty, do nothing (safe default).
    if not contract.keys:
        return False

    if r is None:
        return False

    try:
        blob = r.get(_redis_key(sid))
        if not blob:
            return False

        # Envelope = {"v": int, "ts": int, "compressed": bool, "state": {...}}
        # We store the envelope as JSON (optionally zlib-compressed).
        # To detect compression, we first parse a small header attempt; if not JSON, assume compressed.
        # Simpler: we always store as bytes + a separate meta key would be overkill; so we embed `compressed`.
        # Therefore: if JSON parse fails, try zlib and parse again.
        try:
            env = json.loads(blob.decode("utf-8"))
            compressed = bool(env.get("compressed", False))
            if compressed:
                # If envelope says compressed but we already decoded it, handle robustly:
                # treat current 'blob' as compressed payload.
                env = _decode_payload(blob, compressed=True)
        except Exception:
            env = _decode_payload(blob, compressed=True)

        state = dict(env.get("state", {}))

        # Apply only keys in the contract
        for k in contract.keys:
            if k in state:
                if overwrite_existing or (k not in st.session_state):
                    st.session_state[k] = state[k]

        # Cache hash so we avoid redundant writes on the next run
        st.session_state["_redis_snapshot_hash"] = str(env.get("hash", "")) or _hash_state_subset()

        # Mark restore complete for this sid, so we never clobber widget state on reruns.
        st.session_state[_REDIS_RESTORE_DONE_KEY] = True
        st.session_state[_REDIS_RESTORE_SID_KEY] = str(sid)

        return True
    except Exception:
        return False


def _hash_state_subset() -> str:
    contract = _load_contract()
    subset = {k: st.session_state.get(k) for k in contract.keys}
    return sha256(_json_dumps(subset)).hexdigest()


def maybe_persist_state(*, force: bool = False) -> bool:
    """
    Writes a Redis snapshot if:
      - Redis is configured, and
      - the persisted subset changed since the last stored hash (or force=True)

    Intended usage: call once near the end of each page's main().
    """
    contract = _load_contract()
    if not contract.keys:
        return False

    r = _redis_client()
    if r is None:
        return False

    sid = get_session_id() or init_session_context()

    current_hash = _hash_state_subset()
    last_hash = str(st.session_state.get("_redis_snapshot_hash", ""))

    if (not force) and last_hash == current_hash:
        return False

    envelope = {
        "v": contract.version,
        "ts": int(time.time()),
        "hash": current_hash,
        "compressed": False,  # will be set by encoder if needed
        "state": {k: st.session_state.get(k) for k in contract.keys},
    }

    data, compressed = _encode_payload(envelope)
    if compressed:
        # If we compress, we embed compressed flag inside the payload itself.
        # So we must rebuild with compressed=True and re-encode.
        envelope["compressed"] = True
        data, _ = _encode_payload(envelope)

    try:
        r.setex(_redis_key(sid), _ttl_seconds(), data)
        st.session_state["_redis_snapshot_hash"] = current_hash
        return True
    except Exception:
        return False

def restore_persisted_state_once(*, overwrite_existing: bool = True) -> bool:
    """
    Restore persisted state at most once per Streamlit session (per page load).
    This prevents Redis restore from clobbering widget interactions on every rerun.

    - On a hard refresh/new session_state: this runs once and restores.
    - On ordinary reruns (widget clicks): this is a no-op.
    """
    flag_key = "_redis_restore_done"

    # If we've already attempted restore in this session, do not do it again.
    if bool(st.session_state.get(flag_key, False)):
        return False

    # Mark as done *before* the network call to avoid repeated retries on transient reruns.
    st.session_state[flag_key] = True

    return restore_persisted_state(overwrite_existing=overwrite_existing)