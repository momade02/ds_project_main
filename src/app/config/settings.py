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
