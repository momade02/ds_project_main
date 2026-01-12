from __future__ import annotations

from pathlib import Path

import streamlit as st


# ui/ is located at: src/app/ui
# static/ is located at: src/app/static
_APP_ROOT = Path(__file__).resolve().parents[1]
_CSS_PATH = _APP_ROOT / "static" / "app.css"


@st.cache_data(show_spinner=False)
def _load_css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")


def apply_app_css() -> None:
    """
    Inject global app CSS for all Streamlit pages.
    Call once per page, ideally right after st.set_page_config(...).
    """
    css = _load_css()
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
