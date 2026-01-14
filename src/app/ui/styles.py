from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def _find_css_file() -> Path | None:
    """
    Find app.css without hardcoding absolute paths.

    Expected layout:
      src/app/ui/styles.py
      src/app/static/app.css
    """
    here = Path(__file__).resolve()

    candidates = [
        # Normal case: ./src/app/ui/styles.py -> parents[1] == ./src/app
        here.parents[1] / "static" / "app.css",
        # If someone moved styles.py up/down in the future:
        here.parents[0] / "static" / "app.css",
        here.parents[2] / "static" / "app.css",
        # Fallbacks:
        Path.cwd() / "src" / "app" / "static" / "app.css",
        Path.cwd() / "static" / "app.css",
        Path.cwd() / "app.css",
    ]

    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def _load_css_text(css_path_str: str, mtime: float) -> str:
    # mtime is only here to invalidate the cache automatically when app.css changes
    return Path(css_path_str).read_text(encoding="utf-8")


def apply_app_css() -> None:
    css_path = _find_css_file()
    if not css_path:
        st.warning("app.css not found (apply_app_css). Expected at: src/app/static/app.css")
        return

    # 1) Inject CSS into the main Streamlit document
    css = _load_css_text(str(css_path), css_path.stat().st_mtime)
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)

    # 2) Inject JS that manipulates the *parent* Streamlit document.
    #    Key fix: we FORCE inline styles on the real sidebar element (including transform),
    #    so the drawer opens even if Streamlit's CSS would otherwise override us.
    components.html(
        """
        <script>
        (function () {
          const P = window.parent;
          const doc = P.document;

          const TOGGLE_ID = "mobile-sidebar-toggle";

          function isMobile() { return P.innerWidth <= 768; }

          function qsa(sel) {
            try { return Array.from(doc.querySelectorAll(sel)); }
            catch (e) { return []; }
          }

          function ensureToggle() {
            const body = doc.body;
            if (!body) return null;

            let toggle = doc.getElementById(TOGGLE_ID);
            if (!toggle) {
              toggle = doc.createElement("button");
              toggle.id = TOGGLE_ID;
              toggle.type = "button";
              toggle.setAttribute("aria-label", "Toggle sidebar");
              body.appendChild(toggle);
            }
            return toggle;
          }

          function nativeToggleSelectors() {
            return [
              '[data-testid="collapsedControl"]',
              '[data-testid="stSidebarCollapsedControl"]',
              'button[aria-label="Open sidebar"]',
              'button[aria-label="Close sidebar"]',
              'button[aria-label="Open navigation"]',
              'button[aria-label="Close navigation"]',
              'button[aria-label*="sidebar" i]',
              'button[title*="sidebar" i]'
            ];
          }

          function hideNativeSidebarToggles() {
            // Only hide native controls on MOBILE.
            // On desktop we want Streamlit's normal sidebar behavior (and its own collapse control).
            if (!isMobile()) return;

            nativeToggleSelectors().forEach((sel) => {
              qsa(sel).forEach((el) => {
                if (el && el.id !== TOGGLE_ID) {
                  el.style.setProperty("display", "none", "important");
                }
              });
            });
          }

          function restoreNativeSidebarToggles() {
            // If we previously hid native controls (e.g., during a resize),
            // restore them when leaving mobile.
            if (isMobile()) return;

            nativeToggleSelectors().forEach((sel) => {
              qsa(sel).forEach((el) => {
                if (el && el.id !== TOGGLE_ID) {
                  el.style.removeProperty("display");
                }
              });
            });
          }

          function findOpenButton() {
            const candidates = [
              'button[aria-label="Open sidebar"]',
              'button[aria-label="Open navigation"]',
              '[data-testid="collapsedControl"] button',
              '[data-testid="collapsedControl"]',
              '[data-testid="stSidebarCollapsedControl"] button',
              '[data-testid="stSidebarCollapsedControl"]',
              'button[aria-label*="Open" i][aria-label*="sidebar" i]',
              'button[aria-label*="Open" i][aria-label*="navigation" i]'
            ];
            for (const sel of candidates) {
              const els = qsa(sel);
              if (els.length) return els[0];
            }
            return null;
          }

          function findCloseButton() {
            const candidates = [
              'button[aria-label="Close sidebar"]',
              'button[aria-label="Close navigation"]',
              'button[aria-label*="Close" i][aria-label*="sidebar" i]',
              'button[aria-label*="Close" i][aria-label*="navigation" i]'
            ];
            for (const sel of candidates) {
              const els = qsa(sel);
              if (els.length) return els[0];
            }
            return null;
          }

          function sidebarLooksOpen() {
            // If Streamlit shows a close button, the drawer is open.
            return !!findCloseButton();
          }

          function clickNativeToggle() {
            const closeBtn = findCloseButton();
            if (closeBtn) { try { closeBtn.click(); return; } catch (e) {} }

            const openBtn = findOpenButton();
            if (openBtn) { try { openBtn.click(); return; } catch (e) {} }
          }

          function syncToggleState() {
            const t = doc.getElementById(TOGGLE_ID);
            if (!t) return;

            if (!isMobile()) {
              t.style.display = "none";
              t.removeAttribute("data-state");
              return;
            }

            t.style.display = "inline-flex";
            t.setAttribute("data-state", sidebarLooksOpen() ? "open" : "closed");
          }

          function init() {
            const body = doc.body;
            if (!body) return;

            const toggle = ensureToggle();
            if (!toggle) return;

            // Bind once
            if (!toggle.dataset.bound) {
              toggle.dataset.bound = "1";
              toggle.addEventListener("click", function (e) {
                e.preventDefault();
                // We rely on Streamlit's own sidebar drawer; we just trigger it.
                clickNativeToggle();
                // State update after Streamlit reacts
                P.setTimeout(syncToggleState, 80);
              });
            }

            // Mobile: hide native toggles + show our button.
            // Desktop: restore native toggles + hide our button.
            if (isMobile()) hideNativeSidebarToggles();
            else restoreNativeSidebarToggles();
            syncToggleState();

            // Keep stable across reruns / DOM swaps
            if (!body.dataset.mobileSidebarObserver2) {
              body.dataset.mobileSidebarObserver2 = "1";
              const obs = new MutationObserver(function () {
                if (isMobile()) hideNativeSidebarToggles();
                else restoreNativeSidebarToggles();
                syncToggleState();
              });
              obs.observe(body, { childList: true, subtree: true });
            }

            if (!body.dataset.mobileSidebarResize2) {
              body.dataset.mobileSidebarResize2 = "1";
              P.addEventListener("resize", function () {
                if (isMobile()) hideNativeSidebarToggles();
                else restoreNativeSidebarToggles();
                syncToggleState();
              });
            }
          }

          if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", init);
          else init();
        })();
        </script>
        """,
        height=0,
        width=0,
    )
