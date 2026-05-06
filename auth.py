"""Authentication module for the fence detection system.

Supports four modes (via FENCE_AUTH_MODE env var):
  - none:              dev mode, session-scoped user ID
  - proxy_header:      reverse proxy injects X-Forwarded-User
  - streamlit_oidc:    Streamlit native OIDC (Google/Okta)
  - streamlit_password: shared password gate
"""

from __future__ import annotations

import hashlib
import uuid

import streamlit as st

from config import cfg


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def get_user_email() -> str | None:
    try:
        if cfg.AUTH_MODE == "streamlit_oidc":
            u = getattr(st, "user", None)
            if u is not None and getattr(u, "is_logged_in", False):
                return getattr(u, "email", None)
        elif cfg.AUTH_MODE == "proxy_header":
            try:
                return st.context.headers.get("X-Forwarded-User")
            except Exception:
                return None
        elif cfg.AUTH_MODE == "streamlit_password":
            return st.session_state.get("_auth_email")
    except Exception:
        return None
    return None


def get_user_id() -> str:
    email = get_user_email()
    if email:
        return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]
    if cfg.AUTH_MODE == "streamlit_password" and st.session_state.get("_auth_ok"):
        who = st.session_state.get("_auth_email", "")
        if who:
            return hashlib.sha256(who.strip().lower().encode()).hexdigest()[:16]
    return f"dev_{get_session_id()}"


def _streamlit_password_gate():
    if st.session_state.get("_auth_ok"):
        return
    expected = cfg.APP_PASSWORD
    if not expected:
        try:
            expected = st.secrets.get("auth", {}).get("password", "") if hasattr(st, "secrets") else ""
        except Exception:
            expected = ""
    if not expected:
        st.error("Auth misconfigured: FENCE_AUTH_MODE=streamlit_password requires "
                 "FENCE_APP_PASSWORD env var or [auth].password in secrets.toml.")
        st.stop()
    st.markdown("## Sign in")
    with st.form("_auth_form", clear_on_submit=False):
        email = st.text_input("Email", key="_auth_email_input")
        pw = st.text_input("Password", type="password", key="_auth_pw_input")
        submitted = st.form_submit_button("Log in")
    if submitted:
        if pw == expected and email.strip():
            st.session_state["_auth_ok"] = True
            st.session_state["_auth_email"] = email.strip()
            st.rerun()
        else:
            st.error("Invalid credentials.")
    st.stop()


def require_auth():
    if cfg.AUTH_MODE == "none":
        return
    if cfg.AUTH_MODE == "streamlit_password":
        _streamlit_password_gate()
        return
    if cfg.AUTH_MODE == "streamlit_oidc":
        try:
            if not st.user.is_logged_in:
                st.title("Leo Fence Detection")
                st.markdown("Sign in with your Google account to continue.")
                st.login("google")
                st.stop()
        except AttributeError:
            pass
        return
    if get_user_id().startswith("dev_"):
        st.error("Authentication required. Please sign in and retry.")
        st.stop()


def render_auth_widget():
    if cfg.AUTH_MODE == "none":
        return
    email = get_user_email() or st.session_state.get("_auth_email")
    if not email:
        return
    with st.sidebar:
        st.markdown(f"**Signed in:** `{email}`")
        if cfg.AUTH_MODE == "streamlit_oidc":
            if st.button("Log out", key="_auth_logout"):
                st.logout()
        elif cfg.AUTH_MODE == "streamlit_password":
            if st.button("Log out", key="_auth_logout"):
                for k in ("_auth_ok", "_auth_email"):
                    st.session_state.pop(k, None)
                st.rerun()
