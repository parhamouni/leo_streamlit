"""Supabase JWT verification for the FastAPI backend.

Verifies tokens against the project's JWKS endpoint (asymmetric, ES256).
No shared secret is stored on the server — the public key rotates with
Supabase's key rotation automatically via JWKS.

Two FastAPI dependencies are exposed:

  require_supabase_jwt — always demands a valid Bearer token.
                         Use for endpoints that must authenticate via Supabase
                         (e.g. /api/me).
  get_current_user     — respects cfg.API_AUTH_MODE.
                         Use for endpoints that should follow the migration
                         mode (legacy_header / supabase / both).

Raises HTTPException(401) on missing or invalid credentials.
"""

from __future__ import annotations

import logging
from typing import Optional

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

from config import cfg

log = logging.getLogger("auth_supabase")


_ALLOWED_ALGS = ["ES256", "RS256"]
_EXPECTED_AUD = "authenticated"  # Supabase's default audience claim

_jwks_client: Optional[PyJWKClient] = None


def _jwks_url() -> str:
    if cfg.SUPABASE_JWKS_URL:
        return cfg.SUPABASE_JWKS_URL
    if not cfg.SUPABASE_URL:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL not configured on server",
        )
    return f"{cfg.SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(_jwks_url(), cache_keys=True, lifespan=600)
    return _jwks_client


def _verify(token: str) -> str:
    """Decode and verify a Supabase JWT. Returns the user_id (sub claim)."""
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token).key
    except Exception as e:
        log.warning("JWKS lookup failed: %s", e)
        raise HTTPException(status_code=401, detail="Could not resolve token signing key")

    try:
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=_ALLOWED_ALGS,
            audience=_EXPECTED_AUD,
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidAudienceError:
        raise HTTPException(status_code=401, detail="Invalid token audience")
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing sub claim")
    return str(sub)


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


async def require_supabase_jwt(
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Always require a valid Supabase Bearer token. Returns user_id."""
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization: Bearer token",
        )
    return _verify(token)


async def get_current_user(
    authorization: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
) -> str:
    """AUTH_MODE-aware dependency. Returns user_id."""
    mode = cfg.API_AUTH_MODE
    bearer = _extract_bearer(authorization)

    if mode == "supabase":
        if not bearer:
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        return _verify(bearer)

    if mode == "both":
        if bearer:
            return _verify(bearer)
        if x_user_id:
            return x_user_id
        raise HTTPException(status_code=401, detail="No credentials provided")

    # Default and "legacy_header" — match existing api_server behavior:
    # accept X-User-Id, fall back to "anonymous".
    return x_user_id or "anonymous"
