"""
tokens.py — signed session tokens.

WHY NOT COGNITO
---------------
Cognito's free tier would cover this workload, so the argument is not the
line item. It is that Cognito adds a SECOND source of truth for users beside
the `users` table that already exists here, and the two have to be kept in
step: a signup writes to both, a deletion has to delete from both, and the
`users.cognito_sub` column exists purely to bridge them. It also puts a JWKS
fetch and a key-rotation path in the WebSocket accept path.

For a handful of organizations, HS256 tokens signed against the same database
that already holds the users is less code, no extra service, and one place
where a user exists. The cost of that choice is real and worth naming:

  * No managed MFA, no hosted password reset, no social login. Each is work
    you would otherwise get for free.
  * The signing secret is yours to rotate and to keep out of logs.

If MFA or SSO becomes a requirement, swap this for CognitoPrincipalResolver.
`build_resolver()` is the only place that would change, which is why the
resolver seam was built before either implementation existed.

TWO TOKEN KINDS
---------------
    access      Identifies a user within one organization. Sent on every
                WebSocket connection as ?token=. Short-lived, because a
                WebSocket URL ends up in browser history, proxy logs and ALB
                access logs unless those are explicitly configured not to
                record query strings.

    org_select  Issued only when one email plus password matches users in
                MORE THAN ONE organization. Proves the password was already
                verified, so the chooser step does not ask for it twice. Very
                short-lived and useless for anything else: it carries no
                organization_id, so `resolve()` rejects it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import jwt  # PyJWT

ALGORITHM = "HS256"
ISSUER = "voxlive"

ACCESS_TTL_SEC = int(os.getenv("AUTH_ACCESS_TTL_SEC", str(12 * 3600)))
# Long enough to survive a chooser screen and a slow connection, short enough
# that a leaked one is worthless.
ORG_SELECT_TTL_SEC = int(os.getenv("AUTH_ORG_SELECT_TTL_SEC", "300"))


class TokenError(Exception):
    """The token is missing, malformed, expired or not the kind expected."""


@dataclass(frozen=True)
class AccessClaims:
    user_id: str
    organization_id: str
    role: str
    email: str
    expires_at: float


def require_signing_secret(app_env: str) -> None:
    """Start-up gate. Raises unless a usable signing key exists for `app_env`.

    Called by `build_resolver`, which is handed the SAME `app_env` value the
    rest of the application was configured with. That parameter is the whole
    point: `_secret()` below re-reads `APP_ENV` from the environment, and if
    the two ever disagree — a test, a script, an entrypoint that sets one and
    not the other — the environment read wins and silently downgrades a
    production process to an ephemeral development key. Passing the value in
    explicitly makes the gate agree with the caller by construction instead of
    by coincidence.
    """
    _resolve_secret(app_env)


def _secret() -> str:
    """The HS256 signing key, for the request path.

    Read on every call rather than cached at import, so a rotation that
    restarts the process picks up the new value without a code path that
    remembers the old one.
    """
    return _resolve_secret(os.getenv("APP_ENV", "development").strip().lower())


def _resolve_secret(app_env: str) -> str:
    """Shared body. Absent in development, a stable per-process value is
    generated: every restart invalidates existing tokens, which is annoying on
    a laptop and correct everywhere else.
    """
    secret = os.getenv("AUTH_JWT_SECRET", "").strip()
    if secret:
        if len(secret) < 32:
            raise RuntimeError(
                "AUTH_JWT_SECRET is shorter than 32 characters. Generate one "
                'with: python -c "import secrets;print(secrets.token_urlsafe(48))"'
            )
        return secret

    if app_env != "development":
        raise RuntimeError(
            f"AUTH_JWT_SECRET is required under APP_ENV={app_env!r}. Refusing "
            "to sign tokens with an ephemeral key: every task would sign with "
            "a different one, so a token minted by task A would be rejected "
            "by task B and logins would fail at random behind the load "
            "balancer."
        )
    return _dev_secret()


_DEV_SECRET: Optional[str] = None


def _dev_secret() -> str:
    global _DEV_SECRET
    if _DEV_SECRET is None:
        import secrets

        _DEV_SECRET = secrets.token_urlsafe(48)
    return _DEV_SECRET


def issue_access_token(
    user_id: str, organization_id: str, role: str, email: str
) -> tuple[str, int]:
    """Return `(token, expires_in_seconds)` for an authenticated user."""
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "typ": "access",
        "sub": user_id,
        "org": organization_id,
        "role": role,
        "email": email,
        "iat": now,
        "exp": now + ACCESS_TTL_SEC,
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM), ACCESS_TTL_SEC


def issue_org_select_token(user_ids: list[str], email: str) -> tuple[str, int]:
    """Prove a password was verified, without naming an organization yet."""
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "typ": "org_select",
        "email": email,
        "candidates": list(user_ids),
        "iat": now,
        "exp": now + ORG_SELECT_TTL_SEC,
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM), ORG_SELECT_TTL_SEC


def _decode(token: str, expected_type: str) -> dict:
    if not token:
        raise TokenError("no token supplied")
    try:
        payload = jwt.decode(
            token,
            _secret(),
            algorithms=[ALGORITHM],  # a list, never None: `alg: none` is a bypass
            issuer=ISSUER,
            options={"require": ["exp", "iat", "iss"]},
        )
    except jwt.ExpiredSignatureError:
        raise TokenError("session expired, sign in again") from None
    except jwt.InvalidTokenError as exc:
        raise TokenError(f"invalid token: {exc}") from None

    if payload.get("typ") != expected_type:
        # Without this, an org_select token — issued BEFORE an organization is
        # chosen — would be accepted as an access token.
        raise TokenError("token is not valid for this operation")
    return payload


def read_access_token(token: str) -> AccessClaims:
    payload = _decode(token, "access")
    for field in ("sub", "org", "role"):
        if not payload.get(field):
            raise TokenError("token is missing required claims")
    return AccessClaims(
        user_id=payload["sub"],
        organization_id=payload["org"],
        role=payload["role"],
        email=payload.get("email", ""),
        expires_at=float(payload["exp"]),
    )


def read_org_select_token(token: str) -> tuple[list[str], str]:
    """Return `(candidate_user_ids, email)` from a chooser token."""
    payload = _decode(token, "org_select")
    candidates = payload.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise TokenError("token is missing required claims")
    return [str(c) for c in candidates], str(payload.get("email", ""))
