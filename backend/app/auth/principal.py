"""
principal.py — turning an incoming connection into a TenantContext.

    connection ──► PrincipalResolver ──► TenantContext | AuthenticationError

Three implementations:

    DevPrincipalResolver     development only, trusts query parameters
    LocalPrincipalResolver   verifies an HS256 token minted by routes_auth
    CognitoPrincipalResolver not written — the seam it would occupy is
                             `build_resolver`, and nothing else changes

THE FAIL-CLOSED RULE
--------------------
A resolver that trusts `?organization_id=` from the client is not
authentication — it is the client telling you who it would like to be. So
`DevPrincipalResolver` REFUSES TO CONSTRUCT outside development: a hard
`RuntimeError` at start-up, not a warning and not a config flag.

WHAT CHANGED
------------
`build_resolver` previously raised outside development, which meant
`APP_ENV=production` could not serve WebSocket traffic at all. That was the
correct failure mode while no real resolver existed. `LocalPrincipalResolver`
now fills it, and the fail-closed behaviour moves to the signing secret:
without `AUTH_JWT_SECRET`, token verification raises at start-up rather than
falling back to an ephemeral key.

WHY THE ROLE CHECK IS NOT HERE
------------------------------
Organization status and user status are AUTHENTICATION concerns: a suspended
tenant or a disabled account should not get a context at all. Role is not — a
VIEWER authenticates successfully and is refused at the action. Collapsing the
two would report "who are you?" for what is really "you may not do that", and
would make this resolver unusable for read-only transcript routes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..tenant.models import Organization, User, UserStatus
from ..tenant.repository import TenantRepository
from .context import TenantContext
from .tokens import TokenError, read_access_token, require_signing_secret

log = logging.getLogger("voxlive.auth")


class AuthenticationError(Exception):
    """Could not establish who is connecting. Close with 1008 (policy)."""


class AuthorizationError(Exception):
    """Identity established, but not permitted to do this."""


class PrincipalResolver(ABC):
    @abstractmethod
    async def resolve(self, token: Optional[str], **hints) -> TenantContext:
        """Return the authenticated context, or raise AuthenticationError."""


async def _context_for(
    repo: TenantRepository, user_id: str, organization_id: str
) -> TenantContext:
    """Shared tail of every resolver: load, check status, build the context.

    The user is re-read from the repository on every connection rather than
    reconstructed from token claims. A token is valid for hours; an account
    disabled ten minutes ago must not be able to open a new session with it.
    """
    organization: Optional[Organization] = await repo.get_organization(organization_id)
    if organization is None:
        raise AuthenticationError("unknown organization")

    # Organization-scoped: a user id from another tenant returns None.
    user: Optional[User] = await repo.get_user(user_id, organization_id)
    if user is None:
        raise AuthenticationError("unknown user for this organization")

    if not organization.can_start_sessions:
        raise AuthorizationError(f"organization is {organization.status.value}")
    if user.status is not UserStatus.ACTIVE:
        raise AuthorizationError(f"user account is {user.status.value}")

    return TenantContext.from_records(organization, user)


class LocalPrincipalResolver(PrincipalResolver):
    """Verifies the signed token issued by `POST /auth/login`.

    The token arrives as `?token=` because a browser's WebSocket API cannot
    set an Authorization header. That places a credential in a URL, with two
    deployment consequences that are requirements rather than hardening:

      * the token TTL must stay short (AUTH_ACCESS_TTL_SEC), because URLs
        reach browser history, proxy logs and referrer headers
      * ALB access logging must not record query strings

    Query-parameter identity hints are IGNORED here. If they were read as a
    fallback the endpoint would silently accept unauthenticated connections
    the moment a token failed to parse, which is the exact failure this class
    exists to remove.
    """

    def __init__(self, repo: TenantRepository):
        self.repo = repo

    async def resolve(self, token: Optional[str], **hints) -> TenantContext:
        if not token:
            raise AuthenticationError("sign in to start a session")
        try:
            claims = read_access_token(token)
        except TokenError as exc:
            raise AuthenticationError(str(exc)) from None
        return await _context_for(self.repo, claims.user_id, claims.organization_id)


class DevPrincipalResolver(PrincipalResolver):
    """Development-only resolver. NEVER usable outside APP_ENV=development.

    Accepts a real token if one is supplied, and falls back to
    `?organization_id=&user_id=` so the load harness can exercise
    tenant-aware code paths without a login. It still goes through the
    repository, so an unknown organization or a user from the wrong
    organization is refused — the isolation logic under test is the real one,
    only the identity assertion is fake.
    """

    def __init__(self, repo: TenantRepository, app_env: str):
        if app_env != "development":
            raise RuntimeError(
                f"DevPrincipalResolver refused to start under APP_ENV={app_env!r}. "
                "It trusts client-supplied identity and must never run outside "
                "development."
            )
        self.repo = repo
        self._real = LocalPrincipalResolver(repo)
        log.warning(
            "AUTH IS IN DEVELOPMENT MODE — a connection without a token may "
            "assert its own identity via query parameters. Never deploy this."
        )

    async def resolve(self, token: Optional[str], **hints) -> TenantContext:
        if token:
            return await self._real.resolve(token, **hints)

        organization_id = hints.get("organization_id")
        user_id = hints.get("user_id")
        if not organization_id or not user_id:
            raise AuthenticationError(
                "sign in, or supply ?organization_id= and ?user_id= in development"
            )
        return await _context_for(self.repo, user_id, organization_id)


def build_resolver(repo: TenantRepository, app_env: str) -> PrincipalResolver:
    """Choose a resolver for this environment. Fails closed."""
    if app_env == "development":
        return DevPrincipalResolver(repo, app_env)

    # Check the signing secret at start-up, against the app_env we were
    # HANDED rather than the one in the environment. This raises outside
    # development when AUTH_JWT_SECRET is unset, and it is much better for
    # that to happen here — where ECS reports a failed task and holds the old
    # one in service — than on the first user's first connection.
    require_signing_secret(app_env)
    log.info("auth: local HS256 tokens (APP_ENV=%s)", app_env)
    return LocalPrincipalResolver(repo)
