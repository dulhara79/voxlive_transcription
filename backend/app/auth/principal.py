"""
principal.py — turning an incoming connection into a TenantContext.

    connection ──► PrincipalResolver ──► TenantContext | AuthenticationError

This is the seam Cognito plugs into (the supervisor's step C). Today there are
two implementations:

    DevPrincipalResolver       development only, trusts query parameters
    CognitoPrincipalResolver   the next commit — verifies a JWT against JWKS

THE FAIL-CLOSED RULE
--------------------
The supervisor's §5 is unambiguous: the backend must refuse ORG_002 data, and
the check happens server-side. A resolver that trusts `?organization_id=` from
the client is the exact opposite of that — it is not authentication, it is the
client telling you who it would like to be.

So `DevPrincipalResolver` REFUSES TO CONSTRUCT outside development. Not a
warning, not a config flag that defaults to on — a hard `RuntimeError` at
start-up. `build_resolver()` in staging or production returns the Cognito
resolver or raises.

The consequence is deliberate: until the Cognito commit lands,
`APP_ENV=production` cannot serve WebSocket traffic at all. That is the
correct failure mode. A system that accepts anonymous connections in
production because auth "isn't done yet" is how tenant data leaks, and the
window between "deployed" and "auth finished" is exactly when it happens.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..tenant.models import Organization, User, UserStatus
from ..tenant.repository import TenantRepository
from .context import TenantContext

log = logging.getLogger("voxlive.auth")


class AuthenticationError(Exception):
    """Could not establish who is connecting. Close with 1008 (policy)."""


class AuthorizationError(Exception):
    """Identity established, but not permitted to do this."""


class PrincipalResolver(ABC):
    @abstractmethod
    async def resolve(self, token: Optional[str], **hints) -> TenantContext:
        """Return the authenticated context, or raise AuthenticationError."""


class DevPrincipalResolver(PrincipalResolver):
    """Development-only resolver. NEVER usable outside APP_ENV=development.

    Accepts `?organization_id=&user_id=` so the React frontend and the load
    harness can exercise tenant-aware code paths before Cognito exists. It
    still goes through the repository, so an unknown organization or a user
    from the wrong organization is refused — the isolation logic under test is
    the real one, only the identity assertion is fake.
    """

    def __init__(self, repo: TenantRepository, app_env: str):
        if app_env != "development":
            raise RuntimeError(
                f"DevPrincipalResolver refused to start under APP_ENV={app_env!r}. "
                "It trusts client-supplied identity and must never run outside "
                "development. Configure Cognito for staging/production."
            )
        self.repo = repo
        log.warning(
            "AUTH IS IN DEVELOPMENT MODE — identity is taken from query "
            "parameters and is NOT verified. Never deploy this."
        )

    async def resolve(self, token: Optional[str], **hints) -> TenantContext:
        organization_id = hints.get("organization_id")
        user_id = hints.get("user_id")
        if not organization_id or not user_id:
            raise AuthenticationError(
                "development auth requires ?organization_id= and ?user_id="
            )

        organization: Optional[Organization] = await self.repo.get_organization(
            organization_id
        )
        if organization is None:
            raise AuthenticationError("unknown organization")

        # Organization-scoped: a user id from another tenant returns None.
        user: Optional[User] = await self.repo.get_user(user_id, organization_id)
        if user is None:
            raise AuthenticationError("unknown user for this organization")

        # Organization status and user status are AUTHENTICATION concerns: a
        # suspended tenant or a disabled account should not get a context at
        # all. Role is NOT checked here — a VIEWER authenticates successfully
        # and is refused at the action. Collapsing the two would make this
        # resolver unusable for the read-only transcript routes in §38, and
        # would report "who are you?" for what is really "you may not do that".
        if not organization.can_start_sessions:
            raise AuthorizationError(f"organization is {organization.status.value}")
        if user.status is not UserStatus.ACTIVE:
            raise AuthorizationError(f"user account is {user.status.value}")

        return TenantContext.from_records(organization, user)


def build_resolver(repo: TenantRepository, app_env: str) -> PrincipalResolver:
    """Choose a resolver for this environment. Fails closed."""
    if app_env == "development":
        return DevPrincipalResolver(repo, app_env)
    raise RuntimeError(
        f"no production-grade PrincipalResolver is configured for "
        f"APP_ENV={app_env!r}. Implement CognitoPrincipalResolver "
        "(auth/cognito.py) before deploying. Refusing to start rather than "
        "serving unauthenticated tenant traffic."
    )
