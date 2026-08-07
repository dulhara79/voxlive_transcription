"""
context.py — TenantContext: who is asking, and on whose behalf.

The supervisor's §37: don't make every route repeat

    get user -> get organization -> check organization -> query

Resolve it ONCE at the connection boundary, put it in an immutable object,
and have every service downstream consume that object.

    Cognito JWT
        │
        ▼
    PrincipalResolver          (auth/principal.py)
        │
        ▼
    TenantContext              <- this file
        ├── organization_id
        ├── user_id
        ├── role
        └── session_id
        │
        ├──► QuotaManager      "may this organization start another session?"
        ├──► SessionManager    "register it under this organization"
        ├──► logging           "stamp every line with organization_id"
        └──► persistence       "scope every query to organization_id"

WHY FROZEN
----------
A mutable tenant context is a privilege-escalation bug waiting to be written:
some helper deep in the call stack reassigns `ctx.organization_id` and every
check that already passed becomes meaningless. Frozen means the identity that
was authenticated at connect time is the identity used for the whole session,
and any attempt to change it raises.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Optional

from ..tenant.models import Organization, User, UserRole


@dataclass(frozen=True)
class TenantContext:
    """Immutable identity for one authenticated connection."""

    organization_id: str
    organization_name: str
    user_id: str
    user_email: str
    role: UserRole
    plan_code: str
    session_id: Optional[str] = None

    @classmethod
    def from_records(
        cls,
        organization: Organization,
        user: User,
        session_id: Optional[str] = None,
    ) -> "TenantContext":
        if user.organization_id != organization.id:
            # Defence in depth. The repository already refuses cross-tenant
            # reads; this catches a caller that assembled the pair by hand.
            raise ValueError(
                f"user {user.id} does not belong to organization {organization.id}"
            )
        return cls(
            organization_id=organization.id,
            organization_name=organization.name,
            user_id=user.id,
            user_email=user.email,
            role=user.role,
            plan_code=organization.plan_code,
            session_id=session_id,
        )

    def with_session(self, session_id: str) -> "TenantContext":
        """A copy carrying the session id. Frozen objects are replaced, not
        edited, so the pre-session context can never be mutated underneath a
        check that already used it."""
        return TenantContext(
            organization_id=self.organization_id,
            organization_name=self.organization_name,
            user_id=self.user_id,
            user_email=self.user_email,
            role=self.role,
            plan_code=self.plan_code,
            session_id=session_id,
        )

    @property
    def can_start_session(self) -> bool:
        return self.role in (UserRole.OWNER, UserRole.ADMIN, UserRole.MEMBER)

    def log_fields(self) -> dict:
        """Fields for `observability.logging.bind()`.

        Deliberately excludes the email address. Logs go to CloudWatch and are
        read by operators; an organization id is enough to investigate an
        incident, and personal data in logs is a retention and PDPA problem
        nobody wants to inherit.
        """
        fields = {
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "tenant_id": self.organization_id,  # alias for generic dashboards
        }
        if self.session_id:
            fields["session_id"] = self.session_id
        return fields


# Ambient context for code too deep to be handed the object explicitly.
# asyncio propagates contextvars per task, so concurrent sessions on the same
# worker never observe each other's tenant.
_current: contextvars.ContextVar[Optional[TenantContext]] = contextvars.ContextVar(
    "voxlive_tenant_context", default=None
)


def set_current(ctx: Optional[TenantContext]):
    """Set the ambient context. Returns a token for `reset_current`."""
    return _current.set(ctx)


def reset_current(token) -> None:
    """Restore the previous context. Always pair with `set_current` in a
    `finally`, so a connection cannot leave its tenant bound to the task."""
    _current.reset(token)


def get_current() -> Optional[TenantContext]:
    return _current.get()


def require_current() -> TenantContext:
    """The current context, or raise.

    Used by data-access code that must never run unscoped. Raising here turns
    "forgot to set the tenant" from a silent cross-tenant read into a loud
    failure at the moment the mistake is made.
    """
    ctx = _current.get()
    if ctx is None:
        raise LookupError(
            "no TenantContext is set — refusing to proceed unscoped. "
            "Every data access must run inside an authenticated context."
        )
    return ctx
