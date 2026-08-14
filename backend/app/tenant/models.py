"""
models.py — the tenant domain model.

    Tenant == Organization == Company.

    Organization
        ├── Users
        └── Sessions
                ├── transcript_segments
                └── speakers

These are PLAIN dataclasses with no database dependency, on purpose. The
PostgreSQL commit adds `persistence/models/` (SQLAlchemy) and maps to these;
until then the same objects are served from an in-memory repository. Keeping
the domain free of ORM types means the quota manager, the WebSocket route and
the session layer can all be tested without a database running.

WHY `Plan` IS A LOOKUP, NOT A COLUMN OF NUMBERS
-----------------------------------------------
An organization stores WHICH plan it is on. The limits belong to the plan:

    Starter     max_concurrent_sessions = 5
    Business    max_concurrent_sessions = 25
    Enterprise  max_concurrent_sessions = 100

`Organization.max_concurrent_sessions` exists as an OPTIONAL override for the
one customer who negotiated something different. Without the override field
you end up either editing plan definitions per customer (which breaks every
other customer on that plan) or copying limits onto every row (which drifts).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


def new_id(prefix: str) -> str:
    """Prefixed, sortable identifier: `org_<ms>_<8 hex>`.

    The prefix means an id is self-describing in a log line or a bug report —
    you can tell `org_...` from `usr_...` from `sess_...` without a lookup.
    """
    return f"{prefix}_{int(time.time() * 1000):013d}_{uuid.uuid4().hex[:8]}"


class OrganizationStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"  # non-payment / policy — refuse new sessions
    TRIAL = "trial"
    CLOSED = "closed"


class UserStatus(str, Enum):
    ACTIVE = "active"
    INVITED = "invited"  # created but has never signed in
    DISABLED = "disabled"


class UserRole(str, Enum):
    """Roles are per-organization, never global.

    A platform-wide "superadmin" is deliberately absent. The moment one
    exists, tenant isolation has an exception in it, and exceptions are what
    get exploited. Support access should be a separate, audited mechanism.
    """

    OWNER = "owner"  # billing + can delete the organization
    ADMIN = "admin"  # manage users and settings
    MEMBER = "member"  # can run sessions
    VIEWER = "viewer"  # read transcripts only, cannot start a session


@dataclass(frozen=True)
class Plan:
    """A subscription tier. Frozen: plan limits are configuration, not state."""

    code: str
    name: str
    max_concurrent_sessions: int
    max_session_minutes: int  # per single session
    max_monthly_minutes: int  # per organization, per calendar month
    retention_days: int  # how long transcripts/recordings are kept


# The catalogue. Values here are PLACEHOLDERS pending business requirements —
# the supervisor's §13 is explicit that the real numbers come from the
# business, not from engineering.
PLANS: dict[str, Plan] = {
    "starter": Plan(
        code="starter",
        name="Starter",
        max_concurrent_sessions=5,
        max_session_minutes=60,
        max_monthly_minutes=1_200,
        retention_days=30,
    ),
    "business": Plan(
        code="business",
        name="Business",
        max_concurrent_sessions=25,
        max_session_minutes=180,
        max_monthly_minutes=12_000,
        retention_days=180,
    ),
    "enterprise": Plan(
        code="enterprise",
        name="Enterprise",
        max_concurrent_sessions=100,
        max_session_minutes=480,
        max_monthly_minutes=120_000,
        retention_days=730,
    ),
}

DEFAULT_PLAN = "starter"


@dataclass
class Organization:
    """A tenant. Maps to the `organizations` table."""

    id: str
    name: str
    plan_code: str = DEFAULT_PLAN
    status: OrganizationStatus = OrganizationStatus.TRIAL
    # Per-customer override. None means "use the plan's value".
    max_concurrent_sessions: Optional[int] = None
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        name: str,
        plan_code: str = DEFAULT_PLAN,
        status: OrganizationStatus = OrganizationStatus.TRIAL,
        **kwargs,
    ) -> "Organization":
        if plan_code not in PLANS:
            raise ValueError(f"unknown plan {plan_code!r}; known: {sorted(PLANS)}")
        return cls(
            id=new_id("org"), name=name, plan_code=plan_code, status=status, **kwargs
        )

    @property
    def plan(self) -> Plan:
        # Falls back rather than raising: a plan removed from the catalogue
        # must not take every one of its customers offline mid-request.
        return PLANS.get(self.plan_code, PLANS[DEFAULT_PLAN])

    @property
    def concurrent_session_limit(self) -> int:
        """Effective limit: the override if set, otherwise the plan's."""
        if self.max_concurrent_sessions is not None:
            return self.max_concurrent_sessions
        return self.plan.max_concurrent_sessions

    @property
    def can_start_sessions(self) -> bool:
        return self.status in (OrganizationStatus.ACTIVE, OrganizationStatus.TRIAL)


@dataclass
class User:
    """A member of exactly one organization. Maps to the `users` table.

    `cognito_sub` is the link to the identity provider and is the field the
    Cognito commit will look up on. It is nullable so seed/dev users and
    service accounts can exist before an identity pool does.
    """

    id: str
    organization_id: str
    email: str
    name: str = ""
    role: UserRole = UserRole.MEMBER
    status: UserStatus = UserStatus.ACTIVE
    cognito_sub: Optional[str] = None
    # scrypt hash produced by auth/passwords.py. Nullable because an INVITED
    # user exists before they have chosen one, and because a future Cognito
    # migration would leave this empty for every federated account.
    # `repr=False` keeps it out of tracebacks and log lines that print a User.
    password_hash: Optional[str] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(cls, organization_id: str, email: str, **kwargs) -> "User":
        return cls(
            id=new_id("usr"),
            organization_id=organization_id,
            email=email.strip().lower(),
            **kwargs,
        )

    @property
    def can_start_sessions(self) -> bool:
        return self.status is UserStatus.ACTIVE and self.role in (
            UserRole.OWNER,
            UserRole.ADMIN,
            UserRole.MEMBER,
        )
