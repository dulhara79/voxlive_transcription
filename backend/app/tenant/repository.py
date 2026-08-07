"""
repository.py — the persistence seam for tenant data.

This is the supervisor's "persistence abstraction" (§43 commit 2), delivered
BEFORE PostgreSQL rather than after. The reason is ordering: tenant context
and quotas need somewhere to read organizations from, and if they read from a
concrete database the database ends up wired into the WebSocket route. An
interface here means the PostgreSQL commit changes one line in `main.py` and
nothing else.

    routes_ws / quotas / session
              │
              ▼
      TenantRepository  (interface)
         ├── InMemoryTenantRepository   <- today, and forever in tests
         └── SqlTenantRepository        <- the PostgreSQL commit

TENANT ISOLATION IS ENFORCED HERE, NOT ABOVE
--------------------------------------------
Note the shape of `get_user`: it takes BOTH a user id and an organization id,
and returns None if the user belongs to a different organization. This is the
supervisor's §36 rule expressed as a method signature — the SQL version will
be `WHERE id = :user_id AND organization_id = :org_id`, never a fetch followed
by a check in Python. Making the interface require the tenant id means a
caller CANNOT accidentally write the unscoped query; there is no method that
would let them.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

from .models import Organization, User


class TenantRepository(ABC):
    """Read/write access to organizations and users.

    Every method that touches a user-owned row takes `organization_id`. That
    is not redundancy — it is the isolation boundary.
    """

    @abstractmethod
    async def get_organization(
        self, organization_id: str
    ) -> Optional[Organization]: ...

    @abstractmethod
    async def get_user(self, user_id: str, organization_id: str) -> Optional[User]:
        """Return the user ONLY if they belong to this organization."""

    @abstractmethod
    async def get_user_by_cognito_sub(self, sub: str) -> Optional[User]:
        """Resolve an identity-provider subject to a user.

        Not organization-scoped, because this is what ESTABLISHES the
        organization: the JWT gives a subject, this returns the user, and the
        user carries the organization_id every later call is scoped by.
        """

    @abstractmethod
    async def save_organization(self, organization: Organization) -> Organization: ...

    @abstractmethod
    async def save_user(self, user: User) -> User: ...


class InMemoryTenantRepository(TenantRepository):
    """Dict-backed repository.

    Real in development and in tests; replaced by SQL in production. It is
    NOT a toy — it implements the same isolation rules, so a test that passes
    here would pass against PostgreSQL.
    """

    def __init__(self) -> None:
        self._orgs: dict[str, Organization] = {}
        self._users: dict[str, User] = {}
        self._by_sub: dict[str, str] = {}  # cognito_sub -> user_id
        self._lock = asyncio.Lock()

    async def get_organization(self, organization_id: str) -> Optional[Organization]:
        return self._orgs.get(organization_id)

    async def get_user(self, user_id: str, organization_id: str) -> Optional[User]:
        user = self._users.get(user_id)
        if user is None or user.organization_id != organization_id:
            # Cross-tenant lookup returns None, exactly as the scoped SQL
            # query would return zero rows. It must not raise a distinguishable
            # error, or the difference between "no such user" and "someone
            # else's user" becomes an enumeration oracle.
            return None
        return user

    async def get_user_by_cognito_sub(self, sub: str) -> Optional[User]:
        user_id = self._by_sub.get(sub)
        return self._users.get(user_id) if user_id else None

    async def save_organization(self, organization: Organization) -> Organization:
        async with self._lock:
            self._orgs[organization.id] = organization
        return organization

    async def save_user(self, user: User) -> User:
        async with self._lock:
            self._users[user.id] = user
            if user.cognito_sub:
                self._by_sub[user.cognito_sub] = user.id
        return user

    # ---- helpers that only make sense for the in-memory implementation ----

    async def list_organizations(self) -> list[Organization]:
        return list(self._orgs.values())

    async def count(self) -> tuple[int, int]:
        return len(self._orgs), len(self._users)
