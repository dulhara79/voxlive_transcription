"""
repository.py — the persistence seam for tenant data.

    routes_ws / routes_auth / quotas / session
              │
              ▼
      TenantRepository  (interface)
         ├── InMemoryTenantRepository   <- today, and forever in tests
         └── SqlTenantRepository        <- the PostgreSQL commit

TENANT ISOLATION IS ENFORCED HERE, NOT ABOVE
--------------------------------------------
Note the shape of `get_user`: it takes BOTH a user id and an organization id,
and returns None if the user belongs to a different organization. The SQL
version will be `WHERE id = :user_id AND organization_id = :org_id`, never a
fetch followed by a check in Python. Making the interface require the tenant
id means a caller CANNOT accidentally write the unscoped query; there is no
method that would let them.

THE TWO DELIBERATELY UNSCOPED METHODS
-------------------------------------
`get_user_by_cognito_sub` and `find_users_by_email` do not take an
organization id, because they are what ESTABLISHES one. Everything reached
through them must immediately narrow to a single organization.

`find_users_by_email` returns a LIST, and that is the whole reason the login
flow has a second step. The schema says:

    CONSTRAINT users_email_per_org_uq UNIQUE (organization_id, email)

Email is unique WITHIN an organization, not globally — a contractor may
legitimately hold accounts at two customers. So an email alone does not
identify a user, and any login endpoint that assumes it does will silently
sign people into whichever row the database happened to return first.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

from .models import Organization, User, UserRole, UserStatus


class DuplicateEmailError(ValueError):
    """This email already exists inside this organization."""


class TenantRepository(ABC):
    """Read/write access to organizations and users.

    Every method that touches a user-owned row takes `organization_id`, except
    the two documented above. That is not redundancy — it is the isolation
    boundary.
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
        """Resolve an identity-provider subject to a user."""

    @abstractmethod
    async def find_users_by_email(self, email: str) -> list[User]:
        """Every user row carrying this email, across all organizations.

        Unscoped on purpose: this is the lookup that establishes tenancy. The
        caller must verify a password against each candidate and then narrow
        to exactly one organization before doing anything else.
        """

    @abstractmethod
    async def get_user_in_organization_by_email(
        self, email: str, organization_id: str
    ) -> Optional[User]:
        """Scoped counterpart, used to enforce the per-organization unique."""

    @abstractmethod
    async def save_organization(self, organization: Organization) -> Organization: ...

    @abstractmethod
    async def save_user(self, user: User) -> User: ...

    @abstractmethod
    async def create_organization_with_owner(
        self,
        organization: Organization,
        owner: User,
    ) -> tuple[Organization, User]:
        """Create both, or neither.

        Signup is the only place a new organization appears. If the org write
        succeeds and the user write fails, the result is an organization
        nobody can sign in to and whose name is now taken — a state with no
        recovery path through the UI. In SQL this is one transaction; here it
        is one lock.
        """


class InMemoryTenantRepository(TenantRepository):
    """Dict-backed repository.

    Real in development and in tests; replaced by SQL in production. It is
    NOT a toy — it implements the same isolation and uniqueness rules, so a
    test that passes here would pass against PostgreSQL.
    """

    def __init__(self) -> None:
        self._orgs: dict[str, Organization] = {}
        self._users: dict[str, User] = {}
        self._by_sub: dict[str, str] = {}  # cognito_sub -> user_id
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ read

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

    async def find_users_by_email(self, email: str) -> list[User]:
        needle = email.strip().lower()
        if not needle:
            return []
        # Sorted by creation time so the candidate order a user sees on the
        # organization chooser is stable between requests.
        return sorted(
            (u for u in self._users.values() if u.email == needle),
            key=lambda u: u.created_at,
        )

    async def get_user_in_organization_by_email(
        self, email: str, organization_id: str
    ) -> Optional[User]:
        needle = email.strip().lower()
        for user in self._users.values():
            if user.email == needle and user.organization_id == organization_id:
                return user
        return None

    # ----------------------------------------------------------------- write

    async def save_organization(self, organization: Organization) -> Organization:
        async with self._lock:
            self._orgs[organization.id] = organization
        return organization

    async def save_user(self, user: User) -> User:
        async with self._lock:
            self._assert_email_free(user)
            self._users[user.id] = user
            if user.cognito_sub:
                self._by_sub[user.cognito_sub] = user.id
        return user

    async def create_organization_with_owner(
        self,
        organization: Organization,
        owner: User,
    ) -> tuple[Organization, User]:
        if owner.organization_id != organization.id:
            raise ValueError("owner.organization_id does not match the organization")
        async with self._lock:
            if organization.id in self._orgs:
                raise ValueError(f"organization {organization.id} already exists")
            self._assert_email_free(owner)
            self._orgs[organization.id] = organization
            self._users[owner.id] = owner
            if owner.cognito_sub:
                self._by_sub[owner.cognito_sub] = owner.id
        return organization, owner

    def _assert_email_free(self, user: User) -> None:
        """Mirror `users_email_per_org_uq`. Caller must hold the lock.

        Without this the in-memory repository would accept a duplicate that
        PostgreSQL will later reject, and the bug would only appear after the
        database migration — in production, on the signup path.
        """
        for existing in self._users.values():
            if (
                existing.id != user.id
                and existing.organization_id == user.organization_id
                and existing.email == user.email
            ):
                raise DuplicateEmailError(
                    f"{user.email} already has an account in this organization"
                )

    # ---- helpers that only make sense for the in-memory implementation ----

    async def list_organizations(self) -> list[Organization]:
        return list(self._orgs.values())

    async def list_users(self, organization_id: str) -> list[User]:
        return [u for u in self._users.values() if u.organization_id == organization_id]

    async def count(self) -> tuple[int, int]:
        return len(self._orgs), len(self._users)


__all__ = [
    "DuplicateEmailError",
    "InMemoryTenantRepository",
    "TenantRepository",
    "User",
    "UserRole",
    "UserStatus",
]
