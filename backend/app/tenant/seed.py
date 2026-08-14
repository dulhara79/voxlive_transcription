"""
seed.py — development fixtures.

Two organizations on different plans, with users in each. This exists so the
React frontend and the load harness can exercise tenant-aware code paths
before Cognito and PostgreSQL land, and so the cross-tenant isolation tests
have something concrete to fail against.

Seeding runs ONLY when APP_ENV=development (enforced by the caller in
main.py). Fixed ids, not generated ones, so the frontend can hardcode them
and they stay stable across restarts.

    org_dev_acme       Business    25 concurrent
      usr_dev_alice    admin
      usr_dev_bob      member
      usr_dev_carol    viewer      (cannot start sessions)

    org_dev_globex     Starter      5 concurrent  <- easy to hit in testing
      usr_dev_dave     owner

    org_dev_suspended  Starter      suspended     (refused at admission)
      usr_dev_erin     owner
"""

from __future__ import annotations

import logging

from ..auth.passwords import hash_password
from .models import Organization, OrganizationStatus, User, UserRole, UserStatus
from .repository import TenantRepository

log = logging.getLogger("voxlive.tenant")

# Every seeded user shares this password so the login screen is usable the
# moment the stack starts. Seeding only runs under APP_ENV=development.
DEV_PASSWORD = "voxlive-dev-password"


_CACHED_HASH: str | None = None


def _dev_hash() -> str:
    """Hash the shared dev password once — scrypt is ~60 ms per call."""
    global _CACHED_HASH
    if _CACHED_HASH is None:
        _CACHED_HASH = hash_password(DEV_PASSWORD)
    return _CACHED_HASH


async def seed_development_tenants(repo: TenantRepository) -> None:
    acme = Organization(
        id="org_dev_acme",
        name="ACME Media",
        plan_code="business",
        status=OrganizationStatus.ACTIVE,
    )
    globex = Organization(
        id="org_dev_globex",
        name="Globex Broadcasting",
        plan_code="starter",
        status=OrganizationStatus.ACTIVE,
    )
    suspended = Organization(
        id="org_dev_suspended",
        name="Initech (suspended)",
        plan_code="starter",
        status=OrganizationStatus.SUSPENDED,
    )
    for org in (acme, globex, suspended):
        await repo.save_organization(org)

    users = [
        User(
            id="usr_dev_alice",
            organization_id=acme.id,
            email="alice@acme.example",
            name="Alice",
            role=UserRole.ADMIN,
            password_hash=_dev_hash(),
            cognito_sub="dev-sub-alice",
        ),
        User(
            id="usr_dev_bob",
            organization_id=acme.id,
            email="bob@acme.example",
            name="Bob",
            role=UserRole.MEMBER,
            password_hash=_dev_hash(),
            cognito_sub="dev-sub-bob",
        ),
        User(
            id="usr_dev_carol",
            organization_id=acme.id,
            email="carol@acme.example",
            name="Carol",
            role=UserRole.VIEWER,
            password_hash=_dev_hash(),
            cognito_sub="dev-sub-carol",
        ),
        User(
            id="usr_dev_dave",
            organization_id=globex.id,
            email="dave@globex.example",
            name="Dave",
            role=UserRole.OWNER,
            password_hash=_dev_hash(),
            cognito_sub="dev-sub-dave",
        ),
        User(
            id="usr_dev_erin",
            organization_id=suspended.id,
            email="erin@initech.example",
            name="Erin",
            role=UserRole.OWNER,
            status=UserStatus.ACTIVE,
            password_hash=_dev_hash(),
            cognito_sub="dev-sub-erin",
        ),
    ]
    for user in users:
        await repo.save_user(user)

    log.info("seeded %d development organization(s), %d user(s)", 3, len(users))
