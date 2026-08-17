"""
sqlite_repository.py — a durable TenantRepository for local development.

WHY THIS FILE EXISTS
--------------------
`InMemoryTenantRepository` keeps organizations and users in two dictionaries.
That is correct for tests and wrong for a laptop you restart twenty times a
day: the process dies, the dictionaries go with it, and every account you
registered has to be registered again.

    backend starts -> empty dicts -> you sign up -> account lives in RAM
    backend stops  -> RAM freed   -> account gone

This class stores the same objects in a SQLite file instead, so the sign-up
you did before lunch is still there after it.

WHAT IT DELIBERATELY DOES NOT CHANGE
------------------------------------
The interface. `TenantRepository` already existed as the persistence seam, and
every isolation rule encoded in it is reproduced here rather than reinterpreted:

  * `get_user(user_id, organization_id)` is a SCOPED query
    (`WHERE id = ? AND organization_id = ?`), not a fetch followed by a check
    in Python. A cross-tenant lookup returns None, exactly as it does in the
    in-memory implementation, so the difference between "no such user" and
    "someone else's user" is not observable.

  * `users_email_per_org_uq` is a real UNIQUE constraint here, so the rule the
    in-memory version enforces in `_assert_email_free` is enforced by the
    database rather than by convention. A violation surfaces as
    `DuplicateEmailError`, the same exception `routes_auth.signup` already
    catches and turns into a 409.

  * `create_organization_with_owner` is ONE transaction. Either both rows
    exist or neither does. A half-completed signup leaves an organization
    nobody can sign in to whose name is now taken, and there is no route back
    out of that state through the UI.

WHY STDLIB sqlite3 AND NOT AN ASYNC DRIVER
------------------------------------------
`sqlite3` ships with Python. Adding `aiosqlite` or SQLAlchemy to fix a
development-only persistence problem means a new dependency in
`requirements.txt`, a new thing to install on every machine, and a new version
to keep in step — for a handful of auth queries that run once per login.

The calls are blocking, so each one runs in a worker thread via
`asyncio.to_thread`. That keeps the event loop free, which is what actually
matters here: the same loop is carrying live WebSocket audio.

CONNECTION PER OPERATION
------------------------
Opening a SQLite connection is on the order of tens of microseconds, and a
connection object is not safe to share across threads. Since every call is
already hopping to a worker thread, opening there and closing on the way out
is both simpler and safer than a pool plus `check_same_thread=False`.

WAL is enabled so a reader is never blocked by a writer, and writes use
`BEGIN IMMEDIATE` so two concurrent signups serialise at the database instead
of racing between a SELECT and an INSERT.

RELATIONSHIP TO app/docs/docs_schema.sql
----------------------------------------
That file is the PostgreSQL design for production. The tables below mirror its
`organizations` and `users` definitions, with two deliberate differences:

  * `created_at` is REAL (epoch seconds) rather than TIMESTAMPTZ, because the
    `Organization`/`User` dataclasses hold a float and round-tripping through
    a string only creates a chance to lose precision or timezone.
  * `password_hash` is stored here. It is absent from docs_schema.sql, which
    was written assuming Cognito would own credentials; `auth/passwords.py`
    means this deployment owns them instead, so the column has to exist.

The `sessions`, `speakers` and `transcript_segments` tables from that file are
NOT created here. Nothing reads or writes them yet, and creating empty tables
would imply a persistence guarantee this commit does not make.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from .models import Organization, OrganizationStatus, User, UserRole, UserStatus
from .repository import DuplicateEmailError, TenantRepository

log = logging.getLogger("voxlive.tenant")

SCHEMA = """
CREATE TABLE IF NOT EXISTS organizations (
    id                      TEXT PRIMARY KEY,
    name                    TEXT    NOT NULL,
    plan_code               TEXT    NOT NULL DEFAULT 'starter',
    status                  TEXT    NOT NULL DEFAULT 'trial',
    max_concurrent_sessions INTEGER,
    created_at              REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    email           TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL DEFAULT 'member',
    status          TEXT NOT NULL DEFAULT 'active',
    cognito_sub     TEXT UNIQUE,
    password_hash   TEXT,
    created_at      REAL NOT NULL,
    CONSTRAINT users_email_per_org_uq UNIQUE (organization_id, email)
);

CREATE INDEX IF NOT EXISTS users_org_idx   ON users (organization_id);
CREATE INDEX IF NOT EXISTS users_email_idx ON users (email);
"""


def _row_to_organization(row: sqlite3.Row) -> Organization:
    return Organization(
        id=row["id"],
        name=row["name"],
        plan_code=row["plan_code"],
        # Unknown values fall back rather than raising. A status string this
        # build does not recognise must not take the whole tenant offline in
        # the middle of a request.
        status=_enum_or(OrganizationStatus, row["status"], OrganizationStatus.TRIAL),
        max_concurrent_sessions=row["max_concurrent_sessions"],
        created_at=float(row["created_at"]),
    )


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        organization_id=row["organization_id"],
        email=row["email"],
        name=row["name"] or "",
        role=_enum_or(UserRole, row["role"], UserRole.MEMBER),
        status=_enum_or(UserStatus, row["status"], UserStatus.ACTIVE),
        cognito_sub=row["cognito_sub"],
        password_hash=row["password_hash"],
        created_at=float(row["created_at"]),
    )


def _enum_or(enum_cls, value, default):
    try:
        return enum_cls(value)
    except ValueError:
        log.warning(
            "unknown %s value %r in the database; using %s",
            enum_cls.__name__,
            value,
            default.value,
        )
        return default


class SqliteTenantRepository(TenantRepository):
    """File-backed repository. Same contract as InMemoryTenantRepository."""

    def __init__(self, path: str = "voxlive.db") -> None:
        self.path = str(Path(path).expanduser())
        parent = Path(self.path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        # Writes are serialised in-process as well as in SQLite. The database
        # would serialise them anyway; holding the lock here means a busy
        # writer never turns into an SQLITE_BUSY that surfaces as a 500 on a
        # signup.
        self._write_lock = asyncio.Lock()
        self._ensure_schema()
        log.info("tenant storage: SQLite at %s", os.path.abspath(self.path))

    # ------------------------------------------------------------- plumbing

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Off by default in SQLite, so the REFERENCES clause above is
        # decorative until it is switched on per connection.
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ read

    async def get_organization(self, organization_id: str) -> Optional[Organization]:
        return await asyncio.to_thread(self._get_organization, organization_id)

    def _get_organization(self, organization_id: str) -> Optional[Organization]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM organizations WHERE id = ?", (organization_id,)
            ).fetchone()
            return _row_to_organization(row) if row else None
        finally:
            conn.close()

    async def get_user(self, user_id: str, organization_id: str) -> Optional[User]:
        return await asyncio.to_thread(self._get_user, user_id, organization_id)

    def _get_user(self, user_id: str, organization_id: str) -> Optional[User]:
        conn = self._connect()
        try:
            # BOTH predicates in the query. Fetching by id and comparing the
            # organization afterwards would work until someone forgot the
            # second half; there is no version of this method that can be
            # called without the tenant.
            row = conn.execute(
                "SELECT * FROM users WHERE id = ? AND organization_id = ?",
                (user_id, organization_id),
            ).fetchone()
            return _row_to_user(row) if row else None
        finally:
            conn.close()

    async def get_user_by_cognito_sub(self, sub: str) -> Optional[User]:
        return await asyncio.to_thread(self._get_user_by_cognito_sub, sub)

    def _get_user_by_cognito_sub(self, sub: str) -> Optional[User]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE cognito_sub = ?", (sub,)
            ).fetchone()
            return _row_to_user(row) if row else None
        finally:
            conn.close()

    async def find_users_by_email(self, email: str) -> list[User]:
        return await asyncio.to_thread(self._find_users_by_email, email)

    def _find_users_by_email(self, email: str) -> list[User]:
        needle = email.strip().lower()
        if not needle:
            return []
        conn = self._connect()
        try:
            # Ordered by creation time so the organization chooser lists the
            # candidates in the same order on every request.
            rows = conn.execute(
                "SELECT * FROM users WHERE email = ? ORDER BY created_at ASC",
                (needle,),
            ).fetchall()
            return [_row_to_user(r) for r in rows]
        finally:
            conn.close()

    async def get_user_in_organization_by_email(
        self, email: str, organization_id: str
    ) -> Optional[User]:
        return await asyncio.to_thread(
            self._get_user_in_organization_by_email, email, organization_id
        )

    def _get_user_in_organization_by_email(
        self, email: str, organization_id: str
    ) -> Optional[User]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ? AND organization_id = ?",
                (email.strip().lower(), organization_id),
            ).fetchone()
            return _row_to_user(row) if row else None
        finally:
            conn.close()

    # ----------------------------------------------------------------- write

    async def save_organization(self, organization: Organization) -> Organization:
        async with self._write_lock:
            await asyncio.to_thread(self._save_organization, organization)
        return organization

    def _save_organization(self, organization: Organization) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._upsert_organization(conn, organization)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _upsert_organization(conn: sqlite3.Connection, org: Organization) -> None:
        conn.execute(
            """
            INSERT INTO organizations
                (id, name, plan_code, status, max_concurrent_sessions, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name                    = excluded.name,
                plan_code               = excluded.plan_code,
                status                  = excluded.status,
                max_concurrent_sessions = excluded.max_concurrent_sessions
            """,
            (
                org.id,
                org.name,
                org.plan_code,
                org.status.value,
                org.max_concurrent_sessions,
                org.created_at,
            ),
        )

    async def save_user(self, user: User) -> User:
        async with self._write_lock:
            await asyncio.to_thread(self._save_user, user)
        return user

    def _save_user(self, user: User) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._upsert_user(conn, user)
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise _translate_integrity_error(exc, user) from None
        finally:
            conn.close()

    @staticmethod
    def _upsert_user(conn: sqlite3.Connection, user: User) -> None:
        conn.execute(
            """
            INSERT INTO users
                (id, organization_id, email, name, role, status,
                 cognito_sub, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                organization_id = excluded.organization_id,
                email           = excluded.email,
                name            = excluded.name,
                role            = excluded.role,
                status          = excluded.status,
                cognito_sub     = excluded.cognito_sub,
                password_hash   = excluded.password_hash
            """,
            (
                user.id,
                user.organization_id,
                user.email.strip().lower(),
                user.name,
                user.role.value,
                user.status.value,
                user.cognito_sub,
                user.password_hash,
                user.created_at,
            ),
        )

    async def create_organization_with_owner(
        self,
        organization: Organization,
        owner: User,
    ) -> tuple[Organization, User]:
        if owner.organization_id != organization.id:
            raise ValueError("owner.organization_id does not match the organization")
        async with self._write_lock:
            await asyncio.to_thread(
                self._create_organization_with_owner, organization, owner
            )
        return organization, owner

    def _create_organization_with_owner(
        self, organization: Organization, owner: User
    ) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            exists = conn.execute(
                "SELECT 1 FROM organizations WHERE id = ?", (organization.id,)
            ).fetchone()
            if exists:
                conn.rollback()
                raise ValueError(f"organization {organization.id} already exists")
            self._upsert_organization(conn, organization)
            self._upsert_user(conn, owner)
            # One commit for both rows: the failure mode this prevents is an
            # organization whose name is taken and whose owner does not exist,
            # which the UI offers no way to recover from.
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise _translate_integrity_error(exc, owner) from None
        finally:
            conn.close()

    # ------------------------------------------- parity with the in-memory API

    async def list_organizations(self) -> list[Organization]:
        return await asyncio.to_thread(self._list_organizations)

    def _list_organizations(self) -> list[Organization]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM organizations").fetchall()
            return [_row_to_organization(r) for r in rows]
        finally:
            conn.close()

    async def list_users(self, organization_id: str) -> list[User]:
        return await asyncio.to_thread(self._list_users, organization_id)

    def _list_users(self, organization_id: str) -> list[User]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM users WHERE organization_id = ?", (organization_id,)
            ).fetchall()
            return [_row_to_user(r) for r in rows]
        finally:
            conn.close()

    async def count(self) -> tuple[int, int]:
        return await asyncio.to_thread(self._count)

    def _count(self) -> tuple[int, int]:
        conn = self._connect()
        try:
            orgs = conn.execute("SELECT COUNT(*) FROM organizations").fetchone()[0]
            users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            return int(orgs), int(users)
        finally:
            conn.close()


def _translate_integrity_error(exc: sqlite3.IntegrityError, user: User) -> Exception:
    """Turn a constraint name into the exception the auth routes already handle.

    `routes_auth.signup` catches `DuplicateEmailError` and returns 409. If the
    raw sqlite3 error escaped instead, the same collision would surface as an
    unhandled 500 — the identical bug the in-memory `_assert_email_free`
    exists to prevent, just relocated.
    """
    message = str(exc)
    if "email" in message:
        return DuplicateEmailError(
            f"{user.email} already has an account in this organization"
        )
    if "cognito_sub" in message:
        return ValueError(f"cognito_sub {user.cognito_sub!r} is already in use")
    return exc


__all__ = ["SqliteTenantRepository"]
