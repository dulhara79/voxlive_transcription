"""
routes_auth.py — signup, sign-in, and "who am I".

    POST /auth/signup              create an organization and its owner
    POST /auth/login               email + password
    POST /auth/login/organization  second step, only when the email is
                                   registered in more than one organization
    GET  /auth/me                  the current token's identity

WHY LOGIN CAN NEED TWO STEPS
----------------------------
The schema makes email unique PER ORGANIZATION, not globally:

    CONSTRAINT users_email_per_org_uq UNIQUE (organization_id, email)

That is the right constraint — a contractor may hold accounts at two
customers — but it means an email address does not identify a person. A login
endpoint that looks up one row by email will sign someone into whichever
organization the database returned first, and the bug is invisible until the
day a shared email exists.

So the flow is:

    1. verify the password against EVERY user row carrying that email
    2. exactly one matched   -> issue an access token, done, one step
    3. several matched       -> return the organization names plus a
                                short-lived org_select token, and let the
                                person choose
    4. none matched          -> one generic failure

Step 3 asks WHICH ORGANIZATION only after the password is already proven, so
the organization list is never exposed to someone who cannot authenticate.
Asking for the organization first — the obvious design — would turn this
endpoint into a "does this person work at that company" lookup for anyone.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Password reset and invitations. Both need outbound email, which is a service,
a domain, a DKIM record and a bounce policy. Adding a half-built version now
would leave a live endpoint that mints reset tokens nobody can deliver.
Until then, an owner adds members and sets the initial password.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field

from ..auth.passwords import (
    MIN_PASSWORD_LENGTH,
    PasswordError,
    hash_password,
    needs_rehash,
    verify_password,
)
from ..auth.tokens import (
    TokenError,
    issue_access_token,
    issue_org_select_token,
    read_access_token,
    read_org_select_token,
)
from ..tenant.models import (
    Organization,
    OrganizationStatus,
    User,
    UserRole,
    UserStatus,
)
from ..tenant.repository import DuplicateEmailError, TenantRepository

log = logging.getLogger("voxlive.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

# One message for every authentication failure. "No such account" and "wrong
# password" as separate messages turn this endpoint into a directory of who
# holds an account.
GENERIC_LOGIN_FAILURE = "That email and password combination didn't work."

# A dummy hash with the current parameters. Verified against when no candidate
# user exists, so a request for an unknown email costs the same wall-clock as
# one for a known email. Without it, response time answers the question the
# error message refuses to.
_TIMING_DECOY = hash_password("timing-decoy-not-a-real-password")

# ---------------------------------------------------------------------------
# Rate limiting.
#
# This is PER PROCESS. Behind two Fargate tasks the effective limit doubles,
# and it resets on deploy. It is here because a scrypt verify is ~60 ms of CPU
# and an unthrottled login endpoint is therefore a cheap way to saturate the
# service — not because it is a complete answer to credential stuffing. The
# durable version belongs in front of the app, and on AWS the free option is
# an ALB rule or WAF rate-based rule rather than anything in this file.
# ---------------------------------------------------------------------------
LOGIN_WINDOW_SEC = 300
LOGIN_MAX_ATTEMPTS = 10
_attempts: dict[str, list[float]] = {}
_attempts_lock = asyncio.Lock()


async def _rate_limit(key: str) -> None:
    now = time.monotonic()
    async with _attempts_lock:
        hits = [t for t in _attempts.get(key, ()) if now - t < LOGIN_WINDOW_SEC]
        if len(hits) >= LOGIN_MAX_ATTEMPTS:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many sign-in attempts. Wait a few minutes and try again.",
            )
        hits.append(now)
        _attempts[key] = hits
        # Opportunistic sweep so the dict cannot grow without bound on a
        # long-lived task.
        if len(_attempts) > 4096:
            for stale in [
                k
                for k, v in _attempts.items()
                if not v or now - v[-1] > LOGIN_WINDOW_SEC
            ]:
                _attempts.pop(stale, None)


def _repo(request: Request) -> TenantRepository:
    return request.app.state.tenants


def _client_key(request: Request, email: str) -> str:
    client = request.client.host if request.client else "unknown"
    return f"{client}:{email.strip().lower()}"


# ------------------------------------------------------------------ schemas


class SignupRequest(BaseModel):
    organization_name: str = Field(min_length=2, max_length=120)
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=256)
    # Optional shortcut: a client that already knows which organization the
    # person wants can skip the chooser entirely.
    organization_id: Optional[str] = None


class OrganizationChoice(BaseModel):
    organization_id: str
    organization_name: str
    role: str


class SelectOrganizationRequest(BaseModel):
    org_select_token: str
    organization_id: str


class AuthenticatedResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    email: str
    name: str
    role: str
    organization_id: str
    organization_name: str
    plan_code: str


class ChooseOrganizationResponse(BaseModel):
    """Returned when the password matched in more than one organization."""

    needs_organization: bool = True
    org_select_token: str
    expires_in: int
    organizations: list[OrganizationChoice]


# ------------------------------------------------------------------ helpers


async def _authenticated_response(
    repo: TenantRepository, user: User
) -> AuthenticatedResponse:
    organization = await repo.get_organization(user.organization_id)
    if organization is None:
        # Only reachable if an organization was deleted with users still
        # attached. Fail closed rather than issuing a token scoped to nothing.
        log.error(
            "user %s references missing organization %s",
            user.id,
            user.organization_id,
        )
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="This account's organization is no longer available.",
        )
    if organization.status is OrganizationStatus.CLOSED:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="This organization is closed. Contact your administrator.",
        )
    if user.status is not UserStatus.ACTIVE:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=f"This account is {user.status.value}. Contact your administrator.",
        )

    token, ttl = issue_access_token(
        user_id=user.id,
        organization_id=user.organization_id,
        role=user.role.value,
        email=user.email,
    )
    return AuthenticatedResponse(
        access_token=token,
        expires_in=ttl,
        user_id=user.id,
        email=user.email,
        name=user.name,
        role=user.role.value,
        organization_id=organization.id,
        organization_name=organization.name,
        plan_code=organization.plan_code,
    )


# ------------------------------------------------------------------- routes


@router.post(
    "/signup",
    response_model=AuthenticatedResponse,
    status_code=status.HTTP_201_CREATED,
)
async def signup(request: Request, body: SignupRequest) -> AuthenticatedResponse:
    """Create a new organization and sign in as its owner.

    Joining an EXISTING organization is not possible here on purpose. If it
    were, anyone could attach themselves to a customer's tenant by typing its
    name, and the tenant isolation the rest of the system enforces would have
    a front door. Members arrive by invitation from an owner.
    """
    repo = _repo(request)
    await _rate_limit(_client_key(request, body.email))

    try:
        password_hash = hash_password(body.password)
    except PasswordError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from None

    organization = Organization.create(
        name=body.organization_name.strip(),
        status=OrganizationStatus.TRIAL,
    )
    owner = User.create(
        organization_id=organization.id,
        email=str(body.email),
        name=body.name.strip(),
        role=UserRole.OWNER,
        status=UserStatus.ACTIVE,
        password_hash=password_hash,
    )

    try:
        organization, owner = await repo.create_organization_with_owner(
            organization, owner
        )
    except DuplicateEmailError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from None

    log.info(
        "organization created",
        extra={
            "event": "signup",
            "organization_id": organization.id,
            "user_id": owner.id,
        },
    )
    return await _authenticated_response(repo, owner)


@router.post("/login", response_model=None)
async def login(request: Request, body: LoginRequest):
    """Verify a password across every organization holding this email."""
    repo = _repo(request)
    await _rate_limit(_client_key(request, body.email))

    candidates = await repo.find_users_by_email(str(body.email))
    if body.organization_id:
        candidates = [
            u for u in candidates if u.organization_id == body.organization_id
        ]

    if not candidates:
        # Burn the same CPU a real verify would, so an unknown email and a
        # wrong password are indistinguishable by response time.
        verify_password(body.password, _TIMING_DECOY)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=GENERIC_LOGIN_FAILURE)

    matched = [u for u in candidates if verify_password(body.password, u.password_hash)]

    if not matched:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=GENERIC_LOGIN_FAILURE)

    if len(matched) == 1:
        user = matched[0]
        # Upgrade the stored hash while the plaintext is in hand. This is the
        # only moment it is available, so without it raising the scrypt cost
        # parameters would only ever protect accounts created afterwards.
        if needs_rehash(user.password_hash):
            user.password_hash = hash_password(body.password)
            await repo.save_user(user)
        return await _authenticated_response(repo, user)

    # Same email and same password in several organizations. Ask which one.
    organizations: list[OrganizationChoice] = []
    for user in matched:
        org = await repo.get_organization(user.organization_id)
        if org is None or org.status is OrganizationStatus.CLOSED:
            continue
        organizations.append(
            OrganizationChoice(
                organization_id=org.id,
                organization_name=org.name,
                role=user.role.value,
            )
        )
    if not organizations:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=GENERIC_LOGIN_FAILURE)
    if len(organizations) == 1:
        only = next(
            u for u in matched if u.organization_id == organizations[0].organization_id
        )
        return await _authenticated_response(repo, only)

    token, ttl = issue_org_select_token([u.id for u in matched], str(body.email))
    return ChooseOrganizationResponse(
        org_select_token=token,
        expires_in=ttl,
        organizations=organizations,
    )


@router.post("/login/organization", response_model=AuthenticatedResponse)
async def select_organization(
    request: Request, body: SelectOrganizationRequest
) -> AuthenticatedResponse:
    """Finish a login that needed an organization chosen."""
    repo = _repo(request)
    try:
        candidate_ids, _email = read_org_select_token(body.org_select_token)
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from None

    # The token pins WHICH user ids the password matched. Resolving the
    # organization from the request alone would let a valid token be pointed
    # at an organization whose password was never verified — so the chosen
    # organization must belong to one of the pinned candidates, and
    # `get_user` is scoped, meaning a mismatched pair returns None.
    user: Optional[User] = None
    for user_id in candidate_ids:
        user = await repo.get_user(user_id, body.organization_id)
        if user is not None:
            break

    if user is None:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="That organization isn't available for this sign-in.",
        )
    return await _authenticated_response(repo, user)


@router.get("/me", response_model=AuthenticatedResponse)
async def me(
    request: Request, authorization: str = Header(default="")
) -> AuthenticatedResponse:
    """Re-hydrate a session from a stored token.

    The frontend calls this on load. It re-reads the user from the repository
    rather than trusting the token's claims, so a role change or a disabled
    account takes effect on the next page load instead of at token expiry.
    """
    scheme, _, raw = authorization.partition(" ")
    if scheme.lower() != "bearer" or not raw:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Sign in to continue.")
    try:
        claims = read_access_token(raw.strip())
    except TokenError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from None

    repo = _repo(request)
    user = await repo.get_user(claims.user_id, claims.organization_id)
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="This account no longer exists."
        )
    return await _authenticated_response(repo, user)
