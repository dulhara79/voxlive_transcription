"""
quotas.py — QuotaManager and CapacityManager: may this session start?

TWO SEPARATE LIMITS, CHECKED IN THIS ORDER
------------------------------------------
    1. PLATFORM CAPACITY   "does this task have room at all?"       (§12)
    2. TENANT QUOTA        "has this organization used its share?"  (§13/§14)

They protect against different failures and both are needed:

  * Platform capacity stops a traffic spike becoming a total outage. At
    470/500 the supervisor's §12 is explicit — don't blindly accept the next
    100; reject gracefully so the 470 already running keep working.

  * Tenant quota stops one customer eating everyone else's capacity. §13:
    Company A with 300 users must not make Company B with 20 users unusable.

Order matters. Platform first, because if the task is full the tenant's
remaining quota is irrelevant, and the cheaper check should run first.

    new session
         │
         ▼
    CapacityManager ──── full ────► REJECT (platform_full, retry_after)
         │
      has room
         │
         ▼
     QuotaManager ──── over ──────► REJECT (quota_exceeded)
         │
       within
         │
         ▼
       ACCEPT

SEPARATION OF CONCERNS (§15)
----------------------------
Nothing here knows about speakers, embeddings or ASR. `SpeakerEngine` does
speaker processing; `ASRScheduler` does ASR capacity; this file does tenant
limits. The supervisor's warning about god objects is the reason these are
three files instead of one.

SCOPE: PER PROCESS, FOR NOW
---------------------------
Counts come from this task's SessionManager, so `max_concurrent_sessions`
is currently enforced per ECS task, not fleet-wide. With N tasks behind an
ALB an organization could reach N x its limit. That is a real gap and the
honest fix is a Redis counter (§17: "tenant concurrency counters") — but
it should be added when there is more than one task, with a measured
capacity number, not speculatively now. `QuotaManager` takes its counts
through a callable so the Redis version substitutes without touching callers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from ..auth.context import TenantContext
from .models import Organization
from .repository import TenantRepository

log = logging.getLogger("voxlive.quota")


class RejectReason(str, Enum):
    PLATFORM_FULL = "platform_full"
    QUOTA_EXCEEDED = "quota_exceeded"
    ORGANIZATION_INACTIVE = "organization_inactive"
    UNKNOWN_ORGANIZATION = "unknown_organization"


@dataclass(frozen=True)
class AdmissionDecision:
    """The answer, with enough detail for the client to act on it."""

    allowed: bool
    reason: Optional[RejectReason] = None
    message: str = ""
    retry_after_sec: Optional[int] = None
    current: int = 0
    limit: int = 0

    @classmethod
    def accept(cls, current: int, limit: int) -> "AdmissionDecision":
        return cls(allowed=True, current=current, limit=limit)

    def as_dict(self) -> dict:
        payload = {
            "allowed": self.allowed,
            "current": self.current,
            "limit": self.limit,
        }
        if self.reason:
            payload["reason"] = self.reason.value
            payload["message"] = self.message
        if self.retry_after_sec is not None:
            payload["retry_after_sec"] = self.retry_after_sec
        return payload


class CapacityManager:
    """Platform-wide admission control for one process (§12).

    `max_sessions` is the number of live sessions this task will hold. It is
    NOT a guess to be raised until it reaches 500 — it comes from the
    single-instance benchmark, and the fleet reaches 500 through more tasks,
    not through a larger number here.
    """

    def __init__(self, max_sessions: int, count_fn: Callable[[], int]):
        self.max_sessions = max(1, int(max_sessions))
        self._count = count_fn
        self.accepted = 0
        self.rejected = 0

    def check(self) -> AdmissionDecision:
        current = self._count()
        if current >= self.max_sessions:
            self.rejected += 1
            log.warning(
                "platform at capacity: %d/%d — rejecting new session",
                current,
                self.max_sessions,
                extra={"event": "admission_rejected", "reason": "platform_full"},
            )
            return AdmissionDecision(
                allowed=False,
                reason=RejectReason.PLATFORM_FULL,
                message=("The service is at capacity. Please retry in a few seconds."),
                # Short and fixed: long enough to let a session finish, short
                # enough that a legitimate user is not locked out. Jitter is
                # the client's job, so retries do not synchronise.
                retry_after_sec=15,
                current=current,
                limit=self.max_sessions,
            )
        self.accepted += 1
        return AdmissionDecision.accept(current, self.max_sessions)

    def snapshot(self) -> dict:
        current = self._count()
        return {
            "active_sessions": current,
            "max_sessions": self.max_sessions,
            "utilisation": round(current / self.max_sessions, 3),
            "accepted": self.accepted,
            "rejected": self.rejected,
        }


class QuotaManager:
    """Per-organization concurrency limits (§13/§14)."""

    def __init__(
        self,
        repo: TenantRepository,
        count_for_organization: Callable[[str], int],
    ):
        self.repo = repo
        self._count_for_org = count_for_organization
        self.rejected_by_org: dict[str, int] = {}

    async def check(self, ctx: TenantContext) -> AdmissionDecision:
        organization: Optional[Organization] = await self.repo.get_organization(
            ctx.organization_id
        )
        if organization is None:
            return AdmissionDecision(
                allowed=False,
                reason=RejectReason.UNKNOWN_ORGANIZATION,
                message="Organization not found.",
            )

        if not organization.can_start_sessions:
            return AdmissionDecision(
                allowed=False,
                reason=RejectReason.ORGANIZATION_INACTIVE,
                message=(
                    f"This organization is {organization.status.value} and "
                    "cannot start new sessions."
                ),
            )

        limit = organization.concurrent_session_limit
        current = self._count_for_org(organization.id)
        if current >= limit:
            self.rejected_by_org[organization.id] = (
                self.rejected_by_org.get(organization.id, 0) + 1
            )
            log.warning(
                "organization quota reached: %d/%d concurrent session(s)",
                current,
                limit,
                extra={
                    "event": "quota_exceeded",
                    "organization_id": organization.id,
                    "plan": organization.plan_code,
                },
            )
            return AdmissionDecision(
                allowed=False,
                reason=RejectReason.QUOTA_EXCEEDED,
                message=(
                    f"Your organization is using all {limit} of its concurrent "
                    f"sessions on the {organization.plan.name} plan. End a "
                    "session or upgrade to start another."
                ),
                retry_after_sec=30,
                current=current,
                limit=limit,
            )

        return AdmissionDecision.accept(current, limit)


class AdmissionController:
    """Runs both checks in the right order. One call for the WebSocket route."""

    def __init__(self, capacity: CapacityManager, quotas: QuotaManager):
        self.capacity = capacity
        self.quotas = quotas

    async def admit(self, ctx: TenantContext) -> AdmissionDecision:
        platform = self.capacity.check()
        if not platform.allowed:
            return platform
        return await self.quotas.check(ctx)

    def snapshot(self) -> dict:
        return {
            "platform": self.capacity.snapshot(),
            "quota_rejections_by_organization": dict(self.quotas.rejected_by_org),
        }
