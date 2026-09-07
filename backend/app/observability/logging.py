"""
logging.py — structured application logging.

It answers exactly one question:

    "How do I record operational events?"

WHY JSON
--------
At one user, `ASR attempt 2/3 failed` is readable. At 500 concurrent users it
is noise: you cannot tell which session it belongs to, you cannot count how
many sessions are affected, and CloudWatch Logs Insights cannot query it.

    2026-01-01 10:00:00 WARNING voxlive.session ASR attempt 2/3 failed: 429

becomes

    {"timestamp":"2026-01-01T10:00:00.123Z","level":"WARNING",
     "logger":"voxlive.session","service":"asr","session_id":"sess_...",
     "segment_id":184,"message":"ASR attempt 2/3 failed: 429"}

which supports `stats count(*) by session_id` in Insights and feeds metric
filters for alarms.

HOW CONTEXT GETS IN
-------------------
Two ways, and they compose:

1. Per-call, via the standard `extra=` argument. The session layer already
   does this:

       log.info("segment %d gated", seg_id,
                extra={"session_id": self.session_id, "segment_id": seg_id})

2. Ambiently, via `bind()`. Anything bound for the current asyncio task is
   attached to every record emitted underneath it, so a deep helper does not
   need the session id threaded through five call frames:

       with bind(session_id=sid, tenant_id=tid):
           await do_work()          # every log line inside carries both

`bind()` uses contextvars, which asyncio propagates per task — so two
concurrent sessions never see each other's context.

KEEP IT PLAIN IN DEVELOPMENT
----------------------------
JSON is for machines. `configure_logging(json_format=False)` restores the
human-readable console format, and the development profile selects it.
"""

from __future__ import annotations

import contextlib
import contextvars
import datetime as _dt
import json
import logging
import sys
from typing import Any, Iterator, Optional

# Fields promoted to top-level keys when present. Everything else passed via
# `extra=` still gets included, but these have a guaranteed, queryable name.
CONTEXT_FIELDS = (
    "service",
    "request_id",
    "tenant_id",
    "user_id",
    "session_id",
    "segment_id",
    "event",
)

# LogRecord's own attributes. Anything on a record that is NOT one of these and
# NOT private is something the caller passed via `extra=`, so it belongs in the
# output. Computed once rather than hardcoded, so a Python version that adds a
# record attribute cannot start leaking it into logs.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
    "taskName",
}

_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "voxlive_log_context", default={}
)


@contextlib.contextmanager
def bind(**fields: Any) -> Iterator[None]:
    """Attach fields to every log record emitted inside this block.

    Scoped to the current asyncio task, so concurrent sessions stay isolated.
    Nesting merges, and the outer context is restored on exit even if the body
    raises.
    """
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def get_context() -> dict:
    """The currently bound fields. Useful for attaching the same context to a
    metric or an error response."""
    return dict(_context.get())


class ContextFilter(logging.Filter):
    """Copy the ambient context onto each record.

    A filter rather than a formatter concern, so the fields are available to
    every handler — including ones added later for CloudWatch or OTel.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _context.get().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line — the format CloudWatch Logs parses natively."""

    def __init__(self, service: str = "voxlive") -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _dt.datetime.fromtimestamp(record.created, tz=_dt.timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(record, "service", self.service),
            "message": record.getMessage(),
        }

        for field in CONTEXT_FIELDS:
            if field == "service":
                continue
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        # Anything else the caller passed via extra=.
        for key, value in vars(record).items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # default=str so a stray numpy scalar or Enum can never crash logging.
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human format for local development, with context appended if present."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        bits = []
        for field in CONTEXT_FIELDS:
            value = getattr(record, field, None)
            if value is not None and field != "service":
                # Session ids are long; the tail is the distinguishing part.
                if field == "session_id" and isinstance(value, str):
                    value = value[-8:]
                bits.append(f"{field}={value}")
        return f"{base}  [{' '.join(bits)}]" if bits else base


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    service: str = "voxlive",
    quiet_loggers: Optional[dict[str, str]] = None,
) -> None:
    """Install the root handler. Call ONCE, before anything else logs.

    Replaces the root handlers rather than adding to them, so calling this
    after `logging.basicConfig()` (or twice, as pytest may) cannot produce
    duplicated lines.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service) if json_format else ConsoleFormatter())
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn installs its own handlers; drop them so its lines go through this
    # formatter too instead of appearing twice in two different shapes.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv = logging.getLogger(name)
        uv.handlers.clear()
        uv.propagate = True

    defaults = {"httpx": "WARNING", "httpcore": "WARNING", "urllib3": "WARNING"}
    defaults.update(quiet_loggers or {})
    for name, lvl in defaults.items():
        logging.getLogger(name).setLevel(lvl.upper())


def get_logger(name: str) -> logging.Logger:
    """Module logger. Plain `logging.getLogger` works too — this exists so call
    sites read as intent rather than as stdlib plumbing."""
    return logging.getLogger(name)
