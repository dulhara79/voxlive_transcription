"""
development.py — local laptop defaults.

A profile sets DEFAULTS only. They are applied with `os.environ.setdefault`,
so anything you put in `.env` or export in your shell still wins. A profile
can never silently overrule an explicit choice — it only decides what happens
when you have not chosen.

Development wants: readable logs, a chatty log level, and the Vite dev server
allowed through CORS.
"""

DEFAULTS: dict[str, str] = {
    "LOG_LEVEL": "INFO",
    "LOG_JSON": "false",  # human-readable console output
    "CORS_ORIGINS": "http://localhost:5173,http://127.0.0.1:5173",
    # Small pools: one developer, one browser tab, and a laptop that is also
    # running the frontend, an editor and a browser.
    "ASR_MAX_CONCURRENCY": "6",
    "ASR_QUEUE_MAXSIZE": "64",
    "ASR_TIMEOUT_SEC": "30",
    "DIARIZATION_MAX_CONCURRENCY": "2",
    "DIARIZATION_QUEUE_MAXSIZE": "32",
    "MAX_CONCURRENT_SESSIONS": "50",
}
