"""
production.py — production defaults.

UNVERIFIED CAPACITY NUMBERS
---------------------------
The concurrency values below are placeholders chosen to be SAFE, not to be
right. Nobody has yet measured how many sessions one task sustains, and the
correct source for these numbers is the single-instance benchmark:

    run 1, 5, 10, 25, 50, 100 sessions against one task
        -> find stable_sessions_per_instance
        -> derive ASR_MAX_CONCURRENCY from provider quota and observed latency
        -> derive MAX_CONCURRENT_SESSIONS from that measurement

Until that benchmark exists, do not raise these to hit a target number. A
system that admits 500 sessions and serves them badly is worse than one that
admits 200 and tells the other 300 to retry.

ASR_MAX_CONCURRENCY in particular should be derived from the actual Vertex AI
quota for the project, not guessed. Raising it above the quota converts a
clean queue into a 429 storm.
"""

DEFAULTS: dict[str, str] = {
    "LOG_LEVEL": "INFO",
    "LOG_JSON": "true",
    "CORS_ORIGINS": "https://app.voxlive.example",
    # See the warning above before changing any of these four.
    "ASR_MAX_CONCURRENCY": "32",
    "ASR_QUEUE_MAXSIZE": "512",
    "ASR_TIMEOUT_SEC": "20",
    "DIARIZATION_MAX_CONCURRENCY": "4",
    "DIARIZATION_QUEUE_MAXSIZE": "256",
    "MAX_CONCURRENT_SESSIONS": "150",
    # Never in production: --reload watches the filesystem and restarts the
    # process, dropping every live WebSocket with it.
    "ENABLE_POSTPROCESS": "false",
}
