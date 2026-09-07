"""
Config package. Selects an environment profile from APP_ENV.

    APP_ENV=development   (default)
    APP_ENV=staging
    APP_ENV=production

HOW PRECEDENCE WORKS
--------------------
    shell environment   >   .env file   >   profile DEFAULTS   >   base.py

`.env` is loaded first, then the profile applies its DEFAULTS with
`os.environ.setdefault`, then `base.py` reads the environment as it always
has. `setdefault` never overwrites, so an explicit value always wins and a
profile can only decide what happens when you have not decided.

This is why `base.py` needed no changes: profiles work purely by shaping the
environment before it is read. There is exactly one place that parses
configuration, so a value cannot mean two different things in two files.
"""

import logging
import os

from dotenv import load_dotenv

# Must run BEFORE the profile applies its defaults, so that a value set in
# .env counts as "explicitly chosen" and the profile does not override it.
load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()

_PROFILES = ("development", "staging", "production")

if APP_ENV not in _PROFILES:
    raise ValueError(
        f"APP_ENV={APP_ENV!r} is not one of {_PROFILES}. "
        "Fix it rather than falling back: silently running production with "
        "development defaults is exactly the failure this file exists to stop."
    )

if APP_ENV == "production":
    from .production import DEFAULTS as _DEFAULTS
elif APP_ENV == "staging":
    from .staging import DEFAULTS as _DEFAULTS
else:
    from .development import DEFAULTS as _DEFAULTS

for _key, _value in _DEFAULTS.items():
    os.environ.setdefault(_key, _value)

# Imported only after the environment is fully shaped.
from .base import Settings, settings  # noqa: E402

logging.getLogger("voxlive.config").info(
    "configuration profile: %s (%d default(s) applied)", APP_ENV, len(_DEFAULTS)
)

__all__ = ["APP_ENV", "Settings", "settings"]
