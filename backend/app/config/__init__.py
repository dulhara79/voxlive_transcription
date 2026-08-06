"""Config package. Selects an environment profile from APP_ENV.

P0.3: development | staging | production. `base.py` holds the defaults and the
validation; the profile modules override only what differs, so a value can
never silently disagree between environments.
"""

from .base import Settings, settings

__all__ = ["Settings", "settings"]
