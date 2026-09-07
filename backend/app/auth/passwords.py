"""
passwords.py — password hashing.

WHY scrypt AND NOT bcrypt OR argon2
-----------------------------------
`hashlib.scrypt` is in the Python standard library and is backed by OpenSSL,
which is already in the container because Python links against it. bcrypt and
argon2-cffi both pull a compiled extension, which means either a manylinux
wheel or a C toolchain in the image. On a Fargate task that costs image size
and cold-start time on every deploy and every scale-out event, for a
primitive the standard library already provides at equivalent strength.

scrypt is memory-hard, which is the property that matters: it is what makes a
GPU attack expensive rather than merely parallel.

PARAMETERS
----------
    n = 2**14 (16384), r = 8, p = 1

That is the RFC 7914 "interactive login" set, roughly 16 MB and ~50-80 ms per
hash on a Fargate vCPU. Two consequences to be deliberate about:

  * OpenSSL enforces `maxmem`, and its default is 32 MB. n=2**15 needs ~32 MB
    and fails intermittently. `maxmem` is set explicitly below so the limit is
    a decision rather than a surprise.

  * ~60 ms is a real cost under a login storm. It is also the entire point.
    Rate limiting belongs in front of the endpoint, not in a weaker hash.

STORAGE FORMAT
--------------
    scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>

The parameters travel with the hash, so raising n later does not invalidate
existing passwords: old hashes keep verifying with their own parameters, and
`needs_rehash` tells the login path when to transparently upgrade one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os

_ALGORITHM = "scrypt"
_N = 2**14
_R = 8
_P = 1
_DKLEN = 32
_SALT_BYTES = 16
# n * r * 128 * 2 with headroom. OpenSSL's default of 32 MB is too tight to
# leave implicit.
_MAXMEM = 64 * 1024 * 1024

# Minimum length only. Composition rules (one digit, one symbol) push people
# toward "Password1!" and are worse than length for real-world strength.
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 256  # a hashing-cost DoS bound, not a security rule


class PasswordError(ValueError):
    """The supplied password is not acceptable."""


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def validate_password(password: str) -> None:
    """Raise PasswordError if the password cannot be used. No return value."""
    if not isinstance(password, str) or not password:
        raise PasswordError("Enter a password.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordError(
            f"Use at least {MIN_PASSWORD_LENGTH} characters. "
            "Length matters more than symbols."
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise PasswordError(f"Keep it under {MAX_PASSWORD_LENGTH} characters.")


def hash_password(password: str) -> str:
    """Hash a password for storage. Validates first."""
    validate_password(password)
    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_N,
        r=_R,
        p=_P,
        dklen=_DKLEN,
        maxmem=_MAXMEM,
    )
    return f"{_ALGORITHM}${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str | None) -> bool:
    """Constant-time check of a password against a stored hash.

    Returns False rather than raising on a malformed or missing hash. A user
    row with no password (an invited user who has not set one) must fail the
    same way and in the same time as a wrong password, or the difference is an
    account-enumeration oracle.
    """
    if not encoded or not password:
        return False
    try:
        algorithm, n, r, p, salt_b64, hash_b64 = encoded.split("$")
        if algorithm != _ALGORITHM:
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_unb64(salt_b64),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(_unb64(hash_b64)),
            maxmem=_MAXMEM,
        )
    except (ValueError, TypeError, MemoryError):
        return False
    return hmac.compare_digest(digest, _unb64(hash_b64))


def needs_rehash(encoded: str | None) -> bool:
    """True when a stored hash uses weaker parameters than the current ones.

    Call this after a SUCCESSFUL verify — that is the only moment the plaintext
    is available to re-hash with. Without it, raising `_N` only ever protects
    accounts created after the change.
    """
    if not encoded:
        return False
    try:
        algorithm, n, r, p, _, _ = encoded.split("$")
    except ValueError:
        return True
    return (algorithm, int(n), int(r), int(p)) != (_ALGORITHM, _N, _R, _P)
