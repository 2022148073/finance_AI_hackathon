"""Server-side invitation-code verification and opaque access sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


ACCESS_COOKIE_NAME = "flowbit_access_session"
PBKDF2_ALGORITHM = "pbkdf2_sha256"
DEFAULT_PBKDF2_ITERATIONS = 600_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hash_access_code(
    access_code: str,
    *,
    iterations: int = DEFAULT_PBKDF2_ITERATIONS,
    salt: bytes | None = None,
) -> str:
    """Return a portable PBKDF2 hash suitable for FLOWBIT_ACCESS_CODE_HASH."""
    if not access_code:
        raise ValueError("access_code must not be empty")
    if iterations < 100_000:
        raise ValueError("PBKDF2 iterations must be at least 100000")
    actual_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", access_code.encode("utf-8"), actual_salt, iterations
    )
    return "$".join(
        (
            PBKDF2_ALGORITHM,
            str(iterations),
            base64.urlsafe_b64encode(actual_salt).decode("ascii").rstrip("="),
            base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
        )
    )


def _decode_base64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True)
class AccessCodeVerifier:
    encoded_hash: str | None = None
    plain_code: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.encoded_hash or self.plain_code)

    def verify(self, candidate: str) -> bool:
        if not self.configured or not candidate:
            return False
        if self.encoded_hash:
            try:
                algorithm, raw_iterations, raw_salt, raw_digest = (
                    self.encoded_hash.split("$", 3)
                )
                if algorithm != PBKDF2_ALGORITHM:
                    return False
                iterations = int(raw_iterations)
                salt = _decode_base64(raw_salt)
                expected = _decode_base64(raw_digest)
                actual = hashlib.pbkdf2_hmac(
                    "sha256", candidate.encode("utf-8"), salt, iterations
                )
                return hmac.compare_digest(actual, expected)
            except (TypeError, ValueError):
                return False
        assert self.plain_code is not None
        return hmac.compare_digest(
            candidate.encode("utf-8"), self.plain_code.encode("utf-8")
        )

    @classmethod
    def from_environment(cls) -> "AccessCodeVerifier":
        encoded_hash = os.getenv("FLOWBIT_ACCESS_CODE_HASH", "").strip()
        plain_code = os.getenv("FLOWBIT_ACCESS_CODE", "")
        if encoded_hash and plain_code:
            raise RuntimeError(
                "Set only one of FLOWBIT_ACCESS_CODE_HASH or FLOWBIT_ACCESS_CODE"
            )
        return cls(encoded_hash=encoded_hash or None, plain_code=plain_code or None)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def create_access_session(connection, ttl_seconds: int) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    now = _utc_now()
    expires_at = now + timedelta(seconds=ttl_seconds)
    connection.execute(
        "DELETE FROM access_sessions WHERE expires_at <= ?",
        (now.isoformat(),),
    )
    connection.execute(
        "INSERT INTO access_sessions "
        "(session_token_hash,created_at,expires_at) VALUES (?,?,?)",
        (hash_session_token(token), now.isoformat(), expires_at.isoformat()),
    )
    return token, expires_at


def access_session_is_valid(connection, token: str | None) -> bool:
    if not token:
        return False
    row = connection.execute(
        "SELECT expires_at FROM access_sessions WHERE session_token_hash = ?",
        (hash_session_token(token),),
    ).fetchone()
    if row is None:
        return False
    try:
        expires_at = datetime.fromisoformat(str(row["expires_at"]))
    except ValueError:
        return False
    return expires_at > _utc_now()


class AccessRateLimiter:
    """Small in-process fixed-window limiter for the invitation endpoint."""

    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        if max_attempts < 1 or window_seconds < 1:
            raise ValueError("Rate-limit values must be positive")
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            attempts = self._attempts[client_key]
            while attempts and attempts[0] <= cutoff:
                attempts.popleft()
            if len(attempts) >= self.max_attempts:
                return False
            attempts.append(now)
            return True


if __name__ == "__main__":
    import getpass

    first = getpass.getpass("Access code: ")
    second = getpass.getpass("Confirm access code: ")
    if first != second:
        raise SystemExit("Access codes do not match")
    print(hash_access_code(first))
