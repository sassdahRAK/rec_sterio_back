"""Security helpers: SSRF guard for source URLs and a small in-memory rate limiter."""
from __future__ import annotations

import ipaddress
import socket
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

from app.errors import AppError

ALLOWED_SCHEMES = {"http", "https"}
BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",
    "metadata",
}


def _is_private_host(host: str) -> bool:
    """True when the host resolves to a loopback/private/link-local/reserved address."""
    if not host:
        return True
    lowered = host.lower().strip("[]")
    if lowered in BLOCKED_HOSTNAMES:
        return True
    # Literal IP?
    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        ip = None
    if ip is not None:
        return _ip_is_dangerous(ip)
    # Hostname: resolve it (best effort) and check every address.
    try:
        infos = socket.getaddrinfo(lowered, None)
    except (socket.gaierror, UnicodeError, OSError):
        # Unresolvable host -> let it through; the downloader will fail cleanly.
        return False
    for info in infos:
        addr = info[4][0]
        try:
            if _ip_is_dangerous(ipaddress.ip_address(addr)):
                return True
        except ValueError:
            continue
    return False


def _ip_is_dangerous(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_source_url(url: str) -> str:
    """Validate a user supplied media URL. Raises AppError(422) when unusable."""
    if not url or not url.strip():
        raise AppError(422, "validation_failed", "source_url must not be empty.",
                       [{"field": "source_url", "issue": "empty"}])
    raw = url.strip()
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise AppError(
            422,
            "validation_failed",
            "source_url must be a valid http(s) URL.",
            [{"field": "source_url", "issue": "invalid_scheme"}],
        )
    host = parsed.hostname
    if not host:
        raise AppError(
            422,
            "validation_failed",
            "source_url must be a valid http(s) URL.",
            [{"field": "source_url", "issue": "missing_host"}],
        )
    if _is_private_host(host):
        raise AppError(
            422,
            "validation_failed",
            "source_url points to a private or local address.",
            [{"field": "source_url", "issue": "blocked_host"}],
        )
    return raw


def detect_platform(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if "youtube" in host or "youtu.be" in host:
        return "youtube"
    if "tiktok" in host:
        return "tiktok"
    if "facebook" in host or "fb.watch" in host:
        return "facebook"
    if "instagram" in host:
        return "instagram"
    if host:
        return "direct"
    return "unknown"


class RateLimiter:
    """Fixed-window, per-key in-memory limiter. Good enough for a single-process API."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] > window_seconds:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(1, int(window_seconds - (now - bucket[0])) + 1)
                return False, retry_after
            bucket.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()


def enforce_rate_limit(key: str, limit: int, window_seconds: int) -> None:
    allowed, retry_after = rate_limiter.check(key, limit, window_seconds)
    if not allowed:
        raise AppError(
            429,
            "rate_limited",
            "Too many requests. Please retry later.",
            [{"limit": limit, "window_seconds": window_seconds}],
            headers={"Retry-After": str(retry_after)},
        )
