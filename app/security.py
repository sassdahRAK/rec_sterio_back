"""Security helpers: SSRF guard for source URLs, proxy-aware client address
and a small in-memory rate limiter."""
from __future__ import annotations

import ipaddress
import socket
import threading
import time
from collections import OrderedDict, deque
from urllib.parse import urlparse

from fastapi import Request

from app.config import settings
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


def _is_private_host(host: str, require_resolution: bool = False) -> bool:
    """True when the host resolves to a loopback/private/link-local/reserved address.

    `require_resolution=True` (the fetch-time guard) fails CLOSED: a hostname
    that cannot be resolved at all is refused instead of passed to the
    downloader, so a rebinding name or a typo'd target never becomes a
    best-effort pass.
    """
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
        # At creation time an unresolvable host is let through (the downloader
        # fails cleanly); at fetch time it is refused — decide via the caller.
        return require_resolution
    for info in infos:
        addr = info[4][0]
        try:
            if _ip_is_dangerous(ipaddress.ip_address(addr)):
                return True
        except ValueError:
            continue
    return False


def _ip_is_dangerous(ip: ipaddress._BaseAddress) -> bool:
    # `not is_global` is the catch-all: it refuses carrier-grade NAT
    # (100.64.0.0/10), benchmarking, documentation and the other special-use
    # ranges that are none of private/loopback/link-local/reserved/multicast.
    # The named clauses stay as belt-and-braces for any address where
    # `is_global` is True yet the range is still unroutable-or-local; a normal
    # public address is global, so it is unaffected.
    return (
        not ip.is_global
        or ip.is_private
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


def assert_fetch_url_safe(url: str) -> str:
    """The fetch-time form of the guard, for the moment the bytes are pulled.

    `validate_source_url` runs once at create time; the actual download happens
    later from the worker, so between the two a DNS record can be flipped
    (rebinding) or a redirect can be introduced by the source. This variant is
    fail-closed: the host must resolve NOW to at least one non-private address,
    or the fetch is refused. The downloader calls this immediately before
    handing the URL to yt-dlp and again on the final URL after extraction, so
    every hop the client actually took lands in one of the two checks.
    """
    stripped = validate_source_url(url)
    host = urlparse(stripped).hostname or ""
    if _is_private_host(host, require_resolution=True):
        raise AppError(
            422,
            "validation_failed",
            "source_url resolves to a private or local address, or its host "
            "cannot be resolved.",
            [{"field": "source_url", "issue": "blocked_host"}],
        )
    return stripped


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


def client_ip(request: Request) -> str:
    """The caller's address as the rate limiter should see it.

    Policy, per `settings.trust_proxy_headers` (default False):

    * False — this process faces the client directly (bare uvicorn), so
      `request.client.host` is the only address the caller cannot forge. A
      client-supplied `X-Forwarded-For` is ignored ENTIRELY: rotating the
      header must not mint fresh rate-limit buckets, otherwise the redeem and
      login throttles are advisory.
    * True — a TLS-terminating proxy on the same host (the deploy compose's
      Caddy) is in front of this process: `request.client.host` is then the
      proxy for every visitor, and the real client arrives in
      `X-Forwarded-For`. Only the LAST hop is trusted — the one the proxy
      itself wrote (it overwrites the header for a direct client and appends
      itself at the end when chaining); every earlier hop was typed by whoever
      sent the request and is as fake as it pleases. A header-rotating caller
      still lands in the bucket its real (proxy-seen) address owns.
    """
    peer = request.client.host if request.client else "unknown"
    if not settings.trust_proxy_headers:
        return peer
    forwarded = (request.headers.get("x-forwarded-for") or "").strip()
    hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
    return hops[-1] if hops else peer


class RateLimiter:
    """Fixed-window, per-key in-memory limiter. Good enough for a single-process API.

    The key space is bounded: unauthenticated traffic (redeem endpoints key on
    client-supplied values) could otherwise grow the dict without limit. Keys
    are LRU-touched on every check; when the store exceeds the cap it is first
    swept for buckets whose newest hit is older than the largest window ever
    seen, and whatever remains is evicted oldest-first. Eviction can only
    re-allow a throttled caller — it can never punish one.
    """
    MAX_KEYS = 5000

    def __init__(self) -> None:
        self._hits: "OrderedDict[str, deque[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_window_seen = 0.0

    def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            self._max_window_seen = max(self._max_window_seen, float(window_seconds))
            # pop + reinsert = LRU touch for this key.
            bucket = self._hits.pop(key, deque())
            while bucket and now - bucket[0] > window_seconds:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(1, int(window_seconds - (now - bucket[0])) + 1)
                self._hits[key] = bucket
                return False, retry_after
            bucket.append(now)
            self._hits[key] = bucket
            self._prune(now)
            return True, 0

    def _prune(self, now: float) -> None:
        """Bound the store: lazy expiry sweep, then LRU eviction."""
        if len(self._hits) <= self.MAX_KEYS:
            return
        sweep_to = now - self._max_window_seen
        for key in list(self._hits.keys()):
            bucket = self._hits[key]
            if not bucket or bucket[-1] < sweep_to:
                del self._hits[key]
        while len(self._hits) > self.MAX_KEYS:
            # Oldest-inserted first: least recently touched, so least likely
            # to still be mid-window.
            self._hits.popitem(last=False)
        # A pruned key simply comes back empty on its next use.

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
