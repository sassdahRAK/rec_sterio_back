"""DNS pinning for outbound fetches (closes the DNS-rebinding window).

`app.security.assert_fetch_url_safe` resolves a host and refuses private
answers, but the connection that follows resolves the name a second time: a
record flipped in between (DNS rebinding) sends the fetch to an internal host.
`pin_resolved_host` resolves once and then makes `socket.getaddrinfo` answer
only those addresses for the duration of the block, so the name that was
checked is the address that is used.

What is pinned is the ADDRESS, not the hostname: the caller still fetches the
original URL, so TLS/SNI and certificate validation keep using the hostname
while the socket can only land on an address that was public when the check
ran.

The pin map is thread-local because the API and the worker share one process
and each downloads on its own thread; a pin installed by one download must
never steer another thread's lookups. The wrapper over `socket.getaddrinfo`
itself is installed once per process and delegates to the original resolver
for every lookup that is not pinned on the current thread.

Limitation: the pin only covers lookups that go through Python's
`socket.getaddrinfo`. When yt-dlp impersonates a browser (`YTDLP_IMPERSONATE`,
which routes through curl-cffi / libcurl) the HTTP stack may resolve the
hostname itself, so pinning cannot be guaranteed on that path. The fetch-time
validation and the post-fetch final-URL recheck still apply there.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
from contextlib import contextmanager
from typing import Iterator
from urllib.parse import urlparse

from app.security import _ip_is_dangerous


class UnsafeHostError(RuntimeError):
    """The host cannot be resolved, or resolves to no public address."""


# The resolver as it was when this module loaded. The process-wide wrapper
# delegates here for unpinned lookups; tests may replace it with a fake.
_original_getaddrinfo = socket.getaddrinfo

_install_lock = threading.Lock()
_installed = False

# Per-thread: {lowercased hostname: (address, ...)}. Threads created later get
# their own empty slot automatically; nothing is shared between downloads.
_local = threading.local()

_SOCKTYPE_PROTO = {
    socket.SOCK_STREAM: socket.IPPROTO_TCP,
    socket.SOCK_DGRAM: socket.IPPROTO_UDP,
}


def resolve_public_ips(host: str) -> tuple[str, ...]:
    """Every unique public address `host` resolves to, in resolver order.

    Returns `()` when the host is empty, cannot be resolved, or yields only
    private / loopback / link-local / reserved / multicast / unspecified
    answers. Uses the saved original resolver so an active pin on this thread
    cannot influence what is considered "public".
    """
    text = (host or "").strip().strip("[]")
    if not text:
        return ()
    try:
        infos = _original_getaddrinfo(text, None)
    except (socket.gaierror, UnicodeError, OSError):
        return ()
    seen: dict[str, None] = {}
    for info in infos:
        try:
            addr = info[4][0]
        except (IndexError, TypeError):
            continue
        if not isinstance(addr, str) or not addr:
            continue
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
        except ValueError:
            continue
        if _ip_is_dangerous(ip):
            continue
        seen.setdefault(addr, None)
    return tuple(seen)


def _port_number(port: object) -> int:
    """Accept the int / None / "443" / "https" forms getaddrinfo takes."""
    if port is None:
        return 0
    if isinstance(port, int):
        return port
    text = str(port).strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        pass
    for proto_name in ("tcp", "udp"):
        try:
            return socket.getservbyname(text, proto_name)
        except OSError:
            continue
    return 0


def _pinned_entries(
    addresses: tuple[str, ...],
    port: int,
    family: int,
    socktype: int,
    proto: int,
    canonname: str,
) -> list[tuple]:
    entries: list[tuple] = []
    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr.split("%", 1)[0])
        except ValueError:
            continue
        addr_family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        if family not in (0, socket.AF_UNSPEC, addr_family):
            continue
        sockaddr = (
            (addr, port, 0, 0) if addr_family == socket.AF_INET6 else (addr, port)
        )
        wanted = [socktype] if socktype != 0 else list(_SOCKTYPE_PROTO)
        for stype in wanted:
            stype_proto = _SOCKTYPE_PROTO.get(stype)
            if stype_proto is None:
                continue
            if proto not in (0, stype_proto):
                continue
            entries.append((addr_family, stype, stype_proto, canonname, sockaddr))
    return entries


def _pinned_getaddrinfo(
    host, port=None, family=0, type=0, proto=0, flags=0
):
    """`socket.getaddrinfo` replacement: pinned hosts never hit the resolver."""
    pins = getattr(_local, "pins", None)
    # getaddrinfo accepts a bytes hostname too; decode it so a pinned name
    # written as bytes cannot slip past the pin to the live resolver.
    name = host
    if isinstance(host, bytes):
        try:
            name = host.decode("ascii")
        except UnicodeDecodeError:
            name = host  # not a hostname anyway; let the original resolver judge
    if isinstance(name, str) and pins:
        addresses = pins.get(name.lower())
        if addresses is not None:
            canonname = name if flags & getattr(socket, "AI_CANONNAME", 0) else ""
            entries = _pinned_entries(
                addresses, _port_number(port), family, type, proto, canonname
            )
            if entries:
                return entries
            # Pinned, but nothing matches the requested family/socktype: fail
            # rather than fall through to the live resolver, which is exactly
            # the rebinding path this module exists to close.
            raise socket.gaierror(
                getattr(socket, "EAI_NONAME", -2),
                "no pinned address matches the requested family/socket type",
            )
    return _original_getaddrinfo(host, port, family, type, proto, flags)


def _ensure_installed() -> None:
    global _installed
    if _installed:
        return
    with _install_lock:
        if _installed:
            return
        socket.getaddrinfo = _pinned_getaddrinfo
        _installed = True


@contextmanager
def pin_resolved_host(url: str) -> Iterator[tuple[str, ...]]:
    """Pin `url`'s hostname to its current public addresses for the block.

    Refuses with `UnsafeHostError` when the host is missing, cannot be
    resolved, or has no public address. The pin is always removed on exit,
    including on exception.
    """
    _ensure_installed()
    host = (urlparse(url).hostname or "").strip()
    if not host:
        raise UnsafeHostError("The URL names no host to resolve.")
    addresses = resolve_public_ips(host)
    if not addresses:
        raise UnsafeHostError(
            f"The host {host!r} resolves to no public address."
        )

    pins = getattr(_local, "pins", None)
    if pins is None:
        pins = {}
        _local.pins = pins

    key = host.lower()
    had_previous = key in pins
    previous = pins.get(key)
    pins[key] = addresses
    try:
        yield addresses
    finally:
        if had_previous:
            pins[key] = previous
        else:
            pins.pop(key, None)
        if not pins:
            try:
                del _local.pins
            except AttributeError:  # pragma: no cover - defensive
                pass
