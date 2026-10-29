"""DNS pinning: a checked address is the address actually used.

Every test replaces `app.dns_pin._original_getaddrinfo` with a fake, so no
real DNS lookup ever happens.
"""
from __future__ import annotations

import socket
import threading

import pytest

from app import dns_pin
from app.dns_pin import UnsafeHostError, pin_resolved_host


def _fake_resolver(answers: dict[str, list[str]]):
    """A drop-in for socket.getaddrinfo backed by a mutable answer table."""

    def _getaddrinfo(host, port=None, family=0, type=0, proto=0, flags=0):
        key = host.lower() if isinstance(host, str) else host
        addresses = answers.get(key)
        if addresses is None:
            raise socket.gaierror(socket.EAI_NONAME, "name or service not known")
        resolved = []
        for addr in addresses:
            addr_family = socket.AF_INET6 if ":" in addr else socket.AF_INET
            if addr_family == socket.AF_INET6:
                sockaddr = (addr, port or 0, 0, 0)
            else:
                sockaddr = (addr, port or 0)
            resolved.append(
                (addr_family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)
            )
        return resolved

    return _getaddrinfo


def _patch_resolver(monkeypatch, answers: dict[str, list[str]]) -> None:
    monkeypatch.setattr(dns_pin, "_original_getaddrinfo", _fake_resolver(answers))


def test_pinned_lookup_ignores_a_later_private_answer(monkeypatch):
    answers = {"media.example": ["8.8.8.8"]}
    _patch_resolver(monkeypatch, answers)

    with pin_resolved_host("https://media.example/watch?v=1"):
        # The record flips to an internal address after the check ran.
        answers["media.example"] = ["10.0.0.5"]
        infos = socket.getaddrinfo("media.example", 443)
        assert {info[4][0] for info in infos} == {"8.8.8.8"}
        assert all(info[0] == socket.AF_INET for info in infos)
        assert all(info[4][1] == 443 for info in infos)
        # Host match is case-insensitive and a string port is accepted.
        str_port = socket.getaddrinfo("MEDIA.example", "443")
        assert {info[4][0] for info in str_port} == {"8.8.8.8"}
        assert all(info[4][1] == 443 for info in str_port)


def test_a_bytes_hostname_uses_the_pin_and_never_the_resolver(monkeypatch):
    """getaddrinfo accepts bytes; a bytes host must not bypass the pin."""
    answers = {"media.example": ["8.8.8.8"]}
    live = _fake_resolver(answers)
    calls: list[object] = []

    def _counting_getaddrinfo(host, port=None, family=0, type=0, proto=0, flags=0):
        calls.append(host)
        return live(host, port, family, type, proto, flags)

    monkeypatch.setattr(dns_pin, "_original_getaddrinfo", _counting_getaddrinfo)

    with pin_resolved_host("https://media.example/v"):
        answers["media.example"] = ["10.0.0.5"]  # the record flips after the check
        calls.clear()  # the pin's own resolve went through the fake
        infos = socket.getaddrinfo(b"media.example", 443)
        assert {info[4][0] for info in infos} == {"8.8.8.8"}
        assert calls == []  # the fake original resolver was never consulted


def test_lookup_delegates_to_the_original_after_the_block(monkeypatch):
    answers = {"media.example": ["8.8.8.8"]}
    _patch_resolver(monkeypatch, answers)

    with pin_resolved_host("https://media.example/v"):
        assert {i[4][0] for i in socket.getaddrinfo("media.example", None)} == {
            "8.8.8.8"
        }

    # Out of the block the pin is gone: the original resolver answers again,
    # even with the flipped private answer.
    answers["media.example"] = ["10.0.0.5"]
    assert {i[4][0] for i in socket.getaddrinfo("media.example", None)} == {
        "10.0.0.5"
    }


def test_pinned_ipv6_answers_use_a_four_tuple_sockaddr(monkeypatch):
    _patch_resolver(monkeypatch, {"v6.example": ["2606:4700:4700::1111"]})

    with pin_resolved_host("https://[v6.example]/v"):
        infos = socket.getaddrinfo("v6.example", 8443, socket.AF_INET6)

    assert infos
    assert all(info[0] == socket.AF_INET6 for info in infos)
    assert len(infos[0][4]) == 4
    assert infos[0][4][0] == "2606:4700:4700::1111"
    assert infos[0][4][1] == 8443


def test_a_pinned_host_never_falls_through_on_a_family_mismatch(monkeypatch):
    """Only IPv4 is pinned: an AF_INET6 query must fail, never re-resolve."""
    _patch_resolver(monkeypatch, {"v4.example": ["8.8.8.8"]})

    with pin_resolved_host("https://v4.example/v"):
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("v4.example", 443, socket.AF_INET6)


def test_a_host_with_no_public_address_is_refused_and_leaves_no_pin(monkeypatch):
    answers = {"internal.example": ["127.0.0.1", "10.1.2.3"]}
    _patch_resolver(monkeypatch, answers)

    with pytest.raises(UnsafeHostError):
        with pin_resolved_host("https://internal.example/v"):
            pytest.fail("the block must not be entered")

    # No pin was left behind: the next lookup reaches the original resolver
    # and sees the private answer rather than a pinned (empty) set.
    assert {i[4][0] for i in socket.getaddrinfo("internal.example", None)} == {
        "127.0.0.1",
        "10.1.2.3",
    }


def test_an_unresolvable_host_is_refused(monkeypatch):
    _patch_resolver(monkeypatch, {})

    with pytest.raises(UnsafeHostError):
        with pin_resolved_host("https://missing.example/v"):
            pytest.fail("the block must not be entered")


def test_pins_are_thread_local(monkeypatch):
    """Two threads pinning different hosts must not see each other's pin."""
    answers = {
        "a.example": ["8.8.8.8"],
        "b.example": ["1.1.1.1"],
    }
    _patch_resolver(monkeypatch, answers)
    barrier = threading.Barrier(3)
    seen: dict[str, set[str]] = {}

    def _worker(host: str, other: str, tag: str) -> None:
        with pin_resolved_host(f"https://{host}/v"):
            barrier.wait(timeout=5)
            seen[tag] = {i[4][0] for i in socket.getaddrinfo(host, None)}
            seen[tag + "_other"] = {
                i[4][0] for i in socket.getaddrinfo(other, None)
            }

    first = threading.Thread(target=_worker, args=("a.example", "b.example", "a"))
    second = threading.Thread(target=_worker, args=("b.example", "a.example", "b"))
    first.start()
    second.start()
    barrier.wait(timeout=5)  # both pins are installed
    answers["a.example"] = ["10.0.0.9"]  # flip the live answer for a.example
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    # Each thread keeps its own pin...
    assert seen["a"] == {"8.8.8.8"}
    assert seen["b"] == {"1.1.1.1"}
    # ...and the other thread's host is resolved live, not from a shared pin:
    # b's thread sees the flipped private answer, a's thread does not.
    assert seen["a_other"] == {"1.1.1.1"}
    assert seen["b_other"] == {"10.0.0.9"}


def test_an_exception_inside_the_block_still_unpins(monkeypatch):
    answers = {"media.example": ["8.8.8.8"]}
    _patch_resolver(monkeypatch, answers)

    with pytest.raises(RuntimeError, match="boom"):
        with pin_resolved_host("https://media.example/v"):
            raise RuntimeError("boom")

    answers["media.example"] = ["10.0.0.5"]
    assert {i[4][0] for i in socket.getaddrinfo("media.example", None)} == {
        "10.0.0.5"
    }
