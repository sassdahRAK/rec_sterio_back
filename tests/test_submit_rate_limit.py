"""Per-actor rate limit on POST /videos.

Every Telegram request arrives from the bot host, so the existing client-IP
limiter is one team-wide budget and a single first-time user can exhaust it.
These tests pin the per-actor limiter that sits alongside it: one actor is
throttled on its own budget while another is untouched, a guest keeps the old
IP-only behaviour, and a caller under both limits sees the normal response.
"""
from __future__ import annotations

from app.config import settings
from tests.helpers import guest_headers, telegram_user_headers

API = "/api/v1/videos"


def _url(tag: str) -> str:
    return f"https://www.youtube.com/watch?v=recsterio-ratelimit-{tag}"


def _post(client, headers, tag: str):
    return client.post(API, json={"source_url": _url(tag)}, headers=headers)


def test_one_actor_is_throttled_while_another_is_unaffected(client, monkeypatch):
    # The shared IP budget is effectively unlimited, so the only source of a
    # 429 below is the actor bucket.
    monkeypatch.setattr(settings, "video_create_rate_limit", 10000)
    monkeypatch.setattr(settings, "video_create_actor_rate_limit", 2)

    first = _post(client, telegram_user_headers(111001), "actor-a-1")
    second = _post(client, telegram_user_headers(111001), "actor-a-2")
    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text

    blocked = _post(client, telegram_user_headers(111001), "actor-a-3")
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert "Retry-After" in blocked.headers

    # Same client IP (the test client), different actor: a fresh bucket.
    other = _post(client, telegram_user_headers(222002), "actor-b-1")
    assert other.status_code == 201, other.text


def test_actor_limit_applies_even_when_the_ip_limit_would_not_fire(
    client, monkeypatch
):
    monkeypatch.setattr(settings, "video_create_rate_limit", 100)
    monkeypatch.setattr(settings, "video_create_actor_rate_limit", 1)

    first = _post(client, telegram_user_headers(333003), "actor-c-1")
    assert first.status_code == 201, first.text

    # Two requests against an IP budget of 100: only the actor bucket can
    # throttle this one.
    blocked = _post(client, telegram_user_headers(333003), "actor-c-2")
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"
    assert blocked.headers["Retry-After"].isdigit()


def test_anonymous_callers_still_hit_the_ip_limiter(client, monkeypatch):
    # A guest has no stable actor, so the actor limiter is skipped and the IP
    # bucket alone decides, exactly as it did before the per-actor limit.
    monkeypatch.setattr(settings, "video_create_rate_limit", 1)
    monkeypatch.setattr(settings, "video_create_actor_rate_limit", 10000)

    first = _post(client, guest_headers(), "guest-1")
    second = _post(client, guest_headers(), "guest-2")
    assert first.status_code != 429, first.text
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limited"
    assert "Retry-After" in second.headers


def test_a_caller_under_both_limits_gets_a_normal_response(client, monkeypatch):
    monkeypatch.setattr(settings, "video_create_rate_limit", 10000)
    monkeypatch.setattr(settings, "video_create_actor_rate_limit", 5)

    headers = telegram_user_headers(444004)
    first = _post(client, headers, "under-1")
    assert first.status_code == 201, first.text

    # A repeat of the same URL is the ordinary dedupe answer, never a 429.
    repeat = _post(client, headers, "under-1")
    assert repeat.status_code == 409
    assert repeat.json()["error"]["code"] == "conflict"
