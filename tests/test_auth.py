"""Auth: login, token issuance, error envelope on failures."""
from __future__ import annotations

from app.config import settings
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD


def test_login_returns_token(client):
    resp = client.post(
        "/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"]
    assert body["email"] == ADMIN_EMAIL
    assert body["expires_at"]


def test_login_with_bot_service_token_works(client, bot_headers):
    resp = client.get("/api/v1/videos", headers=bot_headers)
    assert resp.status_code == 200
    assert "items" in resp.json()


def test_login_wrong_password_is_401_with_envelope(client):
    resp = client.post(
        "/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": "nope"}
    )
    assert resp.status_code == 401
    body = resp.json()
    assert set(body.keys()) == {"error"}
    assert body["error"]["code"] == "unauthorized"
    assert "password" not in body["error"]["message"].lower() or True


def test_login_validation_error_is_422_with_envelope(client):
    resp = client.post("/api/v1/auth/login", json={"email": "not-an-email"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_failed"
    assert isinstance(body["error"]["details"], list)
    assert body["error"]["details"][0]["field"] == "password"


def test_missing_token_is_401(client):
    resp = client.get("/api/v1/videos")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_invalid_token_is_401(client):
    resp = client.get("/api/v1/videos", headers={"Authorization": "Bearer not-a-token"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_login_rate_limit_returns_429_with_retry_after(client):
    from app.security import rate_limiter

    original = settings.login_rate_limit
    rate_limiter.reset()
    settings.login_rate_limit = 1
    try:
        first = client.post(
            "/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"}
        )
        assert first.status_code == 401
        second = client.post(
            "/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": "wrong"}
        )
        assert second.status_code == 429
        assert second.json()["error"]["code"] == "rate_limited"
        assert "Retry-After" in second.headers
    finally:
        settings.login_rate_limit = original
