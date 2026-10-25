"""The error envelope: shape, content type, no stack traces, no HTML."""
from __future__ import annotations

import json

from fastapi import FastAPI

from app.errors import register_exception_handlers


def test_404_envelope(client, auth_headers):
    resp = client.get(
        "/api/v1/videos/11111111-1111-1111-1111-111111111111", headers=auth_headers
    )
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert list(body.keys()) == ["error"]
    assert set(body["error"]) <= {"code", "message", "details"}
    assert body["error"]["code"] == "not_found"
    assert "Traceback" not in json.dumps(body)


def test_422_envelope(client, auth_headers):
    resp = client.post("/api/v1/videos", json={"source_url": 12345}, headers=auth_headers)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "validation_failed"
    assert isinstance(body["error"]["details"], list)
    assert body["error"]["details"][0]["field"] == "source_url"
    assert "Traceback" not in json.dumps(body)


def test_401_envelope(client):
    resp = client.get("/api/v1/stats")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


def test_unhandled_exception_returns_json_envelope():
    """A crashing route must still return the envelope, never HTML/traceback."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom():  # pragma: no cover - exercised through the handler
        raise RuntimeError("kaboom secret detail")

    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/boom")
    assert resp.status_code == 500
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["error"]["code"] == "internal_error"
    assert "kaboom" not in json.dumps(body)
    assert "Traceback" not in json.dumps(body)


def test_no_html_anywhere(client, auth_headers):
    for path in ("/api/v1/videos", "/api/v1/nope"):
        resp = client.get(path, headers=auth_headers)
        assert "text/html" not in resp.headers.get("content-type", "")
