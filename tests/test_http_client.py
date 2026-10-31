"""Regression tests for the shared httpx helpers (`http_json` / `http_bytes`).

`httpx.Response` is NOT a context manager — only `httpx.Client` is — so
`with httpx.request(...)` or `with response:` raises
`TypeError: 'httpx.Response' object does not support the context manager
protocol` *after* the request has already completed. That TypeError is not an
`httpx.HTTPError`, so it escaped the helpers' 502 mapping and surfaced as a
500 `internal_error`; every provider call (Google token exchange/userinfo,
Drive upload sessions, Microsoft Graph) went through this path.

These tests drive the REAL helpers through a real `httpx.Client` backed by
`httpx.MockTransport`, so the response object and the context-manager/close
path are genuinely exercised instead of mocking the helper away.
"""
from __future__ import annotations

import httpx
import pytest

from app.errors import AppError
from app.services import http_client

TOKEN_URL = "https://oauth2.googleapis.com/token"


def _patched_module_request(monkeypatch, handler, *, follow_redirects: bool = False) -> httpx.Client:
    """Route `httpx.request(...)` through a real Client + MockTransport.

    Returns the client so the caller can close it; `http_json` calls the
    module-level `httpx.request` helper, so patching that attribute is what
    exercises the helper's own response handling.

    `follow_redirects` configures the *injected* client. Pass True when the
    test must prove the helper's own `follow_redirects=False` is what stops a
    redirect, rather than the client's default.
    """
    client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=follow_redirects
    )

    def fake_request(method: str, url: str, **kwargs) -> httpx.Response:
        return client.request(method, url, **kwargs)

    monkeypatch.setattr(http_client.httpx, "request", fake_request)
    return client


def test_http_json_returns_status_headers_and_body(monkeypatch):
    """The happy path used by the Google token exchange must not raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL(TOKEN_URL)
        assert request.headers["content-type"].startswith("application/json")
        return httpx.Response(
            200,
            headers={"x-provider-test": "google"},
            json={"access_token": "token-value", "refresh_token": "refresh-value"},
        )

    client = _patched_module_request(monkeypatch, handler)
    try:
        status, headers, body = http_client.http_json(
            "POST", TOKEN_URL, json_body={"code": "abc", "grant_type": "authorization_code"}
        )
    finally:
        client.close()

    assert status == 200
    assert headers.get("x-provider-test") == "google"
    assert body == {"access_token": "token-value", "refresh_token": "refresh-value"}


def test_http_json_non_json_body_becomes_empty_dict(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204, content=b"")

    client = _patched_module_request(monkeypatch, handler)
    try:
        status, _, body = http_client.http_json("GET", "https://graph.microsoft.com/v1.0/me")
    finally:
        client.close()

    assert status == 204
    assert body == {}


def test_http_json_does_not_follow_redirects(monkeypatch):
    """`http_json` itself must pass `follow_redirects=False`.

    The injected client is built with `follow_redirects=True` and the transport
    can answer the redirect target, so the only thing that keeps the redirect
    from being followed is the `follow_redirects=False` that `http_json` passes
    on the request. If that kwarg were dropped the client would follow the 302
    and the assertions below would fail.
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.host == "oauth2.googleapis.com":
            return httpx.Response(302, headers={"location": "https://elsewhere.example/token"})
        return httpx.Response(200, json={"followed": True})

    client = _patched_module_request(monkeypatch, handler, follow_redirects=True)
    try:
        status, headers, _ = http_client.http_json("POST", TOKEN_URL, data={"code": "x"})
    finally:
        client.close()

    assert status == 302
    assert headers.get("location") == "https://elsewhere.example/token"
    assert requested == [TOKEN_URL], "http_json followed the redirect"


def test_http_json_maps_transport_error_to_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _patched_module_request(monkeypatch, handler)
    try:
        with pytest.raises(AppError) as excinfo:
            http_client.http_json("GET", "https://openidconnect.googleapis.com/v1/userinfo")
    finally:
        client.close()

    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "upstream_unreachable"


def test_http_bytes_non_stream_returns_bytes(monkeypatch):
    """Drive upload-session creation takes the non-streaming branch.

    The payload is deliberately non-empty so the assertion proves the body is
    fully buffered into `bytes` before the response is closed, not an empty or
    lazily-consumed stream.
    """
    payload = b'{"id": "upload-session-1", "status": "resumable"}'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Location": "https://www.googleapis.com/upload/drive/v3/files?upload_id=1"},
            content=payload,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    monkeypatch.setattr(http_client, "_client", client)
    try:
        status, headers, body = http_client.http_bytes(
            "POST",
            "https://www.googleapis.com/upload/drive/v3/files",
            headers={"Content-Type": "application/json"},
            body=b'{"name": "clip.mp4"}',
        )
    finally:
        client.close()

    assert status == 200
    assert headers.get("location", "").startswith("https://www.googleapis.com/upload/drive")
    assert body == payload


def test_http_bytes_stream_close_releases_without_advancing(monkeypatch):
    """Closing a never-iterated streaming body must release the connection.

    A bare generator's `finally` only runs once it has been started, so the old
    implementation leaked the pooled `_stream_client` connection whenever a
    caller closed the body without reading it — exactly what `_probe` and
    `open_download`'s error branch did.
    """
    state = {"closed": False}

    class TrackingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"payload"

        def close(self):
            state["closed"] = True

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=TrackingStream())

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    monkeypatch.setattr(http_client, "_stream_client", client)
    try:
        status, _, body = http_client.http_bytes(
            "GET",
            "https://www.googleapis.com/drive/v3/files/x?alt=media",
            headers={"Range": "bytes=0-0"},
            stream=True,
        )
        assert status == 200
        # The stream must still be open until the caller consumes or closes it.
        assert state["closed"] is False

        body.close()
        assert state["closed"] is True, "close() did not release the underlying stream"

        # Closing twice must not re-enter the context manager.
        body.close()
        assert state["closed"] is True
    finally:
        client.close()
