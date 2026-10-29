"""Tiny HTTP helpers over httpx (the codebase's HTTP client).

Every external provider call (Google token/userinfo, Microsoft Graph, the
cloud providers) goes through these two functions so there is exactly one
place that turns a transport error or a non-2xx into the codebase's AppError,
and exactly one place tests monkeypatch to fake a provider.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple

import httpx

from app.errors import AppError


def http_json(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """One JSON call; returns (status, response_headers, parsed_body)."""
    try:
        with httpx.request(
            method,
            url,
            headers=headers,
            json=json_body,
            data=data,
            timeout=timeout,
            follow_redirects=False,
        ) as response:
            body: Dict[str, Any] = {}
            try:
                body = response.json() or {}
            except Exception:
                body = {}
            return response.status_code, dict(response.headers), body
    except httpx.HTTPError as exc:
        raise AppError(502, "upstream_unreachable", f"Could not reach {url.split('/')[2]}.") from exc


# Module-level streaming clients: a `client.stream()` response borrows its
# client, so the client must outlive the iterator the caller reads. Keeping two
# (plain + streaming) open for the process lifetime is cheaper than a new
# connection per provider call and keeps the streaming branch simple.
_client = httpx.Client(timeout=60.0, follow_redirects=False)
_stream_client = httpx.Client(timeout=300.0, follow_redirects=False)


def http_bytes(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
    stream: bool = False,
) -> Tuple[int, Dict[str, str], Any]:
    """A call whose body is bytes, or a streaming iterator when `stream`.

    Returns (status, response_headers, body) where body is bytes, or an
    iterator of bytes that closes the httpx stream when exhausted or closed.
    """
    try:
        if stream:
            raw = _stream_client.stream(method, url, headers=headers, content=body)
            response = raw.__enter__()
        else:
            response = _client.request(method, url, headers=headers, content=body)
    except httpx.HTTPError as exc:
        raise AppError(502, "upstream_unreachable", f"Could not reach {url.split('/')[2]}.") from exc

    if not stream:
        with response:
            return response.status_code, dict(response.headers), response.content

    def _iter() -> Iterator[bytes]:
        try:
            for chunk in response.iter_bytes(512 * 1024):
                yield chunk
        finally:
            raw.__exit__(None, None, None)

    return response.status_code, dict(response.headers), _iter()
