"""Uniform error envelope + FastAPI exception handlers.

Every error response in the whole API looks like:

    {"error": {"code": "not_found", "message": "...", "details": [...]}}

Never HTML, never a stack trace.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("recsterio.errors")

STATUS_CODE_MAP: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_failed",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}


def error_payload(code: str, message: str, details: Any = None) -> dict:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return {"error": body}


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code, message, details),
        headers=headers,
    )


class AppError(Exception):
    """Raised by services/routers to produce a specific error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str | None = None,
        message: str = "",
        details: Any = None,
        headers: dict | None = None,
        extra: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code or STATUS_CODE_MAP.get(status_code, "error")
        self.message = message
        self.details = details
        self.headers = headers
        # Optional extra top-level keys alongside the error envelope
        # (e.g. the already existing video on a 409 duplicate create).
        self.extra = extra or {}

    def to_response(self) -> JSONResponse:
        content = error_payload(self.code, self.message, self.details)
        content.update(self.extra)
        return JSONResponse(
            status_code=self.status_code, content=content, headers=self.headers
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):  # noqa: ANN202
        return exc.to_response()

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):  # noqa: ANN202
        code = STATUS_CODE_MAP.get(exc.status_code, "error")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        details = None if isinstance(exc.detail, str) else exc.detail
        headers = dict(exc.headers or {})
        return error_response(exc.status_code, code, message, details, headers)

    @app.exception_handler(StarletteHTTPException)
    async def _starlette_http_error(request: Request, exc: StarletteHTTPException):  # noqa: ANN202
        code = STATUS_CODE_MAP.get(exc.status_code, "error")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        details = None if isinstance(exc.detail, str) else exc.detail
        headers = dict(exc.headers or {})
        return error_response(exc.status_code, code, message, details, headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):  # noqa: ANN202
        details = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body", "query"))
            details.append(
                {
                    "field": loc or "body",
                    "issue": err.get("type", "invalid"),
                    "message": err.get("msg", "invalid value"),
                }
            )
        return error_response(
            422, "validation_failed", "Request validation failed.", details
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # noqa: ANN202
        incident = uuid.uuid4().hex[:12]
        logger.exception("unhandled error incident=%s path=%s", incident, request.url.path)
        return error_response(
            500,
            "internal_error",
            "An internal error occurred.",
            [{"incident": incident}],
        )
