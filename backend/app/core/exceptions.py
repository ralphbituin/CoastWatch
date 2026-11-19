"""Domain exceptions and the shared error envelope.

Every API error is rendered through :func:`register_exception_handlers` as a
consistent JSON body::

    {"error": {"code": "...", "message": "...", "details": {...}}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


class CoastWatchError(Exception):
    """Base class for all CoastWatch domain errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(CoastWatchError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class ValidationError(CoastWatchError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"


class ConflictError(CoastWatchError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"


class DataUnavailableError(CoastWatchError):
    """Requested data is legitimately unavailable (e.g. outside dataset range).

    This must never be silently treated as a normal value.
    """

    status_code = status.HTTP_404_NOT_FOUND
    code = "data_unavailable"


class ModelUnavailableError(CoastWatchError):
    """No validated model artifact is available for this task.

    Never silently replaced by a fabricated output. Train/validate one first
    (see docs/MODEL_DOCUMENTATION.md) or use an operator-labelled input.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_unavailable"


class DatabaseUnavailableError(CoastWatchError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "database_unavailable"


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the consistent error handlers to ``app``."""

    @app.exception_handler(CoastWatchError)
    async def _domain(_request: Request, exc: CoastWatchError) -> JSONResponse:
        logger.warning("domain_error code=%s message=%s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(
                "validation_error",
                "Request validation failed",
                {"errors": jsonable_encoder(exc.errors())},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", "An unexpected error occurred"),
        )
