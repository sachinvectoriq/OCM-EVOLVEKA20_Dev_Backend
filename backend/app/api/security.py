"""Authenticate incoming KA API requests and audit their final HTTP outcome.

This boundary does not run on outbound SDK/service calls or SAML redirects.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from time import perf_counter
from uuid import uuid4

import jwt
from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.routing import Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send


# Exact method/path exceptions: no wildcard bypass for a router or caller header.
# The skill endpoints retain their existing Azure Search integration contract.
TOKEN_EXEMPT_API_ROUTES = frozenset({
    ("GET", "/api/health"),
    ("POST", "/api/enrich"),
    ("POST", "/api/split"),
})


def requires_api_token(method: str, path: str) -> bool:
    return (path == "/api" or path.startswith("/api/")) and (
        method.upper(), path.rstrip("/")
    ) not in TOKEN_EXEMPT_API_ROUTES


class TokenRejected(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _claim_text(value: object) -> str | None:
    # OneLogin SAML attributes commonly arrive as single-element lists.
    if isinstance(value, list):
        value = value[0] if value else None
    return value.strip() if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    email: str
    name: str | None
    group: str | None
    job_title: str | None
    user_data: dict = field(repr=False)


class KATokenValidator:
    def __init__(self, secret: str | None):
        if not isinstance(secret, str) or not secret.strip():
            raise ValueError("JWT_SECRET_KEY must be configured for API authentication")
        self._secret = secret

    def authenticate(self, scope: Scope) -> AuthenticatedUser:
        headers = [value for key, value in scope["headers"] if key.lower() == b"authorization"]
        if not headers:
            raise TokenRejected("missing_token")
        if len(headers) != 1 or len(headers[0]) > 16384:
            raise TokenRejected("invalid_token")
        parts = headers[0].decode("latin-1").split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise TokenRejected("invalid_token")
        try:
            payload = jwt.decode(
                parts[1],
                self._secret,
                algorithms=["HS256"],
                options={"require": ["exp", "user_data"]},
            )
            expiration = payload["exp"]
            if type(expiration) not in (int, float) or not math.isfinite(expiration):
                raise TokenRejected("invalid_token")
            claims = payload["user_data"]
            if not isinstance(claims, dict):
                raise TokenRejected("invalid_token")
            email = _claim_text(claims.get("email"))
            if not email:
                raise TokenRejected("invalid_token")
            return AuthenticatedUser(
                # Existing KA tokens have no stable sub/oid; never use a body user_id.
                user_id=email.casefold(),
                email=email,
                name=_claim_text(claims.get("name")),
                group=_claim_text(claims.get("group")),
                job_title=_claim_text(claims.get("job_title")),
                user_data=claims.copy(),
            )
        except jwt.ExpiredSignatureError:
            raise TokenRejected("expired_token") from None
        except (jwt.InvalidTokenError, ValueError, TypeError, OverflowError):
            raise TokenRejected("invalid_token") from None


def _route_template(scope: Scope) -> str:
    route = scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    # Auth failures happen before routing. Resolve the template without executing
    # a handler, dependency, or request body parser; never log raw IDs/query strings.
    partial = None
    for candidate in scope["app"].router.routes:
        match, _ = candidate.matches(scope)
        if match == Match.FULL:
            return candidate.path
        if match == Match.PARTIAL:
            partial = candidate.path
    return partial or "<unmatched>"


class APIAuthAuditMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not requires_api_token(scope["method"], scope["path"]):
            await self.app(scope, receive, send)
            return

        started = perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        request_id = str(uuid4())
        status_code = 500
        user = None
        auth_result = "authentication_error"
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            try:
                user = scope["app"].state.api_token_validator.authenticate(scope)
            except TokenRejected as error:
                auth_result = error.reason
                response = JSONResponse(
                    {"detail": "Authentication required", "code": error.reason},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send_with_request_id)
                return
            auth_result = "authenticated"
            scope["state"]["authenticated_user"] = user
            await self.app(scope, receive, send_with_request_id)
        finally:
            document = {
                "id": str(uuid4()),
                "record_type": "api_request",
                "timestamp_utc": timestamp,
                "user_id": user.user_id if user else None,
                "user_email": user.email if user else None,
                "user_name": user.name if user else None,
                "user_group": user.group if user else None,
                "job_title": user.job_title if user else None,
                "method": scope["method"],
                "route": _route_template(scope),
                "status_code": status_code,
                "duration_ms": round((perf_counter() - started) * 1000, 3),
                "request_id": request_id,
                "auth_result": auth_result,
            }
            await scope["app"].state.api_audit_store.write(document)


def install_api_security(app: FastAPI) -> None:
    app.add_middleware(APIAuthAuditMiddleware)
    original_openapi = app.openapi

    def authenticated_openapi():
        schema = original_openapi()
        schema.setdefault("components", {}).setdefault("securitySchemes", {})["KABearer"] = {
            "type": "http", "scheme": "bearer", "bearerFormat": "JWT",
            "description": "The KA HS256 token issued after SAML login.",
        }
        for path, operations in schema["paths"].items():
            for method, operation in operations.items():
                if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
                    if requires_api_token(method, path):
                        operation["security"] = [{"KABearer": []}]
        return schema

    app.openapi = authenticated_openapi
