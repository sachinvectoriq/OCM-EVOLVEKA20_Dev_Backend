"""Offline security boundary tests against the actual registered OCM routes.

Run from the repo root: python -B -m unittest discover -s backend/tests -v
No real SAML login, Cosmos writes, or Azure API calls are performed.
"""

import asyncio
from copy import deepcopy
import json
import logging
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.responses import RedirectResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import main
from app.api.routes import auth as auth_routes
from app.api.routes import auth as auth_routes
from app.api.dependencies import get_chat_service, get_logger, get_settings
from app.api.request_audit import APIRequestAuditStore
from app.api.security import KATokenValidator, requires_api_token
from app.core.settings import (
    AIServicesSettings, AzureOpenAISettings, SearchServiceSettings, Settings,
)
from app.utils import saml


SECRET = "offline-test-key-" * 4
logging.getLogger("httpx").setLevel(logging.WARNING)


def token(**overrides):
    payload = {
        "exp": int(time.time()) + 3600,
        "user_data": {
            "name": ["Example Person"],
            "email": ["Example@company.test"],
            "group": "user",
            "job_title": None,
        },
    }
    payload.update(overrides)
    return jwt.encode(payload, SECRET, algorithm="HS256")


def make_test_settings():
    return Settings(
        _env_file=None,
        search_service=SearchServiceSettings(
            _env_file=None, endpoint="https://search.invalid", api_key="test"
        ),
        ai_services=AIServicesSettings(
            _env_file=None, cognitive_services_endpoint="https://ai.invalid"
        ),
        azure_openai=AzureOpenAISettings(_env_file=None, endpoint="https://llm.invalid"),
    )


class RecordingStore:
    def __init__(self):
        self.records = []

    async def write(self, document):
        self.records.append(deepcopy(document))


class APIBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.app = main.create_app()
        self.store = RecordingStore()
        self.app.state.api_token_validator = KATokenValidator(SECRET)
        self.app.state.api_audit_store = self.store
        self.app.dependency_overrides[get_settings] = make_test_settings
        self.app.dependency_overrides[get_logger] = lambda: MagicMock()
        self.chat_service = AsyncMock()
        self.chat_service.list_conversations.return_value = []
        self.app.dependency_overrides[get_chat_service] = lambda: self.chat_service

        @self.app.post("/api/test/{resource_id}")
        async def probe(request: Request, resource_id: str):
            await request.json()
            return {"user_id": request.state.authenticated_user.user_id}

        @self.app.get("/api/failure/{status}")
        async def failure(status: int):
            if status == 500:
                raise RuntimeError("offline handler failure")
            raise HTTPException(status_code=status)

        # No lifespan: initialize security state above and never connect to Azure.
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def tearDown(self):
        self.client.close()

    def headers(self):
        return {"Authorization": "Bearer " + token()}

    def test_every_real_protected_route_rejects_and_audits_missing_token(self):
        count = 0
        for route in self.app.routes:
            if not isinstance(route, APIRoute) or route.path.startswith(("/api/test/", "/api/failure/")):
                continue
            for method in route.methods:
                if not requires_api_token(method, route.path):
                    continue
                with self.subTest(method=method, path=route.path):
                    self.store.records.clear()
                    path = route.path.replace("{user_id}", "untrusted-user").replace("{session_id}", "untrusted-session")
                    response = self.client.request(method, path)
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.headers["www-authenticate"], "Bearer")
                    self.assertEqual(len(self.store.records), 1)
                    record = self.store.records[0]
                    self.assertEqual(record["route"], route.path)
                    self.assertEqual(record["auth_result"], "missing_token")
                    self.assertIsNone(record["user_id"])
                    self.assertEqual(record["status_code"], 401)
                    count += 1
        # Explicit minimum protects against accidentally dropping a business router.
        self.assertGreaterEqual(count, 20)
        self.chat_service.assert_not_called()

    def test_valid_token_logs_verified_identity_not_request_identity_or_secrets(self):
        response = self.client.post(
            "/api/test/sensitive-resource?token=query-secret",
            headers={**self.headers(), "X-Request-ID": "client-spoof", "X-User-ID": "victim"},
            json={"user_id": "victim", "token": "body-secret", "query": "private question"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "example@company.test")
        self.assertEqual(len(self.store.records), 1)
        record = self.store.records[0]
        self.assertEqual(record["user_id"], "example@company.test")
        self.assertEqual(record["user_email"], "Example@company.test")
        self.assertEqual(record["user_name"], "Example Person")
        self.assertEqual(record["user_group"], "user")
        self.assertIsNone(record["job_title"])
        self.assertEqual(record["route"], "/api/test/{resource_id}")
        self.assertEqual(record["request_id"], response.headers["x-request-id"])
        self.assertNotEqual(record["request_id"], "client-spoof")
        self.assertGreaterEqual(record["duration_ms"], 0)
        self.assertEqual(set(record), {
            "id", "record_type", "timestamp_utc", "user_id", "user_email", "user_name",
            "user_group", "job_title", "method", "route", "status_code", "duration_ms",
            "request_id", "auth_result",
        })
        serialized = json.dumps(record)
        for forbidden in ["victim", "sensitive-resource", "query-secret", "body-secret", "private question", token()]:
            self.assertNotIn(forbidden, serialized)

    def test_invalid_tokens_never_reach_handler_or_attribute_identity(self):
        valid_payload = jwt.decode(token(), SECRET, algorithms=["HS256"])
        without_exp = {"user_data": valid_payload["user_data"]}
        cases = {
            "expired": token(exp=int(time.time()) - 1),
            "wrong_key": jwt.encode(valid_payload, "wrong-key-" * 8, algorithm="HS256"),
            "wrong_algorithm": jwt.encode(valid_payload, SECRET, algorithm="HS512"),
            "unsigned": jwt.encode(valid_payload, "", algorithm="none"),
            "malformed": "not.a.jwt",
            "missing_exp": jwt.encode(without_exp, SECRET, algorithm="HS256"),
            "missing_identity": token(user_data={"name": "Someone"}),
            "bad_identity_type": token(user_data=["Someone"]),
            "null_email": token(user_data={"email": [None]}),
            "not_yet_valid": token(nbf=int(time.time()) + 3600),
            "bad_exp": token(exp="not-a-date"),
            "infinite_exp": token(exp=float("inf")),
        }
        for name, encoded in cases.items():
            with self.subTest(name=name):
                self.store.records.clear()
                response = self.client.post("/api/test/1", headers={"Authorization": "Bearer " + encoded}, json={})
                self.assertEqual(response.status_code, 401)
                self.assertEqual(len(self.store.records), 1)
                record = self.store.records[0]
                self.assertIsNone(record["user_id"])
                self.assertIsNone(record["user_email"])
                self.assertEqual(record["auth_result"], "expired_token" if name == "expired" else "invalid_token")

    def test_invalid_headers_query_token_and_spoofed_internal_headers_rejected(self):
        cases = [
            {"Authorization": "Basic " + token()},
            {"Authorization": "Bearer"},
            {"Authorization": "Bearer one two"},
            {"Authorization": "Bearer " + "a" * 16384},
            {"X-Internal-Request": "true", "Origin": "https://trusted.invalid"},
            [("Authorization", "Bearer " + token()), ("Authorization", "Bearer " + token())],
        ]
        for headers in cases:
            response = self.client.post("/api/test/1", headers=headers, json={})
            self.assertEqual(response.status_code, 401)
        self.assertEqual(self.client.post("/api/test/1?token=" + token(), json={}).status_code, 401)

    def test_current_saml_issuer_token_is_accepted_without_claim_changes(self):
        with patch.object(saml, "JWT_SECRET_KEY", SECRET):
            encoded = saml.create_jwt_token({"name": ["Existing User"], "email": ["person@company.test"], "group": "user", "job_title": None})
        response = self.client.post("/api/test/1", headers={"Authorization": "Bearer " + encoded}, json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.records[-1]["user_id"], "person@company.test")

    def test_refresh_renews_only_verified_claims_and_audits_without_token_contents(self):
        old_token = token(exp=int(time.time()) + 45)
        before = jwt.decode(old_token, SECRET, algorithms=["HS256"])
        with patch.object(saml, "JWT_SECRET_KEY", SECRET):
            response = self.client.post(
                "/api/auth/refresh",
                headers={"Authorization": "Bearer " + old_token},
                json={"user_data": {"email": ["victim@company.test"], "group": "admin"}, "expires_in": 999999},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["pragma"], "no-cache")
        refreshed = response.json()["access_token"]
        after = jwt.decode(refreshed, SECRET, algorithms=["HS256"])
        self.assertEqual(after["user_data"], before["user_data"])
        self.assertGreater(after["exp"], before["exp"])
        self.assertLessEqual(abs(after["exp"] - int(time.time()) - 3600), 2)
        self.assertEqual(response.json()["expires_in"], 3600)
        self.assertEqual(response.json()["token_type"], "Bearer")
        self.assertEqual(len(self.store.records), 1)
        self.assertEqual(self.store.records[0]["route"], "/api/auth/refresh")
        self.assertEqual(self.store.records[0]["user_id"], "example@company.test")
        self.assertNotIn(old_token, json.dumps(self.store.records))
        self.assertNotIn(refreshed, json.dumps(self.store.records))
        self.assertEqual(self.client.get(
            "/api/chat/conversations/browser-user-id", headers={"Authorization": "Bearer " + refreshed}
        ).status_code, 200)
        # The replacement can itself be renewed; no absolute session cap is added.
        with patch.object(saml, "JWT_SECRET_KEY", SECRET):
            self.assertEqual(self.client.post(
                "/api/auth/refresh", headers={"Authorization": "Bearer " + refreshed}
            ).status_code, 200)

    def test_refresh_rejects_expired_unsigned_and_missing_tokens_before_issuance(self):
        with patch.object(auth_routes, "create_jwt_token") as issuer:
            for headers in [{}, {"Authorization": "Bearer " + token(exp=int(time.time()) - 5)}, {"Authorization": "Bearer invalid"}]:
                self.store.records.clear()
                response = self.client.post("/api/auth/refresh", headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(len(self.store.records), 1)
                self.assertIsNone(self.store.records[0]["user_id"])
            issuer.assert_not_called()

    def test_refresh_renews_only_verified_claims_and_audits_without_token_contents(self):
        old_token = token(exp=int(time.time()) + 45)
        before = jwt.decode(old_token, SECRET, algorithms=["HS256"])
        with patch.object(saml, "JWT_SECRET_KEY", SECRET):
            response = self.client.post(
                "/api/auth/refresh",
                headers={"Authorization": "Bearer " + old_token},
                json={"user_data": {"email": ["victim@company.test"], "group": "admin"}, "expires_in": 999999},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["pragma"], "no-cache")
        refreshed = response.json()["access_token"]
        after = jwt.decode(refreshed, SECRET, algorithms=["HS256"])
        self.assertEqual(after["user_data"], before["user_data"])
        self.assertGreater(after["exp"], before["exp"])
        self.assertLessEqual(abs(after["exp"] - int(time.time()) - 3600), 2)
        self.assertEqual(response.json()["expires_in"], 3600)
        self.assertEqual(response.json()["token_type"], "Bearer")
        self.assertEqual(len(self.store.records), 1)
        self.assertEqual(self.store.records[0]["route"], "/api/auth/refresh")
        self.assertEqual(self.store.records[0]["user_id"], "example@company.test")
        self.assertNotIn(old_token, json.dumps(self.store.records))
        self.assertNotIn(refreshed, json.dumps(self.store.records))
        self.assertEqual(self.client.get(
            "/api/chat/conversations/browser-user-id", headers={"Authorization": "Bearer " + refreshed}
        ).status_code, 200)
        # The replacement can itself be renewed; no absolute session cap is added.
        with patch.object(saml, "JWT_SECRET_KEY", SECRET):
            self.assertEqual(self.client.post(
                "/api/auth/refresh", headers={"Authorization": "Bearer " + refreshed}
            ).status_code, 200)

    def test_refresh_rejects_expired_unsigned_and_missing_tokens_before_issuance(self):
        with patch.object(auth_routes, "create_jwt_token") as issuer:
            for headers in [{}, {"Authorization": "Bearer " + token(exp=int(time.time()) - 5)}, {"Authorization": "Bearer invalid"}]:
                self.store.records.clear()
                response = self.client.post("/api/auth/refresh", headers=headers)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(len(self.store.records), 1)
                self.assertIsNone(self.store.records[0]["user_id"])
            issuer.assert_not_called()

    def test_authenticated_real_chat_route_executes_and_records(self):
        response = self.client.get("/api/chat/conversations/browser-user-id", headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.chat_service.list_conversations.assert_awaited_once_with("browser-user-id", 100)
        self.assertEqual(self.store.records[-1]["user_id"], "example@company.test")

    def test_saml_and_search_callbacks_and_health_keep_working(self):
        with patch.object(saml, "saml_login", AsyncMock(return_value=RedirectResponse("https://login.invalid"))), patch.object(
            saml, "saml_callback", AsyncMock(return_value=RedirectResponse("https://ui.invalid/?token=new"))
        ):
            self.assertEqual(self.client.get("/saml/login", follow_redirects=False).status_code, 307)
            self.assertEqual(self.client.post("/saml/callback", data={"SAMLResponse": "test"}, follow_redirects=False).status_code, 307)
        with patch.object(saml, "JWT_SECRET_KEY", SECRET):
            self.assertEqual(self.client.get("/saml/token/extract", params={"token": token()}).status_code, 200)
        for path in ["/api/enrich", "/api/split"]:
            response = self.client.post(path, json={"values": [{"recordId": "1", "data": {"opco": "A,B", "persona": "C"}}]})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["values"][0]["data"]["opco_values"], ["A", "B"])
            self.assertEqual(self.client.post(path + "/", json={"values": []}).status_code, 200)
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        self.assertEqual(self.store.records, [])

    def test_exemptions_are_method_and_path_specific(self):
        for method, path in [("POST", "/api/health"), ("GET", "/api/split"), ("POST", "/api/enrich/other"), ("GET", "/api/health/private"), ("GET", "/api/new-business-route")]:
            self.assertEqual(self.client.request(method, path).status_code, 401)

    def test_cors_preflight_and_errors(self):
        response = self.client.options("/api/chat/query", headers={
            "Origin": "https://ui.invalid", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.records, [])
        response = self.client.post("/api/chat/query", headers={"Origin": "https://ui.invalid"})
        self.assertEqual(response.status_code, 401)
        self.assertIn("access-control-allow-origin", response.headers)

    def test_all_http_outcomes_and_validation_errors_logged_once(self):
        for status in [403, 404, 500]:
            self.store.records.clear()
            response = self.client.get(f"/api/failure/{status}", headers=self.headers())
            self.assertEqual(response.status_code, status)
            self.assertEqual(len(self.store.records), 1)
            self.assertEqual(self.store.records[0]["status_code"], status)
            self.assertEqual(self.store.records[0]["auth_result"], "authenticated")
        self.store.records.clear()
        response = self.client.post("/api/chat/query", headers=self.headers(), json={})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(self.store.records), 1)
        self.assertEqual(self.store.records[0]["status_code"], 422)

    def test_openapi_describes_security_on_all_protected_operations(self):
        schema = self.app.openapi()
        for path, operations in schema["paths"].items():
            for method, operation in operations.items():
                if requires_api_token(method, path):
                    self.assertEqual(operation["security"], [{"KABearer": []}])
        self.assertNotIn("security", schema["paths"]["/api/split"]["post"])
        self.assertNotIn("security", schema["paths"]["/saml/login"]["get"])

    def test_no_secret_fails_closed(self):
        for secret in [None, "", " "]:
            with self.assertRaises(ValueError):
                KATokenValidator(secret)

    def test_cosmos_failure_does_not_replace_success_or_auth_failure_response(self):
        cosmos_container = MagicMock()
        cosmos_container.create_item = AsyncMock(side_effect=RuntimeError("offline database"))
        self.app.state.api_audit_store = APIRequestAuditStore(cosmos_container, 0.1)
        with self.assertLogs("knowledge-assistant.api-audit", level="ERROR") as captured:
            self.assertEqual(self.client.post("/api/test/1", headers=self.headers(), json={}).status_code, 200)
            self.assertEqual(self.client.post("/api/test/1", json={}).status_code, 401)
        self.assertEqual(len(captured.output), 2)


class CosmosAuditTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = MagicMock()
        self.database = self.client.get_database_client.return_value
        self.container = self.database.get_container_client.return_value
        self.container.read = AsyncMock(return_value={"partitionKey": {"paths": ["/user_id"]}})
        self.container.create_item = AsyncMock()
        self.client.close = AsyncMock()

    async def test_existing_container_is_validated_then_receives_document(self):
        store = await APIRequestAuditStore.connect(self.client, "agentic_rag", "api_request_audit")
        self.client.get_database_client.assert_called_once_with("agentic_rag")
        self.database.get_container_client.assert_called_once_with("api_request_audit")
        self.client.create_database_if_not_exists.assert_not_called()
        document = {"id": "1", "user_id": None, "auth_result": "missing_token"}
        await store.write(document)
        self.container.create_item.assert_awaited_once_with(body=document)

    async def test_wrong_partition_key_is_rejected_at_startup(self):
        self.container.read.return_value = {"partitionKey": {"paths": ["/user_id", "/session_id"]}}
        with self.assertRaisesRegex(ValueError, "partition key /user_id"):
            await APIRequestAuditStore.connect(self.client, "agentic_rag", "wrong-container")

    async def test_optional_creation_uses_separate_container_and_no_retention_expiry(self):
        self.client.create_database_if_not_exists = AsyncMock(return_value=self.database)
        self.database.create_container_if_not_exists = AsyncMock(return_value=self.container)
        await APIRequestAuditStore.connect(self.client, "agentic_rag", "api_request_audit", auto_create=True)
        kwargs = self.database.create_container_if_not_exists.call_args.kwargs
        self.assertEqual(kwargs["id"], "api_request_audit")
        self.assertEqual(kwargs["partition_key"]["paths"], ["/user_id"])
        self.assertEqual(kwargs["default_ttl"], -1)

    async def test_write_failure_keeps_sanitized_record_without_exception_contents(self):
        self.container.create_item.side_effect = RuntimeError("private-sdk-request-token")
        store = APIRequestAuditStore(self.container, 1)
        with self.assertLogs("knowledge-assistant.api-audit", level="ERROR") as captured:
            await store.write({"id": "test-record", "user_id": None})
        self.assertIn("API_AUDIT_WRITE_FAILED", captured.output[0])
        self.assertIn("test-record", captured.output[0])
        self.assertNotIn("private-sdk-request-token", captured.output[0])

    async def test_write_timeout_does_not_block_the_api_indefinitely(self):
        async def slow_write(**kwargs):
            await asyncio.sleep(1)
        self.container.create_item.side_effect = slow_write
        store = APIRequestAuditStore(self.container, 0.01)
        with self.assertLogs("knowledge-assistant.api-audit", level="ERROR") as captured:
            await store.write({"id": "timed-out-record", "user_id": None})
        self.assertIn("TimeoutError", captured.output[0])

    async def test_lifespan_initializes_security_and_closes_its_client(self):
        app = main.create_app()
        with main.container.config.override(make_test_settings()), main.container.cosmos_client.override(self.client), patch.object(
            saml, "JWT_SECRET_KEY", SECRET
        ), patch.object(main, "_ensure_cosmos_resources", AsyncMock()), patch.object(main, "_configure_foundry_telemetry", AsyncMock()):
            async with main.lifespan(app):
                self.assertIsInstance(app.state.api_token_validator, KATokenValidator)
                self.assertIsInstance(app.state.api_audit_store, APIRequestAuditStore)
            self.client.close.assert_awaited_once()

    async def test_startup_fails_when_audit_container_is_unavailable_and_closes_client(self):
        self.container.read.side_effect = RuntimeError("offline database")
        with main.container.config.override(make_test_settings()), main.container.cosmos_client.override(self.client), patch.object(saml, "JWT_SECRET_KEY", SECRET), self.assertLogs("knowledge-assistant", level="ERROR"):
            with self.assertRaises(RuntimeError):
                async with main.lifespan(main.create_app()):
                    self.fail("Server must not start without configured logging")
        self.client.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
