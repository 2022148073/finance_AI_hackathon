from __future__ import annotations

import re
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from access_gate import ACCESS_COOKIE_NAME, hash_access_code  # noqa: E402
from database import connect  # noqa: E402
from main import create_app  # noqa: E402


class AccessGateApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "access-test.db"
        self.app = create_app(
            database_path=self.database_path,
            scenario_dir=BACKEND_DIR / "scenarios",
            access_code="judge-invitation",
            access_cookie_secure=False,
            access_rate_limit_attempts=3,
            access_rate_limit_window_seconds=300,
        )
        self.context = TestClient(self.app)
        self.client = self.context.__enter__()

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_protected_apis_reject_missing_access_session(self) -> None:
        protected_requests = (
            ("POST", "/api/survey/sessions"),
            ("POST", "/api/survey/submissions"),
            ("POST", "/api/episode1/sessions"),
            ("GET", "/api/episode1/sessions/example"),
            ("POST", "/api/episode3/sessions/example/entry"),
            ("POST", "/api/episode5/sessions/example/pre-decisions"),
            ("POST", "/api/episode5/sessions/example/post-decisions"),
            ("POST", "/api/episode6/sessions/example/decisions"),
            ("POST", "/api/analysis/runs"),
            ("GET", "/api/analysis/runs/example"),
            ("POST", "/api/assessment-attempts"),
        )
        for method, path in protected_requests:
            response = self.client.request(method, path, json={})
            self.assertEqual(response.status_code, 401, (method, path, response.text))
        self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_valid_code_issues_httponly_session_and_unlocks_survey(self) -> None:
        wrong = self.client.post(
            "/api/access/verify", json={"access_code": "wrong"}
        )
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json(), {"detail": "접근 코드를 확인해 주세요."})
        self.assertNotIn("judge-invitation", wrong.text)

        accepted = self.client.post(
            "/api/access/verify", json={"access_code": "judge-invitation"}
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.json(), {"success": True})
        cookie = accepted.headers["set-cookie"]
        self.assertIn(f"{ACCESS_COOKIE_NAME}=", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=lax", cookie)
        self.assertIn("Max-Age=", cookie)
        self.assertNotIn("judge-invitation", cookie)
        self.assertEqual(
            self.client.get("/api/access/session").json(),
            {"authenticated": True},
        )
        survey = self.client.post(
            "/api/survey/sessions", json={"user_id": "access_user"}
        )
        self.assertEqual(survey.status_code, 200, survey.text)

        with closing(connect(self.database_path)) as connection:
            stored = connection.execute(
                "SELECT session_token_hash FROM access_sessions"
            ).fetchone()
        self.assertIsNotNone(stored)
        self.assertRegex(str(stored["session_token_hash"]), r"^[0-9a-f]{64}$")
        self.assertNotIn("judge-invitation", str(stored["session_token_hash"]))

    def test_rate_limit_blocks_even_a_later_correct_code(self) -> None:
        limited_app = create_app(
            database_path=Path(self.temporary.name) / "limited.db",
            scenario_dir=BACKEND_DIR / "scenarios",
            access_code="correct-code",
            access_cookie_secure=False,
            access_rate_limit_attempts=2,
            access_rate_limit_window_seconds=300,
        )
        with TestClient(limited_app) as client:
            for _ in range(2):
                self.assertEqual(
                    client.post(
                        "/api/access/verify", json={"access_code": "wrong"}
                    ).status_code,
                    401,
                )
            limited = client.post(
                "/api/access/verify", json={"access_code": "correct-code"}
            )
        self.assertEqual(limited.status_code, 429)
        self.assertNotIn("correct-code", limited.text)

    def test_pbkdf2_hash_and_production_cookie_are_supported(self) -> None:
        encoded = hash_access_code(
            "hashed-invitation", iterations=100_000, salt=b"0123456789abcdef"
        )
        secure_app = create_app(
            database_path=Path(self.temporary.name) / "secure.db",
            scenario_dir=BACKEND_DIR / "scenarios",
            access_code_hash=encoded,
            access_cookie_secure=True,
        )
        with TestClient(secure_app, base_url="https://testserver") as client:
            response = client.post(
                "/api/access/verify", json={"access_code": "hashed-invitation"}
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Secure", response.headers["set-cookie"])


class AccessGateFrontendContractTests(unittest.TestCase):
    def test_frontend_uses_gate_and_credentialed_requests_without_secret_storage(self) -> None:
        app_source = (WEB_DIR / "frontend" / "src" / "App.jsx").read_text(
            encoding="utf-8"
        )
        gate_source = (WEB_DIR / "frontend" / "src" / "AccessGate.jsx").read_text(
            encoding="utf-8"
        )
        combined = app_source + gate_source
        self.assertIn("대회 심사용 데모 서비스입니다.", gate_source)
        self.assertIn("안내받은 접근 코드를 입력해 주세요.", gate_source)
        self.assertIn("입장하기", gate_source)
        self.assertIn('credentials: "include"', combined)
        self.assertIn("/api/access/session", app_source)
        self.assertIn("/api/access/verify", gate_source)
        self.assertIsNone(
            re.search(r"localStorage\.(?:setItem|getItem)\([^\n]*access", combined)
        )


if __name__ == "__main__":
    unittest.main()
