"""Shared fixtures used only by backend tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


VALID_SURVEY_ANSWERS: dict[str, str | list[str]] = {
    "age_group": "age_41_50",
    "investment_horizon": "years_1_2",
    "investment_experience": ["neutral"],
    "derivative_experience": "years_1_3",
    "loss_tolerance": "partial_principal_loss",
    "investment_asset_ratio": "up_to_50",
    "monthly_income": "up_to_3m",
    "investment_purpose": "above_market_return",
    "financial_knowledge": "deep",
    "vulnerability": "no",
}

TEST_ACCESS_CODE = "flowbit-test-access"


def authorize_access(client: TestClient) -> None:
    response = client.post(
        "/api/access/verify", json={"access_code": TEST_ACCESS_CODE}
    )
    if response.status_code != 200:
        raise AssertionError(response.text)


def complete_survey(client: TestClient, user_id: str) -> None:
    state = client.post("/api/survey/sessions", json={"user_id": user_id})
    if state.status_code != 200:
        raise AssertionError(state.text)
    if state.json()["survey_completed"]:
        return
    response = client.post(
        "/api/survey/submissions",
        json={"user_id": user_id, "answers": VALID_SURVEY_ANSWERS},
    )
    if response.status_code != 200:
        raise AssertionError(response.text)
