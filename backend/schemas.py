from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


class AccessCodeSubmission(BaseModel):
    access_code: str = Field(min_length=1, max_length=256)


class StartSessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class RestartAssessmentRequest(BaseModel):
    participant_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )
    previous_assessment_id: str = Field(
        min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    )


class SurveySubmission(StartSessionRequest):
    answers: dict[str, str | list[str]]


class EntryRiskShareSubmission(BaseModel):
    risk_share: float = Field(ge=0.0, le=1.0)

    @field_validator("risk_share")
    @classmethod
    def validate_five_percent_increment(cls, value: float) -> float:
        units = value / 0.05
        if not math.isclose(units, round(units), abs_tol=1e-8):
            raise ValueError("risk_share must use 0.05 increments")
        return round(value, 2)


class DecisionSubmission(BaseModel):
    scenario_id: str = Field(
        pattern=r"^E[12346]_0[1-3]$"
    )
    decision_point: str = Field(pattern=r"^E[12346]_DP[1-7]$")
    day: int = Field(ge=1, le=60)
    risk_share_after: float = Field(ge=0.0, le=1.0)
    decision_time_ms: int | None = Field(default=None, ge=0, le=86_400_000)

    @field_validator("risk_share_after")
    @classmethod
    def validate_five_percent_increment(cls, value: float) -> float:
        units = value / 0.05
        if not math.isclose(units, round(units), abs_tol=1e-8):
            raise ValueError("risk_share_after must use 0.05 increments")
        return round(value, 2)


class _Episode5DecisionSubmission(BaseModel):
    scenario_id: str = Field(pattern=r"^E5_0[1-3]$")
    decision_point: str = Field(pattern=r"^E5_DP[1-3]$")
    day: int = Field(ge=1, le=60)
    decision_time_ms: int | None = Field(default=None, ge=0, le=86_400_000)

    @staticmethod
    def _validate_share(value: float) -> float:
        units = value / 0.05
        if not math.isclose(units, round(units), abs_tol=1e-8):
            raise ValueError("risk share must use 0.05 increments")
        return round(value, 2)


class Episode5PreSubmission(_Episode5DecisionSubmission):
    risk_share_pre_info: float = Field(ge=0.0, le=1.0)

    @field_validator("risk_share_pre_info")
    @classmethod
    def validate_five_percent_increment(cls, value: float) -> float:
        return cls._validate_share(value)


class Episode5PostSubmission(_Episode5DecisionSubmission):
    risk_share_post_info: float = Field(ge=0.0, le=1.0)

    @field_validator("risk_share_post_info")
    @classmethod
    def validate_five_percent_increment(cls, value: float) -> float:
        return cls._validate_share(value)
