"""KIS-based stated-preference questionnaire and independent scoring.

Question wording and order follow the Korea Investment & Securities investor
information form revised 2026-01-02 and its current online registration page:
https://file.truefriend.com/Storage/formDown/02_02.pdf
https://www.truefriend.com/main/mall/openptrade/TrustTrade01.jsp?cmd=TF02ff010001_Invest1

KIS does not publish its current internal item weights in those public sources.
The score below is therefore explicitly *not* a KIS official score.  It uses a
transparent ordinal sum/max conversion and the five bands requested for this
research baseline, consistent in shape with KOFIA's standard-investor-scoring
approach.  Current KOFIA standard rules (revised 2026-04-09):
https://law.kofia.or.kr/service/law/detailArticlePrint.do?contentSeq=305485&historySeq=1787&seq=149
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


QUESTIONNAIRE_VERSION = "kis_2026_01_02_v1"
STATED_FEATURE_VERSION = "stated_v1"
SCORING_VERSION = "kofia_standard_style_v1"
SCORING_BASIS = "KOFIA-standard-style scoring (not KIS official internal scoring)"

SOURCE_METADATA = {
    "provider": "Korea Investment & Securities",
    "document_name": "투자자정보 확인서 (영업 0125호)",
    "document_revision_date": "2026-01-02",
    "source_url": "https://file.truefriend.com/Storage/formDown/02_02.pdf",
    "online_source_url": (
        "https://www.truefriend.com/main/mall/openptrade/"
        "TrustTrade01.jsp?cmd=TF02ff010001_Invest1"
    ),
    "scoring_reference": "금융투자협회 표준투자권유준칙",
    "scoring_reference_revision_date": "2026-04-09",
    "scoring_reference_url": (
        "https://law.kofia.or.kr/service/law/detailArticlePrint.do?"
        "contentSeq=305485&historySeq=1787&seq=149"
    ),
    "scoring_disclaimer": (
        "한국투자증권의 현행 비공개 내부 배점이 아닌 연구용 baseline 점수"
    ),
}


QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "number": 1,
        "id": "age_group",
        "type": "single",
        "prompt": "고객님의 연령대를 선택해 주세요.",
        "options": (
            {"id": "under_20", "label": "만 19세 이하"},
            {"id": "age_20_40", "label": "만 20세~40세"},
            {"id": "age_41_50", "label": "만 41세~50세"},
            {"id": "age_51_60", "label": "만 51세~60세"},
            {"id": "age_61_plus", "label": "만 61세 이상"},
        ),
    },
    {
        "number": 2,
        "id": "investment_horizon",
        "type": "single",
        "prompt": "투자 예정 기간은 어느 정도입니까?",
        "options": (
            {"id": "years_3_plus", "label": "3년 이상"},
            {"id": "years_2_3", "label": "2년 이상~3년 미만"},
            {"id": "years_1_2", "label": "1년 이상~2년 미만"},
            {"id": "months_6_12", "label": "6개월 이상~1년 미만"},
            {"id": "under_6_months", "label": "6개월 미만"},
        ),
    },
    {
        "number": 3,
        "id": "investment_experience",
        "type": "multiple",
        "prompt": "취득 또는 처분해 본 경험이 있는 상품을 모두 선택해 주세요.",
        "help_text": "복수 선택 가능",
        "options": (
            {
                "id": "aggressive",
                "label": "공격투자형 상품",
                "description": "파생펀드/ELF, 해외주식, 선물옵션, ELW, ETN, 신용거래 등",
            },
            {
                "id": "active",
                "label": "적극투자형 상품",
                "description": "주식형·주식혼합형 펀드, 원금비보장형 ELS, 주식, 신용등급이 낮은 채권 등",
            },
            {
                "id": "neutral",
                "label": "위험중립형 상품",
                "description": "채권혼합형펀드, 원금부분지급형 ELS, 신용등급이 중간인 채권 등",
            },
            {
                "id": "conservative_plus",
                "label": "안정추구형 상품",
                "description": "채권형펀드, 원금지급형 ELB/DLB, 신용등급이 높은 채권 등",
            },
            {
                "id": "conservative",
                "label": "안정형 상품",
                "description": "은행 예·적금, MMF, CMA, RP, 국고채, 통안채, 정부보증채, 특수채, 지방채 등",
            },
        ),
    },
    {
        "number": 4,
        "id": "derivative_experience",
        "type": "single",
        "prompt": "파생상품 등에 투자한 경험은 어느 정도입니까?",
        "help_text": "선물, 옵션, ELW, ETN, 해외선물, FX마진 등",
        "options": (
            {"id": "none_or_under_1_year", "label": "1년 미만 또는 투자경험 없음"},
            {"id": "years_1_3", "label": "1년 이상~3년 미만"},
            {"id": "years_3_plus", "label": "3년 이상"},
        ),
    },
    {
        "number": 5,
        "id": "loss_tolerance",
        "type": "single",
        "prompt": "감내할 수 있는 손실 수준은 어느 정도입니까?",
        "options": (
            {"id": "high_risk_for_return", "label": "기대수익이 높다면 위험이 높아도 상관하지 않음"},
            {"id": "partial_principal_loss", "label": "투자원금 중 일부의 손실을 감수할 수 있음"},
            {"id": "minimal_loss_only", "label": "투자원금에서 최소한의 손실만을 감수할 수 있음"},
            {"id": "principal_preservation", "label": "무슨 일이 있어도 투자원금은 보전되어야 함"},
        ),
    },
    {
        "number": 6,
        "id": "investment_asset_ratio",
        "type": "single",
        "prompt": "총자산 대비 투자성자산의 비중은 어느 정도입니까?",
        "options": tuple(
            {"id": option_id, "label": label}
            for option_id, label in (
                ("up_to_10", "10% 이하"),
                ("up_to_30", "30% 이하"),
                ("up_to_50", "50% 이하"),
                ("up_to_70", "70% 이하"),
                ("over_70", "70% 초과"),
            )
        ),
    },
    {
        "number": 7,
        "id": "monthly_income",
        "type": "single",
        "prompt": "월 소득 현황은 어떻게 됩니까?",
        "options": tuple(
            {"id": option_id, "label": label}
            for option_id, label in (
                ("over_5m", "500만원 초과"),
                ("up_to_5m", "500만원 이하"),
                ("up_to_3m", "300만원 이하"),
                ("up_to_2m", "200만원 이하"),
                ("up_to_1m", "100만원 이하"),
            )
        ),
    },
    {
        "number": 8,
        "id": "investment_purpose",
        "type": "single",
        "prompt": "투자 목적(거래 목적)은 무엇입니까?",
        "options": (
            {"id": "living_or_short_term", "label": "생계(단기)자금 운용"},
            {"id": "deposit_level_return", "label": "예·적금 수준 수익률 기대"},
            {"id": "above_market_return", "label": "시장평균 이상 수익률 기대"},
            {"id": "active_wealth_growth", "label": "적극적인 재산(자산) 증식"},
        ),
    },
    {
        "number": 9,
        "id": "financial_knowledge",
        "type": "single",
        "prompt": "금융지식 수준과 금융상품 이해도는 어느 정도입니까?",
        "options": (
            {"id": "none", "label": "금융투자상품에 투자해 본 경험이 없음"},
            {"id": "partial", "label": "주식·채권·펀드 등의 구조와 위험을 일정 부분 이해하고 있음"},
            {"id": "deep", "label": "주식·채권·펀드 등의 구조와 위험을 깊이 있게 이해하고 있음"},
            {"id": "including_derivatives", "label": "파생상품을 포함한 대부분의 금융투자상품 구조와 위험을 이해하고 있음"},
        ),
    },
    {
        "number": 10,
        "id": "vulnerability",
        "type": "single",
        "prompt": "취약투자자에 해당합니까?",
        "help_text": "예: 고령투자자, 미성년자, 금융투자상품 투자 무경험자, 은퇴자, 주부, 문맹자 등",
        "options": (
            {"id": "yes", "label": "해당"},
            {"id": "no", "label": "미해당"},
        ),
    },
)


QUESTION_BY_ID = {question["id"]: question for question in QUESTIONS}


class SurveyValidationError(ValueError):
    """Raised when raw answers do not match the configured questionnaire."""


@dataclass(frozen=True)
class StatedFeatures:
    risk_capacity_age: float
    investment_horizon: float
    risky_asset_experience: float
    experience_breadth: float
    derivative_experience: float
    stated_loss_tolerance: float
    stated_risk_tolerance: float
    investment_exposure: float
    financial_capacity: float
    return_seeking: float
    financial_literacy: float
    vulnerability_flag: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


AGE_VALUES = {
    "under_20": 1.0,
    "age_20_40": 1.0,
    "age_41_50": 0.75,
    "age_51_60": 0.5,
    "age_61_plus": 0.25,
}
HORIZON_VALUES = {
    "under_6_months": 0.0,
    "months_6_12": 0.25,
    "years_1_2": 0.5,
    "years_2_3": 0.75,
    "years_3_plus": 1.0,
}
EXPERIENCE_VALUES = {
    "conservative": 0.0,
    "conservative_plus": 0.25,
    "neutral": 0.5,
    "active": 0.75,
    "aggressive": 1.0,
}
DERIVATIVE_VALUES = {
    "none_or_under_1_year": 0.0,
    "years_1_3": 0.5,
    "years_3_plus": 1.0,
}
LOSS_VALUES = {
    "principal_preservation": 0.0,
    "minimal_loss_only": 0.33,
    "partial_principal_loss": 0.67,
    "high_risk_for_return": 1.0,
}
EXPOSURE_VALUES = {
    "up_to_10": 0.0,
    "up_to_30": 0.25,
    "up_to_50": 0.5,
    "up_to_70": 0.75,
    "over_70": 1.0,
}
INCOME_VALUES = {
    "up_to_1m": 0.0,
    "up_to_2m": 0.25,
    "up_to_3m": 0.5,
    "up_to_5m": 0.75,
    "over_5m": 1.0,
}
PURPOSE_VALUES = {
    "living_or_short_term": 0.0,
    "deposit_level_return": 0.33,
    "above_market_return": 0.67,
    "active_wealth_growth": 1.0,
}
KNOWLEDGE_VALUES = {
    "none": 0.0,
    "partial": 0.33,
    "deep": 0.67,
    "including_derivatives": 1.0,
}


def public_questionnaire() -> dict[str, Any]:
    """Return render-only configuration with no scoring or feature metadata."""
    return {
        "questionnaire_version": QUESTIONNAIRE_VERSION,
        "title": "투자 성향 설문",
        "questions": [
            {
                key: ([dict(option) for option in value] if key == "options" else value)
                for key, value in question.items()
            }
            for question in QUESTIONS
        ],
    }


def validate_raw_answers(raw_answers: Any) -> dict[str, str | list[str]]:
    if not isinstance(raw_answers, dict):
        raise SurveyValidationError("answers must be an object")
    expected = set(QUESTION_BY_ID)
    received = set(raw_answers)
    missing = expected - received
    unknown = received - expected
    if missing:
        raise SurveyValidationError(f"missing required answers: {', '.join(sorted(missing))}")
    if unknown:
        raise SurveyValidationError(f"unknown answer fields: {', '.join(sorted(unknown))}")

    validated: dict[str, str | list[str]] = {}
    for question_id, question in QUESTION_BY_ID.items():
        value = raw_answers[question_id]
        allowed = {option["id"] for option in question["options"]}
        if question["type"] == "multiple":
            if not isinstance(value, list) or isinstance(value, str) or not value:
                raise SurveyValidationError(f"{question_id} must be a non-empty array")
            if any(not isinstance(item, str) for item in value):
                raise SurveyValidationError(f"{question_id} contains an invalid option")
            if len(value) != len(set(value)):
                raise SurveyValidationError(f"{question_id} contains duplicate options")
            if not set(value).issubset(allowed):
                raise SurveyValidationError(f"{question_id} contains an unknown option")
            validated[question_id] = list(value)
        else:
            if not isinstance(value, str):
                raise SurveyValidationError(f"{question_id} must be a single option")
            if value not in allowed:
                raise SurveyValidationError(f"{question_id} contains an unknown option")
            validated[question_id] = value
    return validated


def calculate_stated_features(
    raw_answers: dict[str, Any],
) -> StatedFeatures:
    answers = validate_raw_answers(raw_answers)
    selected = answers["investment_experience"]
    assert isinstance(selected, list)
    loss_value = LOSS_VALUES[str(answers["loss_tolerance"])]
    features = StatedFeatures(
        risk_capacity_age=AGE_VALUES[str(answers["age_group"])],
        investment_horizon=HORIZON_VALUES[str(answers["investment_horizon"])],
        risky_asset_experience=max(EXPERIENCE_VALUES[item] for item in selected),
        experience_breadth=len(selected) / len(EXPERIENCE_VALUES),
        derivative_experience=DERIVATIVE_VALUES[str(answers["derivative_experience"])],
        stated_loss_tolerance=loss_value,
        stated_risk_tolerance=loss_value,
        investment_exposure=EXPOSURE_VALUES[str(answers["investment_asset_ratio"])],
        financial_capacity=INCOME_VALUES[str(answers["monthly_income"])],
        return_seeking=PURPOSE_VALUES[str(answers["investment_purpose"])],
        financial_literacy=KNOWLEDGE_VALUES[str(answers["financial_knowledge"])],
        vulnerability_flag=int(answers["vulnerability"] == "yes"),
    )
    if any(
        not 0.0 <= float(value) <= 1.0 for value in features.as_dict().values()
    ):
        raise RuntimeError("calculated stated feature is outside [0, 1]")
    return features


# Transparent baseline points, deliberately separate from stated features.
# Vulnerability is investor-protection metadata and is never included.
SURVEY_POINTS: dict[str, dict[str, int]] = {
    "age_group": {"under_20": 4, "age_20_40": 4, "age_41_50": 3, "age_51_60": 2, "age_61_plus": 1},
    "investment_horizon": {"under_6_months": 0, "months_6_12": 1, "years_1_2": 2, "years_2_3": 3, "years_3_plus": 4},
    "investment_experience": {"conservative": 0, "conservative_plus": 1, "neutral": 2, "active": 3, "aggressive": 4},
    "derivative_experience": {"none_or_under_1_year": 0, "years_1_3": 1, "years_3_plus": 2},
    "loss_tolerance": {"principal_preservation": 0, "minimal_loss_only": 1, "partial_principal_loss": 2, "high_risk_for_return": 3},
    "investment_asset_ratio": {"up_to_10": 0, "up_to_30": 1, "up_to_50": 2, "up_to_70": 3, "over_70": 4},
    "monthly_income": {"up_to_1m": 0, "up_to_2m": 1, "up_to_3m": 2, "up_to_5m": 3, "over_5m": 4},
    "investment_purpose": {"living_or_short_term": 0, "deposit_level_return": 1, "above_market_return": 2, "active_wealth_growth": 3},
    "financial_knowledge": {"none": 0, "partial": 1, "deep": 2, "including_derivatives": 3},
}
MAX_SURVEY_POINTS = sum(max(points.values()) for points in SURVEY_POINTS.values())


def calculate_survey_score(raw_answers: dict[str, Any]) -> float:
    answers = validate_raw_answers(raw_answers)
    earned = 0
    for question_id, points in SURVEY_POINTS.items():
        answer = answers[question_id]
        if question_id == "investment_experience":
            assert isinstance(answer, list)
            earned += max(points[item] for item in answer)
        else:
            earned += points[str(answer)]
    return round(earned / MAX_SURVEY_POINTS * 100.0, 6)


def classify_survey_profile(score: float) -> str:
    if not 0.0 <= score <= 100.0:
        raise ValueError("survey score must be within [0, 100]")
    if score <= 20.0:
        return "안정형"
    if score <= 40.0:
        return "안정추구형"
    if score <= 60.0:
        return "위험중립형"
    if score <= 80.0:
        return "적극투자형"
    return "공격투자형"
