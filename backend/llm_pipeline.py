"""Two-call NVIDIA Kimi-K3 analysis pipeline with private audit persistence.

The feature manifest and ``build_llm_input.py`` are the only sources of truth
for feature selection, quantitative anchors, bounded LLM adjustments, and the
deterministic revealed profile.

NVIDIA references (checked 2026-09-01):
- https://build.nvidia.com/moonshotai/kimi-k3
- https://docs.api.nvidia.com/nim/re/reference/moonshotai-kimi-k3
- https://docs.api.nvidia.com/nim/reference/moonshotai-kimi-k3-statuspolling
- https://docs.nvidia.com/nim/large-language-models/1.14.0/structured-generation.html
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx
from dotenv import load_dotenv

from build_llm_input import (
    LlmInputBuildError,
    build_behavioral_input,
    build_comparison_input,
    build_feature_guide,
    load_feature_manifest,
)
from database import connect


BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env", override=False)

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_KIMI_MODEL = "moonshotai/kimi-k3"
DEFAULT_KIMI_REASONING_EFFORT = "low"
DEFAULT_KIMI_TEMPERATURE = 1.0
DEFAULT_KIMI_MAX_TOKENS = 16384
DEFAULT_KIMI_TIMEOUT_SECONDS = 600.0
DEFAULT_KIMI_STATUS_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_KIMI_ANALYSIS_REVISION = "v1"
KIMI_REASONING_EFFORTS = {"low", "high", "max"}
PUBLIC_STATUS_MESSAGES = {
    "queued": "분석을 준비하고 있습니다.",
    "processing": "응답을 분석하고 있습니다.",
    "completed": "분석이 완료되었습니다.",
    "failed": "분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.",
}
PUBLIC_ERROR_MESSAGES = {
    "configuration_error": "분석 서비스 설정을 확인해 주세요.",
    "timeout": "분석 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.",
    "invalid_response": "분석 응답을 검증하지 못했습니다. 다시 시도해 주세요.",
    "upstream_error": "분석 서비스에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.",
    "invalid_source_data": "완료된 응답 데이터를 확인하지 못했습니다.",
    "internal_error": "분석 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
}


class AnalysisEligibilityError(RuntimeError):
    """Raised before a run is created when Survey or E1-E6 is incomplete."""


class AnalysisPipelineError(RuntimeError):
    """A categorized private pipeline failure safe to persist for diagnosis."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class KimiSettings:
    api_key: str
    base_url: str
    model: str
    reasoning_effort: str
    analysis_config_version: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    status_poll_interval_seconds: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_settings() -> KimiSettings:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).strip()
    model = os.getenv("KIMI_MODEL", DEFAULT_KIMI_MODEL).strip()
    reasoning_effort = os.getenv(
        "KIMI_REASONING_EFFORT", DEFAULT_KIMI_REASONING_EFFORT
    ).strip().lower()
    analysis_revision = os.getenv(
        "KIMI_ANALYSIS_REVISION", DEFAULT_KIMI_ANALYSIS_REVISION
    ).strip().lower()
    if not base_url or not model:
        raise AnalysisPipelineError(
            "configuration_error", "NVIDIA_BASE_URL and KIMI_MODEL are required"
        )
    if reasoning_effort not in KIMI_REASONING_EFFORTS:
        raise AnalysisPipelineError(
            "configuration_error",
            "KIMI_REASONING_EFFORT must be low, high, or max",
        )
    if not analysis_revision or not all(
        character.isalnum() or character in {"-", "_"}
        for character in analysis_revision
    ):
        raise AnalysisPipelineError(
            "configuration_error",
            "KIMI_ANALYSIS_REVISION must contain only letters, numbers, - or _",
        )
    try:
        temperature = float(
            os.getenv("KIMI_TEMPERATURE", str(DEFAULT_KIMI_TEMPERATURE))
        )
        max_tokens = int(
            os.getenv("KIMI_MAX_TOKENS", str(DEFAULT_KIMI_MAX_TOKENS))
        )
        timeout = float(
            os.getenv(
                "KIMI_TIMEOUT_SECONDS", str(DEFAULT_KIMI_TIMEOUT_SECONDS)
            )
        )
        poll_interval = float(
            os.getenv(
                "KIMI_STATUS_POLL_INTERVAL_SECONDS",
                str(DEFAULT_KIMI_STATUS_POLL_INTERVAL_SECONDS),
            )
        )
    except ValueError as exc:
        raise AnalysisPipelineError(
            "configuration_error", "Kimi-K3 runtime setting is invalid"
        ) from exc
    if (
        not 0 <= temperature <= 1
        or not 1 <= max_tokens <= 65536
        or timeout <= 0
        or not 0.1 <= poll_interval <= 30
    ):
        raise AnalysisPipelineError(
            "configuration_error", "Kimi-K3 runtime setting is out of range"
        )
    model_token = model.rsplit("/", 1)[-1].replace("-", "_")
    analysis_config_version = (
        f"{model_token}_{reasoning_effort}_{analysis_revision}"
    )
    return KimiSettings(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        reasoning_effort=reasoning_effort,
        analysis_config_version=analysis_config_version,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        status_poll_interval_seconds=poll_interval,
    )


def _assert_completed_user(connection: sqlite3.Connection, user_id: str) -> None:
    survey = connection.execute(
        "SELECT 1 FROM survey_results WHERE user_id = ?", (user_id,)
    ).fetchone()
    completed = {
        str(row["episode"])
        for row in connection.execute(
            "SELECT episode FROM sessions WHERE user_id = ? "
            "AND episode_status = 'completed'",
            (user_id,),
        ).fetchall()
    }
    required = {f"E{index}" for index in range(1, 7)}
    if survey is None or completed != required:
        raise AnalysisEligibilityError(
            "설문과 Episode 1~6를 모두 완료한 뒤 분석할 수 있습니다."
        )


def _is_stale(row: sqlite3.Row, timeout_seconds: float) -> bool:
    try:
        updated_at = datetime.fromisoformat(str(row["updated_at"]))
    except ValueError:
        return True
    # Two model calls plus a small persistence margin.
    return datetime.now(timezone.utc) - updated_at > timedelta(
        seconds=(timeout_seconds * 2) + 60
    )


def create_or_restore_analysis_run(
    database_path: Path, user_id: str
) -> dict[str, Any]:
    """Create one durable run, or restore an active/completed run for the user."""
    settings = _runtime_settings()
    manifest = load_feature_manifest()
    now = _utc_now()
    with closing(connect(database_path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            _assert_completed_user(connection, user_id)
            latest = connection.execute(
                "SELECT * FROM llm_analysis_runs WHERE user_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id,),
            ).fetchone()
            versions = manifest["input_schema_versions"]
            latest_is_current = latest is not None and all(
                (
                    latest["model"] == settings.model,
                    latest["analysis_config_version"]
                    == settings.analysis_config_version,
                    latest["manifest_schema_version"] == manifest["schema_version"],
                    latest["behavioral_input_schema_version"]
                    == versions["behavioral"],
                    latest["comparison_input_schema_version"]
                    == versions["comparison"],
                )
            )
            if (
                latest is not None
                and not latest_is_current
                and latest["status"] in {"queued", "processing"}
            ):
                connection.execute(
                    "UPDATE llm_analysis_runs SET status = 'failed', updated_at = ?, "
                    "completed_at = ?, error_code = ?, internal_error = ? "
                    "WHERE analysis_id = ?",
                    (
                        now,
                        now,
                        "configuration_error",
                        "Analysis schema or model changed before completion",
                        latest["analysis_id"],
                    ),
                )
            if latest is not None and latest["status"] in {
                "queued",
                "processing",
                "completed",
            } and latest_is_current:
                if latest["status"] in {"queued", "processing"} and _is_stale(
                    latest, settings.timeout_seconds
                ):
                    connection.execute(
                        "UPDATE llm_analysis_runs SET status = 'failed', "
                        "updated_at = ?, completed_at = ?, error_code = ?, "
                        "internal_error = ? WHERE analysis_id = ?",
                        (
                            now,
                            now,
                            "timeout",
                            "Analysis worker did not finish before stale deadline",
                            latest["analysis_id"],
                        ),
                    )
                else:
                    connection.commit()
                    return _public_run(connection, str(latest["analysis_id"]))

            analysis_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO llm_analysis_runs (analysis_id,user_id,status,model,"
                "analysis_config_version,"
                "manifest_schema_version,behavioral_input_schema_version,"
                "comparison_input_schema_version,created_at,updated_at) "
                "VALUES (?,?,'queued',?,?,?,?,?,?,?)",
                (
                    analysis_id,
                    user_id,
                    settings.model,
                    settings.analysis_config_version,
                    manifest["schema_version"],
                    versions["behavioral"],
                    versions["comparison"],
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO llm_analysis_artifacts (analysis_id) VALUES (?)",
                (analysis_id,),
            )
            connection.commit()
            return _public_run(connection, analysis_id)
        except Exception:
            connection.rollback()
            raise


def _behavioral_response_schema(
    behavioral_input: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    ordinal_levels = list(manifest["revealed_profile_scoring"]["ordinal_values"])
    confidence_levels = list(
        manifest["revealed_profile_scoring"]["confidence_levels"]
    )
    baselines = behavioral_input["quantitative_baselines"]
    template = behavioral_input["analysis_request"]["required_output_format"]
    dimensions = template["revealed_behavioral_dimensions"]
    properties: dict[str, Any] = {}
    for name in dimensions:
        base_level = baselines[name]["base_level"]
        maximum = int(baselines[name]["max_llm_adjustment_steps"])
        suggested_adjustment = baselines[name].get("suggested_adjustment")
        rubric = manifest["behavioral_dimension_rubrics"]["dimensions"][name]
        allowed_evidence = list(
            dict.fromkeys(
                list(rubric["primary_evidence"])
                + list(rubric["supporting_evidence"])
            )
        )
        properties[name] = {
            "type": "object",
            "properties": {
                "base_level": {"enum": [base_level]},
                "adjustment": {
                    "enum": (
                        [None]
                        if base_level is None
                        else (
                            [suggested_adjustment]
                            if suggested_adjustment is not None
                            else list(range(-maximum, maximum + 1))
                        )
                    )
                },
                "confidence_level": {
                    "enum": [None] if base_level is None else confidence_levels
                },
                "reason": {"type": "string", "minLength": 1},
                "evidence_fields": {
                    "type": "array",
                    "items": {"type": "string", "enum": allowed_evidence},
                    "minItems": 1,
                    # "uniqueItems": True,
                },
            },
            "required": list(dimensions[name]),
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "revealed_behavioral_dimensions": {
                "type": "object",
                "properties": properties,
                "required": list(dimensions),
                "additionalProperties": False,
            }
        },
        "required": ["revealed_behavioral_dimensions"],
        "additionalProperties": False,
    }


def _comparison_response_schema(
    comparison_input: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    template = comparison_input["analysis_request"]["required_output_format"]

    def field_schema(name: str, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return {
                "type": "object",
                "properties": {
                    child: field_schema(child, child_value)
                    for child, child_value in value.items()
                },
                "required": list(value),
                "additionalProperties": False,
            }
        if isinstance(value, list):
            item_schema: dict[str, Any] = {"type": "string", "minLength": 1}
            if name == "key_behavioral_evidence":
                item_schema = {
                    "type": "string",
                    "enum": list(
                        comparison_input["public_behavioral_observations"]
                    ),
                }
            return {
                "type": "array",
                "items": item_schema,
                **({"minItems": 1} if name == "key_behavioral_evidence" else {}),
            }
        if name == "investor_type":
            return {"type": "string", "enum": [str(value)]}
        if name == "confidence_level":
            return {
                "type": "string",
                "enum": list(
                    manifest["revealed_profile_scoring"]["confidence_levels"]
                ),
            }
        return {"type": "string", "minLength": 1}

    return field_schema("root", template)


def _validate_response_shape(
    value: Any,
    template: Any,
    manifest: Mapping[str, Any],
    path: str = "response",
) -> None:
    """Defensively validate the manifest/builder output shape after parsing."""
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(value) != set(template):
            raise AnalysisPipelineError(
                "invalid_response", f"{path} fields do not match the required format"
            )
        for name, child_template in template.items():
            _validate_response_shape(
                value[name], child_template, manifest, f"{path}.{name}"
            )
        return
    if isinstance(template, list):
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise AnalysisPipelineError(
                "invalid_response", f"{path} must be an array of non-empty strings"
            )
        if path.endswith("key_behavioral_evidence") and not value:
            raise AnalysisPipelineError(
                "invalid_response", f"{path} must contain at least one field"
            )
        if path.endswith("key_behavioral_evidence") and any(
            re.search(r"[가-힣]", item) is None for item in value
        ):
            raise AnalysisPipelineError(
                "invalid_response",
                f"{path} must contain Korean user-facing strings",
            )
        return
    if path.endswith("investor_type"):
        if not isinstance(value, str) or value != template:
            raise AnalysisPipelineError(
                "invalid_response",
                f"{path} must preserve the fixed Python investor type",
            )
        return
    if path.endswith("confidence_level"):
        confidence_levels = set(
            manifest["revealed_profile_scoring"]["confidence_levels"]
        )
        if value not in confidence_levels:
            raise AnalysisPipelineError(
                "invalid_response",
                f"{path} must match a manifest confidence level",
            )
        return
    if not isinstance(value, str) or not value.strip():
        raise AnalysisPipelineError(
            "invalid_response", f"{path} must be a non-empty string"
        )


def _behavioral_feature_guide(manifest: Mapping[str, Any]) -> dict[str, Any]:
    guide = build_feature_guide(manifest)
    # Call 1 must not receive stated-preference values or even its feature guide.
    guide.pop("stated_preference", None)
    return guide


def _comparison_llm_payload(
    comparison_input: Mapping[str, Any]
) -> dict[str, Any]:
    """Build Call 2 input without raw feature names, paths, or numeric scores."""
    request = comparison_input["analysis_request"]
    return {
        "source_user_facing_results": {
            "stated_profile": comparison_input["stated_preference"][
                "survey_baseline"
            ]["profile"],
            "revealed_profile": comparison_input["revealed_profile"]["profile"],
        },
        "confidence_anchor": comparison_input["cross_context_calibration"][
            "behavioral_confidence_base"
        ],
        "verified_behavioral_observations": comparison_input[
            "public_behavioral_observations"
        ],
        "rules": request["rules"],
        "required_output_format": request["required_output_format"],
    }


USER_FACING_INTERNAL_PATTERNS = (
    re.compile(r"\b(?:very_low|very_high|low|medium|high)\b", re.IGNORECASE),
    re.compile(r"\b(?:Episode|에피소드)\s*[1-6]?\b", re.IGNORECASE),
    re.compile(r"\bE[1-6]\b", re.IGNORECASE),
    re.compile(r"\b0\.\d+\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s*점\b"),
    re.compile(r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+"),
    re.compile(
        r"\b(?:rubric|base\s*level|adjustment|Call\s*[12])\b",
        re.IGNORECASE,
    ),
)
DIRECT_ADVICE_PATTERNS = (
    re.compile(r"(?:매수|매도|투자).*(?:추천(?:합니다|해요)|권(?:합니다|해요)|하세요)"),
    re.compile(r"비중을\s*(?:늘리세요|줄이세요|조정하세요)"),
)
USER_FACING_SIMULATION_PATTERNS = (
    re.compile(r"(?:실제|실전)\s*(?:투자|거래)"),
)
VERIFIABLE_ACTION_PATTERNS = {
    "increase": re.compile(
        r"(?:(?:비중|노출).{0,12}(?:높이|늘리|확대|증가)|"
        r"(?:높이|늘리|확대|증가).{0,12}(?:비중|노출))"
    ),
    "decrease": re.compile(
        r"(?:(?:비중|노출).{0,12}(?:낮추|줄이|축소|감소)|"
        r"(?:낮추|줄이|축소|감소).{0,12}(?:비중|노출))"
    ),
    "hold": re.compile(r"유지"),
    "reentry": re.compile(r"재진입|다시\s*(?:늘리|높이|확대)"),
    "recovery": re.compile(r"회복"),
}


def _validate_user_facing_analysis(
    value: Mapping[str, Any], verified_observations: list[str]
) -> None:
    text_fields = (
        "stated_preference_summary",
        "revealed_preference_summary",
        "stated_revealed_gap",
        "final_analysis",
    )
    texts = [str(value[field]) for field in text_fields]
    texts.extend(str(item) for item in value["key_behavioral_evidence"])
    if sum(text.count("사용자님") for text in texts) > 1:
        raise AnalysisPipelineError(
            "invalid_response",
            "Kimi-K3 user-facing output repeated 사용자님 unnecessarily",
        )
    if value.get("confidence_level") in {"low", "medium"} and any(
        re.search(r"(?:일관된|명확한|확실한)", text) for text in texts
    ):
        raise AnalysisPipelineError(
            "invalid_response",
            "Kimi-K3 user-facing wording was too strong for the confidence level",
        )
    verified_text = " ".join(verified_observations)
    for text in texts:
        if any(pattern.search(text) for pattern in USER_FACING_INTERNAL_PATTERNS):
            raise AnalysisPipelineError(
                "invalid_response",
                "Kimi-K3 user-facing output exposed internal analysis metadata",
            )
        if any(pattern.search(text) for pattern in DIRECT_ADVICE_PATTERNS):
            raise AnalysisPipelineError(
                "invalid_response",
                "Kimi-K3 user-facing output contained direct investment advice",
            )
        if any(pattern.search(text) for pattern in USER_FACING_SIMULATION_PATTERNS):
            raise AnalysisPipelineError(
                "invalid_response",
                "Kimi-K3 user-facing output described simulated choices as real trading",
            )
        if re.search(r"[가-힣]", text) is None or re.search(
            r"요[.!?]?\s*$", text
        ) is None:
            raise AnalysisPipelineError(
                "invalid_response",
                "Kimi-K3 user-facing prose must use natural Korean 해요체",
            )
        for concept, pattern in VERIFIABLE_ACTION_PATTERNS.items():
            if pattern.search(text) and not pattern.search(verified_text):
                raise AnalysisPipelineError(
                    "invalid_response",
                    f"Kimi-K3 invented an unverified behavioral action: {concept}",
                )


def _call_structured(
    *,
    settings: KimiSettings,
    schema_name: str,
    schema: Mapping[str, Any],
    system_prompt: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not settings.api_key:
        raise AnalysisPipelineError(
            "configuration_error", "NVIDIA_API_KEY is not configured"
        )
    request_body = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": dict(schema),
            },
        },
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": False,
        "reasoning_effort": settings.reasoning_effort,
    }
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    deadline = time.monotonic() + settings.timeout_seconds

    def remaining_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AnalysisPipelineError(
                "timeout", "Kimi-K3 pending result exceeded the polling deadline"
            )
        return remaining

    def response_json(response: httpx.Response, stage: str) -> dict[str, Any]:
        try:
            value = response.json()
        except ValueError as exc:
            raise AnalysisPipelineError(
                "invalid_response", f"NVIDIA {stage} response was not valid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise AnalysisPipelineError(
                "invalid_response", f"NVIDIA {stage} response must be an object"
            )
        return value

    def request_id_from(value: Mapping[str, Any]) -> str:
        request_id = value.get("requestId")
        if not isinstance(request_id, str) or len(request_id) > 36:
            raise AnalysisPipelineError(
                "invalid_response",
                "NVIDIA pending response did not contain a valid requestId",
            )
        try:
            uuid.UUID(request_id)
        except ValueError as exc:
            raise AnalysisPipelineError(
                "invalid_response", "NVIDIA requestId was not a UUID"
            ) from exc
        return request_id

    def ensure_supported_status(response: httpx.Response, stage: str) -> None:
        if response.status_code in {200, 202}:
            return
        detail = response.text[:1000]
        raise AnalysisPipelineError(
            "upstream_error",
            f"NVIDIA {stage} returned HTTP {response.status_code}: {detail}",
        )

    try:
        with httpx.Client(headers=headers) as client:
            response = client.post(
                f"{settings.base_url}/chat/completions",
                json=request_body,
                timeout=remaining_timeout(),
            )
            ensure_supported_status(response, "chat completion")
            response_payload = response_json(response, "chat completion")

            if response.status_code == 202:
                request_id = request_id_from(response_payload)
                status_url = f"{settings.base_url}/status/{request_id}"
                while True:
                    sleep_seconds = min(
                        settings.status_poll_interval_seconds,
                        remaining_timeout(),
                    )
                    time.sleep(sleep_seconds)
                    status_response = client.get(
                        status_url,
                        timeout=remaining_timeout(),
                    )
                    ensure_supported_status(status_response, "status polling")
                    response_payload = response_json(
                        status_response, "status polling"
                    )
                    if status_response.status_code == 200:
                        break
                    pending_id = request_id_from(response_payload)
                    if pending_id != request_id:
                        raise AnalysisPipelineError(
                            "invalid_response",
                            "NVIDIA status response changed requestId",
                        )
    except httpx.TimeoutException as exc:
        raise AnalysisPipelineError("timeout", str(exc)) from exc
    except httpx.RequestError as exc:
        raise AnalysisPipelineError("upstream_error", str(exc)) from exc

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AnalysisPipelineError(
            "invalid_response", "Kimi-K3 response did not contain a choice"
        )
    first_choice = choices[0]
    message = first_choice.get("message") if isinstance(first_choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        raise AnalysisPipelineError(
            "invalid_response", "Kimi-K3 response content was empty"
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AnalysisPipelineError(
            "invalid_response", "Kimi-K3 response was not valid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise AnalysisPipelineError(
            "invalid_response", "Kimi-K3 response root must be an object"
        )
    return parsed


def _update_artifact(
    database_path: Path, analysis_id: str, column: str, payload: Mapping[str, Any]
) -> None:
    allowed = {
        "behavioral_input_json",
        "call1_raw_response_json",
        "revealed_result_json",
        "comparison_input_json",
        "call2_raw_response_json",
        "public_result_json",
    }
    if column not in allowed:
        raise ValueError(f"Unsupported artifact column: {column}")
    with closing(connect(database_path)) as connection:
        connection.execute(
            f"UPDATE llm_analysis_artifacts SET {column} = ? WHERE analysis_id = ?",
            (json.dumps(payload, ensure_ascii=False), analysis_id),
        )
        connection.execute(
            "UPDATE llm_analysis_runs SET updated_at = ? WHERE analysis_id = ?",
            (_utc_now(), analysis_id),
        )
        connection.commit()


def _public_result(
    comparison_input: Mapping[str, Any],
    comparison_result: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    revealed_profile = comparison_input["revealed_profile"]
    confidence_labels = {"low": "낮음", "medium": "보통", "high": "높음"}
    public_analysis = {
        key: value
        for key, value in comparison_result.items()
        if key != "confidence_level"
    }
    public_analysis["confidence"] = confidence_labels[
        comparison_result["confidence_level"]
    ]
    return {
        "stated_profile": comparison_input["stated_preference"][
            "survey_baseline"
        ]["profile"],
        "revealed_profile": revealed_profile["profile"],
        "analysis": public_analysis,
    }


def execute_analysis_run(
    database_path: Path, analysis_id: str, user_id: str
) -> None:
    """Run Call 1 and Call 2, persisting each audit stage before continuing."""
    now = _utc_now()
    with closing(connect(database_path)) as connection:
        claimed = connection.execute(
            "UPDATE llm_analysis_runs SET status = 'processing', updated_at = ?, "
            "error_code = NULL, internal_error = NULL WHERE analysis_id = ? "
            "AND user_id = ? AND status = 'queued'",
            (now, analysis_id, user_id),
        )
        connection.commit()
        if claimed.rowcount != 1:
            return

    try:
        settings = _runtime_settings()
        manifest = load_feature_manifest()
        behavioral_input = build_behavioral_input(
            database_path, user_id, manifest
        )
        call1_payload = {
            "feature_guide": _behavioral_feature_guide(manifest),
            "user_behavioral_input": behavioral_input,
        }
        _update_artifact(
            database_path,
            analysis_id,
            "behavioral_input_json",
            call1_payload,
        )
        call1_result = _call_structured(
            settings=settings,
            schema_name="revealed_behavioral_dimensions",
            schema=_behavioral_response_schema(behavioral_input, manifest),
            system_prompt=(
                "You analyze investment behavior using only the supplied behavioral "
                "evidence and versioned rubric. Never infer or request stated survey "
                "answers. Preserve Python base levels and apply only the allowed "
                "adjustment. A non-null suggested_adjustment is a mandatory Python "
                "calibration result and must be echoed exactly. Information "
                "sensitivity is direction-neutral and may never raise or lower any "
                "risk-profile dimension. Return only schema-valid JSON."
            ),
            payload=call1_payload,
        )
        _update_artifact(
            database_path, analysis_id, "call1_raw_response_json", call1_result
        )

        # This call invokes the existing Python validator and deterministically fixes
        # final levels, the revealed risk score, and the revealed profile.
        comparison_input = build_comparison_input(
            database_path, user_id, call1_result, manifest
        )
        finalized_revealed = {
            "revealed_profile": comparison_input["revealed_profile"],
            "behavioral_evidence": comparison_input["behavioral_evidence"],
            "behavioral_modifiers": comparison_input["behavioral_modifiers"],
        }
        _update_artifact(
            database_path,
            analysis_id,
            "revealed_result_json",
            finalized_revealed,
        )

        call2_payload = _comparison_llm_payload(comparison_input)
        _update_artifact(
            database_path,
            analysis_id,
            "comparison_input_json",
            call2_payload,
        )

        call2_result = _call_structured(
            settings=settings,
            schema_name="stated_revealed_interpretation",
            schema=_comparison_response_schema(comparison_input, manifest),
            system_prompt=(
                "Write a respectful Korean user-facing interpretation using only the "
                "verified Korean observations supplied in this request. Every prose "
                "field and every evidence item must be a non-empty natural Korean "
                "해요체 sentence ending in -요. Never use -습니다/-입니다 style or "
                "plain -다/-한다 style. Avoid repeating 사용자님 and omit the subject "
                "naturally whenever possible. This is a simulated market-choice "
                "assessment, so never describe it as 실제 투자, 실제 거래, or 실전 투자. "
                "investor_type is the only fixed output field and must exactly match "
                "the Python result. Choose confidence_level only from the allowed "
                "manifest enum and use confidence_anchor as the primary calibration "
                "evidence. Matching stated and revealed profile labels never proves "
                "behavioral consistency; infer consistency only from cross-context "
                "calibration and verified observations. For high confidence, wording "
                "may indicate a comparatively clear or similar tendency across "
                "situations. For medium confidence, mention observed differences and "
                "avoid strong claims such as 일관된, 명확한, or 확실한. For low "
                "confidence, state that the evidence is limited or varies by context "
                "and requires cautious interpretation. Never expose English ordinal "
                "levels, decimal "
                "values, point scores, experiment or episode names, feature names, "
                "JSON paths, rubrics, base levels, cutoffs, or adjustment mechanics. "
                "Do not invent an allocation increase, decrease, hold, re-entry, or "
                "recovery action that is absent from verified_behavioral_observations. "
                "Information responsiveness describes only reaction magnitude and "
                "must never be interpreted as aggressive or conservative direction. "
                "Cross-market conflict controls confidence only. Never recommend a "
                "product, allocation, purchase, sale, or investment strategy. Return "
                "only schema-valid JSON."
            ),
            payload=call2_payload,
        )
        _validate_response_shape(
            call2_result,
            comparison_input["analysis_request"]["required_output_format"],
            manifest,
        )
        _update_artifact(
            database_path, analysis_id, "call2_raw_response_json", call2_result
        )
        _validate_user_facing_analysis(
            call2_result,
            comparison_input["public_behavioral_observations"],
        )
        public_result = _public_result(comparison_input, call2_result, manifest)
        _update_artifact(
            database_path, analysis_id, "public_result_json", public_result
        )

        completed_at = _utc_now()
        with closing(connect(database_path)) as connection:
            connection.execute(
                "UPDATE llm_analysis_runs SET status = 'completed', updated_at = ?, "
                "completed_at = ?, error_code = NULL, internal_error = NULL "
                "WHERE analysis_id = ? AND user_id = ?",
                (completed_at, completed_at, analysis_id, user_id),
            )
            connection.commit()
    except LlmInputBuildError as exc:
        _mark_failed(database_path, analysis_id, "invalid_source_data", str(exc))
    except AnalysisPipelineError as exc:
        _mark_failed(database_path, analysis_id, exc.code, str(exc))
    except Exception as exc:  # pragma: no cover - final worker safety boundary
        _mark_failed(
            database_path,
            analysis_id,
            "internal_error",
            f"{type(exc).__name__}: {exc}",
        )


def _mark_failed(
    database_path: Path, analysis_id: str, error_code: str, internal_error: str
) -> None:
    completed_at = _utc_now()
    with closing(connect(database_path)) as connection:
        connection.execute(
            "UPDATE llm_analysis_runs SET status = 'failed', updated_at = ?, "
            "completed_at = ?, error_code = ?, internal_error = ? "
            "WHERE analysis_id = ?",
            (
                completed_at,
                completed_at,
                error_code,
                internal_error[:4000],
                analysis_id,
            ),
        )
        connection.commit()


def _public_run(connection: sqlite3.Connection, analysis_id: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT run.analysis_id,run.status,run.error_code,artifact.public_result_json "
        "FROM llm_analysis_runs AS run JOIN llm_analysis_artifacts AS artifact "
        "ON artifact.analysis_id = run.analysis_id WHERE run.analysis_id = ?",
        (analysis_id,),
    ).fetchone()
    if row is None:
        raise KeyError(analysis_id)
    status = str(row["status"])
    response: dict[str, Any] = {
        "analysis_id": str(row["analysis_id"]),
        "status": status,
        "message": PUBLIC_STATUS_MESSAGES[status],
        "result": None,
    }
    if status == "completed" and row["public_result_json"]:
        response["result"] = json.loads(str(row["public_result_json"]))
    elif status == "failed":
        response["message"] = PUBLIC_ERROR_MESSAGES.get(
            str(row["error_code"]), PUBLIC_ERROR_MESSAGES["internal_error"]
        )
    return response


def get_public_analysis_run(
    database_path: Path, analysis_id: str, user_id: str
) -> dict[str, Any] | None:
    with closing(connect(database_path)) as connection:
        owner = connection.execute(
            "SELECT 1 FROM llm_analysis_runs WHERE analysis_id = ? AND user_id = ?",
            (analysis_id, user_id),
        ).fetchone()
        if owner is None:
            return None
        return _public_run(connection, analysis_id)
