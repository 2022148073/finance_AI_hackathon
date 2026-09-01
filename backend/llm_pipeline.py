"""Two-call NVIDIA Kimi-K3 analysis pipeline with private audit persistence.

The feature manifest and ``build_llm_input.py`` are the only sources of truth
for feature selection, quantitative anchors, bounded LLM adjustments, and the
deterministic revealed profile.

NVIDIA references (checked 2026-09-01):
- https://build.nvidia.com/moonshotai/kimi-k3
- https://docs.api.nvidia.com/nim/re/reference/moonshotai-kimi-k3
- https://docs.nvidia.com/nim/large-language-models/1.14.0/structured-generation.html
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

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
DEFAULT_KIMI_TIMEOUT_SECONDS = 120.0
DEFAULT_KIMI_MAX_RETRIES = 1
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
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_settings() -> KimiSettings:
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).strip()
    model = os.getenv("KIMI_MODEL", DEFAULT_KIMI_MODEL).strip()
    reasoning_effort = os.getenv(
        "KIMI_REASONING_EFFORT", DEFAULT_KIMI_REASONING_EFFORT
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
        max_retries = int(
            os.getenv("KIMI_MAX_RETRIES", str(DEFAULT_KIMI_MAX_RETRIES))
        )
    except ValueError as exc:
        raise AnalysisPipelineError(
            "configuration_error", "Kimi-K3 runtime setting is invalid"
        ) from exc
    if (
        not 0 <= temperature <= 1
        or not 1 <= max_tokens <= 65536
        or timeout <= 0
        or max_retries < 0
    ):
        raise AnalysisPipelineError(
            "configuration_error", "Kimi-K3 runtime setting is out of range"
        )
    return KimiSettings(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        max_retries=max_retries,
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
                "manifest_schema_version,behavioral_input_schema_version,"
                "comparison_input_schema_version,created_at,updated_at) "
                "VALUES (?,?,'queued',?,?,?,?,?,?)",
                (
                    analysis_id,
                    user_id,
                    settings.model,
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
        properties[name] = {
            "type": "object",
            "properties": {
                "base_level": {"enum": [base_level]},
                "adjustment": {
                    "enum": (
                        [None]
                        if base_level is None
                        else list(range(-maximum, maximum + 1))
                    )
                },
                "confidence_level": {
                    "enum": [None] if base_level is None else confidence_levels
                },
                "reason": {"type": "string", "minLength": 1},
                "evidence_fields": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
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
            return {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                **({"minItems": 1} if name == "key_behavioral_evidence" else {}),
            }
        if name == "investor_type":
            return {"type": "string", "enum": [str(value)]}
        if name == "confidence":
            return {"type": "number", "minimum": 0, "maximum": 1}
        return {"type": "string", "minLength": 1}

    return field_schema("root", template)


def _validate_response_shape(
    value: Any, template: Any, path: str = "response"
) -> None:
    """Defensively validate the manifest/builder output shape after parsing."""
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(value) != set(template):
            raise AnalysisPipelineError(
                "invalid_response", f"{path} fields do not match the required format"
            )
        for name, child_template in template.items():
            _validate_response_shape(value[name], child_template, f"{path}.{name}")
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
        return
    if isinstance(template, float):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= float(value) <= 1
        ):
            raise AnalysisPipelineError(
                "invalid_response", f"{path} must be a number between 0 and 1"
        )
        return
    if isinstance(template, str) and value != template:
        raise AnalysisPipelineError(
            "invalid_response", f"{path} must preserve the fixed Python result"
        )
    if not isinstance(value, str) or not value.strip():
        raise AnalysisPipelineError(
            "invalid_response", f"{path} must be a non-empty string"
        )


def _behavioral_feature_guide(manifest: Mapping[str, Any]) -> dict[str, Any]:
    guide = build_feature_guide(manifest)
    # Call 1 must not receive stated-preference values or even its feature guide.
    guide.pop("stated_preference", None)
    return guide


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
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            OpenAI,
            RateLimitError,
        )
    except ImportError as exc:
        raise AnalysisPipelineError(
            "configuration_error", "The openai Python package is not installed"
        ) from exc

    # NVIDIA NIM exposes an OpenAI-compatible Chat Completions endpoint. The
    # SDK is only the transport client; requests are sent to NVIDIA, not OpenAI.
    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
    try:
        response = client.chat.completions.create(
            model=settings.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                },
            },
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            stream=False,
            extra_body={"reasoning_effort": settings.reasoning_effort},
        )
    except APITimeoutError as exc:
        raise AnalysisPipelineError("timeout", str(exc)) from exc
    except (APIConnectionError, RateLimitError, APIStatusError) as exc:
        raise AnalysisPipelineError("upstream_error", str(exc)) from exc

    if not response.choices:
        raise AnalysisPipelineError(
            "invalid_response", "Kimi-K3 response did not contain a choice"
        )
    content = response.choices[0].message.content
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
        "call1_raw_response_json",
        "revealed_result_json",
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
    dimensions = {
        **comparison_input["behavioral_evidence"],
        **{
            name: comparison_input["behavioral_modifiers"][name]
            for name in manifest["revealed_profile_scoring"]["modifier_dimensions"]
        },
    }
    return {
        "stated_profile": comparison_input["stated_preference"][
            "survey_baseline"
        ]["profile"],
        "revealed_profile": revealed_profile["profile"],
        "classification_status": revealed_profile["classification_status"],
        "behavioral_traits": {
            name: {"level": value["final_level"]}
            for name, value in dimensions.items()
        },
        "analysis": dict(comparison_result),
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
        call1_result = _call_structured(
            settings=settings,
            schema_name="revealed_behavioral_dimensions",
            schema=_behavioral_response_schema(behavioral_input, manifest),
            system_prompt=(
                "You analyze investment behavior using only the supplied behavioral "
                "evidence and versioned rubric. Never infer or request stated survey "
                "answers. Preserve Python base levels and apply only the allowed "
                "adjustment. Return only schema-valid JSON."
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

        call2_result = _call_structured(
            settings=settings,
            schema_name="stated_revealed_interpretation",
            schema=_comparison_response_schema(comparison_input, manifest),
            system_prompt=(
                "You write a Korean user-facing interpretation of fixed stated and "
                "revealed results. The revealed profile is immutable: never change, "
                "recalculate, or replace it. Do not expose raw feature names, field "
                "paths, quantitative cutoffs, base levels, or adjustment mechanics in "
                "any user-facing output field, including key_behavioral_evidence and "
                "summaries. investor_type must exactly echo the fixed revealed profile "
                "provided in the required output format. "
                "Treat survey and revealed numeric scores as different, non-calibrated "
                "scales. Write every user-facing prose field in Korean. Return only "
                "schema-valid JSON."
            ),
            payload={
                "feature_guide": build_feature_guide(manifest),
                "comparison_input": comparison_input,
            },
        )
        _validate_response_shape(
            call2_result,
            comparison_input["analysis_request"]["required_output_format"],
        )
        _update_artifact(
            database_path, analysis_id, "call2_raw_response_json", call2_result
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
