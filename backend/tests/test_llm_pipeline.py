from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
import uuid
from contextlib import closing
from pathlib import Path
from unittest.mock import MagicMock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from build_llm_input import (  # noqa: E402
    LlmInputBuildError,
    _comparison_request,
    _extract_dimension_result,
    _stated_preference,
    calculate_cross_context_calibration,
    load_feature_manifest,
)
from database import connect, initialize_database  # noqa: E402
from llm_pipeline import (  # noqa: E402
    AnalysisPipelineError,
    _call_structured,
    _comparison_llm_payload,
    _comparison_response_schema,
    _public_result,
    _runtime_settings,
    _update_artifact,
    _validate_user_facing_analysis,
    _validate_response_shape,
    create_or_restore_analysis_run,
)


class LlmPipelineTargetedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary.name) / "llm-test.db"
        initialize_database(self.database_path)
        self.manifest = load_feature_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _insert_eligible_user(self, user_id: str) -> None:
        now = "2026-09-02T00:00:00+00:00"
        survey_id = f"survey-{user_id}"
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO survey_responses VALUES (?,?,?,?,?,?)",
                (survey_id, user_id, "test", "{}", "{}", now),
            )
            connection.execute(
                "INSERT INTO survey_results VALUES (?,?,?,?,?,?,?)",
                (
                    survey_id,
                    user_id,
                    "test",
                    "test",
                    50.0,
                    "위험중립형",
                    now,
                ),
            )
            for index in range(1, 7):
                connection.execute(
                    "INSERT INTO sessions "
                    "(session_id,user_id,episode,scenario_id,episode_status,"
                    "created_at,updated_at,completed_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        f"session-{user_id}-{index}",
                        user_id,
                        f"E{index}",
                        f"scenario-{index}",
                        "completed",
                        now,
                        now,
                        now,
                    ),
                )
            connection.commit()

    def test_reasoning_effort_creates_a_distinct_analysis_config(self) -> None:
        self._insert_eligible_user("cache-user")
        environment = {
            "KIMI_MODEL": "moonshotai/kimi-k3",
            "KIMI_ANALYSIS_REVISION": "v1",
        }
        with (
            patch(
                "llm_pipeline.build_analysis_input_fingerprint",
                return_value="fingerprint-a",
            ),
            patch.dict(
                os.environ, {**environment, "KIMI_REASONING_EFFORT": "low"}
            ),
        ):
            low = create_or_restore_analysis_run(self.database_path, "cache-user")
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "UPDATE llm_analysis_runs SET status = 'completed' "
                "WHERE analysis_id = ?",
                (low["analysis_id"],),
            )
            connection.commit()

        with (
            patch(
                "llm_pipeline.build_analysis_input_fingerprint",
                return_value="fingerprint-a",
            ),
            patch.dict(
                os.environ, {**environment, "KIMI_REASONING_EFFORT": "max"}
            ),
        ):
            maximum = create_or_restore_analysis_run(
                self.database_path, "cache-user"
            )

        self.assertNotEqual(low["analysis_id"], maximum["analysis_id"])
        with closing(connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT analysis_config_version FROM llm_analysis_runs "
                "ORDER BY created_at, rowid"
            ).fetchall()
        self.assertEqual(
            [row["analysis_config_version"] for row in rows],
            ["kimi_k3_low_v1", "kimi_k3_max_v1"],
        )

    def test_stated_preference_exports_configured_financial_context_labels(self) -> None:
        now = "2026-09-04T00:00:00+00:00"
        answers = {
            "monthly_income": "up_to_3m",
            "investment_asset_ratio": "up_to_50",
        }
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "INSERT INTO survey_responses VALUES (?,?,?,?,?,?)",
                (
                    "survey-financial-context",
                    "financial-context-user",
                    "test",
                    "{}",
                    json.dumps(answers, ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO survey_results VALUES (?,?,?,?,?,?,?)",
                (
                    "survey-financial-context",
                    "financial-context-user",
                    "test",
                    "test",
                    50.0,
                    "위험중립형",
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO stated_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "survey-financial-context",
                    "financial-context-user",
                    "stated_v1",
                    now,
                    0.5,
                    0.5,
                    0.5,
                    0.4,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0.5,
                    0,
                ),
            )
            stated = _stated_preference(
                connection, "financial-context-user", self.manifest
            )
        self.assertEqual(
            stated["financial_context"],
            {
                "monthly_income_range": "300만원 이하",
                "investment_asset_ratio_range": "50% 이하",
            },
        )

    def test_kimi_timeout_and_pending_poll_interval_are_configured(self) -> None:
        with patch.dict(
            os.environ,
            {
                "KIMI_TIMEOUT_SECONDS": "600",
                "KIMI_STATUS_POLL_INTERVAL_SECONDS": "2",
            },
        ):
            settings = _runtime_settings()
        self.assertEqual(settings.timeout_seconds, 600.0)
        self.assertEqual(settings.status_poll_interval_seconds, 2.0)

    def test_completed_analysis_cache_requires_matching_input_fingerprint(self) -> None:
        self._insert_eligible_user("fingerprint-user")
        with patch(
            "llm_pipeline.build_analysis_input_fingerprint",
            return_value="fingerprint-one",
        ):
            first = create_or_restore_analysis_run(
                self.database_path, "fingerprint-user"
            )
        with closing(connect(self.database_path)) as connection:
            connection.execute(
                "UPDATE llm_analysis_runs SET status = 'completed' "
                "WHERE analysis_id = ?",
                (first["analysis_id"],),
            )
            connection.commit()

        with patch(
            "llm_pipeline.build_analysis_input_fingerprint",
            return_value="fingerprint-one",
        ):
            restored = create_or_restore_analysis_run(
                self.database_path, "fingerprint-user"
            )
        self.assertEqual(first["analysis_id"], restored["analysis_id"])

        with patch(
            "llm_pipeline.build_analysis_input_fingerprint",
            return_value="fingerprint-two",
        ):
            changed = create_or_restore_analysis_run(
                self.database_path, "fingerprint-user"
            )
        self.assertNotEqual(first["analysis_id"], changed["analysis_id"])
        with closing(connect(self.database_path)) as connection:
            fingerprints = {
                row["input_fingerprint"]
                for row in connection.execute(
                    "SELECT input_fingerprint FROM llm_analysis_runs "
                    "WHERE user_id = ?",
                    ("fingerprint-user",),
                )
            }
        self.assertEqual(fingerprints, {"fingerprint-one", "fingerprint-two"})

    @staticmethod
    def _http_response(status_code: int, payload: dict[str, object]) -> MagicMock:
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = payload
        response.text = str(payload)
        return response

    def test_kimi_202_is_polled_until_final_200(self) -> None:
        request_id = str(uuid.uuid4())
        pending = self._http_response(202, {"requestId": request_id})
        pending_again = self._http_response(202, {"requestId": request_id})
        expected = {"result": "ok"}
        completed = self._http_response(
            200,
            {
                "choices": [
                    {"message": {"content": json.dumps(expected)}}
                ]
            },
        )
        client = MagicMock()
        client.post.return_value = pending
        client.get.side_effect = [pending_again, completed]
        client_context = MagicMock()
        client_context.__enter__.return_value = client
        with patch.dict(
            os.environ,
            {
                "NVIDIA_API_KEY": "test-key",
                "KIMI_STATUS_POLL_INTERVAL_SECONDS": "0.1",
            },
        ):
            settings = _runtime_settings()
        with (
            patch("llm_pipeline.httpx.Client", return_value=client_context),
            patch("llm_pipeline.time.sleep"),
        ):
            result = _call_structured(
                settings=settings,
                schema_name="test_schema",
                schema={"type": "object"},
                system_prompt="test",
                payload={"value": 1},
            )

        self.assertEqual(result, expected)
        client.post.assert_called_once()
        self.assertEqual(client.get.call_count, 2)
        status_url = client.get.call_args_list[0].args[0]
        self.assertEqual(
            status_url,
            f"https://integrate.api.nvidia.com/v1/status/{request_id}",
        )
        posted_json = client.post.call_args.kwargs["json"]
        self.assertEqual(posted_json["reasoning_effort"], "low")
        self.assertEqual(posted_json["response_format"]["type"], "json_schema")

    def test_kimi_202_without_uuid_request_id_is_rejected(self) -> None:
        client = MagicMock()
        client.post.return_value = self._http_response(
            202, {"requestId": "not-a-uuid"}
        )
        client_context = MagicMock()
        client_context.__enter__.return_value = client
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
            settings = _runtime_settings()
        with patch("llm_pipeline.httpx.Client", return_value=client_context):
            with self.assertRaisesRegex(AnalysisPipelineError, "UUID"):
                _call_structured(
                    settings=settings,
                    schema_name="test_schema",
                    schema={"type": "object"},
                    system_prompt="test",
                    payload={"value": 1},
                )
        client.get.assert_not_called()

    def test_private_input_snapshot_columns_are_migrated_and_writable(self) -> None:
        legacy_path = Path(self.temporary.name) / "legacy-llm.db"
        legacy = sqlite3.connect(legacy_path)
        legacy.executescript(
            """
            CREATE TABLE llm_analysis_runs (
                analysis_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                status TEXT NOT NULL, model TEXT NOT NULL,
                manifest_schema_version TEXT NOT NULL,
                behavioral_input_schema_version TEXT NOT NULL,
                comparison_input_schema_version TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                completed_at TEXT, error_code TEXT, internal_error TEXT
            );
            CREATE TABLE llm_analysis_artifacts (
                analysis_id TEXT PRIMARY KEY,
                call1_raw_response_json TEXT, revealed_result_json TEXT,
                call2_raw_response_json TEXT, public_result_json TEXT
            );
            """
        )
        legacy.close()
        initialize_database(legacy_path)
        with closing(connect(legacy_path)) as connection:
            run_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(llm_analysis_runs)")
            }
            artifact_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(llm_analysis_artifacts)"
                )
            }
        self.assertIn("analysis_config_version", run_columns)
        self.assertIn("input_fingerprint", run_columns)
        self.assertIn("behavioral_input_json", artifact_columns)
        self.assertIn("comparison_input_json", artifact_columns)

        self._insert_eligible_user("snapshot-user")
        with patch(
            "llm_pipeline.build_analysis_input_fingerprint",
            return_value="fingerprint-snapshot",
        ):
            run = create_or_restore_analysis_run(self.database_path, "snapshot-user")
        _update_artifact(
            self.database_path,
            run["analysis_id"],
            "behavioral_input_json",
            {"stage": "call1"},
        )
        _update_artifact(
            self.database_path,
            run["analysis_id"],
            "comparison_input_json",
            {"stage": "call2"},
        )
        with closing(connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT behavioral_input_json,comparison_input_json "
                "FROM llm_analysis_artifacts WHERE analysis_id = ?",
                (run["analysis_id"],),
            ).fetchone()
        self.assertEqual(row["behavioral_input_json"], '{"stage": "call1"}')
        self.assertEqual(row["comparison_input_json"], '{"stage": "call2"}')

    def test_evidence_must_be_manifest_allowed_and_exist_in_actual_input(self) -> None:
        rubrics = self.manifest["behavioral_dimension_rubrics"]["dimensions"]
        baselines = {
            name: {"base_level": "medium", "max_llm_adjustment_steps": 1}
            for name in rubrics
        }
        behavioral: dict[str, object] = {}
        for rubric in rubrics.values():
            path = str(rubric["primary_evidence"][0]).split(".")[1:]
            cursor = behavioral
            for segment in path[:-1]:
                cursor = cursor.setdefault(segment, {})  # type: ignore[assignment]
            cursor[path[-1]] = 0.5

        dimensions = {
            name: {
                "base_level": "medium",
                "adjustment": 0,
                "confidence_level": "medium",
                "reason": "test",
                "evidence_fields": [rubric["primary_evidence"][0]],
            }
            for name, rubric in rubrics.items()
        }
        result = {"revealed_behavioral_dimensions": dimensions}
        cleaned = _extract_dimension_result(
            result, self.manifest, baselines, behavioral
        )
        self.assertEqual(cleaned["risk_engagement"]["final_level"], "medium")

        dimensions["risk_engagement"]["evidence_fields"] = [
            "behavioral_analysis.episode3.fake_feature"
        ]
        with self.assertRaisesRegex(LlmInputBuildError, "outside its manifest"):
            _extract_dimension_result(result, self.manifest, baselines, behavioral)

        allowed_but_missing = rubrics["risk_engagement"]["supporting_evidence"][4]
        dimensions["risk_engagement"]["evidence_fields"] = [allowed_but_missing]
        with self.assertRaisesRegex(LlmInputBuildError, "actual input"):
            _extract_dimension_result(result, self.manifest, baselines, behavioral)

    def test_large_e6_risk_gap_fixes_downward_adjustment_and_low_confidence(self) -> None:
        behavioral = {
            "episode3": {
                "adaptive_context": {"routing_score": 0.70},
                "summary_features": {"behavior_resilience_score": 0.80},
            },
            "episode6": {
                "summary_features": {
                    "anchor_risk_exposure_auc": 0.10,
                    "risk_engagement_consistency": 0.40,
                    "e6_behavior_resilience_score": 0.20,
                    "loss_response_consistency": 0.40,
                    "cross_context_consistency": 0.40,
                }
            },
        }
        calibration = calculate_cross_context_calibration(
            behavioral, self.manifest
        )
        risk = calibration["risk_engagement"]
        loss = calibration["loss_resilience"]
        self.assertAlmostEqual(risk["gap"], 0.60)
        self.assertEqual(risk["direction"], "strongly_lower")
        self.assertEqual(risk["suggested_adjustment"], -1)
        self.assertAlmostEqual(loss["gap"], 0.60)
        self.assertEqual(loss["direction"], "strongly_lower")
        self.assertEqual(calibration["behavioral_confidence_base"], "low")

    def test_python_suggested_adjustment_cannot_be_overridden(self) -> None:
        rubrics = self.manifest["behavioral_dimension_rubrics"]["dimensions"]
        baselines = {
            name: {"base_level": "medium", "max_llm_adjustment_steps": 1}
            for name in rubrics
        }
        baselines["risk_engagement"]["suggested_adjustment"] = -1
        behavioral: dict[str, object] = {}
        for rubric in rubrics.values():
            path = str(rubric["primary_evidence"][0]).split(".")[1:]
            cursor = behavioral
            for segment in path[:-1]:
                cursor = cursor.setdefault(segment, {})  # type: ignore[assignment]
            cursor[path[-1]] = 0.5
        dimensions = {
            name: {
                "base_level": "medium",
                "adjustment": 0,
                "confidence_level": "medium",
                "reason": "test",
                "evidence_fields": [rubric["primary_evidence"][0]],
            }
            for name, rubric in rubrics.items()
        }
        with self.assertRaisesRegex(LlmInputBuildError, "Python calibration"):
            _extract_dimension_result(
                {"revealed_behavioral_dimensions": dimensions},
                self.manifest,
                baselines,
                behavioral,
            )

    def test_call2_payload_and_public_result_exclude_internal_analysis(self) -> None:
        template = {
            "investor_type": "위험중립형",
            "confidence_level": "low",
            "stated_preference_summary": "설문 응답 성향을 해요체로 요약해요.",
            "revealed_preference_summary": "시장 선택 성향을 해요체로 요약해요.",
            "stated_revealed_gap": "설문과 시장 선택의 차이를 해요체로 설명해요.",
            "key_behavioral_evidence": [],
            "final_analysis": "검증된 근거로 행동 특성을 종합 설명해요.",
            "personalized_guidance": "재정 여건과 행동 성향을 함께 점검해요.",
        }
        comparison_input = {
            "stated_preference": {
                "survey_baseline": {"score": 90, "profile": "공격투자형"},
                "financial_context": {
                    "monthly_income_range": "300만원 이하",
                    "investment_asset_ratio_range": "50% 이하",
                },
            },
            "revealed_profile": {
                "profile": "위험중립형",
                "classification_status": "classified",
                "core_dimension_values": {"risk_engagement": 30},
            },
            "cross_context_calibration": {
                "behavioral_confidence_base": "low",
                "risk_engagement": {"gap": 0.60},
            },
            "public_behavioral_observations": [
                "공통 시장 조건에서는 위험자산 노출을 낮췄어요."
            ],
            "analysis_request": {
                "rules": ["존댓말을 사용합니다."],
                "required_output_format": template,
            },
        }
        payload = _comparison_llm_payload(comparison_input)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("survey_baseline", serialized)
        self.assertNotIn("core_dimension_values", serialized)
        self.assertNotIn("risk_engagement", serialized)
        self.assertNotIn("0.6", serialized)
        self.assertIn("300만원 이하", serialized)
        self.assertIn("50% 이하", serialized)

        raw_result = {
            "investor_type": "위험중립형",
            "confidence_level": "low",
            "stated_preference_summary": "설문 결과에서는 공격투자형으로 나타났어요.",
            "revealed_preference_summary": "행동에서는 위험자산 노출을 낮췄어요.",
            "stated_revealed_gap": "설문과 행동 사이에 차이가 관찰됐어요.",
            "key_behavioral_evidence": [
                "공통 시장 조건에서는 위험자산 노출을 낮췄어요."
            ],
            "final_analysis": "행동 성향은 시장 조건에 따라 달라졌어요.",
            "personalized_guidance": "유동성과 비상자금 여건을 함께 점검해 볼 필요가 있어요.",
        }
        public = _public_result(
            comparison_input, raw_result, self.manifest
        )
        self.assertNotIn("classification_status", public)
        self.assertNotIn("behavioral_traits", public)
        self.assertNotIn("confidence_level", public["analysis"])
        self.assertEqual(public["analysis"]["confidence"], "낮음")

    def test_call2_guidance_uses_relationship_based_reasoning_contract(self) -> None:
        request = _comparison_request(
            self.manifest,
            {"profile": "위험중립형"},
            "medium",
        )
        rules = " ".join(request["rules"])
        self.assertIn("Reason in this order", rules)
        self.assertIn("compare their relationship", rules)
        self.assertIn("aggressive but current investment exposure is low", rules)
        self.assertIn("conservative but current investment exposure is high", rules)
        self.assertIn("2 to 4 natural Korean sentences", rules)
        self.assertIn("current-state assessment", rules)
        self.assertIn("2~4개", request["required_output_format"]["personalized_guidance"])

    def test_user_facing_output_enforces_prose_and_guidance_policy(self) -> None:
        verified = ["위험자산 노출을 낮췄어요."]
        valid = {
            "stated_preference_summary": "설문 응답에서 나타난 성향을 요약했어요.",
            "revealed_preference_summary": "위험자산 노출을 낮췄어요.",
            "stated_revealed_gap": "설문과 행동 사이에 차이가 관찰됐어요.",
            "key_behavioral_evidence": ["위험자산 노출을 낮췄어요."],
            "final_analysis": "시장 조건에 따른 차이를 함께 고려해야 해요.",
            "personalized_guidance": "현재 정보만으로 단정하지 않고 재정 여건을 점검해 봐요.",
        }
        _validate_user_facing_analysis(valid, verified)

        leaked = dict(valid)
        leaked["final_analysis"] = (
            "behavioral_analysis.episode3.adaptive_context.routing_score는 0.3278이에요."
        )
        with self.assertRaisesRegex(AnalysisPipelineError, "internal"):
            _validate_user_facing_analysis(leaked, verified)

        invented = dict(valid)
        invented["final_analysis"] = "회복 구간에서 다시 재진입했어요."
        _validate_user_facing_analysis(invented, verified)

        unverified_evidence = dict(valid)
        unverified_evidence["key_behavioral_evidence"] = [
            "검증 목록에 없는 행동 근거예요."
        ]
        with self.assertRaisesRegex(AnalysisPipelineError, "verified enum"):
            _validate_user_facing_analysis(unverified_evidence, verified)

        advice = dict(valid)
        advice["final_analysis"] = "위험자산을 매수하세요."
        with self.assertRaisesRegex(AnalysisPipelineError, "advice"):
            _validate_user_facing_analysis(advice, verified)

        formal = dict(valid)
        formal["final_analysis"] = "시장 조건에 따른 차이를 고려해야 합니다."
        with self.assertLogs("llm_pipeline", level="WARNING") as captured:
            _validate_user_facing_analysis(formal, verified)
        self.assertIn("style-only deviation", captured.output[0])

        noun_ending = dict(valid)
        noun_ending["final_analysis"] = "시장 조건에 따른 행동 차이"
        _validate_user_facing_analysis(noun_ending, verified)

        non_korean = dict(valid)
        non_korean["final_analysis"] = "Behavioral profile"
        with self.assertRaisesRegex(AnalysisPipelineError, "Korean text"):
            _validate_user_facing_analysis(non_korean, verified)

        real_trading = dict(valid)
        real_trading["final_analysis"] = "실제 투자에서 보인 선택으로 해석해요."
        with self.assertRaisesRegex(AnalysisPipelineError, "real trading"):
            _validate_user_facing_analysis(real_trading, verified)

        repeated_subject = dict(valid)
        repeated_subject["stated_preference_summary"] = "사용자님의 설문 성향을 요약했어요."
        repeated_subject["final_analysis"] = "사용자님의 행동을 함께 해석했어요."
        with self.assertRaisesRegex(AnalysisPipelineError, "repeated"):
            _validate_user_facing_analysis(repeated_subject, verified)

        overconfident = {**valid, "confidence_level": "medium"}
        overconfident["final_analysis"] = "명확한 행동 성향이 관찰됐어요."
        with self.assertRaisesRegex(AnalysisPipelineError, "too strong"):
            _validate_user_facing_analysis(overconfident, verified)

        leveraged = dict(valid)
        leveraged["personalized_guidance"] = "대출을 활용한 투자를 추천해요."
        with self.assertRaisesRegex(AnalysisPipelineError, "advice"):
            _validate_user_facing_analysis(leveraged, verified)

        precise_allocation = dict(valid)
        precise_allocation["personalized_guidance"] = "위험자산 비중을 30%로 줄이세요."
        with self.assertRaisesRegex(AnalysisPipelineError, "advice"):
            _validate_user_facing_analysis(precise_allocation, verified)

        guaranteed_return = dict(valid)
        guaranteed_return["personalized_guidance"] = "이 선택은 수익률이 확실해요."
        with self.assertRaisesRegex(AnalysisPipelineError, "advice"):
            _validate_user_facing_analysis(guaranteed_return, verified)

        general_review = dict(valid)
        general_review["personalized_guidance"] = (
            "현재 투자에 배분된 비중이 높은 편이라면 생활비와 비상자금 "
            "여유를 함께 점검해 보는 것이 좋아요."
        )
        _validate_user_facing_analysis(general_review, verified)

        missing_context_review = dict(valid)
        missing_context_review["personalized_guidance"] = (
            "월소득뿐 아니라 고정 지출이나 부채도 함께 고려해 현재 투자 "
            "규모를 감당할 수 있는지 확인해 보는 것이 좋아요."
        )
        _validate_user_facing_analysis(missing_context_review, verified)

    def test_comparison_confidence_is_manifest_ordinal(self) -> None:
        template = {
            "investor_type": "위험중립형",
            "confidence_level": "high",
            "stated_preference_summary": "설문 응답 성향을 해요체로 요약해요.",
            "revealed_preference_summary": "시장 선택 성향을 해요체로 요약해요.",
            "stated_revealed_gap": "설문과 시장 선택의 차이를 해요체로 설명해요.",
            "key_behavioral_evidence": [],
            "final_analysis": "검증된 근거로 행동 특성을 종합 설명해요.",
            "personalized_guidance": "재정 여건과 행동 성향을 함께 점검해요.",
        }
        comparison_input = {
            "analysis_request": {"required_output_format": template},
            "public_behavioral_observations": ["위험자산 노출을 유지했어요."],
        }
        schema = _comparison_response_schema(comparison_input, self.manifest)
        self.assertEqual(
            schema["properties"]["confidence_level"]["enum"],
            self.manifest["revealed_profile_scoring"]["confidence_levels"],
        )
        response = {
            **template,
            "confidence_level": "medium",
            "stated_preference_summary": "설문에서 확인된 성향을 요약했어요.",
            "revealed_preference_summary": "시장 선택에서 나타난 성향을 요약했어요.",
            "stated_revealed_gap": "설문과 시장 선택 사이에 차이가 관찰됐어요.",
            "key_behavioral_evidence": ["위험자산 노출을 유지했어요."],
            "final_analysis": "검증된 행동을 바탕으로 종합적으로 해석했어요.",
            "personalized_guidance": "유동성과 위험 감내 수준을 함께 점검해 봐요.",
        }
        _validate_response_shape(response, template, self.manifest)
        response["confidence_level"] = "very_high"
        with self.assertRaisesRegex(AnalysisPipelineError, "confidence"):
            _validate_response_shape(response, template, self.manifest)

    def test_call2_null_prose_is_rejected(self) -> None:
        template = {
            "investor_type": "위험중립형",
            "confidence_level": "medium",
            "stated_preference_summary": "설문 응답 성향을 해요체로 요약해요.",
            "revealed_preference_summary": "시장 선택 성향을 해요체로 요약해요.",
            "stated_revealed_gap": "설문과 시장 선택의 차이를 해요체로 설명해요.",
            "key_behavioral_evidence": [],
            "final_analysis": "검증된 근거로 행동 특성을 종합 설명해요.",
            "personalized_guidance": "재정 여건과 행동 성향을 함께 점검해요.",
        }
        response = {
            **template,
            "stated_preference_summary": None,
            "key_behavioral_evidence": ["위험자산 노출을 유지했어요."],
        }
        with self.assertRaisesRegex(AnalysisPipelineError, "non-empty string"):
            _validate_response_shape(response, template, self.manifest)

        response = {
            **template,
            "personalized_guidance": None,
            "key_behavioral_evidence": ["위험자산 노출을 유지했어요."],
        }
        with self.assertRaisesRegex(AnalysisPipelineError, "non-empty string"):
            _validate_response_shape(response, template, self.manifest)


if __name__ == "__main__":
    unittest.main()
