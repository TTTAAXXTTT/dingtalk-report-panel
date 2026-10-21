from __future__ import annotations

import json
import datetime as dt
import pathlib
import tempfile
import threading
import time
import unittest
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from app.dws_adapter import FixtureDwsAdapter
from app.llm_config_store import LLMConfigStore
from app.llm_engine import (
    LLMConfig,
    LLMOutputError,
    StructuredReport,
    render_completed_items,
    render_weekly_report,
)
from app.main import create_app
from app.storage import Storage


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class FakeLLMEngine:
    """Network-free model double used only by API integration tests."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def test_configuration(self) -> dict:
        return {
            "ok": True,
            "connection_ok": True,
            "model_ok": True,
            "structured_output_ok": True,
            "message": "合成模型测试通过",
            "tested_at": "2026-08-24T00:00:00+00:00",
            "config_fingerprint": self.config.fingerprint(),
        }

    async def generate(
        self,
        messages,
        *,
        report_subject_name="",
        only_subject_items=False,
        progress_callback=None,
        checkpoint_dir=None,
    ) -> dict:
        del progress_callback, checkpoint_dir
        rows = list(messages)
        if only_subject_items and report_subject_name:
            related_rows = [
                row
                for row in rows
                if str(row.get("sender_name") or row.get("sender") or "").strip()
                == report_subject_name
                or report_subject_name in str(row.get("text") or "")
            ]
            rows = related_rows
        if not rows:
            report = StructuredReport.empty()
        else:
            def item(row, prefix):
                message_id = str(row.get("id") or row["message_id"])
                source_text = str(row.get("text") or "有证据的测试事项")
                return {
                    "title": f"{prefix}{row.get('conversation_title') or row['conversation_id']}",
                    "detail": source_text,
                    "source_message_ids": [message_id],
                    "evidence_quotes": [
                        {
                            "source_message_id": message_id,
                            "quote": source_text[:160],
                        }
                    ],
                }

            completed_row = next(
                (row for row in rows if any(word in str(row.get("text")) for word in ("完成", "确认"))),
                rows[0],
            )
            unfinished_row = next(
                (row for row in rows if any(word in str(row.get("text")) for word in ("风险", "请", "待"))),
                rows[-1],
            )
            completed = item(completed_row, "完成")
            unfinished = item(unfinished_row, "跟进")
            report = StructuredReport.model_validate(
                {
                    "completed_items": [completed],
                    "weekly_report": {
                        "this_week": [completed, unfinished],
                        "next_week": [unfinished],
                    },
                }
            )
        return {
            "schema_version": 1,
            "engine": "fake-llm-evidence-v1",
            "structured": report.model_dump(mode="json"),
            "documents": {
                "completed_items_markdown": render_completed_items(
                    report, report_subject_name=report_subject_name
                ),
                "weekly_report_markdown": render_weekly_report(
                    report, report_subject_name=report_subject_name
                ),
            },
            "model": {
                "protocol": self.config.protocol,
                "model": self.config.model,
                "config_fingerprint": self.config.fingerprint(),
                "message_batch_count": 1 if rows else 0,
                "request_count": 1 if rows else 0,
                "report_subject": {
                    "name": report_subject_name,
                    "only_subject_items": only_subject_items,
                },
            },
        }


class FailingLLMEngine(FakeLLMEngine):
    async def generate(
        self,
        messages,
        *,
        report_subject_name="",
        only_subject_items=False,
        progress_callback=None,
        checkpoint_dir=None,
    ) -> dict:
        del report_subject_name, only_subject_items, progress_callback, checkpoint_dir
        raise LLMOutputError(
            "模型返回了不存在的证据 ID",
            code="LLM_EVIDENCE_OUT_OF_SCOPE",
        )


class ApiFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Storage(pathlib.Path(self.temporary.name) / "state")
        self.adapter = FixtureDwsAdapter(PROJECT_ROOT / "fixtures")
        self.llm_store = LLMConfigStore(
            pathlib.Path(self.temporary.name) / "llm-config.json"
        )
        llm_config = self.llm_store.save_config(
            {
                "protocol": "openai_responses",
                "base_url": "https://model.test/v1",
                "model": "fixture-model",
                "api_key": "fixture-secret",
                "analysis_instruction": "只根据测试消息整理",
                "weekly_report_template": "本周工作：\n\n1，事项：内容\n\n下周工作计划：\n\n1，事项：计划",
            }
        )
        self.llm_store.record_test(
            {
                "ok": True,
                "connection_ok": True,
                "model_ok": True,
                "structured_output_ok": True,
                "message": "已预置合成测试模型",
                "tested_at": "2026-08-24T00:00:00+00:00",
            },
            llm_config,
        )
        self.client = TestClient(
            create_app(
                adapter=self.adapter,
                storage=self.storage,
                mode="fixture",
                llm_config_store=self.llm_store,
                llm_engine_factory=FakeLLMEngine,
            )
        )
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        self.temporary.cleanup()

    @staticmethod
    def configuration(*, include_direct: bool = True) -> dict:
        ids = ["cid-fixture-project"]
        if include_direct:
            ids.append("cid-fixture-direct")
        return {
            "conversation_ids": ids,
            "start": "2026-08-17T00:00:00",
            "end": "2026-08-24T00:00:00",
            "timezone": "Asia/Shanghai",
            "filters": {
                "message_types": ["text"],
                "include_system_messages": False,
                "include_bot_messages": False,
                "include_recalled": True,
                "casual_filter": "conservative",
            },
            "report": {
                "subject_name": "测试用户",
                "title": "Fixture 手动报告",
                "grouping": "conversation",
                "sections": ["关键决策", "风险与阻塞", "待跟进事项", "其他重要信息"],
                "max_length_chars": 8000,
                "language": "zh-CN",
                "inference_policy": "facts_only",
                "custom_instruction": "只写事实，不明确信息标记为待确认。",
                "evidence_enabled": True,
            },
        }

    def test_status_and_refresh_never_create_a_report_job(self) -> None:
        status = self.client.get("/api/status")
        self.assertEqual(status.status_code, 200)
        body = status.json()
        self.assertEqual(body["data_mode"], "fixture")
        self.assertEqual(body["generation_mode"], "manual")
        self.assertFalse(body["scheduled"])
        self.assertTrue(body["dws"]["authenticated"])

        refreshed = self.client.post("/api/conversations/refresh")
        self.assertEqual(refreshed.status_code, 200)
        self.assertEqual(len(refreshed.json()["items"]), 3)
        self.assertEqual(self.client.get("/api/jobs").json()["total"], 0)

    def test_llm_configuration_is_editable_testable_and_never_echoes_key(self) -> None:
        initial = self.client.get("/api/llm/config")
        self.assertEqual(200, initial.status_code)
        self.assertTrue(initial.json()["test_status"]["ok"])
        self.assertTrue(initial.json()["config"]["api_key_configured"])
        self.assertNotIn("fixture-secret", initial.text)
        self.assertNotIn("api_key", initial.json()["config"])

        updated = self.client.put(
            "/api/llm/config",
            json={
                "protocol": "openai_chat_completions",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "custom-local-model",
                "api_key": "",
                "timeout_seconds": 180,
                "temperature": 0.3,
                "max_output_tokens": 16000,
                "analysis_instruction": "合并同类事项，只写明确事实",
                "weekly_report_template": "本周工作：\n\n1，事项：内容\n\n下周工作计划：\n\n1，事项：计划",
            },
        )
        self.assertEqual(200, updated.status_code, updated.text)
        self.assertFalse(updated.json()["test_status"]["ok"])
        self.assertTrue(updated.json()["config"]["api_key_configured"])
        self.assertNotIn("fixture-secret", updated.text)

        tested = self.client.post("/api/llm/test")
        self.assertEqual(200, tested.status_code, tested.text)
        self.assertTrue(tested.json()["ok"])
        self.assertTrue(tested.json()["connection_ok"])
        self.assertTrue(tested.json()["model_ok"])
        self.assertTrue(tested.json()["structured_output_ok"])

        changed_again = self.client.put(
            "/api/llm/config",
            json={
                "protocol": "openai_chat_completions",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "custom-local-model-v2",
                "timeout_seconds": 180,
                "temperature": 0.3,
                "max_output_tokens": 16000,
                "analysis_instruction": "合并同类事项，只写明确事实",
                "weekly_report_template": "本周工作：\n\n1，事项：内容\n\n下周工作计划：\n\n1，事项：计划",
            },
        )
        self.assertFalse(changed_again.json()["test_status"]["ok"])

    def test_preview_and_job_are_blocked_until_model_test_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = LLMConfigStore(pathlib.Path(temporary) / "llm-config.json")
            store.save_config(
                {
                    "base_url": "https://model.test/v1",
                    "model": "untested-model",
                    "analysis_instruction": "只写事实",
                    "weekly_report_template": "本周工作：\n下周工作计划：",
                }
            )
            storage = Storage(pathlib.Path(temporary) / "state")
            client = TestClient(
                create_app(
                    adapter=FixtureDwsAdapter(PROJECT_ROOT / "fixtures"),
                    storage=storage,
                    mode="fixture",
                    llm_config_store=store,
                    llm_engine_factory=FakeLLMEngine,
                )
            )
            with client:
                client.post("/api/conversations/refresh")
                configuration = self.configuration(include_direct=False)
                preview = client.post("/api/preview", json=configuration)
                self.assertEqual(200, preview.status_code)
                self.assertFalse(preview.json()["can_generate"])
                self.assertTrue(
                    any("大模型配置" in item for item in preview.json()["blocking_errors"])
                )
                configuration.update(
                    {"explicit_confirm": True, "uncertainty_ack": True}
                )
                created = client.post("/api/jobs", json=configuration)
                self.assertEqual(409, created.status_code)
                self.assertEqual(0, client.get("/api/jobs").json()["total"])

    def test_model_output_failure_never_publishes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = Storage(pathlib.Path(temporary) / "state")
            client = TestClient(
                create_app(
                    adapter=FixtureDwsAdapter(PROJECT_ROOT / "fixtures"),
                    storage=storage,
                    mode="fixture",
                    llm_config_store=self.llm_store,
                    llm_engine_factory=FailingLLMEngine,
                )
            )
            with client:
                client.post("/api/conversations/refresh")
                configuration = self.configuration(include_direct=False)
                configuration.update(
                    {"explicit_confirm": True, "uncertainty_ack": True}
                )
                created = client.post("/api/jobs", json=configuration)
                self.assertEqual(202, created.status_code, created.text)
                job_id = created.json()["id"]
                final = None
                for _ in range(100):
                    final = client.get(f"/api/jobs/{job_id}").json()
                    if final["status"] in {"COMPLETED", "FAILED"}:
                        break
                    time.sleep(0.03)
                self.assertEqual("FAILED", final["status"])
                self.assertEqual("LLM_EVIDENCE_OUT_OF_SCOPE", final["error"]["code"])
                self.assertFalse(
                    storage.artifact_path(
                        job_id, "report", "weekly_report.md"
                    ).exists()
                )
                self.assertFalse(
                    storage.artifact_path(
                        job_id, "report", "completed_items.md"
                    ).exists()
                )

    def test_preview_requires_known_conversations_and_explicit_confirmation(self) -> None:
        self.client.post("/api/conversations/refresh")
        configuration = self.configuration()
        preview = self.client.post("/api/preview", json=configuration)
        self.assertEqual(preview.status_code, 200)
        preview_body = preview.json()
        self.assertTrue(preview_body["can_generate"])
        self.assertFalse(preview_body["requires_uncertainty_ack"])
        self.assertEqual([], preview_body["warnings"])
        self.assertFalse(
            preview_body["effective_report"]["include_full_review_appendix"]
        )
        self.assertFalse(preview_body["effective_report"]["source_details_visible"])
        self.assertEqual("测试用户", preview_body["effective_report"]["subject_name"])

        not_confirmed = self.client.post("/api/jobs", json=configuration)
        self.assertEqual(not_confirmed.status_code, 409)

        configuration["explicit_confirm"] = True
        acknowledged_not_required = self.client.post("/api/jobs", json=configuration)
        self.assertEqual(acknowledged_not_required.status_code, 202)

        unknown = self.configuration(include_direct=False)
        unknown["conversation_ids"] = ["cid-does-not-exist"]
        unknown_preview = self.client.post("/api/preview", json=unknown)
        self.assertEqual(unknown_preview.status_code, 200)
        unknown_body = unknown_preview.json()
        self.assertTrue(unknown_body["can_generate"])
        self.assertTrue(unknown_body["requires_uncertainty_ack"])
        self.assertEqual(unknown_body["resolved_conversations"][0]["source"], "manual")
        self.assertTrue(any("采集时" in warning for warning in unknown_body["warnings"]))

    def test_manual_job_generates_immutable_grounded_report(self) -> None:
        self.client.post("/api/conversations/refresh")
        configuration = self.configuration()
        configuration.update({"explicit_confirm": True, "uncertainty_ack": True})
        created = self.client.post("/api/jobs", json=configuration)
        self.assertEqual(created.status_code, 202, created.text)
        job_id = created.json()["id"]

        final = None
        for _ in range(100):
            response = self.client.get(f"/api/jobs/{job_id}")
            self.assertEqual(response.status_code, 200)
            final = response.json()
            if final["status"] in {"COMPLETED", "FAILED"}:
                break
            time.sleep(0.03)
        self.assertIsNotNone(final)
        self.assertEqual(final["status"], "COMPLETED", final)
        self.assertFalse(final["data_complete"])
        self.assertEqual(final["snapshot_status"], "PROVISIONAL_SOURCE_LIMITATION")
        self.assertEqual(
            final["integrity"]["collection"]["required_stable_passes"], 3
        )

        self.assertEqual(["completed_items", "weekly_report"], final["documents"])
        completed_response = self.client.get(
            f"/api/jobs/{job_id}/report?document=completed_items"
        )
        weekly = self.client.get(
            f"/api/jobs/{job_id}/report?document=weekly_report"
        )
        self.assertEqual(completed_response.status_code, 200)
        self.assertEqual(weekly.status_code, 200)
        completed_markdown = completed_response.json()["markdown"]
        weekly_markdown = weekly.json()["markdown"]
        self.assertIn("报告本人：测试用户", completed_markdown)
        self.assertIn("报告本人：测试用户", weekly_markdown)
        self.assertIn("本周已完成事项：", completed_markdown)
        self.assertNotIn("未完成事项：", completed_markdown)
        self.assertIn("本周工作：", weekly_markdown)
        self.assertIn("下周工作计划：", weekly_markdown)
        self.assertNotIn("全量人工核对清单", weekly_markdown)
        self.assertNotIn("消息出处", weekly_markdown)
        self.assertNotIn("消息 ID：", weekly_markdown)
        self.assertNotIn("引用原文：", weekly_markdown)
        self.assertNotIn("消息出处", completed_markdown)
        self.assertNotIn("消息 ID：", completed_markdown)
        self.assertNotIn("引用原文：", completed_markdown)
        self.assertNotEqual(completed_markdown, weekly_markdown)

        default_completed = self.client.get(f"/api/jobs/{job_id}/report")
        self.assertEqual(default_completed.json()["markdown"], completed_markdown)
        for document, expected in (
            ("completed_items", completed_markdown),
            ("weekly_report", weekly_markdown),
        ):
            download = self.client.get(
                f"/api/jobs/{job_id}/download?document={document}"
            )
            self.assertEqual(download.status_code, 200)
            self.assertTrue(
                download.headers["content-type"].startswith("text/markdown")
            )
            self.assertEqual(download.content.decode("utf-8"), expected)

        analysis_path = self.storage.artifact_path(
            job_id, "analysis", "llm_analysis"
        )
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        self.assertEqual("fake-llm-evidence-v1", analysis["engine"])
        self.assertEqual("测试用户", analysis["report_subject"]["name"])
        manual_review = analysis["manual_review_appendix"]
        self.assertFalse(manual_review["enabled"])
        self.assertEqual(
            "disabled_by_report_display_policy", manual_review["mode"]
        )
        self.assertIsNone(manual_review["coverage_validated"])
        self.assertFalse(manual_review["ai_selection_applied"])
        self.assertEqual(0, manual_review["rendered_message_count"])
        self.assertNotIn("人工核对消息", final["counts"])
        evidence_ids = {
            source_id
            for item in analysis["structured"]["completed_items"]
            for source_id in item["source_message_ids"]
        }
        self.assertTrue(evidence_ids)

        snapshot_path = self.storage.artifact_path(job_id, "snapshot")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        message_ids = {message["id"] for message in snapshot["messages"]}
        self.assertTrue(evidence_ids.issubset(message_ids))
        self.assertIn("fx-start", message_ids)
        self.assertIn("fx-dm-targeted-only", message_ids)
        self.assertNotIn("fx-before", message_ids)
        self.assertNotIn("fx-end", message_ids)

        manifest = json.loads(
            self.storage.artifact_path(job_id, "manifest").read_text(encoding="utf-8")
        )
        self.assertTrue(manifest["explicit_confirm"])
        self.assertFalse(manifest["scheduled"])
        self.assertEqual(manifest["generation_mode"], "manual")
        self.assertEqual("large_model", manifest["analysis_engine"])
        self.assertNotIn("fixture-secret", json.dumps(manifest, ensure_ascii=False))

    def test_legacy_full_review_request_is_normalised_to_hidden_sources(self) -> None:
        self.client.post("/api/conversations/refresh")
        configuration = self.configuration(include_direct=False)
        configuration["report"]["include_full_review_appendix"] = True
        preview = self.client.post("/api/preview", json=configuration)
        self.assertEqual(200, preview.status_code, preview.text)
        self.assertFalse(
            preview.json()["effective_report"]["include_full_review_appendix"]
        )
        configuration.update({"explicit_confirm": True, "uncertainty_ack": True})
        created = self.client.post("/api/jobs", json=configuration)
        self.assertEqual(202, created.status_code, created.text)
        job_id = created.json()["id"]

        final = None
        for _ in range(100):
            final = self.client.get(f"/api/jobs/{job_id}").json()
            if final["status"] in {"COMPLETED", "FAILED"}:
                break
            time.sleep(0.03)
        self.assertEqual("COMPLETED", final["status"], final)
        weekly = self.client.get(
            f"/api/jobs/{job_id}/report?document=weekly_report"
        ).json()["markdown"]
        self.assertNotIn("全量人工核对清单", weekly)
        self.assertNotIn("消息出处", weekly)
        self.assertNotIn("消息 ID：", weekly)
        self.assertNotIn("引用原文：", weekly)
        analysis = json.loads(
            self.storage.artifact_path(
                job_id, "analysis", "llm_analysis"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(analysis["manual_review_appendix"]["enabled"])
        self.assertNotIn("人工核对消息", final["counts"])

    def test_future_end_and_empty_selection_fail_validation(self) -> None:
        bad = self.configuration(include_direct=False)
        bad["conversation_ids"] = []
        self.assertEqual(self.client.post("/api/preview", json=bad).status_code, 422)

        future = self.configuration(include_direct=False)
        future["end"] = "2099-01-01T00:00:00"
        self.assertEqual(self.client.post("/api/preview", json=future).status_code, 422)

        timezone = ZoneInfo("Asia/Shanghai")
        now = dt.datetime.now(timezone)
        near_future = self.configuration(include_direct=False)
        near_future["start"] = (now - dt.timedelta(hours=1)).isoformat(timespec="seconds")
        near_future["end"] = (now + dt.timedelta(minutes=4)).isoformat(timespec="seconds")
        self.assertEqual(
            self.client.post("/api/preview", json=near_future).status_code,
            422,
            "即使只晚于当前四分钟也必须拒绝，否则期间新消息会被确定性漏掉",
        )

    def test_report_subject_is_required_and_subject_only_mode_is_previewed(self) -> None:
        self.client.post("/api/conversations/refresh")
        configured = self.configuration(include_direct=False)
        configured["report"]["subject_name"] = "张三"
        configured["filters"]["only_related_to_me"] = True

        response = self.client.post("/api/preview", json=configured)
        self.assertEqual(200, response.status_code, response.text)
        body = response.json()
        self.assertEqual("张三", body["effective_report"]["subject_name"])
        self.assertTrue(body["effective_report"]["only_subject_items"])
        self.assertTrue(body["effective_filters"]["only_related_to_me"])

        missing = self.configuration(include_direct=False)
        del missing["report"]["subject_name"]
        self.assertEqual(422, self.client.post("/api/preview", json=missing).status_code)

        generic = self.configuration(include_direct=False)
        generic["report"]["subject_name"] = "本人"
        self.assertEqual(422, self.client.post("/api/preview", json=generic).status_code)

    def test_api_rejects_options_that_the_accuracy_mode_cannot_execute(self) -> None:
        cases: list[dict] = []

        expand = self.configuration(include_direct=False)
        expand["filters"]["expand_replies"] = True
        cases.append(expand)

        aggressive = self.configuration(include_direct=False)
        aggressive["filters"]["casual_filter"] = "aggressive"
        cases.append(aggressive)

        english = self.configuration(include_direct=False)
        english["report"]["language"] = "en-US"
        cases.append(english)

        inferred = self.configuration(include_direct=False)
        inferred["report"]["inference_policy"] = "limited"
        cases.append(inferred)

        no_evidence = self.configuration(include_direct=False)
        no_evidence["report"]["evidence_enabled"] = False
        cases.append(no_evidence)

        unknown = self.configuration(include_direct=False)
        unknown["report"]["silently_ignored_option"] = True
        cases.append(unknown)

        for payload in cases:
            with self.subTest(payload=payload):
                response = self.client.post("/api/preview", json=payload)
                self.assertEqual(response.status_code, 422, response.text)

        unsupported_instruction = self.configuration(include_direct=False)
        unsupported_instruction["report"]["custom_instruction"] = "请像专家一样自由发挥并综合判断"
        preview = self.client.post("/api/preview", json=unsupported_instruction)
        self.assertEqual(200, preview.status_code)
        self.assertTrue(preview.json()["can_generate"])
        self.assertEqual([], preview.json()["instruction_interpretation"]["unsupported"])
        self.assertIn(
            "请像专家一样自由发挥并综合判断",
            preview.json()["instruction_interpretation"]["applied"],
        )

        conflicting_scope = self.configuration(include_direct=False)
        conflicting_scope["report"]["sections"] = ["风险与阻塞"]
        conflicting_scope["report"]["custom_instruction"] = "只整理待办事项。只写事实。"
        preview = self.client.post("/api/preview", json=conflicting_scope)
        self.assertEqual(200, preview.status_code)
        self.assertTrue(preview.json()["can_generate"])

        conflicting_controls = self.configuration(include_direct=False)
        conflicting_controls["report"]["grouping"] = "date"
        conflicting_controls["report"]["max_length_chars"] = 5000
        conflicting_controls["filters"]["casual_filter"] = "off"
        conflicting_controls["report"]["custom_instruction"] = (
            "按会话分组。过滤普通闲聊。不超过6000字。只写事实。"
        )
        preview = self.client.post("/api/preview", json=conflicting_controls)
        self.assertEqual(200, preview.status_code)
        conflict_body = preview.json()
        self.assertTrue(conflict_body["can_generate"])
        self.assertEqual([], conflict_body["instruction_interpretation"]["conflicts"])
        self.assertEqual("date", conflict_body["effective_report"]["grouping"])
        self.assertEqual(5000, conflict_body["effective_report"]["max_length_chars"])
        self.assertFalse(conflict_body["effective_filters"]["exclude_casual"])

    def test_real_preview_blocks_before_job_creation_when_dws_is_not_ready(self) -> None:
        class UnreadyAdapter(FixtureDwsAdapter):
            def preflight(self):
                return {
                    "ok": False,
                    "version_ok": True,
                    "version_pinned": True,
                    "auth_command_ok": True,
                    "authenticated": False,
                    "errors": [{"code": "NOT_AUTHENTICATED", "message": "未登录"}],
                }

        with tempfile.TemporaryDirectory() as temporary:
            storage = Storage(pathlib.Path(temporary) / "real")
            client = TestClient(
                create_app(
                    adapter=UnreadyAdapter(PROJECT_ROOT / "fixtures"),
                    storage=storage,
                    mode="real",
                    llm_config_store=self.llm_store,
                    llm_engine_factory=FakeLLMEngine,
                )
            )
            with client:
                configuration = self.configuration(include_direct=False)
                preview = client.post("/api/preview", json=configuration)
                self.assertEqual(200, preview.status_code)
                self.assertFalse(preview.json()["can_generate"])
                self.assertTrue(
                    any("数据连接未就绪" in item for item in preview.json()["blocking_errors"])
                )
                configuration.update({"explicit_confirm": True, "uncertainty_ack": True})
                created = client.post("/api/jobs", json=configuration)
                self.assertEqual(422, created.status_code)
                self.assertEqual(0, client.get("/api/jobs").json()["total"])

    def test_mode_filters_distinct_source_rows_and_jobs_in_shared_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            storage = Storage(pathlib.Path(temporary) / "shared")
            storage.upsert_conversations(
                [
                    {
                        "id": "fixture-row",
                        "title": "合成会话",
                        "kind": "group",
                        "participants": [],
                        "available": True,
                        "source": "fixture",
                        "raw": {},
                    },
                    {
                        "id": "real-row",
                        "title": "真实来源会话",
                        "kind": "group",
                        "participants": [],
                        "available": True,
                        "source": "dws",
                        "raw": {},
                    },
                ]
            )
            fixture_job = storage.create_job({"data_mode": "fixture"})
            real_job = storage.create_job({"data_mode": "real"})
            real_client = TestClient(
                create_app(
                    adapter=FixtureDwsAdapter(PROJECT_ROOT / "fixtures"),
                    storage=storage,
                    mode="real",
                )
            )
            with real_client:
                conversations = real_client.get("/api/conversations").json()
                self.assertEqual(["real-row"], [item["id"] for item in conversations["items"]])
                self.assertEqual(1, real_client.get("/api/status").json()["conversation_count"])
                jobs = real_client.get("/api/jobs").json()
                self.assertEqual([real_job["id"]], [item["id"] for item in jobs["items"]])
                self.assertEqual(
                    404, real_client.get(f"/api/jobs/{fixture_job['id']}").status_code
                )

    def test_real_directory_warning_clears_only_after_verified_complete_refresh(self) -> None:
        class CompleteRealAdapter(FixtureDwsAdapter):
            def refresh_conversations(self):
                return {
                    "items": [
                        {
                            "id": "current-real-row",
                            "title": "当前真实会话",
                            "kind": "direct",
                            "last_active": None,
                            "participants": [],
                            "available": True,
                            "source": "dws",
                            "raw": {"openConversationId": "current-real-row"},
                        }
                    ],
                    "warnings": [],
                    "integrity": {
                        "complete": True,
                        "data_complete": True,
                        "pagination_known": True,
                        "source_limited": False,
                        "directory_incomplete": False,
                    },
                }

        with tempfile.TemporaryDirectory() as temporary:
            storage = Storage(pathlib.Path(temporary) / "real")
            storage.upsert_conversations(
                [
                    {
                        "id": "stale-real-row",
                        "title": "旧缓存会话",
                        "kind": "group",
                        "participants": [],
                        "available": True,
                        "source": "dws",
                        "raw": {"openConversationId": "stale-real-row"},
                    }
                ]
            )
            client = TestClient(
                create_app(
                    adapter=CompleteRealAdapter(PROJECT_ROOT / "fixtures"),
                    storage=storage,
                    mode="real",
                    llm_config_store=self.llm_store,
                    llm_engine_factory=FakeLLMEngine,
                )
            )
            with client:
                before = client.get("/api/conversations").json()
                self.assertFalse(before["directory"]["verified"])
                self.assertTrue(any("刷新聊天列表" in item for item in before["warnings"]))

                refreshed = client.post("/api/conversations/refresh")
                self.assertEqual(200, refreshed.status_code, refreshed.text)
                self.assertEqual(1, refreshed.json()["persistence"]["marked_unavailable"])
                self.assertTrue(refreshed.json()["directory"]["verified"])

                after = client.get("/api/conversations").json()
                self.assertEqual([], after["warnings"])
                self.assertTrue(after["directory"]["verified"])
                self.assertEqual(
                    ["current-real-row"], [item["id"] for item in after["items"]]
                )

    def test_conversation_refresh_uses_the_same_adapter_lock_as_collection(self) -> None:
        lock = self.client.app.state.context.worker_lock
        completed = threading.Event()
        responses: list[int] = []

        def refresh() -> None:
            responses.append(self.client.post("/api/conversations/refresh").status_code)
            completed.set()

        lock.acquire()
        try:
            worker = threading.Thread(target=refresh, daemon=True)
            worker.start()
            time.sleep(0.08)
            self.assertFalse(completed.is_set(), "刷新没有等待采集共用锁")
        finally:
            lock.release()
        worker.join(timeout=3)
        self.assertTrue(completed.is_set())
        self.assertEqual([200], responses)


if __name__ == "__main__":
    unittest.main()
