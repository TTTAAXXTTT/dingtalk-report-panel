from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import httpx
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.llm_engine as llm_engine_module  # noqa: E402
from app.llm_engine import (  # noqa: E402
    LLMConfig,
    LLMConfigurationError,
    LLMOutputError,
    LLMReportEngine,
    LLMTransportError,
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_RESPONSES,
    StructuredReport,
    candidate_json_schema,
    render_completed_items,
    render_weekly_report,
    report_json_schema,
)


def structured_payload(
    *,
    completed_ids: list[str] | None = None,
    unfinished_ids: list[str] | None = None,
    completed_quotes: dict[str, str] | None = None,
    unfinished_quotes: dict[str, str] | None = None,
    completed_title: str = "部署完成",
) -> dict:
    completed_ids = completed_ids or []
    unfinished_ids = unfinished_ids or []
    completed_quotes = completed_quotes or {
        message_id: "部署已经完成" for message_id in completed_ids
    }
    unfinished_quotes = unfinished_quotes or {
        message_id: "接口尚未完成，下周继续处理" for message_id in unfinished_ids
    }
    completed = [
        {
            "title": completed_title,
            "detail": "完成测试部署。",
            "source_message_ids": completed_ids,
            "evidence_quotes": [
                {
                    "source_message_id": message_id,
                    "quote": completed_quotes[message_id],
                }
                for message_id in completed_ids
            ],
        }
    ] if completed_ids else []
    next_week = [
        {
            "title": "接口待处理",
            "detail": "下周继续处理接口工作。",
            "source_message_ids": unfinished_ids,
            "evidence_quotes": [
                {
                    "source_message_id": message_id,
                    "quote": unfinished_quotes[message_id],
                }
                for message_id in unfinished_ids
            ],
        }
    ] if unfinished_ids else []
    return {
        "completed_items": completed,
        "weekly_report": {
            "this_week": completed,
            "next_week": next_week,
        },
    }


def responses_result(value: dict) -> httpx.Response:
    return httpx.Response(200, json={"output_text": json.dumps(value, ensure_ascii=False)})


def chat_result(value: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)}}
            ]
        },
    )


def chat_text_result(value: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": value}}]},
    )


def message(
    message_id: str,
    text: str,
    *,
    time: str = "2026-08-20T10:00:00+08:00",
    sender: str = "张三",
) -> dict:
    return {
        "conversation_id": "conv-a",
        "conversation_name": "项目群",
        "message_id": message_id,
        "sender": sender,
        "time": time,
        "text": text,
    }


def model_row_id(row) -> str:
    return row[0] if isinstance(row, list) else row["message_id"]


def model_row_text(row) -> str:
    return row[5] if isinstance(row, list) else row["text"]


class LLMConfigTests(unittest.TestCase):
    def test_temperature_defaults_to_automatic_omission(self):
        config = LLMConfig()

        self.assertIsNone(config.temperature)
        payload = config.public_dict()
        self.assertIsNone(payload["temperature"])

    def test_config_masks_key_and_fingerprint_changes_without_leaking_secret(self):
        first = LLMConfig(api_key="sk-super-secret-one")
        second = LLMConfig(api_key="sk-super-secret-two")

        public = first.public_dict()
        self.assertNotIn("api_key", public)
        self.assertNotIn("sk-super-secret-one", json.dumps(public, ensure_ascii=False))
        self.assertTrue(public["api_key_configured"])
        self.assertTrue(public["api_key_masked"].endswith("-one"))
        self.assertIn("本周工作：", public["weekly_report_template"])
        self.assertIn("下周工作计划：", public["weekly_report_template"])
        self.assertTrue(public["analysis_instruction"])
        self.assertNotEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual("sk-super-secret-one", first.storage_dict()["api_key"])
        self.assertNotIn("sk-super-secret-one", repr(first))

    def test_config_validates_url_and_header(self):
        with self.assertRaises(ValidationError):
            LLMConfig(base_url="ftp://example.com")
        with self.assertRaises(ValidationError):
            LLMConfig(api_key_header="Authorization\r\nInjected")

    def test_protocol_endpoint_and_custom_api_key_header(self):
        config = LLMConfig(
            protocol=PROTOCOL_CHAT_COMPLETIONS,
            base_url="http://127.0.0.1:9999/v1/",
            api_key="local-key",
            api_key_header="X-API-Key",
            api_key_prefix="",
        )
        self.assertEqual("http://127.0.0.1:9999/v1/chat/completions", config.endpoint())
        self.assertEqual("local-key", config.request_headers()["X-API-Key"])

    def test_schema_is_strict_at_every_object_level(self):
        schema = report_json_schema()
        self.assertFalse(schema["additionalProperties"])
        completed_item = schema["properties"]["completed_items"]["items"]
        weekly = schema["properties"]["weekly_report"]
        next_week_item = weekly["properties"]["next_week"]["items"]
        self.assertFalse(completed_item["additionalProperties"])
        self.assertFalse(weekly["additionalProperties"])
        self.assertFalse(next_week_item["additionalProperties"])
        self.assertEqual(1, completed_item["properties"]["source_message_ids"]["minItems"])
        quote_item = completed_item["properties"]["evidence_quotes"]["items"]
        self.assertFalse(quote_item["additionalProperties"])
        self.assertIn("source_message_id", quote_item["properties"])
        self.assertNotIn("message_id", quote_item["properties"])
        self.assertEqual(160, quote_item["properties"]["quote"]["maxLength"])

        candidate_schema = candidate_json_schema()
        self.assertFalse(candidate_schema["additionalProperties"])
        candidate_item = candidate_schema["properties"]["items"]["items"]
        self.assertFalse(candidate_item["additionalProperties"])
        self.assertEqual(
            ["completed", "performed", "open_next"],
            candidate_item["properties"]["kind"]["enum"],
        )
        self.assertEqual(40, candidate_schema["properties"]["items"]["maxItems"])

    def test_evidence_quote_accepts_provider_message_id_alias_but_serialises_canonically(self):
        report = StructuredReport.model_validate(
            {
                "completed_items": [
                    {
                        "title": "部署完成",
                        "detail": "完成测试部署。",
                        "source_message_ids": ["m1"],
                        "evidence_quotes": [
                            {"message_id": "m1", "quote": "部署已经完成"}
                        ],
                    }
                ],
                "weekly_report": {"this_week": [], "next_week": []},
            }
        )

        dumped = report.model_dump(mode="json")
        evidence = dumped["completed_items"][0]["evidence_quotes"][0]
        self.assertEqual("m1", evidence["source_message_id"])
        self.assertNotIn("message_id", evidence)


class LLMEngineAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_candidate_contract_is_split_into_two_documents(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return responses_result(
                {
                    "items": [
                        {
                            "kind": "completed",
                            "title": "部署完成",
                            "detail": "完成部署。",
                            "source_message_ids": ["m1"],
                            "evidence_quotes": [
                                {
                                    "source_message_id": "m1",
                                    "quote": "部署已经完成",
                                }
                            ],
                        },
                        {
                            "kind": "performed",
                            "title": "接口延迟排查",
                            "detail": "排查接口延迟。",
                            "source_message_ids": ["m2"],
                            "evidence_quotes": [
                                {
                                    "source_message_id": "m2",
                                    "quote": "已排查接口延迟",
                                }
                            ],
                        },
                        {
                            "kind": "open_next",
                            "title": "接口待处理",
                            "detail": "下周继续处理。",
                            "source_message_ids": ["m3"],
                            "evidence_quotes": [
                                {
                                    "source_message_id": "m3",
                                    "quote": "接口尚未完成，下周继续处理",
                                }
                            ],
                        },
                    ]
                }
            )

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate(
            [
                message("m1", "部署已经完成。"),
                message("m2", "已排查接口延迟。", time="2026-08-20T11:00:00+08:00"),
                message(
                    "m3",
                    "接口尚未完成，下周继续处理。",
                    time="2026-08-20T12:00:00+08:00",
                ),
            ]
        )

        schema = captured["payload"]["text"]["format"]["schema"]
        self.assertIn("items", schema["properties"])
        self.assertNotIn("completed_items", schema["properties"])
        self.assertEqual(
            ["m1"],
            result["structured"]["completed_items"][0]["source_message_ids"],
        )
        this_week_ids = {
            source_id
            for item in result["structured"]["weekly_report"]["this_week"]
            for source_id in item["source_message_ids"]
        }
        self.assertEqual({"m1", "m2"}, this_week_ids)
        self.assertEqual(
            ["m3"],
            result["structured"]["weekly_report"]["next_week"][0][
                "source_message_ids"
            ],
        )

    async def test_native_completed_without_result_is_demoted_to_performed(self):
        value = {
            "items": [
                {
                    "kind": "completed",
                    "title": "接口延迟排查",
                    "detail": "排查接口延迟。",
                    "source_message_ids": ["m1"],
                    "evidence_quotes": [
                        {
                            "source_message_id": "m1",
                            "quote": "已排查接口延迟",
                        }
                    ],
                }
            ]
        }

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "已排查接口延迟。")])

        self.assertEqual([], result["structured"]["completed_items"])
        self.assertEqual(
            ["m1"],
            result["structured"]["weekly_report"]["this_week"][0][
                "source_message_ids"
            ],
        )
        self.assertIn(
            "unsupported_completed_demoted",
            result["model"]["output_normalizations"],
        )

    async def test_native_unsupported_completed_claim_is_removed(self):
        value = {
            "items": [
                {
                    "kind": "completed",
                    "title": "测试结论",
                    "detail": "确认测试结论。",
                    "source_message_ids": ["m1"],
                    "evidence_quotes": [
                        {
                            "source_message_id": "m1",
                            "quote": "请在周五前确认测试结论",
                        }
                    ],
                }
            ]
        }

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "请在周五前确认测试结论。")])

        self.assertEqual([], result["structured"]["completed_items"])
        self.assertEqual([], result["structured"]["weekly_report"]["this_week"])
        self.assertIn(
            "unsupported_completed_removed",
            result["model"]["output_normalizations"],
        )

    async def test_large_scope_prefilter_scans_all_and_sends_only_wide_candidates(self):
        observed_ids: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            data = json.loads(payload["input"].split("\n", 1)[1])
            rows = data["messages"]
            observed_ids.extend(model_row_id(row) for row in rows)
            return responses_result(
                structured_payload(
                    completed_ids=[model_row_id(rows[0])],
                    completed_quotes={model_row_id(rows[0]): "部署已经完成"},
                )
            )

        messages = [
            message(f"noise-{index:03d}", "今天天气晴朗。")
            for index in range(54)
        ]
        messages.append(message("work-001", "部署已经完成。"))
        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate(messages)

        self.assertEqual(["work-001"], observed_ids)
        self.assertTrue(result["model"]["candidate_prefilter_applied"])
        self.assertEqual(55, result["model"]["input_message_count"])
        self.assertEqual(1, result["model"]["candidate_message_count"])
        self.assertEqual(1, result["model"]["message_batch_count"])

    async def test_large_scope_prefilter_keeps_short_first_person_commitment(self):
        observed_ids: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            data = json.loads(payload["input"].split("\n", 1)[1])
            observed_ids.extend(model_row_id(row) for row in data["messages"])
            return responses_result({"items": []})

        messages = [
            message(f"noise-{index:03d}", "今天天气晴朗。", sender="李四")
            for index in range(54)
        ]
        messages.append(message("subject-short", "我来。", sender="张三"))
        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate(
            messages,
            report_subject_name="张三",
            only_subject_items=True,
        )

        self.assertEqual(["subject-short"], observed_ids)
        self.assertEqual(1, result["model"]["selected_subject_related_count"])

    async def test_responses_configuration_test_checks_schema_and_classification(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["authorization"] = request.headers.get("Authorization")
            captured["payload"] = json.loads(request.content)
            return responses_result(
                structured_payload(
                    completed_ids=["config-test-completed"],
                    unfinished_ids=["config-test-unfinished"],
                )
            )

        config = LLMConfig(api_key="sk-test", model="gpt-test", temperature=None)
        engine = LLMReportEngine(config, transport=httpx.MockTransport(handler))
        result = await engine.test_configuration()

        self.assertTrue(result["ok"])
        self.assertTrue(result["connection_validated"])
        self.assertTrue(result["structured_output_validated"])
        self.assertTrue(result["connection_ok"])
        self.assertTrue(result["model_ok"])
        self.assertTrue(result["structured_output_ok"])
        self.assertTrue(result["tested_at"].endswith("+00:00"))
        self.assertEqual("https://api.openai.com/v1/responses", captured["url"])
        self.assertEqual("Bearer sk-test", captured["authorization"])
        payload = captured["payload"]
        self.assertEqual("json_schema", payload["text"]["format"]["type"])
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertNotIn("temperature", payload)
        self.assertIn("合成测试", payload["instructions"])

    async def test_provider_rejected_temperature_is_automatically_omitted_and_retested(self):
        captured: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            captured.append(payload)
            if len(captured) == 1:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "type": "invalid_request_error",
                            "param": "temperature",
                            "message": (
                                "Error from provider: Unsupported parameter: "
                                "'temperature' is not supported with this model. "
                                "private prompt must never be surfaced"
                            ),
                        }
                    },
                )
            return responses_result(
                structured_payload(
                    completed_ids=["config-test-completed"],
                    unfinished_ids=["config-test-unfinished"],
                )
            )

        result = await LLMReportEngine(
            LLMConfig(
                api_key="sk-test",
                model="reasoning-model",
                temperature=0.1,
                max_retries=0,
            ),
            transport=httpx.MockTransport(handler),
        ).test_configuration()

        self.assertTrue(result["ok"])
        self.assertEqual(2, len(captured))
        self.assertEqual(0.1, captured[0]["temperature"])
        self.assertNotIn("temperature", captured[1])
        self.assertEqual(2, result["http_attempt_count"])
        self.assertEqual(1, result["compatibility_retry_count"])
        self.assertEqual(0, result["transient_retry_count"])
        self.assertEqual(["temperature"], result["omitted_optional_parameters"])
        self.assertIn("自动省略", result["message"])
        self.assertNotIn("private prompt", result["message"])
        self.assertNotIn("sk-private", result["message"])

    async def test_semantically_invalid_output_is_retried_once_with_safe_correction(self):
        payloads: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            payloads.append(payload)
            if len(payloads) == 1:
                return responses_result(
                    structured_payload(
                        completed_ids=["m1"],
                        completed_quotes={"m1": "正在讨论部署方案"},
                        completed_title="部署方案",
                    )
                )
            return responses_result(
                structured_payload(
                    completed_ids=["m1"],
                    completed_quotes={"m1": "部署已经完成"},
                    completed_title="部署完成",
                )
            )

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "正在讨论部署方案，随后部署已经完成。")])

        self.assertEqual(2, len(payloads))
        self.assertEqual(2, result["model"]["http_attempt_count"])
        self.assertEqual(1, result["model"]["validation_retry_count"])
        self.assertIn(
            "LLM_EVIDENCE_COMPLETION_UNSUPPORTED", payloads[1]["instructions"]
        )
        self.assertNotIn("正在讨论部署方案", payloads[1]["instructions"])

    async def test_temperature_compatibility_fallback_is_reused_for_later_batches(self):
        payloads: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            payloads.append(payload)
            if len(payloads) == 1:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": "Parameter temperature is not supported.",
                        }
                    },
                )
            input_data = json.loads(str(payload["input"]).split("\n", 1)[1])
            rows = input_data.get("messages") or []
            if rows:
                completed_ids = [model_row_id(row) for row in rows]
                completed_quotes = {
                    model_row_id(row): model_row_text(row)[:8] for row in rows
                }
            else:
                completed_ids = []
                completed_quotes = {}
                for candidate in input_data.get("candidate_reports") or []:
                    for item in candidate["items"]:
                        completed_ids.extend(item["source_message_ids"])
                        completed_quotes.update(
                            {
                                evidence["source_message_id"]: evidence["quote"]
                                for evidence in item["evidence_quotes"]
                            }
                        )
            return responses_result(
                structured_payload(
                    completed_ids=completed_ids,
                    completed_quotes=completed_quotes,
                    completed_title="完成事项",
                )
            )

        messages = [
            message("m1", "已经完成事项 A。" + "甲" * 1300),
            message("m2", "已经完成事项 B。" + "乙" * 1300),
        ]
        result = await LLMReportEngine(
            LLMConfig(max_input_chars=2000, temperature=0.2, max_retries=0),
            transport=httpx.MockTransport(handler),
        ).generate(messages)

        self.assertEqual(3, len(payloads))
        self.assertIn("temperature", payloads[0])
        self.assertTrue(all("temperature" not in payload for payload in payloads[1:]))
        self.assertEqual(1, result["model"]["compatibility_retry_count"])
        self.assertEqual(["temperature"], result["model"]["omitted_optional_parameters"])
        self.assertIn(
            "validated_map_union_without_reduce",
            result["model"]["output_normalizations"],
        )

    async def test_chat_completions_generation_and_business_markdown(self):
        captured: dict = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return chat_result(structured_payload(completed_ids=["m1"], unfinished_ids=["m2"]))

        config = LLMConfig(
            protocol=PROTOCOL_CHAT_COMPLETIONS,
            base_url="https://compatible.example/v1",
            model="custom-model",
            api_key="secret",
        )
        result = await LLMReportEngine(
            config, transport=httpx.MockTransport(handler)
        ).generate(
            [
                message("m1", "部署已经完成。"),
                message("m2", "接口尚未完成，下周继续处理。", time="2026-08-20T11:00:00+08:00"),
            ]
        )

        payload = captured["payload"]
        self.assertEqual("json_schema", payload["response_format"]["type"])
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertIn("不得补写", payload["messages"][0]["content"])
        self.assertIn("title/detail 只写整理后的业务内容", payload["messages"][0]["content"])
        self.assertIn("消息出处", payload["messages"][0]["content"])
        self.assertEqual(1, result["model"]["request_count"])
        self.assertEqual(
            {"completed_items_markdown", "weekly_report_markdown"},
            set(result["documents"]),
        )
        self.assertIn("本周已完成事项：", result["documents"]["completed_items_markdown"])
        self.assertNotIn("未完成事项：", result["documents"]["completed_items_markdown"])
        self.assertIn("本周工作：", result["documents"]["weekly_report_markdown"])
        self.assertIn("下周工作计划：", result["documents"]["weekly_report_markdown"])
        self.assertIn("1，部署完成：完成测试部署。", result["documents"]["weekly_report_markdown"])
        self.assertNotIn("会话：项目群（conv-a）", result["documents"]["weekly_report_markdown"])
        self.assertNotIn("发送人：张三", result["documents"]["weekly_report_markdown"])
        self.assertNotIn("消息 ID：m1", result["documents"]["weekly_report_markdown"])
        self.assertNotIn("引用原文：部署已经完成", result["documents"]["weekly_report_markdown"])
        internal_item = result["structured"]["completed_items"][0]
        self.assertEqual(["m1"], internal_item["source_message_ids"])
        self.assertEqual("部署已经完成", internal_item["evidence_quotes"][0]["quote"])

    async def test_known_reasoning_and_module_wrappers_are_normalised_then_fully_validated(self):
        value = json.loads(
            json.dumps(
                structured_payload(
                    completed_ids=["config-test-completed"],
                    unfinished_ids=["config-test-unfinished"],
                ),
                ensure_ascii=False,
            )
        )
        for item in value["completed_items"]:
            item["module"] = "测试模块"
            for quote in item["evidence_quotes"]:
                quote["message_id"] = quote.pop("source_message_id")
        for section in ("this_week", "next_week"):
            items = value["weekly_report"][section]
            for item in items:
                for quote in item["evidence_quotes"]:
                    quote["message_id"] = quote.pop("source_message_id")
            value["weekly_report"][section] = [
                {"module": "测试模块", "items": items}
            ]
        wrapped = (
            "<think>synthetic reasoning that must not be published</think>\n"
            "```json\n"
            + json.dumps(value, ensure_ascii=False)
            + "\n```"
        )

        async def handler(_request: httpx.Request) -> httpx.Response:
            return chat_text_result(wrapped)

        result = await LLMReportEngine(
            LLMConfig(protocol=PROTOCOL_CHAT_COMPLETIONS),
            transport=httpx.MockTransport(handler),
        ).test_configuration()

        self.assertTrue(result["ok"])
        self.assertEqual(
            {
                "evidence_message_id_alias",
                "json_fence_removed",
                "legacy_full_report_converted",
                "module_field_ignored",
                "module_group_flattened",
                "reasoning_wrapper_removed",
            },
            set(result["output_normalizations"]),
        )
        self.assertNotIn("synthetic reasoning", json.dumps(result, ensure_ascii=False))

    async def test_unknown_provider_item_field_still_fails_closed(self):
        value = structured_payload(completed_ids=["m1"])
        value["completed_items"][0]["unverified_claim"] = "must not be accepted"

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        with self.assertRaises(LLMOutputError) as context:
            await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(handler)
            ).generate([message("m1", "部署已经完成。")])
        self.assertEqual("LLM_OUTPUT_SCHEMA_INVALID", context.exception.code)

    async def test_unknown_source_id_fails_closed(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(structured_payload(completed_ids=["invented-id"]))

        engine = LLMReportEngine(LLMConfig(), transport=httpx.MockTransport(handler))
        with self.assertRaisesRegex(LLMOutputError, "不存在的消息 ID") as context:
            await engine.generate([message("m1", "部署已经完成。")])
        self.assertEqual("LLM_EVIDENCE_ID_UNKNOWN", context.exception.code)

    async def test_non_verbatim_evidence_quote_fails_closed(self):
        value = structured_payload(completed_ids=["m1"])
        value["completed_items"][0]["evidence_quotes"][0]["quote"] = "不存在于消息中的原文"
        value["weekly_report"]["this_week"][0]["evidence_quotes"][0]["quote"] = (
            "不存在于消息中的原文"
        )

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        with self.assertRaises(LLMOutputError) as context:
            await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(handler)
            ).generate([message("m1", "部署已经完成。")])
        self.assertEqual("LLM_EVIDENCE_QUOTE_MISMATCH", context.exception.code)

    async def test_title_from_an_uncited_message_is_replaced_with_evidence(self):
        calls = 0
        value = structured_payload(
            completed_ids=["m1"],
            completed_quotes={"m1": "测试结论已经完成"},
            completed_title="边界用例补充",
        )

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "测试结论已经完成。")])

        completed = result["structured"]["completed_items"][0]
        self.assertEqual("测试结论已经完成", completed["title"])
        self.assertNotIn("边界用例补充", result["documents"]["completed_items_markdown"])
        self.assertIn(
            "ungrounded_title_replaced_with_evidence",
            result["model"]["output_normalizations"],
        )
        self.assertEqual(1, calls)
        self.assertEqual(0, result["model"]["validation_retry_count"])

    async def test_real_restart_wording_keeps_item_and_repairs_summary_title(self):
        value = {
            "items": [
                {
                    "kind": "performed",
                    "title": "主机恢复结果",
                    "detail": "已执行主机重启，登录结果仍需观察。",
                    "source_message_ids": ["m1", "m2"],
                    "evidence_quotes": [
                        {
                            "source_message_id": "m1",
                            "quote": "十分钟后重新登录一下，我重启了",
                        },
                        {
                            "source_message_id": "m2",
                            "quote": "演示主机需要重启一下，有个测试客户端还是登录不了",
                        },
                    ],
                }
            ]
        }

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate(
            [
                message("m1", "@邱金玉 十分钟后重新登录一下，我重启了"),
                message("m2", "@张三 演示主机需要重启一下，有个测试客户端还是登录不了"),
            ]
        )

        this_week = result["structured"]["weekly_report"]["this_week"]
        self.assertEqual(1, len(this_week))
        self.assertEqual("十分钟后重新登录一下，我重启了", this_week[0]["title"])
        self.assertEqual(["m1", "m2"], this_week[0]["source_message_ids"])
        self.assertEqual(
            ["十分钟后重新登录一下，我重启了", "演示主机需要重启一下，有个测试客户端还是登录不了"],
            [evidence["quote"] for evidence in this_week[0]["evidence_quotes"]],
        )

    async def test_completed_item_without_completion_words_fails_closed(self):
        value = structured_payload(
            completed_ids=["m1"],
            completed_quotes={"m1": "正在讨论部署方案"},
            completed_title="部署方案讨论",
        )

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        with self.assertRaises(LLMOutputError) as context:
            await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(handler)
            ).generate([message("m1", "正在讨论部署方案。")])
        self.assertEqual("LLM_EVIDENCE_COMPLETION_UNSUPPORTED", context.exception.code)

    async def test_negated_conditional_question_and_rolled_back_completion_fail_closed(self):
        cases = (
            ("当前尚未通过验收。", "通过验收", "验收"),
            ("待完成并上线后再通知。", "完成并上线", "上线"),
            ("如果修复了再发布。", "如果修复了", "修复"),
            ("部署已经完成了吗？", "已经完成", "部署"),
            ("服务已完成，但随后回滚失败需要重新处理。", "服务已完成", "服务"),
        )
        for source, quote, title in cases:
            with self.subTest(source=source):
                value = structured_payload(
                    completed_ids=["m1"],
                    completed_quotes={"m1": quote},
                    completed_title=title,
                )

                async def handler(_request: httpx.Request) -> httpx.Response:
                    return responses_result(value)

                with self.assertRaises(LLMOutputError) as context:
                    await LLMReportEngine(
                        LLMConfig(), transport=httpx.MockTransport(handler)
                    ).generate([message("m1", source)])
                self.assertEqual(
                    "LLM_EVIDENCE_COMPLETION_UNSUPPORTED", context.exception.code
                )

    async def test_explicit_result_phrases_are_accepted_as_completion_evidence(self):
        cases = (
            ("通过验收", "数据核对已经完成并通过验收。", "数据核对"),
            ("并完成材料梳理", "已推进培训准备并完成材料梳理。", "培训材料梳理"),
        )
        for quote, source, title in cases:
            with self.subTest(quote=quote):
                value = structured_payload(
                    completed_ids=["m1"],
                    completed_quotes={"m1": quote},
                    completed_title=title,
                )

                async def handler(_request: httpx.Request) -> httpx.Response:
                    return responses_result(value)

                result = await LLMReportEngine(
                    LLMConfig(), transport=httpx.MockTransport(handler)
                ).generate([message("m1", source)])
                self.assertEqual(
                    ["m1"],
                    result["structured"]["completed_items"][0][
                        "source_message_ids"
                    ],
                )

    async def test_next_week_item_without_plan_or_open_work_is_removed(self):
        value = structured_payload(
            unfinished_ids=["m1"],
            unfinished_quotes={"m1": "正在排查接口延迟"},
        )

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "正在排查接口延迟。")])
        self.assertEqual([], result["structured"]["weekly_report"]["next_week"])
        self.assertIn(
            "unsupported_next_week_items_removed",
            result["model"]["output_normalizations"],
        )

    async def test_cancelled_questioned_or_already_resolved_plan_is_removed(self):
        cases = (
            ("原计划下周处理接口，现已取消。", "下周处理接口"),
            ("下周是否处理接口？", "下周是否处理接口"),
            ("接口尚未完成，但今天已经完成并上线。", "接口尚未完成"),
        )
        for source, quote in cases:
            with self.subTest(source=source):
                value = structured_payload(
                    unfinished_ids=["m1"],
                    unfinished_quotes={"m1": quote},
                )

                async def handler(_request: httpx.Request) -> httpx.Response:
                    return responses_result(value)

                result = await LLMReportEngine(
                    LLMConfig(), transport=httpx.MockTransport(handler)
                ).generate([message("m1", source)])
                self.assertEqual(
                    [], result["structured"]["weekly_report"]["next_week"]
                )

    async def test_requested_action_is_removed_from_performed_this_week(self):
        item = {
            "title": "接口排查",
            "detail": "推进接口排查。",
            "source_message_ids": ["m1"],
            "evidence_quotes": [
                {"source_message_id": "m1", "quote": "请继续推进接口排查"}
            ],
        }
        value = {
            "completed_items": [],
            "weekly_report": {"this_week": [item], "next_week": []},
        }

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "请继续推进接口排查。")])
        self.assertEqual([], result["structured"]["weekly_report"]["this_week"])
        self.assertIn(
            "unsupported_this_week_items_removed",
            result["model"]["output_normalizations"],
        )

    async def test_validated_completed_item_is_deterministically_added_to_this_week(self):
        value = structured_payload(completed_ids=["m1"])
        value["weekly_report"]["this_week"] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "部署已经完成。")])
        self.assertEqual(
            ["m1"],
            result["structured"]["weekly_report"]["this_week"][0][
                "source_message_ids"
            ],
        )
        self.assertIn("部署完成", result["documents"]["weekly_report_markdown"])

    async def test_empty_source_id_is_rejected_by_schema(self):
        value = structured_payload(completed_ids=["m1"])
        value["completed_items"][0]["source_message_ids"] = []

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        with self.assertRaises(LLMOutputError) as context:
            await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(handler)
            ).generate([message("m1", "部署已经完成。")])
        self.assertEqual("LLM_OUTPUT_SCHEMA_INVALID", context.exception.code)

    async def test_stable_map_batches_use_lossless_local_union_when_it_fits(self):
        seen_stages: list[tuple[str, list[str]]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            user_input = payload["input"]
            data = json.loads(user_input.split("\n", 1)[1])
            if "messages" in data:
                ids = [model_row_id(item) for item in data["messages"]]
                quotes = {
                    model_row_id(item): model_row_text(item)[:80]
                    for item in data["messages"]
                }
                seen_stages.append(("map", ids))
                return responses_result(
                    structured_payload(
                        completed_ids=ids,
                        completed_quotes=quotes,
                        completed_title="完成事项",
                    )
                )
            ids: list[str] = []
            quotes: dict[str, str] = {}
            for candidate in data["candidate_reports"]:
                for item in candidate["items"]:
                    ids.extend(item["source_message_ids"])
                    quotes.update(
                        {
                            evidence["source_message_id"]: evidence["quote"]
                            for evidence in item["evidence_quotes"]
                        }
                    )
            seen_stages.append(("reduce", ids))
            return responses_result(
                structured_payload(
                    completed_ids=ids,
                    completed_quotes=quotes,
                    completed_title="完成事项",
                )
            )

        config = LLMConfig(max_input_chars=2000, temperature=None)
        messages = [
            message("m1", "已经完成事项 A。" + "甲" * 1300),
            message("m2", "已经完成事项 B。" + "乙" * 1300, time="2026-08-20T11:00:00+08:00"),
        ]
        result = await LLMReportEngine(
            config, transport=httpx.MockTransport(handler)
        ).generate(messages)

        self.assertEqual([("map", ["m1"]), ("map", ["m2"])], seen_stages)
        self.assertEqual(2, result["model"]["message_batch_count"])
        self.assertEqual(2, result["model"]["request_count"])
        self.assertEqual(
            [["m1"], ["m2"]],
            [
                item["source_message_ids"]
                for item in result["structured"]["completed_items"]
            ],
        )
        self.assertIn(
            "validated_map_union_without_reduce",
            result["model"]["output_normalizations"],
        )

    async def test_identical_broadcasts_are_merged_without_losing_evidence(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            data = json.loads(payload["input"].split("\n", 1)[1])
            row = data["messages"][0]
            return responses_result(
                structured_payload(
                    completed_ids=[model_row_id(row)],
                    completed_quotes={
                        model_row_id(row): "企微后台升级已完成"
                    },
                    completed_title="企微后台升级",
                )
            )

        result = await LLMReportEngine(
            LLMConfig(max_messages_per_batch=1),
            transport=httpx.MockTransport(handler),
        ).generate(
            [
                message("m1", "企微后台升级已完成。"),
                message("m2", "企微后台升级已完成。", time="2026-08-20T11:00:00+08:00"),
            ]
        )

        completed = result["structured"]["completed_items"]
        self.assertEqual(1, len(completed))
        self.assertEqual(["m1", "m2"], completed[0]["source_message_ids"])
        self.assertEqual(
            ["m1", "m2"],
            [item["source_message_id"] for item in completed[0]["evidence_quotes"]],
        )
        self.assertIn(
            "identical_evidence_items_merged",
            result["model"]["output_normalizations"],
        )

    async def test_long_opaque_ids_are_aliased_then_losslessly_restored(self):
        observed_ids: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            data = json.loads(payload["input"].split("\n", 1)[1])
            row = data["messages"][0]
            observed_ids.append(model_row_id(row))
            return responses_result(
                structured_payload(
                    completed_ids=[model_row_id(row)],
                    completed_quotes={model_row_id(row): "部署已经完成"},
                    completed_title="部署完成",
                )
            )

        original_id = "opaque-message-" + "x" * 64
        original = message(original_id, "部署已经完成。")
        original["conversation_id"] = "opaque-conversation-" + "y" * 64
        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([original])

        self.assertEqual(["m000001"], observed_ids)
        self.assertEqual(
            [original_id],
            result["structured"]["completed_items"][0]["source_message_ids"],
        )
        self.assertEqual(
            original_id,
            result["structured"]["completed_items"][0]["evidence_quotes"][0][
                "source_message_id"
            ],
        )
        self.assertTrue(result["model"]["input_id_aliasing"])

    async def test_validated_batch_checkpoint_is_reused_without_network(self):
        calls = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return responses_result(structured_payload(completed_ids=["m1"]))

        with tempfile.TemporaryDirectory() as temporary:
            config = LLMConfig()
            first = await LLMReportEngine(
                config, transport=httpx.MockTransport(handler)
            ).generate(
                [message("m1", "部署已经完成。")],
                checkpoint_dir=temporary,
            )
            second = await LLMReportEngine(
                config, transport=httpx.MockTransport(handler)
            ).generate(
                [message("m1", "部署已经完成。")],
                checkpoint_dir=temporary,
            )

        self.assertEqual(1, calls)
        self.assertEqual(1, first["model"]["checkpoint_write_count"])
        self.assertEqual(1, second["model"]["checkpoint_hit_count"])
        self.assertEqual(0, second["model"]["request_count"])
        self.assertEqual(first["structured"], second["structured"])

    async def test_v12_validated_checkpoint_is_reused_after_multi_quote_upgrade(self):
        calls = 0

        async def first_handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return responses_result(structured_payload(completed_ids=["m1"]))

        original_engine_version = llm_engine_module.ENGINE_VERSION
        with tempfile.TemporaryDirectory() as temporary:
            try:
                llm_engine_module.ENGINE_VERSION = "llm-evidence-v12-provider-output"
                await LLMReportEngine(
                    LLMConfig(), transport=httpx.MockTransport(first_handler)
                ).generate(
                    [message("m1", "部署已经完成。")],
                    checkpoint_dir=temporary,
                )
            finally:
                llm_engine_module.ENGINE_VERSION = original_engine_version

            async def no_network(_request: httpx.Request) -> httpx.Response:
                raise AssertionError("compatible validated checkpoint should be reused")

            resumed = await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(no_network)
            ).generate(
                [message("m1", "部署已经完成。")],
                checkpoint_dir=temporary,
            )

        self.assertEqual(1, calls)
        self.assertEqual(1, resumed["model"]["checkpoint_hit_count"])
        self.assertEqual(0, resumed["model"]["request_count"])
        self.assertIn(
            "compatible_legacy_checkpoint_reused",
            resumed["model"]["output_normalizations"],
        )

    async def test_reduce_that_silently_drops_candidate_evidence_fails_closed(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            data = json.loads(payload["input"].split("\n", 1)[1])
            if "messages" in data:
                row = data["messages"][0]
                source_id = model_row_id(row)
                module = "A" if source_id == "m1" else "B"
                return responses_result(
                    {
                        "items": [
                            {
                                "kind": "completed",
                                "title": f"模块{module}{index:02d}",
                                "detail": f"模块{module}{index:02d}已经完成。",
                                "source_message_ids": [source_id],
                                "evidence_quotes": [
                                    {
                                        "source_message_id": source_id,
                                        "quote": f"模块{module}{index:02d}已经完成",
                                    }
                                ],
                            }
                            for index in range(21)
                        ]
                    }
                )
            return responses_result(
                structured_payload(
                    completed_ids=["m1"],
                    completed_quotes={"m1": "模块A00已经完成"},
                    completed_title="模块A00",
                )
            )

        with self.assertRaises(LLMOutputError) as context:
            await LLMReportEngine(
                LLMConfig(max_input_chars=20_000, max_messages_per_batch=1),
                transport=httpx.MockTransport(handler),
            ).generate(
                [
                    message(
                        "m1",
                        "；".join(f"模块A{index:02d}已经完成" for index in range(21)),
                    ),
                    message(
                        "m2",
                        "；".join(f"模块B{index:02d}已经完成" for index in range(21)),
                    ),
                ]
            )
        self.assertEqual("LLM_REDUCE_EVIDENCE_DROPPED", context.exception.code)

    async def test_generation_retries_one_transient_http_400(self):
        calls = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(400, json={"error": {"message": "temporary"}})
            return responses_result(structured_payload(completed_ids=["m1"]))

        result = await LLMReportEngine(
            LLMConfig(max_retries=1), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "部署已经完成。")])

        self.assertEqual(2, calls)
        self.assertEqual(1, result["model"]["transient_retry_count"])
        self.assertEqual(["m1"], result["structured"]["completed_items"][0]["source_message_ids"])

    async def test_conflicting_duplicate_message_id_is_rejected_before_network(self):
        call_count = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return responses_result(structured_payload())

        engine = LLMReportEngine(LLMConfig(), transport=httpx.MockTransport(handler))
        with self.assertRaises(LLMConfigurationError) as context:
            await engine.generate(
                [message("same", "内容 A"), message("same", "内容 B")]
            )
        self.assertEqual("LLM_INPUT_MESSAGE_ID_CONFLICT", context.exception.code)
        self.assertEqual(0, call_count)

    async def test_http_error_does_not_echo_response_or_api_key(self):
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                text="bad sk-do-not-leak and private chat text",
            )

        engine = LLMReportEngine(
            LLMConfig(api_key="sk-do-not-leak"),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(LLMTransportError) as context:
            await engine.generate([message("m1", "private chat text")])
        rendered = str(context.exception)
        self.assertIn("HTTP 401", rendered)
        self.assertNotIn("sk-do-not-leak", rendered)
        self.assertNotIn("private chat text", rendered)

    async def test_transient_protocol_disconnect_is_retried_and_audited(self):
        call_count = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.RemoteProtocolError("temporary disconnect")
            return responses_result(structured_payload(completed_ids=["m1"]))

        result = await LLMReportEngine(
            LLMConfig(max_retries=1), transport=httpx.MockTransport(handler)
        ).generate([message("m1", "部署已经完成。")])
        self.assertEqual(2, call_count)
        self.assertEqual(1, result["model"]["request_count"])
        self.assertEqual(2, result["model"]["http_attempt_count"])
        self.assertEqual(1, result["model"]["retry_count"])

    async def test_empty_model_content_is_validation_retried_once(self):
        call_count = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": ""},
                                "finish_reason": "length",
                            }
                        ]
                    },
                )
            return chat_result(structured_payload(completed_ids=["m1"]))

        result = await LLMReportEngine(
            LLMConfig(protocol=PROTOCOL_CHAT_COMPLETIONS),
            transport=httpx.MockTransport(handler),
        ).generate([message("m1", "部署已经完成。")])

        self.assertEqual(2, call_count)
        self.assertEqual(1, result["model"]["validation_retry_count"])

    async def test_slow_multi_message_batch_adaptively_splits_to_single_messages(self):
        call_sizes: list[int] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            data = json.loads(payload["input"].split("\n", 1)[1])
            rows = data["messages"]
            call_sizes.append(len(rows))
            if len(rows) > 1:
                raise httpx.ReadTimeout("synthetic slow batch")
            row = rows[0]
            message_id = model_row_id(row)
            quote = model_row_text(row)[:8]
            return responses_result(
                structured_payload(
                    completed_ids=[message_id],
                    completed_quotes={message_id: quote},
                    completed_title="完成事项",
                )
            )

        result = await LLMReportEngine(
            LLMConfig(max_messages_per_batch=5, max_retries=1),
            transport=httpx.MockTransport(handler),
        ).generate(
            [
                message("m1", "已经完成事项 A。"),
                message("m2", "已经完成事项 B。"),
            ]
        )

        self.assertEqual([2, 1, 1], call_sizes)
        self.assertEqual(1, result["model"]["adaptive_batch_split_count"])
        self.assertEqual(3, result["model"]["http_attempt_count"])
        completed_ids = {
            source_id
            for item in result["structured"]["completed_items"]
            for source_id in item["source_message_ids"]
        }
        self.assertEqual({"m1", "m2"}, completed_ids)

    async def test_restart_rebuilds_adaptive_parent_from_child_checkpoints(self):
        first_call_sizes: list[int] = []

        async def first_handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            data = json.loads(payload["input"].split("\n", 1)[1])
            rows = data["messages"]
            first_call_sizes.append(len(rows))
            if len(rows) > 1:
                raise httpx.ReadTimeout("synthetic slow batch")
            row = rows[0]
            message_id = model_row_id(row)
            return responses_result(
                structured_payload(
                    completed_ids=[message_id],
                    completed_quotes={message_id: model_row_text(row)[:8]},
                    completed_title="完成事项",
                )
            )

        source_messages = [
            message("m1", "已经完成事项 A。"),
            message("m2", "已经完成事项 B。"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            await LLMReportEngine(
                LLMConfig(max_messages_per_batch=5, max_retries=1),
                transport=httpx.MockTransport(first_handler),
            ).generate(source_messages, checkpoint_dir=checkpoint_dir)

            parent_checkpoints: list[Path] = []
            for checkpoint in checkpoint_dir.glob("*.json"):
                envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
                ids = {
                    source_id
                    for item in envelope["report"]["items"]
                    for source_id in item["source_message_ids"]
                }
                if ids == {"m1", "m2"}:
                    parent_checkpoints.append(checkpoint)
            self.assertEqual(1, len(parent_checkpoints))
            parent_checkpoints[0].unlink()

            second_call_count = 0

            async def second_handler(_request: httpx.Request) -> httpx.Response:
                nonlocal second_call_count
                second_call_count += 1
                raise AssertionError("restart must not call the model")

            resumed = await LLMReportEngine(
                LLMConfig(max_messages_per_batch=5, max_retries=1),
                transport=httpx.MockTransport(second_handler),
            ).generate(source_messages, checkpoint_dir=checkpoint_dir)

        self.assertEqual([2, 1, 1], first_call_sizes)
        self.assertEqual(0, second_call_count)
        self.assertGreaterEqual(resumed["model"]["checkpoint_hit_count"], 2)
        self.assertEqual(1, resumed["model"]["checkpoint_write_count"])

    async def test_subject_only_mode_keeps_wang_qi_and_removes_other_people_work(self):
        captured: dict = {}
        value = {
            "items": [
                {
                    "kind": "completed",
                    "title": "接口排查已经完成",
                    "detail": "张三完成接口排查。",
                    "source_message_ids": ["m-wang"],
                    "evidence_quotes": [
                        {
                            "source_message_id": "m-wang",
                            "quote": "接口排查已经完成",
                        }
                    ],
                },
                {
                    "kind": "completed",
                    "title": "数据清理已经完成",
                    "detail": "李四完成数据清理。",
                    "source_message_ids": ["m-li"],
                    "evidence_quotes": [
                        {
                            "source_message_id": "m-li",
                            "quote": "数据清理已经完成",
                        }
                    ],
                },
                {
                    "kind": "open_next",
                    "title": "接口尚未完成",
                    "detail": "明确分配给张三，下周继续处理。",
                    "source_message_ids": ["m-assigned"],
                    "evidence_quotes": [
                        {
                            "source_message_id": "m-assigned",
                            "quote": "@张三 接口尚未完成，下周继续处理",
                        }
                    ],
                },
            ]
        }

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate(
            [
                message("m-wang", "接口排查已经完成。", sender="张三"),
                message("m-li", "数据清理已经完成。", sender="李四"),
                message(
                    "m-assigned",
                    "@张三 接口尚未完成，下周继续处理。",
                    sender="李四",
                ),
            ],
            report_subject_name="张三",
            only_subject_items=True,
        )

        completed_ids = {
            source_id
            for item in result["structured"]["completed_items"]
            for source_id in item["source_message_ids"]
        }
        next_week_ids = {
            source_id
            for item in result["structured"]["weekly_report"]["next_week"]
            for source_id in item["source_message_ids"]
        }
        self.assertEqual({"m-wang"}, completed_ids)
        self.assertEqual({"m-assigned"}, next_week_ids)
        self.assertIn(
            "non_subject_completed_removed",
            result["model"]["output_normalizations"],
        )
        self.assertEqual("张三", result["model"]["report_subject"]["name"])
        self.assertEqual(
            1, result["model"]["report_subject"]["direct_sender_message_count"]
        )
        self.assertEqual(
            1, result["model"]["report_subject"]["explicit_mention_message_count"]
        )
        self.assertIn("报告本人：张三", result["documents"]["weekly_report_markdown"])
        self.assertIn("其他发送人的第一人称", captured["payload"]["instructions"])

    async def test_subject_only_mode_accepts_explicit_third_party_attribution(self):
        value = {
            "items": [
                {
                    "kind": "completed",
                    "title": "张三已经完成部署",
                    "detail": "张三完成部署。",
                    "source_message_ids": ["m1"],
                    "evidence_quotes": [
                        {
                            "source_message_id": "m1",
                            "quote": "张三已经完成部署",
                        }
                    ],
                }
            ]
        }

        async def handler(_request: httpx.Request) -> httpx.Response:
            return responses_result(value)

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate(
            [message("m1", "张三已经完成部署。", sender="李四")],
            report_subject_name="张三",
            only_subject_items=True,
        )
        self.assertEqual(
            ["m1"],
            result["structured"]["completed_items"][0]["source_message_ids"],
        )

    async def test_subject_only_mode_fails_closed_when_identity_is_absent(self):
        call_count = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return responses_result(structured_payload())

        with self.assertRaises(LLMConfigurationError) as raised:
            await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(handler)
            ).generate(
                [message("m1", "李四已经完成部署。", sender="李四")],
                report_subject_name="张三",
                only_subject_items=True,
            )
        self.assertEqual("LLM_REPORT_SUBJECT_NOT_FOUND", raised.exception.code)
        self.assertEqual(0, call_count)

    async def test_subject_identity_changes_checkpoint_key(self):
        call_count = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return responses_result(
                {
                    "items": [
                        {
                            "kind": "completed",
                            "title": "部署已经完成",
                            "detail": "完成部署。",
                            "source_message_ids": ["m1"],
                            "evidence_quotes": [
                                {
                                    "source_message_id": "m1",
                                    "quote": "部署已经完成",
                                }
                            ],
                        }
                    ]
                }
            )

        source = [message("m1", "部署已经完成。", sender="张三")]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(handler)
            ).generate(
                source,
                report_subject_name="张三",
                only_subject_items=True,
                checkpoint_dir=checkpoint_dir,
            )
            await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(handler)
            ).generate(
                source,
                report_subject_name="李四",
                only_subject_items=False,
                checkpoint_dir=checkpoint_dir,
            )
        self.assertEqual(2, call_count)

    async def test_empty_input_returns_empty_documents_without_network(self):
        call_count = 0

        async def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return responses_result(structured_payload())

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([])
        self.assertEqual(0, call_count)
        self.assertEqual(0, result["model"]["request_count"])
        self.assertIn("暂无有明确消息依据", result["documents"]["weekly_report_markdown"])


class RenderingTests(unittest.TestCase):
    def test_renderers_use_prior_gpt_headings_and_chinese_numbering(self):
        report = StructuredReport.model_validate(
            structured_payload(completed_ids=["m1"], unfinished_ids=["m2"])
        )

        completed = render_completed_items(report)
        weekly = render_weekly_report(report)

        self.assertTrue(completed.startswith("本周已完成事项："))
        self.assertIn("1，部署完成：完成测试部署。", completed)
        self.assertNotIn("未完成事项：", completed)
        self.assertEqual(1, weekly.count("本周工作："))
        self.assertEqual(1, weekly.count("下周工作计划："))
        self.assertNotIn("message_id", weekly)

    def test_extra_model_fields_are_rejected(self):
        value = structured_payload(completed_ids=["m1"])
        value["unexpected"] = "not allowed"
        with self.assertRaises(ValidationError):
            StructuredReport.model_validate(value)


if __name__ == "__main__":
    unittest.main()
