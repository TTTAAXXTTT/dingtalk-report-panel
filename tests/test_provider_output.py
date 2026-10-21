"""Regression checks based on official DeepSeek wire responses."""
import json
import unittest

import httpx

from app.llm_engine import (
    LLMConfig, LLMOutputError, LLMReportEngine,
    PROTOCOL_CHAT_COMPLETIONS, PROTOCOL_RESPONSES,
    _extract_response_text, _payload_for_protocol,
    candidate_json_schema, report_json_schema,
)


class ProviderOutputTests(unittest.TestCase):
    def test_responses_reads_only_assistant_output_text(self):
        body = {
            "status": "completed",
            "output": [
                {"type": "reasoning", "content": [
                    {"type": "reasoning_text", "text": "analysis that is not JSON"}]},
                {"type": "function_call", "content": [
                    {"type": "output_text", "text": "tool text"}]},
                {"type": "message", "role": "user", "content": [
                    {"type": "output_text", "text": "echoed input"}]},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "reasoning_text", "text": "another reasoning field"},
                    {"type": "output_text", "text": '{"items":'},
                    {"type": "output_text", "text": '[]}'},
                ]},
            ],
        }
        self.assertEqual('{"items":[]}', _extract_response_text(body, PROTOCOL_RESPONSES))

    def test_reasoning_only_is_never_accepted_even_if_it_contains_valid_json(self):
        body = {"output": [{"type": "reasoning", "content": [
            {"type": "reasoning_text", "text": '{"items":[]}'}]}]}
        with self.assertRaises(LLMOutputError) as caught:
            _extract_response_text(body, PROTOCOL_RESPONSES)
        self.assertEqual("LLM_OUTPUT_TEXT_MISSING", caught.exception.code)

    def test_responses_status_is_checked_before_accepting_valid_json(self):
        for status, code in (
            ("incomplete", "LLM_OUTPUT_TRUNCATED"),
            ("failed", "LLM_RESPONSE_FAILED"),
            ("cancelled", "LLM_RESPONSE_FAILED"),
            ("in_progress", "LLM_RESPONSE_NOT_COMPLETED"),
        ):
            with self.subTest(status=status), self.assertRaises(LLMOutputError) as caught:
                _extract_response_text(
                    {"status": status, "output_text": '{"items":[]}'}, PROTOCOL_RESPONSES)
            self.assertEqual(code, caught.exception.code)

    def test_chat_truncated_json_is_not_accepted(self):
        with self.assertRaises(LLMOutputError) as caught:
            _extract_response_text({"choices": [{
                "finish_reason": "length",
                "message": {"content": '{"items":[]}'},
            }]}, PROTOCOL_CHAT_COMPLETIONS)
        self.assertEqual("LLM_OUTPUT_TRUNCATED", caught.exception.code)

    def test_refusal_is_not_used_as_report_content(self):
        with self.assertRaises(LLMOutputError) as caught:
            _extract_response_text({"output": [{
                "type": "message", "role": "assistant", "content": [
                    {"type": "refusal", "refusal": "private refusal explanation"}],
            }]}, PROTOCOL_RESPONSES)
        self.assertEqual("LLM_OUTPUT_REFUSED", caught.exception.code)
        self.assertNotIn("private refusal explanation", str(caught.exception))

    def test_official_chat_includes_full_local_contract_in_json_mode(self):
        for base in ("https://api.deepseek.com", "https://api.deepseek.com/v1"):
            for contract, schema in (("report", report_json_schema()),
                                     ("candidates", candidate_json_schema())):
                with self.subTest(base=base, contract=contract):
                    payload = _payload_for_protocol(
                        LLMConfig(base_url=base, model="deepseek-flash",
                                  protocol=PROTOCOL_CHAT_COMPLETIONS),
                        system_prompt="只返回 JSON", user_input="synthetic data",
                        output_contract=contract)
                    self.assertEqual({"type": "json_object"}, payload["response_format"])
                    self.assertIn(
                        json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
                        payload["messages"][0]["content"])

    def test_gateway_and_responses_keep_native_json_schema(self):
        for base, protocol in (
            ("https://opencode.ai/zen/go/v1", PROTOCOL_CHAT_COMPLETIONS),
            ("https://api.deepseek.com.example.test", PROTOCOL_CHAT_COMPLETIONS),
            ("https://api.deepseek.com", PROTOCOL_RESPONSES),
        ):
            with self.subTest(base=base, protocol=protocol):
                payload = _payload_for_protocol(
                    LLMConfig(base_url=base, protocol=protocol),
                    system_prompt="JSON", user_input="synthetic data")
                actual = payload.get("response_format") or payload["text"]["format"]
                self.assertEqual("json_schema", actual["type"])


class ProviderValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_distinct_quotes_from_one_message_are_preserved(self):
        payload = {"items": [{
            "title": "报表字段修复",
            "detail": "完成报表字段修复并通过验证。",
            "kind": "completed",
            "source_message_ids": ["m1"],
            "evidence_quotes": [
                {"source_message_id": "m1", "quote": "报表字段修复已经完成"},
                {"source_message_id": "m1", "quote": "报表字段验证已经通过"},
            ],
        }]}

        def handler(_request):
            return httpx.Response(200, json={
                "status": "completed", "output": [{
                    "type": "message", "role": "assistant", "content": [{
                        "type": "output_text", "text": json.dumps(payload),
                    }],
                }],
            })

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([{
            "message_id": "m1", "conversation_id": "test-c", "sender": "张三",
            "time": "2026-09-07T09:00:00+08:00",
            "text": "报表字段修复已经完成，报表字段验证已经通过。",
        }])
        item = result["structured"]["completed_items"][0]
        self.assertEqual(2, len(item["evidence_quotes"]))
        self.assertEqual(0, result["model"]["validation_retry_count"])

    async def test_exact_duplicate_quote_is_removed_without_losing_evidence(self):
        evidence = {"source_message_id": "m1", "quote": "部署已经完成"}
        payload = {"items": [{
            "title": "部署完成", "detail": "部署已经完成。",
            "kind": "completed", "source_message_ids": ["m1"],
            "evidence_quotes": [evidence, dict(evidence)],
        }]}

        def handler(_request):
            return httpx.Response(200, json={
                "status": "completed", "output": [{
                    "type": "message", "role": "assistant", "content": [{
                        "type": "output_text", "text": json.dumps(payload),
                    }],
                }],
            })

        result = await LLMReportEngine(
            LLMConfig(), transport=httpx.MockTransport(handler)
        ).generate([{
            "message_id": "m1", "conversation_id": "test-c", "sender": "张三",
            "time": "2026-09-07T09:00:00+08:00", "text": "部署已经完成。",
        }])
        self.assertEqual(
            1, len(result["structured"]["completed_items"][0]["evidence_quotes"])
        )
        self.assertIn(
            "duplicate_evidence_quote_removed",
            result["model"]["output_normalizations"],
        )

    async def test_quote_for_unlisted_source_still_fails_closed(self):
        payload = {"items": [{
            "title": "报表字段修复", "detail": "完成报表字段修复。",
            "kind": "completed", "source_message_ids": ["m1"],
            "evidence_quotes": [
                {"source_message_id": "m1", "quote": "完成报表字段修复"},
                {"source_message_id": "m2", "quote": "额外消息证据"},
            ],
        }]}

        def handler(_request):
            return httpx.Response(200, json={
                "status": "completed", "output": [{
                    "type": "message", "role": "assistant", "content": [{
                        "type": "output_text", "text": json.dumps(payload),
                    }],
                }],
            })

        with self.assertRaises(LLMOutputError) as caught:
            await LLMReportEngine(
                LLMConfig(), transport=httpx.MockTransport(handler)
            ).generate([{
                "message_id": "m1", "conversation_id": "test-c", "sender": "张三",
                "time": "2026-09-07T09:00:00+08:00",
                "text": "完成报表字段修复。",
            }])
        self.assertEqual("LLM_OUTPUT_SCHEMA_INVALID", caught.exception.code)

    async def test_thinking_response_passes_configuration_and_internal_evidence_checks(self):
        def handler(request):
            # Same shape as the official /responses response: reasoning followed
            # by a valid assistant JSON message.
            items = []
            for ident, title, quote, kind in (
                ("config-test-completed", "测试部署", "测试部署已经完成", "completed"),
                ("config-test-unfinished", "测试接口", "测试接口尚未完成，下周继续处理", "open_next"),
            ):
                items.append({
                    "title": title, "detail": quote, "kind": kind,
                    "source_message_ids": [ident],
                    "evidence_quotes": [{"source_message_id": ident, "quote": quote}],
                })
            return httpx.Response(200, json={
                "status": "completed", "output": [
                    {"type": "reasoning", "content": [
                        {"type": "reasoning_text", "text": "synthetic non-JSON analysis"}]},
                    {"type": "message", "role": "assistant", "content": [
                        {"type": "output_text", "text": json.dumps({"items": items})}]},
                ]})
        engine = LLMReportEngine(
            LLMConfig(base_url="https://api.deepseek.com", model="deepseek-flash"),
            transport=httpx.MockTransport(handler))
        result = await engine.test_configuration()
        self.assertTrue(result["ok"])
        self.assertTrue(result["evidence_quote_validation_validated"])
        self.assertTrue(result["subject_identity_validation_validated"])
        self.assertEqual(1, result["http_attempt_count"])

    async def test_official_json_mode_still_rejects_bad_schema_and_unknown_evidence(self):
        for bad, expected in (
            ({"items": [], "unexpected": 1}, "LLM_OUTPUT_SCHEMA_INVALID"),
            ({"items": [{
                "title": "部署", "detail": "部署已经完成", "kind": "completed",
                "source_message_ids": ["invented-message"],
                "evidence_quotes": [{"source_message_id": "invented-message",
                                     "quote": "部署已经完成"}],
            }]}, "LLM_EVIDENCE_ID_UNKNOWN"),
        ):
            with self.subTest(expected=expected):
                def handler(request):
                    self.assertEqual(
                        {"type": "json_object"}, json.loads(request.content)["response_format"])
                    return httpx.Response(200, json={"choices": [{
                        "finish_reason": "stop", "message": {"content": json.dumps(bad)}}]})
                engine = LLMReportEngine(
                    LLMConfig(base_url="https://api.deepseek.com", model="deepseek-flash",
                              protocol=PROTOCOL_CHAT_COMPLETIONS),
                    transport=httpx.MockTransport(handler))
                with self.assertRaises(LLMOutputError) as caught:
                    await engine.generate([{
                        "message_id": "input-message", "conversation_id": "test-c",
                        "sender": "张三", "time": "2026-09-07T09:00:00+08:00",
                        "text": "部署已经完成。",
                    }])
                self.assertEqual(expected, caught.exception.code)


if __name__ == "__main__":
    unittest.main()
