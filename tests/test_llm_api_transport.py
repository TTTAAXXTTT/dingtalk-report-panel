from __future__ import annotations

import json
import pathlib
import tempfile
import time
import unittest

import httpx
from fastapi.testclient import TestClient

from app.dws_adapter import FixtureDwsAdapter
from app.llm_config_store import LLMConfigStore
from app.llm_engine import LLMReportEngine
from app.main import create_app
from app.storage import Storage


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class LLMApiTransportIntegrationTests(unittest.TestCase):
    def test_real_engine_wire_protocol_runs_through_api_and_job_pipeline(self) -> None:
        requests: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual("/v1/responses", request.url.path)
            self.assertEqual("Bearer mock-secret", request.headers["authorization"])
            payload = json.loads(request.content.decode("utf-8"))
            requests.append(payload)
            user_input = str(payload.get("input") or "")
            input_data = json.loads(user_input.split("\n", 1)[1])
            messages = input_data["messages"]
            self.assertTrue(messages)

            def row_id(item: list | dict) -> str:
                return str(item[0] if isinstance(item, list) else item["message_id"])

            def row_text(item: list | dict) -> str:
                return str(item[5] if isinstance(item, list) else item["text"])

            by_id = {row_id(item): item for item in messages}
            if "config-test-completed" in by_id:
                completed_row = by_id["config-test-completed"]
                unfinished_row = by_id["config-test-unfinished"]
                completed = {
                    "title": "测试部署完成",
                    "detail": "测试部署已经完成。",
                    "source_message_ids": [row_id(completed_row)],
                    "evidence_quotes": [
                        {
                            "source_message_id": row_id(completed_row),
                            "quote": row_text(completed_row).rstrip("。"),
                        }
                    ],
                }
                next_week = {
                    "title": "测试接口待处理",
                    "detail": "测试接口尚未完成，下周继续处理。",
                    "source_message_ids": [row_id(unfinished_row)],
                    "evidence_quotes": [
                        {
                            "source_message_id": row_id(unfinished_row),
                            "quote": row_text(unfinished_row).rstrip("。"),
                        }
                    ],
                }
                structured = {
                    "completed_items": [completed],
                    "weekly_report": {
                        "this_week": [completed],
                        "next_week": [next_week],
                    },
                }
            else:
                risk_row = next(
                    item for item in messages if "边界用例" in row_text(item)
                )
                follow_up = {
                    "title": "边界用例补充",
                    "detail": "测试接口需要补充边界用例。",
                    "source_message_ids": [row_id(risk_row)],
                    "evidence_quotes": [
                        {
                            "source_message_id": row_id(risk_row),
                            "quote": row_text(risk_row),
                        }
                    ],
                }
                structured = {
                    "completed_items": [],
                    "weekly_report": {
                        "this_week": [follow_up],
                        "next_week": [follow_up],
                    },
                }
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "content": [
                                {"type": "reasoning_text", "text": "Synthetic reasoning, not report JSON."}
                            ],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(structured, ensure_ascii=False),
                                }
                            ],
                        }
                    ]
                },
            )

        transport = httpx.MockTransport(handler)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            store = LLMConfigStore(root / "llm-config.json")
            store.save_config(
                {
                    "protocol": "openai_responses",
                    "base_url": "https://mock-model.test/v1",
                    "model": "mock-structured-model",
                    "api_key": "mock-secret",
                    "temperature": 0.1,
                    "max_output_tokens": 6000,
                    "analysis_instruction": "只根据输入消息整理",
                    "weekly_report_template": "本周工作：\n下周工作计划：",
                }
            )
            storage = Storage(root / "state")
            app = create_app(
                adapter=FixtureDwsAdapter(PROJECT_ROOT / "fixtures"),
                storage=storage,
                mode="fixture",
                llm_config_store=store,
                llm_engine_factory=lambda config: LLMReportEngine(
                    config, transport=transport
                ),
            )
            with TestClient(app) as client:
                tested = client.post("/api/llm/test")
                self.assertEqual(200, tested.status_code, tested.text)
                self.assertTrue(tested.json()["ok"])
                self.assertTrue(tested.json()["structured_output_ok"])
                self.assertEqual(0, tested.json()["retry_count"])

                client.post("/api/conversations/refresh")
                configuration = {
                    "conversation_ids": ["cid-fixture-project"],
                    "start": "2026-08-17T00:00:00",
                    "end": "2026-08-24T00:00:00",
                    "timezone": "Asia/Shanghai",
                    "filters": {
                        "message_types": ["text"],
                        "include_system_messages": False,
                        "include_bot_messages": False,
                        "include_recalled": True,
                        "only_related_to_me": True,
                        "casual_filter": "conservative",
                    },
                    "report": {
                        "subject_name": "测试用户",
                        "title": "模型传输集成测试",
                        "grouping": "conversation",
                        "sections": ["完成事项", "待跟进事项", "下一步计划"],
                        "max_length_chars": 8000,
                        "language": "zh-CN",
                        "inference_policy": "facts_only",
                        "custom_instruction": "合并同类事项，只写有消息依据的内容。",
                        "evidence_enabled": True,
                    },
                    "explicit_confirm": True,
                    "uncertainty_ack": False,
                }
                preview = client.post("/api/preview", json=configuration)
                self.assertEqual(200, preview.status_code, preview.text)
                self.assertTrue(preview.json()["can_generate"])
                self.assertEqual(
                    "测试用户",
                    preview.json()["effective_report"]["subject_name"],
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
                self.assertEqual("COMPLETED", final["status"], final)
                self.assertEqual(
                    ["completed_items", "weekly_report"], final["documents"]
                )
                self.assertIn(
                    "本周已完成事项：",
                    client.get(
                        f"/api/jobs/{job_id}/report?document=completed_items"
                    ).json()["markdown"],
                )
                self.assertIn(
                    "本周工作：",
                    client.get(
                        f"/api/jobs/{job_id}/report?document=weekly_report"
                    ).json()["markdown"],
                )

        self.assertGreaterEqual(len(requests), 2)
        self.assertTrue(all(request["text"]["format"]["strict"] for request in requests))


if __name__ == "__main__":
    unittest.main()
