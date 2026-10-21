from __future__ import annotations

import json
import pathlib
import unittest

from app.dws_adapter import FixtureDwsAdapter
from app.report_engine import (
    ReportEngineError,
    analyze_messages,
    generate_report,
    render_full_review_appendix,
)


FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"


def message(
    message_id: str,
    text: str,
    *,
    conversation_id: str = "conv-a",
    conversation_name: str = "项目群",
    sender: str = "张三",
    time: str = "2026-08-20T10:00:00+08:00",
    **extra,
):
    value = {
        "conversation_id": conversation_id,
        "conversation_name": conversation_name,
        "message_id": message_id,
        "sender": sender,
        "time": time,
        "text": text,
        "is_bot": False,
        "is_system": False,
    }
    value.update(extra)
    return value


class ReportEngineTests(unittest.TestCase):
    def test_every_finding_has_complete_evidence_and_no_unbacked_inferences(self):
        result = generate_report(
            [
                message("m1", "已经完成登录页上线。"),
                message("m2", "接口存在超时风险，需要继续检查。", time="2026-08-20T11:00:00+08:00"),
            ],
            {"title": "测试报告"},
        )
        analysis = result["analysis"]

        self.assertEqual(3, len(analysis["facts"]))
        self.assertEqual(2, analysis["summary"]["included_message_count"])
        self.assertEqual(3, analysis["summary"]["finding_count"])
        self.assertEqual([], analysis["inferences"])
        self.assertEqual(0, analysis["summary"]["inference_count"])
        for finding in analysis["facts"]:
            self.assertEqual("fact", finding["fact_or_inference"])
            self.assertEqual("extractive", finding["statement_mode"])
            self.assertEqual(1, len(finding["evidence"]))
            evidence = finding["evidence"][0]
            for field in ("conversation_id", "message_id", "sender", "time"):
                self.assertTrue(evidence[field])
            self.assertIn(f"message_id={evidence['message_id']}", result["markdown"])

    def test_filters_exclude_system_bot_sender_keyword_type_and_time(self):
        messages = [
            message("keep", "项目存在风险，需要跟进。"),
            message("system", "系统通知风险", message_type="system", is_system=True),
            message("bot", "机器人发现风险", is_bot=True),
            message("sender", "测试账号报告风险", sender="测试账号"),
            message("keyword", "红包风险通知"),
            message("type", "语音里提到风险", message_type="audio"),
            message("early", "历史风险", time="2026-08-19T23:59:59+08:00"),
            message("end", "边界风险", time="2026-08-21T00:00:00+08:00"),
        ]
        analysis = analyze_messages(
            messages,
            {
                "filters": {
                    "exclude_senders": ["测试账号"],
                    "exclude_keywords": ["红包"],
                    "message_types": ["text"],
                    "start": "2026-08-20T00:00:00+08:00",
                    "end": "2026-08-21T00:00:00+08:00",
                }
            },
        )

        self.assertEqual(
            {"keep"},
            {fact["evidence"][0]["message_id"] for fact in analysis["facts"]},
        )
        self.assertEqual(7, analysis["summary"]["excluded_message_count"])
        self.assertEqual(1, analysis["exclusion_counts"]["系统消息已排除"])
        self.assertEqual(1, analysis["exclusion_counts"]["机器人消息已排除"])
        self.assertEqual(1, analysis["exclusion_counts"]["达到或晚于结束时间"])

    def test_unknown_origin_is_not_silently_treated_as_human(self):
        unknown = message("unknown", "接口存在风险，需要继续检查。")
        unknown.pop("is_bot")
        unknown.pop("is_system")

        excluded = analyze_messages([unknown])
        self.assertEqual(0, excluded["summary"]["finding_count"])
        self.assertEqual(1, excluded["summary"]["excluded_message_count"])
        self.assertTrue(
            any(reason.startswith("消息来源未验证（") for reason in excluded["exclusion_counts"])
        )
        self.assertTrue(any("未将来源未知" in warning for warning in excluded["warnings"]))

        included = analyze_messages(
            [unknown],
            {"filters": {"include_bot_messages": True, "include_system_messages": True}},
        )
        self.assertGreater(included["summary"]["finding_count"], 0)
        self.assertTrue(any("不据此断言" in warning for warning in included["warnings"]))

    def test_recalled_messages_are_included_by_default_and_explicitly_excluded_when_disabled(self):
        recalled = message("m1", "这是一条已撤回的风险信息。", status="RECALLED")

        included = analyze_messages([recalled])
        self.assertEqual(1, included["summary"]["finding_count"])
        self.assertTrue(included["config"]["filters"]["include_recalled"])

        excluded = analyze_messages(
            [recalled, message("m2", "撤回通知", status="RECALL_NOTICE_UNRESOLVED")],
            {"filters": {"include_recalled": False}},
        )
        self.assertEqual(0, excluded["summary"]["finding_count"])
        self.assertEqual(2, excluded["exclusion_counts"]["撤回消息已排除"])
        self.assertFalse(excluded["config"]["filters"]["include_recalled"])

        with self.assertRaises(ReportEngineError):
            analyze_messages([recalled], {"filters": {"include_recalled": "false"}})

    def test_markdown_honours_hard_length_without_cutting_evidence(self):
        messages = [
            message(
                f"m{index}",
                f"接口存在风险，需要继续跟进第 {index} 项，" + "详细内容" * 20,
                time=f"2026-08-20T10:{index:02d}:00+08:00",
            )
            for index in range(8)
        ]
        result = generate_report(messages, {"max_length_chars": 700})

        self.assertLessEqual(len(result["markdown"]), 700)
        self.assertTrue(result["analysis"]["rendering"]["truncated"])
        self.assertGreater(result["analysis"]["rendering"]["omitted_finding_count"], 0)
        for line in result["markdown"].splitlines():
            if line.startswith("- ") and not line.startswith("- 输入") and not line.startswith("- 进入") and not line.startswith("- 排除") and not line.startswith("- 待人工") and not line.startswith("- 分组"):
                # Every rendered conclusion block has a complete source line later.
                tail = result["markdown"][result["markdown"].find(line) :]
                self.assertIn("conversation_id=", tail)
                self.assertIn("message_id=", tail)

    def test_same_identity_is_deduplicated_but_same_text_different_ids_is_retained(self):
        base = message("m1", "已经完成部署。")
        analysis = analyze_messages(
            [base, dict(base), message("m2", "已经完成部署。")],
        )

        self.assertEqual(1, analysis["summary"]["duplicate_observation_count"])
        self.assertEqual(2, analysis["summary"]["finding_count"])
        ids = [fact["evidence"][0]["message_id"] for fact in analysis["facts"]]
        self.assertEqual(["m1", "m2"], ids)

    def test_conflicting_same_identity_is_not_guessed(self):
        analysis = analyze_messages(
            [message("m1", "已经完成 A。"), message("m1", "已经完成 B。")]
        )

        self.assertEqual(0, analysis["summary"]["finding_count"])
        self.assertEqual(1, analysis["summary"]["conflicting_identity_count"])
        self.assertIn("冲突内容", analysis["review_items"][0]["reason"])

    def test_same_conversation_name_with_different_ids_stays_separate(self):
        analysis = analyze_messages(
            [
                message("m1", "已经完成 A。", conversation_id="cid-1", conversation_name="同名群"),
                message("m2", "已经完成 B。", conversation_id="cid-2", conversation_name="同名群"),
            ],
            {"grouping": "conversation"},
        )
        completed = next(section for section in analysis["sections"] if section["id"] == "completed")

        self.assertEqual(2, len(completed["groups"]))
        self.assertEqual({"cid-1", "cid-2"}, {group["key"] for group in completed["groups"]})
        self.assertTrue(all("同名群" in group["title"] for group in completed["groups"]))

    def test_sender_is_not_inferred_as_owner_and_explicit_owner_is_extracted(self):
        analysis = analyze_messages(
            [
                message("m1", "待办：需要处理接口问题。", sender="李四"),
                message("m2", "待办：负责人：王五，截止日期：2026-08-25。"),
            ]
        )
        by_id = {fact["evidence"][0]["message_id"]: fact for fact in analysis["facts"]}

        self.assertIsNone(by_id["m1"]["owner"]["value"])
        self.assertEqual("unknown", by_id["m1"]["owner"]["certainty"])
        self.assertNotEqual("李四", by_id["m1"]["owner"]["value"])
        self.assertEqual("王五", by_id["m2"]["owner"]["value"])
        self.assertEqual("2026-08-25", by_id["m2"]["deadline"]["value"])

    def test_unmatched_message_goes_to_manual_review_section(self):
        analysis = analyze_messages([message("m1", "本次讨论涉及蓝色按钮。")])
        finding = analysis["facts"][0]
        other = next(section for section in analysis["sections"] if "other" in section["categories"])

        self.assertEqual("other", finding["category"])
        self.assertEqual(1, other["finding_count"])
        self.assertIn("人工审阅", finding["classification_basis"][0])

    def test_topic_grouping_without_rules_uses_only_deterministic_categories(self):
        analysis = analyze_messages(
            [message("m1", "部署存在风险。")],
            {"grouping": "topic"},
        )

        self.assertEqual("topic", analysis["config"]["effective_grouping"])
        risk = next(section for section in analysis["sections"] if section["id"] == "risk")
        self.assertEqual("风险与阻塞", risk["groups"][0]["title"])
        self.assertTrue(any("不进行开放式主题推断" in warning for warning in analysis["warnings"]))

    def test_topic_and_time_grouping_with_explicit_rules(self):
        topic = analyze_messages(
            [
                message("m1", "登录模块存在风险。"),
                message("m2", "支付模块存在风险。", time="2026-08-21T10:00:00+08:00"),
            ],
            {
                "grouping": "topic",
                "topic_rules": {"登录": ["登录"], "支付": ["支付"]},
            },
        )
        risk = next(section for section in topic["sections"] if section["id"] == "risk")
        self.assertEqual(["登录", "支付"], [group["title"] for group in risk["groups"]])

        by_time = analyze_messages(
            [
                message("m1", "已经完成 A。"),
                message("m2", "已经完成 B。", time="2026-08-21T10:00:00+08:00"),
            ],
            {"grouping": "time"},
        )
        completed = next(section for section in by_time["sections"] if section["id"] == "completed")
        self.assertEqual(["2026-08-20", "2026-08-21"], [group["key"] for group in completed["groups"]])

    def test_custom_section_routes_only_explicit_categories(self):
        analysis = analyze_messages(
            [
                message("m1", "已经完成上线。"),
                message("m2", "系统存在延期风险。"),
            ],
            {
                "sections": [
                    {
                        "id": "delivery",
                        "title": "交付与风险",
                        "categories": ["completed", "risk"],
                    }
                ]
            },
        )
        custom = next(section for section in analysis["sections"] if section["id"] == "delivery")

        self.assertEqual(2, custom["finding_count"])
        self.assertTrue(any("other" in section["categories"] for section in analysis["sections"]))

    def test_provided_sections_are_a_true_category_scope_but_unmatched_other_is_retained(self):
        analysis = analyze_messages(
            [
                message("completed", "已经完成上线。"),
                message("risk", "接口存在超时风险。", time="2026-08-20T10:01:00+08:00"),
                message("other", "按钮颜色为蓝色。", time="2026-08-20T10:02:00+08:00"),
            ],
            {"sections": ["风险与阻塞"]},
        )

        self.assertEqual(
            {("risk", "risk"), ("other", "other")},
            {
                (fact["evidence"][0]["message_id"], fact["category"])
                for fact in analysis["facts"]
            },
        )
        self.assertEqual(["risk"], analysis["config"]["selected_explicit_categories"])
        self.assertEqual(1, analysis["exclusion_counts"]["分类不在报告包含范围"])
        self.assertEqual(1, analysis["summary"]["excluded_message_count"])
        other = next(section for section in analysis["sections"] if "other" in section["categories"])
        self.assertEqual(["other"], [fact["evidence"][0]["message_id"] for fact in other["findings"]])

    def test_multicategory_message_creates_one_evidence_bound_fact_per_selected_category(self):
        analysis = analyze_messages(
            [message("multi", "接口存在超时风险，需要继续检查处理。")],
            {"sections": ["风险与阻塞", "待跟进事项"]},
        )

        self.assertEqual(1, analysis["summary"]["included_message_count"])
        self.assertEqual(2, analysis["summary"]["finding_count"])
        self.assertEqual({"risk", "todo"}, {fact["category"] for fact in analysis["facts"]})
        self.assertEqual(2, len({fact["id"] for fact in analysis["facts"]}))
        self.assertEqual(
            {("conv-a", "multi")},
            {
                (
                    fact["evidence"][0]["conversation_id"],
                    fact["evidence"][0]["message_id"],
                )
                for fact in analysis["facts"]
            },
        )

        risk_only = analyze_messages(
            [message("multi", "接口存在超时风险，需要继续检查处理。")],
            {"sections": ["风险与阻塞"]},
        )
        self.assertEqual(["risk"], [fact["category"] for fact in risk_only["facts"]])
        self.assertEqual(1, risk_only["summary"]["included_message_count"])
        self.assertEqual(0, risk_only["summary"]["excluded_message_count"])
        self.assertEqual(1, risk_only["exclusion_counts"]["分类不在报告包含范围"])
        self.assertEqual("todo", next(
            item["category"] for item in risk_only["exclusions"]
            if item["reason"] == "分类不在报告包含范围"
        ))

    def test_free_instruction_applies_only_supported_directives(self):
        result = generate_report(
            [
                message("m1", "系统存在风险。"),
                message("m2", "已经完成上线。"),
                message("m3", "收到。"),
            ],
            {
                "grouping": "time",
                "max_length_chars": 600,
                "filters": {"exclude_casual": True},
                "custom_instruction": "按时间分组，控制在600字以内，不要推断，附消息出处，不要写闲聊；使用董事会风格",
            },
        )
        config = result["analysis"]["config"]

        self.assertEqual("time", config["effective_grouping"])
        self.assertEqual(600, config["max_length_chars"])
        self.assertTrue(config["custom_instruction"]["applied"])
        self.assertTrue(config["custom_instruction"]["unsupported"])
        self.assertTrue(any("未假装执行" in warning for warning in result["analysis"]["warnings"]))
        self.assertNotIn("收到。", [fact["statement"] for fact in result["analysis"]["facts"]])

    def test_free_instruction_cannot_silently_override_structured_controls(self):
        analysis = analyze_messages(
            [message("m1", "接口存在风险，需要继续检查。")],
            {
                "grouping": "time",
                "max_length_chars": 5000,
                "filters": {"exclude_casual": False},
                "custom_instruction": "按会话分组。过滤普通闲聊。不超过6000字。只写事实。",
            },
        )
        config = analysis["config"]
        self.assertEqual("time", config["effective_grouping"])
        self.assertEqual(5000, config["max_length_chars"])
        self.assertFalse(config["filters"]["exclude_casual"])
        self.assertEqual(3, len(config["custom_instruction"]["conflicts"]))
        self.assertTrue(any("结构化选项未被" in warning for warning in analysis["warnings"]))

    def test_invalid_evidence_never_becomes_a_finding(self):
        analysis = analyze_messages(
            [
                message("m1", "已经完成上线。"),
                {"conversation_id": "c", "message_id": "bad", "time": "not-a-time", "text": "风险"},
            ]
        )

        self.assertEqual(1, analysis["summary"]["finding_count"])
        self.assertEqual(1, analysis["summary"]["invalid_message_count"])
        self.assertIn("sender", analysis["review_items"][0]["reason"])
        self.assertIn("time", analysis["review_items"][0]["reason"])

    def test_left_closed_right_open_filter(self):
        analysis = analyze_messages(
            [
                message("start", "开始边界存在风险。", time="2026-08-20T00:00:00+08:00"),
                message("inside", "范围内存在风险。", time="2026-08-20T23:59:59+08:00"),
                message("end", "结束边界存在风险。", time="2026-08-21T00:00:00+08:00"),
            ],
            {
                "filters": {
                    "start": "2026-08-20T00:00:00+08:00",
                    "end": "2026-08-21T00:00:00+08:00",
                }
            },
        )

        self.assertEqual(
            ["start", "inside"],
            [fact["evidence"][0]["message_id"] for fact in analysis["facts"]],
        )

    def test_stable_ids_remain_case_sensitive_in_filters(self):
        analysis = analyze_messages(
            [
                message("M1", "已经完成 A。", conversation_id="CID-A"),
                message("m1", "已经完成 B。", conversation_id="cid-a"),
            ],
            {"filters": {"include_conversation_ids": ["CID-A"], "include_message_ids": ["M1"]}},
        )

        self.assertEqual(
            [("CID-A", "M1")],
            [
                (fact["evidence"][0]["conversation_id"], fact["evidence"][0]["message_id"])
                for fact in analysis["facts"]
            ],
        )

    def test_invalid_config_fails_closed(self):
        with self.assertRaises(ReportEngineError):
            analyze_messages([], {"max_length_chars": 100})
        with self.assertRaises(ReportEngineError):
            analyze_messages([], {"filters": {"start": "2026-08-21", "end": "2026-08-20"}})

    def test_collection_audit_is_fixed_at_markdown_head_and_structured(self):
        messages = [
            message(
                f"m{index}",
                "接口存在超时风险，需要继续检查处理。" + "详细说明" * 20,
                time=f"2026-08-20T10:{index:02d}:00+08:00",
            )
            for index in range(6)
        ]
        audit = {
            "data_complete": False,
            "status": "PROVISIONAL_SOURCE_LIMITATION",
            "sealed": False,
            "warnings": ["DWS issue 249：部分单聊可能遗漏"],
            "source_difference": {
                "only_list_all_count": 1,
                "only_targeted_count": 2,
            },
            "coverage": [
                {
                    "conversation_id": "cid-empty",
                    "conversation_name": "空结果单聊",
                    "message_count": 0,
                    "uncertainty": "EMPTY_SELECTED_DIRECT_CONVERSATION_UNCERTAIN",
                }
            ],
            "job_id": "job123",
            "dataset_digest": "abc123",
        }
        result = generate_report(
            messages,
            {
                "max_length_chars": 1500,
                "filters": {
                    "include_conversation_ids": ["conv-a"],
                    "start": "2026-08-20T00:00:00+08:00",
                    "end": "2026-08-21T00:00:00+08:00",
                },
                "audit": audit,
            },
        )
        markdown = result["markdown"]
        rendered_text = markdown.replace("\\", "")

        self.assertFalse(result["analysis"]["audit"]["data_complete"])
        self.assertEqual("job123", result["analysis"]["config"]["audit"]["job_id"])
        for expected in (
            "数据完整性审计（固定，不可裁剪）",
            "data_complete：false",
            "status：PROVISIONAL_SOURCE_LIMITATION",
            "sealed：false",
            "job_id：job123",
            "dataset_digest：abc123",
            "only_list_all_count=1",
            "only_targeted_count=2",
            "DWS issue 249：部分单聊可能遗漏",
            "cid-empty",
            "EMPTY_SELECTED_DIRECT_CONVERSATION_UNCERTAIN",
            "数据范围：[",
            "会话范围：conv-a",
        ):
            self.assertIn(expected, rendered_text)
        self.assertLess(markdown.index("数据完整性审计"), markdown.index("## 风险与阻塞"))
        self.assertTrue(result["analysis"]["rendering"]["truncated"])
        self.assertIn("受最大长度限制", markdown)

    def test_too_small_limit_fails_closed_instead_of_dropping_audit(self):
        audit = {
            "data_complete": False,
            "status": "PROVISIONAL_SOURCE_LIMITATION",
            "sealed": False,
            "warnings": ["不可省略的长采集警告" * 20],
            "source_difference": {"only_targeted_count": 3},
            "zero_message_uncertain_conversations": [
                {
                    "conversation_id": "cid-empty",
                    "uncertainty": "EMPTY_SELECTED_DIRECT_CONVERSATION_UNCERTAIN",
                }
            ],
            "job_id": "job123",
            "dataset_digest": "abc123",
        }
        with self.assertRaisesRegex(ReportEngineError, "无法容纳不可裁剪的数据审计头"):
            generate_report(
                [message("m1", "接口存在风险。")],
                {"max_length_chars": 300, "audit": audit},
            )

    def test_invalid_audit_fails_closed(self):
        with self.assertRaisesRegex(ReportEngineError, "audit.data_complete"):
            analyze_messages(
                [],
                {
                    "audit": {
                        "data_complete": "false",
                        "status": "PROVISIONAL",
                        "sealed": False,
                        "job_id": "job1",
                        "dataset_digest": "digest1",
                    }
                },
            )

    def test_owner_grouping_uses_only_explicit_owner(self):
        analysis = analyze_messages(
            [
                message("m1", "待办：负责人：王五，需要处理接口。"),
                message("m2", "待办：需要处理文档。", sender="李四"),
            ],
            {"grouping": "owner"},
        )
        todo = next(section for section in analysis["sections"] if section["id"] == "todo")

        self.assertEqual(
            ["负责人：王五", "待确认负责人（消息中未明确）"],
            [group["title"] for group in todo["groups"]],
        )

    def test_category_filter_routes_multicategory_message_to_requested_category(self):
        analysis = analyze_messages(
            [message("m1", "问题已解决，但仍有超时风险。")],
            {"include_categories": ["risk"]},
        )

        self.assertEqual(1, analysis["summary"]["finding_count"])
        self.assertEqual("risk", analysis["facts"][0]["category"])
        risk = next(section for section in analysis["sections"] if section["id"] == "risk")
        self.assertEqual(1, risk["finding_count"])

    def test_rendering_escapes_message_markdown_and_html_without_changing_structured_fact(self):
        raw_text = "已经完成 <script>alert(1)</script>\n# 不是报告标题"
        result = generate_report([message("m1", raw_text)], {"max_length_chars": 1200})

        self.assertEqual(raw_text, result["analysis"]["facts"][0]["statement"])
        self.assertNotIn("<script>", result["markdown"])
        self.assertNotIn("\n# 不是报告标题", result["markdown"])
        self.assertIn("&lt;script&gt;", result["markdown"])

    def test_full_review_appendix_renders_every_unique_message_without_ai_selection(self):
        raw_text = "需要人工判断 <script>风险</script>\n第二行" + ("甲" * 5000)
        rows = [
            message(
                "m-late",
                raw_text,
                time="2026-08-20T11:00:00+08:00",
            ),
            message(
                "m-other",
                "另一个会话也必须保留",
                conversation_id="conv-b",
                conversation_name="支持群",
                time="2026-08-20T08:00:00+08:00",
            ),
            message(
                "m-early",
                "较早消息",
                time="2026-08-20T09:00:00+08:00",
            ),
        ]

        markdown, audit = render_full_review_appendix(rows)

        self.assertIn("全量人工核对清单（不经 AI 取舍）", markdown)
        self.assertNotIn("<script>", markdown)
        self.assertIn("&lt;script&gt;风险&lt;/script&gt;", markdown)
        self.assertIn("甲" * 5000, markdown)
        self.assertLess(markdown.index(r"m\-early"), markdown.index(r"m\-late"))
        self.assertEqual(3, audit["input_observation_count"])
        self.assertEqual(3, audit["unique_identity_count"])
        self.assertEqual(3, audit["rendered_message_count"])
        self.assertEqual(2, audit["conversation_count"])
        self.assertTrue(audit["coverage_validated"])
        self.assertFalse(audit["ai_selection_applied"])
        self.assertFalse(audit["text_truncated"])
        self.assertEqual(
            {("conv-a", "m-early"), ("conv-a", "m-late"), ("conv-b", "m-other")},
            {
                (item["conversation_id"], item["message_id"])
                for item in audit["message_identities"]
            },
        )

    def test_full_review_appendix_deduplicates_observations_and_rejects_conflicts(self):
        original = message("same", "同一条消息")
        markdown, audit = render_full_review_appendix([original, dict(original)])
        self.assertEqual(2, audit["input_observation_count"])
        self.assertEqual(1, audit["duplicate_observation_count"])
        self.assertEqual(1, audit["rendered_message_count"])
        self.assertEqual(1, markdown.count("消息 ID：same"))

        conflict = dict(original)
        conflict["text"] = "冲突版本"
        with self.assertRaisesRegex(ReportEngineError, "冲突内容"):
            render_full_review_appendix([original, conflict])

    def test_panel_todo_section_alias_is_supported(self):
        analysis = analyze_messages(
            [message("m1", "待办：需要处理接口问题。")],
            {"sections": ["完成事项", "待跟进事项"]},
        )

        todo = next(section for section in analysis["sections"] if "todo" in section["categories"])
        self.assertEqual(1, todo["finding_count"])

    def test_output_of_fixture_dws_adapter_is_accepted_and_json_serialisable(self):
        adapter = FixtureDwsAdapter(FIXTURES)
        adapter.refresh_conversations()
        collected = adapter.collect_selected(
            ["cid-fixture-project", "cid-fixture-direct"],
            "2026-08-17T00:00:00+08:00",
            "2026-08-24T00:00:00+08:00",
        )
        result = generate_report(
            collected["messages"],
            {"grouping": "conversation", "max_length_chars": 4000},
        )

        self.assertEqual(len(collected["messages"]), result["analysis"]["summary"]["valid_message_count"])
        self.assertTrue(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
