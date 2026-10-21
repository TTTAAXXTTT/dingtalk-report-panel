from __future__ import annotations

import copy
import datetime as dt
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

from app.dws_adapter import (
    DwsAdapter,
    DwsAdapterError,
    FixtureDwsAdapter,
    ISSUE_249,
    _normalize_conversation,
    _validate_conversation_page,
    _validate_group_page,
    _validate_message_detail_batch,
    _validate_message_page,
    _validate_scoped_message_page,
    _validate_typed_conversation_page,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = PROJECT_ROOT / "fixtures"


def message(
    message_id: str | None = "m-1",
    *,
    conversation_id: str = "cid-direct",
    created: str = "2026-08-20T10:00:00+08:00",
    text: str = "测试消息",
) -> dict:
    value = {
        "sender": {"openDingTalkId": "u-1", "name": "测试用户"},
        "msgType": "text",
        "content": {"text": text},
        "createTime": created,
        "openConversationId": conversation_id,
    }
    if message_id is not None:
        value["openMessageId"] = message_id
    return value


def payload(rows: list[dict], *, conversation_id: str = "cid-direct") -> dict:
    return {
        "success": True,
        "paging": {
            "pages": 1,
            "total": len(rows),
            "truncated": False,
            "hasMore": False,
        },
        "result": {
            "hasMore": False,
            "nextCursor": "",
            "conversationMessagesList": [
                {
                    "openConversationId": conversation_id,
                    "title": "测试单聊",
                    "messages": rows,
                }
            ],
        },
    }


def wrapper(rows: list[dict], *, targeted_rows: list[dict] | None = None) -> dict:
    return {
        "list_all": payload(rows),
        "targeted": payload(rows if targeted_rows is None else targeted_rows),
    }


def typed_directory_payload(rows: list[dict], *, limited: bool = False) -> dict:
    payload = {
        "chats": rows,
        "complete": not limited,
        "count": len(rows),
        "effectivePageSize": 100,
        "failedCount": 0,
        "failures": [],
        "hasMore": False,
        "nextCursor": 0,
        "pagesFetched": 1,
        "paginationKnown": not limited,
        "partial": False,
        "requestedPageSize": 100,
        "requestedTypes": ["group", "p2p"],
        "stopReason": "single_page_full_untrusted" if limited else "source_complete",
        "truncatedByPageLimit": False,
    }
    if not limited:
        payload["completionEvidence"] = "backend_has_more_false_short_page"
    return payload


def complete_directory_payload(rows: list[dict], *, rows_key: str) -> dict:
    return {
        rows_key: rows,
        "complete": True,
        "count": len(rows),
        "failedCount": 0,
        "failures": [],
        "hasMore": False,
        "nextCursor": 0,
        "pagesFetched": 1,
        "paginationKnown": True,
        "partial": False,
        "stopReason": "source_complete",
        "truncated": False,
        "truncatedByPageLimit": False,
        "truncatedByResultLimit": False,
    }


def conversation_directory_payload(rows: list[dict]) -> dict:
    return complete_directory_payload(rows, rows_key="conversations")


def group_directory_payload(rows: list[dict]) -> dict:
    return complete_directory_payload(rows, rows_key="groups")


class FixtureAdapterTests(unittest.TestCase):
    def test_fixture_preflight_and_conversation_directory(self) -> None:
        adapter = FixtureDwsAdapter(FIXTURES)
        readiness = adapter.preflight()
        self.assertTrue(readiness["ok"])
        self.assertTrue(readiness["version_pinned"])
        directory = adapter.refresh_conversations()
        self.assertEqual(3, len(directory["items"]))
        self.assertTrue(directory["integrity"]["complete"])
        self.assertEqual("direct", directory["items"][1]["kind"])

    def test_collection_is_scoped_half_open_and_provisional(self) -> None:
        adapter = FixtureDwsAdapter(FIXTURES)
        adapter.refresh_conversations()
        result = adapter.collect_selected(
            ["cid-fixture-project", "cid-fixture-direct"],
            "2026-08-17T00:00:00+08:00",
            "2026-08-24T00:00:00+08:00",
        )
        self.assertTrue(result["execution_success"])
        self.assertFalse(result["data_complete"])
        self.assertFalse(result["sealed"])
        self.assertEqual("PROVISIONAL_SOURCE_LIMITATION", result["status"])
        self.assertEqual(ISSUE_249, result["source_limitation"])
        self.assertEqual(
            {"fx-start", "fx-project-2", "fx-dm-duplicate", "fx-dm-targeted-only"},
            {item["id"] for item in result["messages"]},
        )
        self.assertNotIn("fx-before", {item["id"] for item in result["messages"]})
        self.assertNotIn("fx-end", {item["id"] for item in result["messages"]})
        difference = result["integrity"]["source_difference"]
        self.assertEqual(1, difference["only_targeted_count"])
        self.assertEqual(2, result["integrity"]["passes_executed"])
        self.assertTrue(result["integrity"]["strong_identity_only"])

    def test_selected_empty_direct_chat_is_uncertain(self) -> None:
        adapter = FixtureDwsAdapter(FIXTURES)
        adapter.refresh_conversations()
        result = adapter.collect_selected(
            ["cid-fixture-empty-direct"],
            "2026-08-17T00:00:00+08:00",
            "2026-08-24T00:00:00+08:00",
        )
        self.assertEqual([], result["messages"])
        coverage = result["integrity"]["coverage"][0]
        self.assertEqual("direct", coverage["kind"])
        self.assertEqual("EMPTY_SELECTED_DIRECT_CONVERSATION_UNCERTAIN", coverage["uncertainty"])
        self.assertTrue(any("not proof" in warning for warning in result["warnings"]))

    def test_invalid_selection_and_window_fail_closed(self) -> None:
        adapter = FixtureDwsAdapter(FIXTURES)
        with self.assertRaisesRegex(DwsAdapterError, "select at least one"):
            adapter.collect_selected([], "2026-08-17", "2026-08-18")
        with self.assertRaisesRegex(DwsAdapterError, "end must be after start"):
            adapter.collect_selected(["cid"], "2026-08-18", "2026-08-17")
        with self.assertRaisesRegex(DwsAdapterError, "at least 2"):
            adapter.collect_selected(["cid"], "2026-08-17", "2026-08-18", passes=1)
        with self.assertRaisesRegex(DwsAdapterError, "whole seconds"):
            adapter.collect_selected(
                ["cid"],
                "2026-08-17T00:00:00.500000+08:00",
                "2026-08-18T00:00:00+08:00",
            )

    def test_weak_in_scope_message_id_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "weak.json"
            path.write_text(
                json.dumps(wrapper([message(None)]), ensure_ascii=False), encoding="utf-8"
            )
            adapter = FixtureDwsAdapter(FIXTURES, pass_paths=[path])
            with self.assertRaises(DwsAdapterError) as raised:
                adapter.collect_selected(
                    ["cid-direct"],
                    "2026-08-17T00:00:00+08:00",
                    "2026-08-24T00:00:00+08:00",
                )
            self.assertEqual("WEAK_MESSAGE_ID", raised.exception.code)

    def test_source_content_conflict_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "collision.json"
            path.write_text(
                json.dumps(
                    wrapper([message(text="源 A")], targeted_rows=[message(text="源 B")]),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            adapter = FixtureDwsAdapter(FIXTURES, pass_paths=[path])
            with self.assertRaises(DwsAdapterError) as raised:
                adapter.collect_selected(
                    ["cid-direct"],
                    "2026-08-17T00:00:00+08:00",
                    "2026-08-24T00:00:00+08:00",
                )
            self.assertEqual("SOURCE_CONTENT_CONFLICT", raised.exception.code)

    def test_verified_bot_and_system_markers_are_preserved(self) -> None:
        explicit_bot = message("m-bot", text="机器人测试正文")
        explicit_bot["isBot"] = True
        explicit_bot["sender"]["isBot"] = True
        typed_bot = message("m-typed-bot", text="类型机器人正文")
        typed_bot["sender"]["senderType"] = "robot"
        system = message("m-system", text="系统测试正文")
        system["msgType"] = "system"
        system["isSystem"] = True
        explicit_human = message("m-human", text="人工测试正文")
        explicit_human["sender_is_bot"] = False
        explicit_human["sender"]["isBot"] = False
        explicit_human["is_system"] = False
        unknown_origin = message("m-unknown", text="来源标记未知正文")
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "markers.json"
            path.write_text(
                json.dumps(
                    wrapper([explicit_bot, typed_bot, system, explicit_human, unknown_origin]),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            adapter = FixtureDwsAdapter(FIXTURES, pass_paths=[path])
            result = adapter.collect_selected(
                ["cid-direct"],
                "2026-08-17T00:00:00+08:00",
                "2026-08-24T00:00:00+08:00",
            )
        by_id = {item["id"]: item for item in result["messages"]}
        self.assertTrue(by_id["m-bot"]["is_bot"])
        self.assertTrue(by_id["m-bot"]["sender_is_bot"])
        self.assertTrue(by_id["m-bot"]["origin_verification"]["bot_verified"])
        self.assertTrue(by_id["m-typed-bot"]["is_bot"])
        self.assertTrue(by_id["m-typed-bot"]["sender_is_bot"])
        self.assertEqual(
            "senderType",
            by_id["m-typed-bot"]["origin_verification"]["sender_bot_type_key"],
        )
        self.assertTrue(by_id["m-system"]["is_system"])
        self.assertFalse(by_id["m-human"]["is_bot"])
        self.assertFalse(by_id["m-human"]["sender_is_bot"])
        self.assertFalse(by_id["m-human"]["is_system"])
        self.assertTrue(by_id["m-human"]["origin_verification"]["bot_verified"])
        self.assertTrue(by_id["m-human"]["origin_verification"]["system_verified"])
        self.assertIsNone(by_id["m-unknown"]["is_bot"])
        self.assertIsNone(by_id["m-unknown"]["sender_is_bot"])
        self.assertIsNone(by_id["m-unknown"]["is_system"])
        self.assertFalse(by_id["m-unknown"]["origin_verification"]["bot_verified"])
        self.assertFalse(by_id["m-unknown"]["origin_verification"]["system_verified"])

    def test_non_boolean_or_conflicting_origin_markers_fail_closed(self) -> None:
        invalid = message("m-invalid")
        invalid["sender"]["isBot"] = "true"
        conflict = message("m-conflict")
        conflict["isBot"] = False
        conflict["sender"]["senderType"] = "robot"
        for expected_code, row in (
            ("INVALID_BOOLEAN_MARKER", invalid),
            ("BOOLEAN_MARKER_CONFLICT", conflict),
        ):
            with self.subTest(expected_code=expected_code), tempfile.TemporaryDirectory() as temporary:
                path = pathlib.Path(temporary) / "marker_failure.json"
                path.write_text(json.dumps(wrapper([row])), encoding="utf-8")
                adapter = FixtureDwsAdapter(FIXTURES, pass_paths=[path])
                with self.assertRaises(DwsAdapterError) as raised:
                    adapter.collect_selected(
                        ["cid-direct"],
                        "2026-08-17T00:00:00+08:00",
                        "2026-08-24T00:00:00+08:00",
                    )
                self.assertEqual(expected_code, raised.exception.code)

    def test_disappearing_message_is_not_accepted_as_stable_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = []
            for index, rows in enumerate(([message()], [], []), start=1):
                path = root / f"pass_{index}.json"
                path.write_text(json.dumps(wrapper(list(rows))), encoding="utf-8")
                paths.append(path)
            adapter = FixtureDwsAdapter(FIXTURES, pass_paths=paths, max_passes=3)
            with self.assertRaises(DwsAdapterError) as raised:
                adapter.collect_selected(
                    ["cid-direct"],
                    "2026-08-17T00:00:00+08:00",
                    "2026-08-24T00:00:00+08:00",
                )
            self.assertEqual("MESSAGES_DISAPPEARED", raised.exception.code)


class StrictLedgerTests(unittest.TestCase):
    def test_dws_command_retries_only_explicit_retryable_read_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = pathlib.Path(temporary) / "dws.exe"
            executable.write_bytes(b"fixture")
            retryable = subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr=json.dumps(
                    {"error": {"reason": "chat_messages_incomplete", "retryable": True}}
                ),
            )
            success = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"ok":true}', stderr=""
            )
            adapter = DwsAdapter(
                executable,
                command_retries=2,
                command_retry_delay=0,
            )
            with mock.patch(
                "app.dws_adapter.subprocess.run", side_effect=[retryable, success]
            ) as invoked:
                self.assertTrue(adapter._run(["chat"], operation="test")["ok"])
            self.assertEqual(2, invoked.call_count)

            non_retryable = copy.deepcopy(retryable)
            non_retryable.stderr = json.dumps(
                {"error": {"reason": "business_error", "retryable": False}}
            )
            with mock.patch(
                "app.dws_adapter.subprocess.run", return_value=non_retryable
            ) as invoked:
                with self.assertRaises(DwsAdapterError):
                    adapter._run(["chat"], operation="test")
            self.assertEqual(1, invoked.call_count)

    def test_normal_chat_scope_requires_exact_complete_half_open_ledger(self) -> None:
        start = dt.datetime(2026, 8, 17, tzinfo=dt.timezone(dt.timedelta(hours=8)))
        end = dt.datetime(2026, 8, 24, tzinfo=dt.timezone(dt.timedelta(hours=8)))
        valid = {
            "complete": True,
            "contractVersion": "im.message-list.v1",
            "count": 1,
            "failedCount": 0,
            "failures": [],
            "hasMore": False,
            "messages": [
                {
                    "conversationId": "cid-normal",
                    "createTime": "2026-08-20T10:00:00+08:00",
                    "messageId": "m-normal",
                    "sender": "测试用户",
                    "senderId": "u-normal",
                    "text": "测试正文",
                }
            ],
            "pagesFetched": 1,
            "paginationKnown": True,
            "partial": False,
            "queryRange": {
                "startTime": start.isoformat(),
                "endTime": end.isoformat(),
                "order": "asc",
                "semantics": "[start,end)",
            },
            "stopReason": "range_end",
            "truncated": False,
            "truncatedByPageLimit": False,
            "truncatedByResultLimit": False,
        }
        rows, ledger = _validate_scoped_message_page(
            valid,
            conversation_id="cid-normal",
            start=start,
            end=end,
            source="test",
        )
        self.assertEqual(1, len(rows))
        self.assertTrue(ledger["complete"])

        for mutation, expected_code in (
            (("complete", False), "PAGINATION_UNPROVEN"),
            (("conversationId", "cid-other"), "CONVERSATION_ID_CONFLICT"),
            (("endTime", "2026-08-23T00:00:00+08:00"), "PAGINATION_UNPROVEN"),
        ):
            invalid = copy.deepcopy(valid)
            field, value = mutation
            if field == "conversationId":
                invalid["messages"][0][field] = value
            elif field == "endTime":
                invalid["queryRange"][field] = value
            else:
                invalid[field] = value
            with self.subTest(field=field), self.assertRaises(DwsAdapterError) as raised:
                _validate_scoped_message_page(
                    invalid,
                    conversation_id="cid-normal",
                    start=start,
                    end=end,
                    source="test",
                )
            self.assertEqual(expected_code, raised.exception.code)

    def test_message_detail_batch_requires_every_requested_strong_id(self) -> None:
        valid = {
            "success": True,
            "errorCode": None,
            "result": {
                "messages": [
                    {
                        "openMessageId": "m-1",
                        "openConversationId": "cid-normal",
                    },
                    {
                        "openMessageId": "m-2",
                        "openConversationId": "cid-normal",
                    },
                ]
            },
        }
        parsed = _validate_message_detail_batch(valid, ["m-1", "m-2"], source="test")
        self.assertEqual({"m-1", "m-2"}, set(parsed))
        missing = copy.deepcopy(valid)
        missing["result"]["messages"].pop()
        with self.assertRaises(DwsAdapterError) as raised:
            _validate_message_detail_batch(missing, ["m-1", "m-2"], source="test")
        self.assertEqual("MESSAGE_DETAILS_INCOMPLETE", raised.exception.code)

    def test_message_pagination_rejects_string_false_and_total_mismatch(self) -> None:
        valid = payload([message()])
        bad_boolean = copy.deepcopy(valid)
        bad_boolean["paging"]["truncated"] = "false"
        with self.assertRaises(DwsAdapterError) as raised:
            _validate_message_page(bad_boolean, source="test")
        self.assertEqual("PAGINATION_UNPROVEN", raised.exception.code)

        bad_total = copy.deepcopy(valid)
        bad_total["paging"]["total"] = 2
        with self.assertRaises(DwsAdapterError) as raised:
            _validate_message_page(bad_total, source="test")
        self.assertEqual("PAGINATION_UNPROVEN", raised.exception.code)

    def test_competing_message_containers_are_ambiguous(self) -> None:
        for label, competing_value in (
            ("top-level list", [message("m-other")]),
            ("result object", {"items": [message("m-other")]}),
        ):
            with self.subTest(label=label):
                invalid = payload([message()])
                if label == "top-level list":
                    invalid["messages"] = competing_value
                else:
                    invalid["result"]["messages"] = competing_value
                with self.assertRaises(DwsAdapterError) as raised:
                    _validate_message_page(invalid, source="test")
                self.assertEqual("AMBIGUOUS_MESSAGE_CONTAINERS", raised.exception.code)

    def test_group_and_row_conversation_id_conflict_fails_closed(self) -> None:
        invalid = payload([message(conversation_id="cid-other")], conversation_id="cid-direct")
        with self.assertRaises(DwsAdapterError) as raised:
            _validate_message_page(invalid, source="test")
        self.assertEqual("CONVERSATION_ID_CONFLICT", raised.exception.code)

    def test_conversation_pagination_requires_exact_completion(self) -> None:
        valid = json.loads((FIXTURES / "conversations.json").read_text(encoding="utf-8"))
        rows, ledger = _validate_conversation_page(valid)
        self.assertEqual(3, len(rows))
        self.assertTrue(ledger["complete"])
        invalid = copy.deepcopy(valid)
        invalid["paginationKnown"] = 1
        with self.assertRaises(DwsAdapterError) as raised:
            _validate_conversation_page(invalid)
        self.assertEqual("PAGINATION_UNPROVEN", raised.exception.code)

        groups = group_directory_payload(
            [{"openConversationId": "cid-group", "name": "项目群"}]
        )
        groups["nextCursor"] = None
        group_rows, group_ledger = _validate_group_page(groups)
        self.assertEqual(1, len(group_rows))
        self.assertTrue(group_ledger["complete"])
        invalid_groups = copy.deepcopy(groups)
        invalid_groups["partial"] = True
        with self.assertRaises(DwsAdapterError) as group_raised:
            _validate_group_page(invalid_groups)
        self.assertEqual("PAGINATION_UNPROVEN", group_raised.exception.code)
        invalid_cursor = copy.deepcopy(groups)
        invalid_cursor["nextCursor"] = 1
        with self.assertRaises(DwsAdapterError) as cursor_raised:
            _validate_group_page(invalid_cursor)
        self.assertEqual("PAGINATION_UNPROVEN", cursor_raised.exception.code)

    def test_normal_conversation_single_chat_fields_are_normalized(self) -> None:
        item = _normalize_conversation(
            {
                "openConversationId": "cid-normal-direct",
                "title": "普通单聊",
                "singleChat": True,
                "lastMsgCreateAt": "2026-08-23T12:00:00+08:00",
            },
            source="test",
        )
        self.assertEqual("direct", item["kind"])
        self.assertEqual("2026-08-23T12:00:00+08:00", item["last_active"])

    def test_typed_directory_accepts_only_the_exact_disclosed_limited_shape(self) -> None:
        rows = [
            {
                "openConversationId": f"cid-{index}",
                "name": f"会话 {index}",
                "conversationType": "direct" if index % 2 else "group",
                "chatMode": "p2p" if index % 2 else "group",
            }
            for index in range(100)
        ]
        valid = typed_directory_payload(rows, limited=True)
        parsed, ledger = _validate_typed_conversation_page(valid)
        self.assertEqual(100, len(parsed))
        self.assertFalse(ledger["complete"])
        self.assertTrue(ledger["source_limited"])
        self.assertTrue(ledger["typed"])

        # count is post type-filtering; the lower source can have an untrusted
        # full page while unknown-type rows are absent from this projection.
        filtered_rows = rows[:73]
        filtered, filtered_ledger = _validate_typed_conversation_page(
            typed_directory_payload(filtered_rows, limited=True)
        )
        self.assertEqual(73, len(filtered))
        self.assertTrue(filtered_ledger["directory_incomplete"])

        for field, value, expected_code in (
            ("partial", True, "PAGINATION_UNPROVEN"),
            ("stopReason", "pagination_error", "PAGINATION_UNPROVEN"),
            ("requestedTypes", ["group"], "INVALID_OUTPUT_SHAPE"),
            ("nextCursor", None, "PAGINATION_UNPROVEN"),
        ):
            invalid = copy.deepcopy(valid)
            invalid[field] = value
            with self.subTest(field=field), self.assertRaises(DwsAdapterError) as raised:
                _validate_typed_conversation_page(invalid)
            self.assertEqual(expected_code, raised.exception.code)

        invalid_type = copy.deepcopy(valid)
        invalid_type["chats"][0]["chatMode"] = "p2p"
        with self.assertRaises(DwsAdapterError) as raised:
            _validate_typed_conversation_page(invalid_type)
        self.assertEqual("CONVERSATION_TYPE_UNPROVEN", raised.exception.code)

        weak_id = copy.deepcopy(valid)
        weak_id["chats"][0]["openConversationId"] = ""
        with self.assertRaises(DwsAdapterError) as raised:
            _validate_typed_conversation_page(weak_id)
        self.assertEqual("WEAK_CONVERSATION_ID", raised.exception.code)

        generic_id_fallback = copy.deepcopy(valid)
        del generic_id_fallback["chats"][0]["openConversationId"]
        generic_id_fallback["chats"][0]["id"] = "generic-id-is-not-open-conversation-id"
        with self.assertRaises(DwsAdapterError) as raised:
            _validate_typed_conversation_page(generic_id_fallback)
        self.assertEqual("WEAK_CONVERSATION_ID", raised.exception.code)

    def test_missing_real_dws_is_explicit_preflight_failure(self) -> None:
        adapter = DwsAdapter(PROJECT_ROOT / "does-not-exist.exe")
        result = adapter.preflight()
        self.assertFalse(result["ok"])
        self.assertEqual("DWS_NOT_FOUND", result["errors"][0]["code"])

    def test_real_directory_unions_complete_conversations_and_joined_groups(self) -> None:
        conversation_payload = conversation_directory_payload(
            [
                *[
                    {
                        "openConversationId": f"cid-direct-{index}",
                        "conversationName": f"单聊 {index}",
                    }
                    for index in range(72)
                ],
                *[
                    {
                        "openConversationId": f"cid-group-{index}",
                        "conversationName": f"活动群 {index}",
                    }
                    for index in range(28)
                ],
            ]
        )
        group_payload = group_directory_payload(
            [
                {
                    "openConversationId": f"cid-group-{index}",
                    "name": f"已加入群 {index}",
                }
                for index in range(46)
            ]
        )

        class ReadyAdapter(DwsAdapter):
            def preflight(self) -> dict:
                return {
                    "ok": True,
                    "expected_version": "v1.0.59",
                    "version_pinned": True,
                    "authenticated": True,
                    "version": "Version: v1.0.59",
                }

            def _run(self, arguments, *, operation):
                del arguments
                if operation == "complete conversation directory":
                    return conversation_payload
                if operation == "complete group directory":
                    return group_payload
                raise AssertionError(operation)

        directory = ReadyAdapter(PROJECT_ROOT / "unused.exe").refresh_conversations()
        self.assertTrue(directory["integrity"]["data_complete"])
        self.assertTrue(directory["integrity"]["complete"])
        self.assertFalse(directory["integrity"]["source_limited"])
        self.assertEqual(118, len(directory["items"]))
        self.assertEqual(72, directory["integrity"]["direct_count"])
        self.assertEqual(46, directory["integrity"]["group_count"])
        self.assertEqual(18, directory["integrity"]["additional_joined_group_count"])
        self.assertTrue(all(item["kind"] in {"direct", "group"} for item in directory["items"]))
        self.assertEqual([], directory["warnings"])

    def test_real_commands_use_current_normal_chat_without_org_profile_filter(self) -> None:
        conversations_payload = conversation_directory_payload(
            [
                {
                    "openConversationId": "cid-fixture-project",
                    "conversationName": "项目群",
                },
                {
                    "openConversationId": "cid-fixture-direct",
                    "conversationName": "普通单聊",
                },
            ]
        )
        groups_payload = group_directory_payload(
            [{"openConversationId": "cid-fixture-project", "name": "项目群"}]
        )
        typed_row = {
            "conversationId": "cid-fixture-project",
            "createTime": "2026-08-20T10:00:00+08:00",
            "messageId": "normal-message-1",
            "sender": "测试用户",
            "senderId": "u-1",
            "text": "部署已经完成。",
        }
        normal_messages_payload = {
            "complete": True,
            "contractVersion": "im.message-list.v1",
            "count": 1,
            "failedCount": 0,
            "failures": [],
            "hasMore": False,
            "messages": [typed_row],
            "pagesFetched": 1,
            "paginationKnown": True,
            "partial": False,
            "queryRange": {
                "startTime": "2026-08-17T00:00:00+08:00",
                "endTime": "2026-08-24T00:00:00+08:00",
                "order": "asc",
                "semantics": "[start,end)",
            },
            "stopReason": "range_end",
            "truncated": False,
            "truncatedByPageLimit": False,
            "truncatedByResultLimit": False,
        }
        detail_payload = {
            "success": True,
            "errorCode": None,
            "errorMsg": None,
            "result": {
                "messages": [
                    {
                        "content": "部署已经完成。",
                        "createTime": "2026-08-20T10:00:00+08:00",
                        "openConversationId": "cid-fixture-project",
                        "openMessageId": "normal-message-1",
                        "sender": "测试用户",
                        "senderOpenDingTalkId": "u-1",
                    }
                ]
            },
        }
        member_payload = {
            "bots": [],
            "complete": True,
            "contractVersion": "im.group-members.v1",
            "conversationId": "cid-fixture-project",
            "counts": {"bots": 0, "total": 1, "users": 1},
            "failedCount": 0,
            "failures": [],
            "hasMore": False,
            "partial": False,
            "users": [{"name": "测试用户", "openDingtalkId": "u-1"}],
        }

        class RecordingAdapter(DwsAdapter):
            def __init__(self) -> None:
                super().__init__(PROJECT_ROOT / "unused.exe", max_passes=2, settle_seconds=0)
                self.commands: list[list[str]] = []

            def preflight(self) -> dict:
                return {
                    "ok": True,
                    "expected_version": "v1.0.59",
                    "version_pinned": True,
                    "authenticated": True,
                    "version": "Version: v1.0.59",
                }

            def _run(self, arguments, *, operation):
                command = list(arguments)
                self.commands.append(command)
                if operation == "complete conversation directory":
                    return conversations_payload
                if operation == "complete group directory":
                    return groups_payload
                if "+chat-messages" in command:
                    return normal_messages_payload
                if "list-by-ids" in command:
                    return detail_payload
                if "+chat-members-list" in command:
                    return member_payload
                raise AssertionError((operation, command))

        adapter = RecordingAdapter()
        adapter.refresh_conversations()
        result = adapter.collect_selected(
            ["cid-fixture-project"],
            "2026-08-17T00:00:00+08:00",
            "2026-08-24T00:00:00+08:00",
        )
        self.assertTrue(result["data_complete"])
        self.assertTrue(result["sealed"])
        self.assertEqual("COMPLETE_SELECTED_SCOPE", result["status"])
        self.assertEqual("text", result["messages"][0]["type"])
        self.assertFalse(result["messages"][0]["is_bot"])
        self.assertFalse(result["messages"][0]["is_system"])
        self.assertTrue(all("--profile" not in command for command in adapter.commands))
        conversation_directory = adapter.commands[0]
        group_directory = adapter.commands[1]
        self.assertEqual(["chat", "+conversation-list"], conversation_directory[:2])
        self.assertEqual("100", conversation_directory[conversation_directory.index("--limit") + 1])
        self.assertIn("--page-all", conversation_directory)
        self.assertEqual(["chat", "+chat-list-all"], group_directory[:2])
        self.assertEqual("200", group_directory[group_directory.index("--limit") + 1])
        self.assertIn("--page-all", group_directory)
        normal_reads = [command for command in adapter.commands if "+chat-messages" in command]
        self.assertEqual(2, len(normal_reads))
        self.assertTrue(all("--page-all" in command for command in normal_reads))
        self.assertTrue(all("--start" in command and "--end" in command for command in normal_reads))
        self.assertFalse(any("list-all" in command for command in adapter.commands))
        self.assertFalse(any("search-advanced" in command for command in adapter.commands))

    def test_transient_contact_identity_failure_is_retried_in_later_pass(self) -> None:
        typed_row = {
            "conversationId": "cid-direct",
            "createTime": "2026-08-20T10:00:00+08:00",
            "messageId": "normal-message-1",
            "sender": "测试用户",
            "senderId": "u-1",
            "text": "部署已经完成。",
        }
        normal_messages_payload = {
            "complete": True,
            "contractVersion": "im.message-list.v1",
            "count": 1,
            "failedCount": 0,
            "failures": [],
            "hasMore": False,
            "messages": [typed_row],
            "pagesFetched": 1,
            "paginationKnown": True,
            "partial": False,
            "queryRange": {
                "startTime": "2026-08-17T00:00:00+08:00",
                "endTime": "2026-08-24T00:00:00+08:00",
                "order": "asc",
                "semantics": "[start,end)",
            },
            "stopReason": "range_end",
            "truncated": False,
            "truncatedByPageLimit": False,
            "truncatedByResultLimit": False,
        }
        detail_payload = {
            "success": True,
            "errorCode": None,
            "errorMsg": None,
            "result": {
                "messages": [
                    {
                        "content": "部署已经完成。",
                        "createTime": "2026-08-20T10:00:00+08:00",
                        "openConversationId": "cid-direct",
                        "openMessageId": "normal-message-1",
                        "sender": "测试用户",
                        "senderOpenDingTalkId": "u-1",
                    }
                ]
            },
        }

        class RetryingIdentityAdapter(DwsAdapter):
            def __init__(self) -> None:
                super().__init__(
                    PROJECT_ROOT / "unused.exe", max_passes=4, settle_seconds=0
                )
                self.contact_calls = 0

            def preflight(self) -> dict:
                return {
                    "ok": True,
                    "expected_version": "v1.0.59",
                    "version_pinned": True,
                    "authenticated": True,
                    "version": "Version: v1.0.59",
                }

            def _run(self, arguments, *, operation):
                command = list(arguments)
                if "+chat-messages" in command:
                    return normal_messages_payload
                if "list-by-ids" in command:
                    return detail_payload
                if command[:3] == ["contact", "user", "search"]:
                    self.contact_calls += 1
                    if self.contact_calls == 1:
                        raise DwsAdapterError(
                            "temporary identity lookup failure", code="TEMPORARY"
                        )
                    return {
                        "success": True,
                        "result": [{"openDingTalkId": "u-1"}],
                    }
                raise AssertionError((operation, command))

        adapter = RetryingIdentityAdapter()
        result = adapter.collect_selected(
            ["cid-direct"],
            "2026-08-17T00:00:00+08:00",
            "2026-08-24T00:00:00+08:00",
            passes=2,
            conversation_metadata=[
                {"id": "cid-direct", "title": "测试单聊", "kind": "direct"}
            ],
        )
        self.assertEqual(2, adapter.contact_calls)
        self.assertEqual(3, result["integrity"]["passes_executed"])
        self.assertEqual(0, result["integrity"]["passes"][-1]["normal_chat_paging"]["origin"]["unknown"])
        self.assertFalse(result["messages"][0]["is_bot"])

    def test_each_pass_ledger_has_utc_timing_and_content_free_scope_summary(self) -> None:
        adapter = FixtureDwsAdapter(FIXTURES)
        result = adapter.collect_selected(
            ["cid-fixture-project", "cid-fixture-direct"],
            "2026-08-17T00:00:00+08:00",
            "2026-08-24T00:00:00+08:00",
        )
        self.assertEqual(2, len(result["integrity"]["passes"]))
        for ledger in result["integrity"]["passes"]:
            started = dt.datetime.fromisoformat(ledger["started_at"].replace("Z", "+00:00"))
            finished = dt.datetime.fromisoformat(ledger["finished_at"].replace("Z", "+00:00"))
            self.assertEqual(dt.timedelta(0), started.utcoffset())
            self.assertEqual(dt.timedelta(0), finished.utcoffset())
            self.assertLessEqual(started, finished)
            self.assertIs(type(ledger["duration_ms"]), int)
            self.assertGreaterEqual(ledger["duration_ms"], 0)
            self.assertEqual("left_closed_right_open", ledger["window"]["interval"])
            self.assertEqual(2, ledger["selection"]["conversation_count"])
            self.assertEqual(64, len(ledger["selection"]["conversation_ids_digest"]))
            serialized = json.dumps(ledger, ensure_ascii=False)
            self.assertNotIn("确认本次测试范围和验收标准", serialized)
            self.assertNotIn("风险：测试接口需要补充边界用例", serialized)


if __name__ == "__main__":
    unittest.main()
