"""Fail-closed DingTalk Workspace CLI adapters.

The panel deliberately treats DWS as an untrusted changing data source.  This
module validates the CLI version, authentication state, output shape and every
pagination signal before exposing conversations or messages.  The real
conversation directory is built from the fully paged ordinary-conversation and
joined-group sources; either source failing its completion ledger aborts the
refresh instead of publishing a partial directory.

The real adapter reads every explicitly selected ordinary conversation through
``chat +chat-messages``.  That route works for both direct and group chats and
does not require the optional global message-search entitlement.  Every
conversation must return the typed ``im.message-list.v1`` completion ledger;
message identities are then re-read in batches and sender identity is checked
before a frozen projection is exposed.  Repeated full rounds still have to
converge, so messages arriving while collection is running cannot silently
produce a transient snapshot.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any


EXPECTED_DWS_VERSION = "v1.0.59"
SHANGHAI = dt.timezone(dt.timedelta(hours=8), name="Asia/Shanghai")
ISSUE_249 = (
    "DWS upstream issue #249 may omit some direct conversations even when "
    "pagination reports hasMore=false; the selected-scope result is provisional."
)

NORMAL_MESSAGE_CONTRACT = "im.message-list.v1"
GROUP_MEMBERS_CONTRACT = "im.group-members.v1"
NORMAL_SCOPE_STATUS = "COMPLETE_SELECTED_SCOPE"


class DwsAdapterError(RuntimeError):
    """Raised when source completeness or data identity cannot be proven."""

    def __init__(self, message: str, *, code: str = "DWS_ERROR", details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "details": self.details}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _iso_utc(value: dt.datetime | None = None) -> str:
    current = value or dt.datetime.now(dt.timezone.utc)
    return current.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _parse_time(value: Any, *, field: str = "time") -> dt.datetime:
    """Parse a DWS timestamp and normalize it to the Shanghai fixed offset."""

    if isinstance(value, bool) or value in (None, ""):
        raise DwsAdapterError(f"missing {field}", code="INVALID_TIME", details=value)
    parsed: dt.datetime | None = None
    if isinstance(value, (int, float)):
        number = float(value)
        while abs(number) > 100_000_000_000:
            number /= 1000.0
        try:
            parsed = dt.datetime.fromtimestamp(number, tz=dt.timezone.utc)
        except (OSError, OverflowError, ValueError) as exc:
            raise DwsAdapterError(
                f"invalid {field}", code="INVALID_TIME", details=value
            ) from exc
    else:
        raw = str(value).strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
            return _parse_time(float(raw), field=field)
        normalized = raw.replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%Y-%m-%d",
            ):
                try:
                    parsed = dt.datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    continue
    if parsed is None:
        raise DwsAdapterError(f"invalid {field}", code="INVALID_TIME", details=value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHANGHAI)
    parsed = parsed.astimezone(SHANGHAI)
    if not (
        dt.datetime(2000, 1, 1, tzinfo=SHANGHAI)
        <= parsed
        <= dt.datetime(2100, 1, 1, tzinfo=SHANGHAI)
    ):
        raise DwsAdapterError(f"out-of-range {field}", code="INVALID_TIME", details=value)
    return parsed


def _window(start: str | dt.datetime, end: str | dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    parsed_start = _parse_time(start, field="start")
    parsed_end = _parse_time(end, field="end")
    if parsed_end <= parsed_start:
        raise DwsAdapterError("end must be after start", code="INVALID_WINDOW")
    if parsed_start.microsecond or parsed_end.microsecond:
        raise DwsAdapterError(
            "start and end must use whole seconds because DWS list-all has second precision",
            code="INVALID_WINDOW",
        )
    return parsed_start, parsed_end


def _recursive_text(value: Any) -> str:
    chunks: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            text = item.strip()
            if text:
                chunks.append(text)
            return
        if isinstance(item, Mapping):
            preferred = ("text", "plainText", "richText", "title", "content", "caption", "fileName")
            visited: set[str] = set()
            for key in preferred:
                if key in item:
                    visited.add(key)
                    visit(item[key])
            for key, nested in item.items():
                if key not in visited and key not in {
                    "mediaId", "fileId", "openMessageId", "openConversationId",
                    "senderOpenDingTalkId", "downloadUrl", "url",
                }:
                    visit(nested)
            return
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            for nested in item:
                visit(nested)

    visit(value)
    compact: list[str] = []
    for chunk in chunks:
        if not compact or compact[-1] != chunk:
            compact.append(chunk)
    return "\n".join(compact)


def _decode_single_json(stdout: str, *, operation: str) -> Mapping[str, Any]:
    text = stdout.lstrip("\ufeff \t\r\n")
    if not text:
        raise DwsAdapterError(
            f"{operation} returned no JSON", code="INVALID_JSON_OUTPUT"
        )
    decoder = json.JSONDecoder()
    try:
        value, offset = decoder.raw_decode(text)
    except json.JSONDecodeError as exc:
        raise DwsAdapterError(
            f"{operation} returned invalid JSON",
            code="INVALID_JSON_OUTPUT",
            details={"line": exc.lineno, "column": exc.colno},
        ) from exc
    if text[offset:].strip():
        raise DwsAdapterError(
            f"{operation} returned trailing non-JSON output",
            code="INVALID_JSON_OUTPUT",
        )
    if not isinstance(value, Mapping):
        raise DwsAdapterError(
            f"{operation} JSON root is not an object", code="INVALID_OUTPUT_SHAPE"
        )
    return value


def _strict_bool(mapping: Mapping[str, Any], key: str, expected: bool, *, where: str) -> None:
    if key not in mapping or mapping.get(key) is not expected:
        raise DwsAdapterError(
            f"{where}.{key} is not exactly {str(expected).lower()}",
            code="PAGINATION_UNPROVEN",
            details=mapping.get(key),
        )


def _conversation_id(mapping: Mapping[str, Any]) -> str | None:
    value = _first(
        mapping,
        ("openConversationId", "conversationId", "conversation_id", "chatId", "id"),
    )
    return str(value).strip() if value not in (None, "") else None


def _conversation_kind(raw: Mapping[str, Any]) -> str:
    single_chat = _first(raw, ("singleChat", "single_chat", "isSingleChat"))
    if single_chat is True:
        return "direct"
    if single_chat is False:
        return "group"
    value = _first(raw, ("kind", "conversationType", "conversation_type", "type", "chatType"))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # DWS/IM schemas commonly use 1 for direct and 2 for group.  Unknown
        # numeric values remain unknown instead of being guessed.
        return {1: "direct", 2: "group"}.get(int(value), "unknown")
    lowered = str(value or "").strip().lower()
    if lowered in {"1", "direct", "single", "private", "p2p", "one_to_one", "one-to-one"}:
        return "direct"
    if lowered in {"2", "group", "groupchat", "group_chat", "multi"}:
        return "group"
    if raw.get("isGroup") is True:
        return "group"
    if raw.get("isGroup") is False:
        return "direct"
    return "unknown"


def _participant_list(raw: Mapping[str, Any]) -> list[dict[str, str]]:
    value = _first(raw, ("participants", "members", "users", "memberList"))
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, Mapping):
            identifier = _first(item, ("openDingTalkId", "userId", "id", "staffId"))
            name = _first(item, ("name", "nick", "displayName", "title"))
            if identifier in (None, "") and name in (None, ""):
                continue
            key = str(identifier or name)
            if key in seen:
                continue
            seen.add(key)
            result.append({"id": str(identifier or ""), "name": str(name or identifier or "")})
        elif isinstance(item, str) and item.strip() and item not in seen:
            seen.add(item)
            result.append({"id": "", "name": item.strip()})
    return result


def _normalize_conversation(raw: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    conversation_id = _conversation_id(raw)
    if not conversation_id:
        raise DwsAdapterError(
            "conversation has no stable id", code="WEAK_CONVERSATION_ID", details=raw
        )
    title = _first(raw, ("title", "name", "conversationTitle", "conversationName", "displayName"))
    last_active_raw = _first(
        raw,
        (
            "lastActive", "lastActiveTime", "latestMessageTime", "lastMessageTime",
            "lastMsgCreateAt", "lastMsgCreateTime", "updatedAt",
        ),
    )
    last_active: str | None = None
    if last_active_raw not in (None, ""):
        last_active = _parse_time(last_active_raw, field="conversation.last_active").isoformat()
    available_raw = raw.get("available", True)
    if type(available_raw) is not bool:
        raise DwsAdapterError(
            "conversation.available is not a boolean",
            code="INVALID_OUTPUT_SHAPE",
            details=available_raw,
        )
    return {
        "id": conversation_id,
        "title": str(title or conversation_id),
        "kind": _conversation_kind(raw),
        "last_active": last_active,
        "participants": _participant_list(raw),
        "available": available_raw,
        "source": source,
        "raw": dict(raw),
    }


def _validate_complete_directory_page(
    payload: Mapping[str, Any],
    *,
    rows_key: str,
    where: str,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if "error" in payload:
        raise DwsAdapterError(
            f"DWS {where} returned an error",
            code="DWS_RESPONSE_ERROR",
            details=payload["error"],
        )
    _strict_bool(payload, "complete", True, where=where)
    _strict_bool(payload, "paginationKnown", True, where=where)
    _strict_bool(payload, "partial", False, where=where)
    _strict_bool(payload, "truncated", False, where=where)
    _strict_bool(payload, "truncatedByPageLimit", False, where=where)
    _strict_bool(payload, "truncatedByResultLimit", False, where=where)
    _strict_bool(payload, "hasMore", False, where=where)
    pages = payload.get("pagesFetched")
    if type(pages) is not int or pages < 1:
        raise DwsAdapterError(
            f"{where} has no valid page count", code="PAGINATION_UNPROVEN"
        )
    count = payload.get("count")
    if type(count) is not int or count < 0:
        raise DwsAdapterError(
            f"{where} has no valid item count", code="PAGINATION_UNPROVEN"
        )
    failed_count = payload.get("failedCount")
    failures = payload.get("failures")
    if type(failed_count) is not int or failed_count != 0:
        raise DwsAdapterError(
            f"{where} reported failed pages",
            code="PAGINATION_UNPROVEN",
            details={"failedCount": failed_count, "failures": failures},
        )
    if not isinstance(failures, list) or failures:
        raise DwsAdapterError(
            f"{where} contains failures", code="PAGINATION_UNPROVEN"
        )
    next_cursor = payload.get("nextCursor")
    if not (
        next_cursor is None
        or next_cursor == ""
        or (type(next_cursor) is int and next_cursor == 0)
    ):
        raise DwsAdapterError(
            f"{where} still exposes a non-terminal cursor", code="PAGINATION_UNPROVEN"
        )
    if payload.get("stopReason") != "source_complete":
        raise DwsAdapterError(
            f"{where} has no source-complete stop reason",
            code="PAGINATION_UNPROVEN",
            details=payload.get("stopReason"),
        )
    rows = payload.get(rows_key)
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise DwsAdapterError(
            f"{where} has no valid {rows_key} array",
            code="INVALID_OUTPUT_SHAPE",
        )
    if count != len(rows):
        raise DwsAdapterError(
            f"{where} count does not match parsed rows",
            code="PAGINATION_UNPROVEN",
            details={"declared": count, "parsed": len(rows)},
        )
    for index, row in enumerate(rows):
        conversation_id = row.get("openConversationId")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise DwsAdapterError(
                f"{where} row has no stable openConversationId",
                code="WEAK_CONVERSATION_ID",
                details={"row": index},
            )
    return rows, {
        "complete": True,
        "pages": pages,
        "count": count,
        "pagination_known": True,
        "has_more": False,
        "truncated": False,
        "partial": False,
        "failed_count": 0,
        "stop_reason": "source_complete",
        "next_cursor": next_cursor,
    }


def _validate_conversation_page(
    payload: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    return _validate_complete_directory_page(
        payload,
        rows_key="conversations",
        where="conversation_page",
    )


def _validate_group_page(
    payload: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    return _validate_complete_directory_page(
        payload,
        rows_key="groups",
        where="group_page",
    )


def _validate_typed_conversation_page(
    payload: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Validate the typed ``+chat-list`` directory projection.

    DWS v1.0.59 deliberately reports a full first page with no usable cursor as
    ``complete=false`` / ``single_page_full_untrusted``.  That exact response is
    useful as a *limited discovery directory* because every returned row has a
    stable ID and a projected direct/group type.  It must not be accepted for
    any other incomplete, partial, failed, or ambiguous pagination state.
    """

    if "error" in payload:
        raise DwsAdapterError(
            "DWS typed conversation directory returned an error",
            code="DWS_RESPONSE_ERROR",
            details=payload["error"],
        )
    for key in ("complete", "paginationKnown", "partial", "hasMore", "truncatedByPageLimit"):
        if type(payload.get(key)) is not bool:
            raise DwsAdapterError(
                f"typed_conversation_page.{key} is not a boolean",
                code="PAGINATION_UNPROVEN",
                details=payload.get(key),
            )
    _strict_bool(payload, "partial", False, where="typed_conversation_page")
    _strict_bool(
        payload, "truncatedByPageLimit", False, where="typed_conversation_page"
    )
    _strict_bool(payload, "hasMore", False, where="typed_conversation_page")

    pages = payload.get("pagesFetched")
    count = payload.get("count")
    requested_page_size = payload.get("requestedPageSize")
    effective_page_size = payload.get("effectivePageSize")
    if type(pages) is not int or pages != 1:
        raise DwsAdapterError(
            "typed conversation directory must contain exactly one audited page",
            code="PAGINATION_UNPROVEN",
            details=pages,
        )
    if type(count) is not int or not 0 <= count <= 100:
        raise DwsAdapterError(
            "typed conversation directory has no valid item count",
            code="PAGINATION_UNPROVEN",
            details=count,
        )
    if requested_page_size != 100 or effective_page_size != 100:
        raise DwsAdapterError(
            "typed conversation directory did not use the pinned 100-row window",
            code="PAGINATION_UNPROVEN",
            details={
                "requestedPageSize": requested_page_size,
                "effectivePageSize": effective_page_size,
            },
        )

    failed_count = payload.get("failedCount")
    failures = payload.get("failures")
    if type(failed_count) is not int or failed_count != 0:
        raise DwsAdapterError(
            "typed conversation directory reported failed pages",
            code="PAGINATION_UNPROVEN",
            details={"failedCount": failed_count, "failures": failures},
        )
    if not isinstance(failures, list) or failures:
        raise DwsAdapterError(
            "typed conversation directory contains failures",
            code="PAGINATION_UNPROVEN",
            details=failures,
        )

    requested_types = payload.get("requestedTypes")
    if requested_types != ["group", "p2p"]:
        raise DwsAdapterError(
            "typed conversation directory did not prove both group and direct scope",
            code="INVALID_OUTPUT_SHAPE",
            details=requested_types,
        )

    rows = payload.get("chats")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise DwsAdapterError(
            "typed conversation directory has no valid chats array",
            code="INVALID_OUTPUT_SHAPE",
        )
    if count != len(rows):
        raise DwsAdapterError(
            "typed conversation count does not match parsed rows",
            code="PAGINATION_UNPROVEN",
            details={"declared": count, "parsed": len(rows)},
        )

    for index, row in enumerate(rows):
        open_conversation_id = row.get("openConversationId")
        if (
            not isinstance(open_conversation_id, str)
            or not open_conversation_id.strip()
        ):
            raise DwsAdapterError(
                "typed conversation row has no stable openConversationId",
                code="WEAK_CONVERSATION_ID",
                details={"row": index},
            )
        conversation_type = row.get("conversationType")
        chat_mode = row.get("chatMode")
        expected_mode = {"direct": "p2p", "group": "group"}.get(conversation_type)
        if expected_mode is None or chat_mode != expected_mode:
            raise DwsAdapterError(
                "typed conversation row has no consistent direct/group classification",
                code="CONVERSATION_TYPE_UNPROVEN",
                details={
                    "row": index,
                    "conversationType": conversation_type,
                    "chatMode": chat_mode,
                },
            )

    complete = payload["complete"]
    pagination_known = payload["paginationKnown"]
    stop_reason = payload.get("stopReason")
    next_cursor = payload.get("nextCursor")
    if type(next_cursor) is not int or next_cursor != 0:
        raise DwsAdapterError(
            "typed conversation directory has no exact zero terminal cursor",
            code="PAGINATION_UNPROVEN",
            details=next_cursor,
        )
    if complete:
        if (
            pagination_known is not True
            or count >= effective_page_size
            or stop_reason != "source_complete"
            or payload.get("completionEvidence") != "backend_has_more_false_short_page"
        ):
            raise DwsAdapterError(
                "typed conversation completion ledger is internally inconsistent",
                code="PAGINATION_UNPROVEN",
                details={
                    "count": count,
                    "paginationKnown": pagination_known,
                    "stopReason": stop_reason,
                    "nextCursor": next_cursor,
                    "completionEvidence": payload.get("completionEvidence"),
                },
            )
    elif not (
        pagination_known is False
        and stop_reason == "single_page_full_untrusted"
    ):
        raise DwsAdapterError(
            "typed conversation directory has an unsupported incomplete ledger",
            code="PAGINATION_UNPROVEN",
            details={
                "count": count,
                "paginationKnown": pagination_known,
                "stopReason": stop_reason,
                "nextCursor": next_cursor,
            },
        )

    return rows, {
        "complete": complete,
        "pages": pages,
        "count": count,
        "pagination_known": pagination_known,
        "has_more": False,
        "truncated": False,
        "source_limited": not complete,
        "directory_incomplete": not complete,
        "stop_reason": stop_reason,
        "next_cursor": next_cursor,
        "source_count": count,
        "typed": True,
    }


def _validate_message_page(payload: Mapping[str, Any], *, source: str) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], dict[str, Any]]:
    if "error" in payload:
        raise DwsAdapterError(
            f"{source} returned an error",
            code="DWS_RESPONSE_ERROR",
            details=payload["error"],
        )
    if "success" in payload and payload.get("success") is not True:
        raise DwsAdapterError(
            f"{source} returned a non-success envelope",
            code="DWS_RESPONSE_ERROR",
            details=payload,
        )
    paging = payload.get("paging")
    if not isinstance(paging, Mapping):
        raise DwsAdapterError(
            f"{source} has no paging ledger", code="PAGINATION_UNPROVEN"
        )
    _strict_bool(paging, "truncated", False, where=f"{source}.paging")
    _strict_bool(paging, "hasMore", False, where=f"{source}.paging")
    if "partial" in paging:
        _strict_bool(paging, "partial", False, where=f"{source}.paging")
    pages = paging.get("pages")
    total = paging.get("total")
    if type(pages) is not int or pages < 1:
        raise DwsAdapterError(
            f"{source} has no valid page count", code="PAGINATION_UNPROVEN"
        )
    if type(total) is not int or total < 0:
        raise DwsAdapterError(
            f"{source} has no valid total", code="PAGINATION_UNPROVEN"
        )
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise DwsAdapterError(
            f"{source} has no result object", code="INVALID_OUTPUT_SHAPE"
        )
    hint = _first(payload, ("friendly_hint", "friendlyHint", "action_url", "actionUrl"))
    hint = hint or _first(result, ("friendly_hint", "friendlyHint", "action_url", "actionUrl"))
    if hint:
        raise DwsAdapterError(
            f"{source} requires an entitlement or permission",
            code="ENTITLEMENT_REQUIRED",
            details=hint,
        )
    _strict_bool(result, "hasMore", False, where=f"{source}.result")
    if result.get("nextCursor") not in (None, "", 0):
        raise DwsAdapterError(
            f"{source} still exposes a result cursor", code="PAGINATION_UNPROVEN"
        )
    groups = result.get("conversationMessagesList")
    if not isinstance(groups, list):
        raise DwsAdapterError(
            f"{source} has no conversationMessagesList", code="INVALID_OUTPUT_SHAPE"
        )

    def competing_container(value: Any) -> bool:
        # Empty arrays are commonly emitted as compatibility placeholders.  A
        # non-empty array, object, or any other non-null structured-looking
        # value competes with conversationMessagesList and makes row ownership
        # ambiguous.
        if value is None:
            return False
        if isinstance(value, list):
            return bool(value)
        if isinstance(value, Mapping):
            return True
        return value != ""

    competing: list[str] = []
    for label, candidate in (
        ("payload.messages", payload.get("messages")),
        ("payload.conversationMessagesList", payload.get("conversationMessagesList")),
        ("result.messages", result.get("messages")),
    ):
        if competing_container(candidate):
            competing.append(label)
    if competing:
        raise DwsAdapterError(
            f"{source} exposes competing message containers",
            code="AMBIGUOUS_MESSAGE_CONTAINERS",
            details={"primary": "result.conversationMessagesList", "competing": competing},
        )
    flattened: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise DwsAdapterError(
                f"{source} group {group_index} is malformed", code="INVALID_OUTPUT_SHAPE"
            )
        if not _conversation_id(group):
            raise DwsAdapterError(
                f"{source} group {group_index} has no conversation id",
                code="WEAK_CONVERSATION_ID",
            )
        if "hasMore" in group:
            _strict_bool(group, "hasMore", False, where=f"{source}.group[{group_index}]")
        if group.get("nextCursor") not in (None, "", 0):
            raise DwsAdapterError(
                f"{source} group {group_index} still exposes a cursor",
                code="PAGINATION_UNPROVEN",
            )
        messages = group.get("messages")
        if not isinstance(messages, list) or any(not isinstance(row, Mapping) for row in messages):
            raise DwsAdapterError(
                f"{source} group {group_index} has no valid messages array",
                code="INVALID_OUTPUT_SHAPE",
            )
        group_conversation_id = _conversation_id(group)
        for row_index, row in enumerate(messages):
            row_conversation_id = _conversation_id(row)
            if (
                row_conversation_id is not None
                and group_conversation_id is not None
                and row_conversation_id != group_conversation_id
            ):
                raise DwsAdapterError(
                    f"{source} group and message row disagree on conversation id",
                    code="CONVERSATION_ID_CONFLICT",
                    details={
                        "group_index": group_index,
                        "row_index": row_index,
                        "group_conversation_id": group_conversation_id,
                        "row_conversation_id": row_conversation_id,
                    },
                )
        flattened.extend((group, row) for row in messages)
    if len(flattened) != total:
        raise DwsAdapterError(
            f"{source} total does not match parsed messages",
            code="PAGINATION_UNPROVEN",
            details={"declared": total, "parsed": len(flattened)},
        )
    return flattened, {
        "complete": True,
        "pages": pages,
        "total": total,
        "has_more": False,
        "truncated": False,
    }


def _message_id(raw: Mapping[str, Any]) -> str | None:
    value = _first(raw, ("openMessageId", "messageId", "openMsgId", "msgId"))
    return str(value).strip() if value not in (None, "") else None


def _sender_id(raw: Mapping[str, Any]) -> str | None:
    value = _first(raw, ("senderOpenDingTalkId", "senderId", "senderUserId"))
    return str(value).strip() if value not in (None, "") else None


def _validate_scoped_message_page(
    payload: Mapping[str, Any],
    *,
    conversation_id: str,
    start: dt.datetime,
    end: dt.datetime,
    source: str,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Validate one fully paged ``chat +chat-messages`` response."""

    if payload.get("contractVersion") != NORMAL_MESSAGE_CONTRACT:
        raise DwsAdapterError(
            f"{source} has an unexpected message contract",
            code="INVALID_OUTPUT_SHAPE",
            details=payload.get("contractVersion"),
        )
    for key, expected in (
        ("complete", True),
        ("paginationKnown", True),
        ("partial", False),
        ("truncated", False),
        ("truncatedByPageLimit", False),
        ("truncatedByResultLimit", False),
        ("hasMore", False),
    ):
        _strict_bool(payload, key, expected, where=source)

    pages = payload.get("pagesFetched")
    count = payload.get("count")
    if type(pages) is not int or pages < 1:
        raise DwsAdapterError(
            f"{source} has no valid page count", code="PAGINATION_UNPROVEN"
        )
    if type(count) is not int or count < 0:
        raise DwsAdapterError(
            f"{source} has no valid message count", code="PAGINATION_UNPROVEN"
        )
    if payload.get("failedCount") != 0:
        raise DwsAdapterError(
            f"{source} reported failed pages",
            code="PAGINATION_UNPROVEN",
            details={
                "failedCount": payload.get("failedCount"),
                "failures": payload.get("failures"),
            },
        )
    failures = payload.get("failures")
    if not isinstance(failures, list) or failures:
        raise DwsAdapterError(
            f"{source} contains page failures",
            code="PAGINATION_UNPROVEN",
            details=failures,
        )
    if payload.get("stopReason") not in {"source_complete", "range_end"}:
        raise DwsAdapterError(
            f"{source} has no terminal stop reason",
            code="PAGINATION_UNPROVEN",
            details=payload.get("stopReason"),
        )
    if payload.get("nextPage") is not None:
        raise DwsAdapterError(
            f"{source} still exposes a next page",
            code="PAGINATION_UNPROVEN",
        )

    query_range = payload.get("queryRange")
    if not isinstance(query_range, Mapping):
        raise DwsAdapterError(
            f"{source} has no query range", code="INVALID_OUTPUT_SHAPE"
        )
    if query_range.get("semantics") != "[start,end)" or query_range.get("order") != "asc":
        raise DwsAdapterError(
            f"{source} did not preserve the requested half-open ascending range",
            code="PAGINATION_UNPROVEN",
            details={
                "semantics": query_range.get("semantics"),
                "order": query_range.get("order"),
            },
        )
    actual_start = _parse_time(query_range.get("startTime"), field=f"{source}.start")
    actual_end = _parse_time(query_range.get("endTime"), field=f"{source}.end")
    if actual_start != start or actual_end != end:
        raise DwsAdapterError(
            f"{source} returned a different time range",
            code="PAGINATION_UNPROVEN",
            details={
                "requested_start": start.isoformat(),
                "requested_end": end.isoformat(),
                "actual_start": actual_start.isoformat(),
                "actual_end": actual_end.isoformat(),
            },
        )

    rows = payload.get("messages")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise DwsAdapterError(
            f"{source} has no valid messages array", code="INVALID_OUTPUT_SHAPE"
        )
    if len(rows) != count:
        raise DwsAdapterError(
            f"{source} count does not match parsed messages",
            code="PAGINATION_UNPROVEN",
            details={"declared": count, "parsed": len(rows)},
        )

    seen: set[str] = set()
    for index, row in enumerate(rows):
        if _conversation_id(row) != conversation_id:
            raise DwsAdapterError(
                f"{source} returned a message from another conversation",
                code="CONVERSATION_ID_CONFLICT",
                details={"row": index},
            )
        message_id = _message_id(row)
        if not message_id:
            raise DwsAdapterError(
                f"{source} returned a message without a stable id",
                code="WEAK_MESSAGE_ID",
                details={"row": index},
            )
        if message_id in seen:
            raise DwsAdapterError(
                f"{source} returned a duplicate message id",
                code="MESSAGE_ID_COLLISION",
                details={"row": index},
            )
        seen.add(message_id)
        if not _sender_id(row):
            raise DwsAdapterError(
                f"{source} returned a message without a stable sender id",
                code="WEAK_SENDER_ID",
                details={"row": index, "message_id": message_id},
            )
        created = _parse_time(row.get("createTime"), field=f"{source}.message_time")
        if not (start <= created < end):
            raise DwsAdapterError(
                f"{source} returned a message outside the requested range",
                code="OUT_OF_SCOPE_MESSAGE",
                details={"row": index, "message_id": message_id},
            )

    return rows, {
        "complete": True,
        "pages": pages,
        "count": count,
        "pagination_known": True,
        "has_more": False,
        "truncated": False,
        "partial": False,
        "failed_count": 0,
        "stop_reason": payload.get("stopReason"),
        "contract_version": NORMAL_MESSAGE_CONTRACT,
    }


def _validate_message_detail_batch(
    payload: Mapping[str, Any], requested_ids: Sequence[str], *, source: str
) -> dict[str, Mapping[str, Any]]:
    """Require an exact ``message list-by-ids`` response for one <=50 batch."""

    if payload.get("success") is not True:
        raise DwsAdapterError(
            f"{source} returned a non-success envelope",
            code="DWS_RESPONSE_ERROR",
            details={"errorCode": payload.get("errorCode")},
        )
    if payload.get("errorCode") not in (None, "", 0, "0"):
        raise DwsAdapterError(
            f"{source} returned an error code",
            code="DWS_RESPONSE_ERROR",
            details=payload.get("errorCode"),
        )
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise DwsAdapterError(
            f"{source} has no result object", code="INVALID_OUTPUT_SHAPE"
        )
    rows = result.get("messages")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise DwsAdapterError(
            f"{source} has no valid messages array", code="INVALID_OUTPUT_SHAPE"
        )
    expected = list(requested_ids)
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise DwsAdapterError(
            f"{source} was called with duplicate message ids", code="MESSAGE_ID_COLLISION"
        )
    output: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        message_id = _message_id(row)
        if not message_id or message_id not in expected_set:
            raise DwsAdapterError(
                f"{source} returned an unrequested or weak message id",
                code="WEAK_MESSAGE_ID",
                details={"row": index},
            )
        if message_id in output:
            raise DwsAdapterError(
                f"{source} returned a duplicate message id",
                code="MESSAGE_ID_COLLISION",
                details={"row": index},
            )
        output[message_id] = row
    if set(output) != expected_set:
        raise DwsAdapterError(
            f"{source} did not return every requested message detail",
            code="MESSAGE_DETAILS_INCOMPLETE",
            details={
                "requested_count": len(expected_set),
                "returned_count": len(output),
                "missing_count": len(expected_set - set(output)),
            },
        )
    return output


def _validate_group_identity_page(
    payload: Mapping[str, Any], *, conversation_id: str, source: str
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Return disjoint, fully paged human and bot member identity sets."""

    if payload.get("contractVersion") != GROUP_MEMBERS_CONTRACT:
        raise DwsAdapterError(
            f"{source} has an unexpected member contract",
            code="INVALID_OUTPUT_SHAPE",
            details=payload.get("contractVersion"),
        )
    for key, expected in (("complete", True), ("partial", False), ("hasMore", False)):
        _strict_bool(payload, key, expected, where=source)
    if payload.get("conversationId") != conversation_id:
        raise DwsAdapterError(
            f"{source} returned another conversation",
            code="CONVERSATION_ID_CONFLICT",
        )
    if payload.get("failedCount") != 0:
        raise DwsAdapterError(
            f"{source} reported member page failures",
            code="PAGINATION_UNPROVEN",
            details=payload.get("failures"),
        )
    failures = payload.get("failures")
    if not isinstance(failures, list) or failures:
        raise DwsAdapterError(
            f"{source} contains member failures",
            code="PAGINATION_UNPROVEN",
            details=failures,
        )
    users = payload.get("users")
    bots = payload.get("bots")
    if (
        not isinstance(users, list)
        or not isinstance(bots, list)
        or any(not isinstance(row, Mapping) for row in [*users, *bots])
    ):
        raise DwsAdapterError(
            f"{source} has no valid user/bot buckets", code="INVALID_OUTPUT_SHAPE"
        )

    def identity_set(rows: Sequence[Mapping[str, Any]], bucket: str) -> set[str]:
        output: set[str] = set()
        for index, row in enumerate(rows):
            value = _first(row, ("openDingTalkId", "openDingtalkId"))
            if value in (None, ""):
                raise DwsAdapterError(
                    f"{source} {bucket} member has no stable openDingTalkId",
                    code="WEAK_SENDER_ID",
                    details={"row": index},
                )
            identifier = str(value).strip()
            if identifier in output:
                raise DwsAdapterError(
                    f"{source} contains a duplicate {bucket} member",
                    code="SENDER_ID_COLLISION",
                    details={"row": index},
                )
            output.add(identifier)
        return output

    user_ids = identity_set(users, "user")
    bot_ids = identity_set(bots, "bot")
    if user_ids & bot_ids:
        raise DwsAdapterError(
            f"{source} classifies one identity as both user and bot",
            code="SENDER_ID_COLLISION",
        )
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or (
        counts.get("users") != len(user_ids)
        or counts.get("bots") != len(bot_ids)
        or counts.get("total") != len(user_ids) + len(bot_ids)
    ):
        raise DwsAdapterError(
            f"{source} member counts do not match the buckets",
            code="PAGINATION_UNPROVEN",
            details=counts,
        )
    return user_ids, bot_ids, {
        "complete": True,
        "user_count": len(user_ids),
        "bot_count": len(bot_ids),
        "failed_count": 0,
        "contract_version": GROUP_MEMBERS_CONTRACT,
    }


def _validate_contact_search_page(payload: Mapping[str, Any], *, source: str) -> set[str]:
    """Return positive human identity matches; absence is never proof of a bot."""

    if payload.get("success") is not True:
        raise DwsAdapterError(
            f"{source} returned a non-success envelope", code="DWS_RESPONSE_ERROR"
        )
    rows = payload.get("result")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise DwsAdapterError(
            f"{source} has no valid result array", code="INVALID_OUTPUT_SHAPE"
        )
    output: set[str] = set()
    for row in rows:
        value = _first(row, ("openDingTalkId", "openDingtalkId"))
        if value not in (None, ""):
            output.add(str(value).strip())
    return output


def _projected_message_type(
    row: Mapping[str, Any], detail: Mapping[str, Any]
) -> str:
    explicit = _first(row, ("msgType", "messageType", "type"))
    if explicit not in (None, ""):
        return str(explicit)
    resource_types: set[str] = set()
    resources = detail.get("resources")
    if isinstance(resources, list):
        for resource in resources:
            if isinstance(resource, Mapping):
                value = resource.get("resourceType")
                if isinstance(value, str) and value.strip():
                    resource_types.add(value.strip().casefold())
    if len(resource_types) == 1:
        return next(iter(resource_types))
    if len(resource_types) > 1:
        return "mixed_resource"
    references = row.get("resourceRefs")
    if isinstance(references, list) and references:
        reference_types = {
            str(reference.get("type") or "").casefold()
            for reference in references
            if isinstance(reference, Mapping)
        }
        if "fileid" in reference_types:
            return "file"
        return "resource"
    return "text"


def _message_status(raw: Mapping[str, Any], msg_type: str, text: str) -> str:
    status = str(_first(raw, ("status", "messageStatus", "lifecycle")) or "").upper()
    lowered_type = msg_type.lower()
    if (
        raw.get("isRecalled") is True
        or raw.get("recalled") is True
        or status in {"RECALL", "RECALLED", "WITHDRAWN", "DELETED"}
        or lowered_type in {"recall", "recalled", "withdraw", "withdrawn"}
    ):
        return "RECALLED"
    if "撤回了一条消息" in text and lowered_type in {"system", "recall", "recalled"}:
        return "RECALL_NOTICE_UNRESOLVED"
    return "ACTIVE"


def _explicit_boolean(
    mapping: Mapping[str, Any], keys: Sequence[str], *, field: str
) -> tuple[bool | None, list[str]]:
    values: list[tuple[str, bool]] = []
    for key in keys:
        if key not in mapping or mapping.get(key) is None:
            continue
        value = mapping.get(key)
        if type(value) is not bool:
            raise DwsAdapterError(
                f"{field} marker {key} is not a boolean",
                code="INVALID_BOOLEAN_MARKER",
                details={"field": field, "key": key, "value": value},
            )
        values.append((key, value))
    distinct = {value for _key, value in values}
    if len(distinct) > 1:
        raise DwsAdapterError(
            f"conflicting explicit {field} markers",
            code="BOOLEAN_MARKER_CONFLICT",
            details={"field": field, "markers": dict(values)},
        )
    return (values[0][1] if values else None), [key for key, _value in values]


def _actor_type_marker(sender: Mapping[str, Any]) -> tuple[bool | None, str | None]:
    robot_types = {
        "bot", "robot", "chatbot", "chat_bot", "ai_bot", "dingbot", "application",
    }
    human_types = {"human", "person", "user", "staff", "employee"}
    for key in ("senderType", "sender_type", "userType", "user_type", "actorType", "type"):
        if key not in sender or sender.get(key) in (None, ""):
            continue
        value = sender.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
        if normalized in robot_types:
            return True, key
        if normalized in human_types:
            return False, key
    return None, None


def _verified_origin_flags(
    raw: Mapping[str, Any], sender: Mapping[str, Any], msg_type: str
) -> dict[str, Any]:
    message_bot, message_bot_keys = _explicit_boolean(
        raw,
        ("is_bot", "isBot", "isRobot"),
        field="is_bot",
    )
    raw_sender_bot, raw_sender_bot_keys = _explicit_boolean(
        raw,
        ("sender_is_bot", "senderIsBot", "senderIsRobot"),
        field="sender_is_bot",
    )
    nested_sender_bot, sender_bot_keys = _explicit_boolean(
        sender,
        ("is_bot", "isBot", "isRobot"),
        field="sender_is_bot",
    )
    actor_bot, actor_type_key = _actor_type_marker(sender)
    sender_evidence = [
        value
        for value in (raw_sender_bot, nested_sender_bot, actor_bot)
        if value is not None
    ]
    if len(set(sender_evidence)) > 1:
        raise DwsAdapterError(
            "sender boolean and sender type disagree on robot identity",
            code="BOOLEAN_MARKER_CONFLICT",
            details={"field": "sender_is_bot", "actor_type_key": actor_type_key},
        )
    verified_sender_bot = sender_evidence[0] if sender_evidence else None
    overall_evidence = [value for value in (message_bot, verified_sender_bot) if value is not None]
    if len(set(overall_evidence)) > 1:
        raise DwsAdapterError(
            "message and sender markers disagree on robot identity",
            code="BOOLEAN_MARKER_CONFLICT",
            details={"field": "is_bot"},
        )
    verified_bot = overall_evidence[0] if overall_evidence else None

    explicit_system, system_keys = _explicit_boolean(
        raw,
        ("is_system", "isSystem", "isSystemMessage", "systemMessage"),
        field="is_system",
    )
    sender_system, sender_system_keys = _explicit_boolean(
        sender,
        ("is_system", "isSystem", "isSystemActor"),
        field="sender_is_system",
    )
    type_system = msg_type.casefold() in {"system", "system_message", "systemmessage"}
    system_evidence = [
        value for value in (explicit_system, sender_system) if value is not None
    ]
    if type_system:
        system_evidence.append(True)
    if len(set(system_evidence)) > 1:
        raise DwsAdapterError(
            "explicit markers and message type disagree on system identity",
            code="BOOLEAN_MARKER_CONFLICT",
            details={"field": "is_system", "message_type": msg_type},
        )
    verified_system = system_evidence[0] if system_evidence else None
    return {
        # Preserve the tri-state value.  ``None`` means the source did not
        # provide enough evidence; downstream filters must not silently turn
        # that into a verified human/non-system message.
        "is_bot": verified_bot,
        "sender_is_bot": verified_sender_bot,
        "is_system": verified_system,
        "origin_verification": {
            "bot_verified": verified_bot is not None,
            "sender_bot_verified": verified_sender_bot is not None,
            "system_verified": verified_system is not None,
            "message_bot_boolean_keys": message_bot_keys,
            "sender_bot_boolean_keys": [*raw_sender_bot_keys, *sender_bot_keys],
            "sender_bot_type_key": actor_type_key,
            "system_boolean_keys": system_keys,
            "sender_system_boolean_keys": sender_system_keys,
            "system_from_message_type": type_system,
        },
    }


def _normalize_selected_messages(
    rows: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    selected: set[str],
    start: dt.datetime,
    end: dt.datetime,
    *,
    source: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    messages: dict[str, dict[str, Any]] = {}
    counters = {"source_rows": len(rows), "outside_selection": 0, "outside_window": 0, "duplicates": 0}
    for group, raw in rows:
        row_conversation_id = _conversation_id(raw)
        group_conversation_id = _conversation_id(group)
        if (
            row_conversation_id is not None
            and group_conversation_id is not None
            and row_conversation_id != group_conversation_id
        ):
            raise DwsAdapterError(
                f"{source} group and message row disagree on conversation id",
                code="CONVERSATION_ID_CONFLICT",
                details={
                    "group_conversation_id": group_conversation_id,
                    "row_conversation_id": row_conversation_id,
                },
            )
        conversation_id = row_conversation_id or group_conversation_id
        if not conversation_id or conversation_id not in selected:
            counters["outside_selection"] += 1
            continue
        raw_time = _first(raw, ("createTime", "createdAt", "timestamp", "time", "sendTime"))
        created = _parse_time(raw_time, field=f"{source}.message.create_time")
        if not (start <= created < end):
            counters["outside_window"] += 1
            continue
        message_id_raw = _first(raw, ("openMessageId", "messageId", "openMsgId", "msgId"))
        if message_id_raw in (None, ""):
            raise DwsAdapterError(
                f"{source} contains an in-scope message without a stable id",
                code="WEAK_MESSAGE_ID",
                details={"conversation_id": conversation_id, "raw": raw},
            )
        message_id = str(message_id_raw)
        sender_value = raw.get("sender")
        sender = sender_value if isinstance(sender_value, Mapping) else {}
        sender_id_raw = _first(raw, ("senderOpenDingTalkId", "senderId", "senderUserId"))
        sender_id_raw = sender_id_raw or _first(sender, ("openDingTalkId", "id", "userId"))
        if sender_id_raw in (None, ""):
            raise DwsAdapterError(
                f"{source} contains an in-scope message without a stable sender id",
                code="WEAK_SENDER_ID",
                details={"conversation_id": conversation_id, "message_id": message_id},
            )
        sender_id = str(sender_id_raw)
        sender_name = str(
            _first(raw, ("senderName", "fromName"))
            or _first(sender, ("name", "nick", "displayName"))
            or sender_id
        )
        title = str(
            _first(raw, ("conversationTitle", "conversationName", "title"))
            or _first(group, ("title", "name", "conversationTitle"))
            or conversation_id
        )
        msg_type = str(_first(raw, ("msgType", "messageType", "type")) or "unknown")
        content = raw.get("content")
        if content is None:
            content = _first(raw, ("text", "body", "richText", "title")) or ""
        text = str(raw.get("text") or _recursive_text(content) or "").strip()
        status = _message_status(raw, msg_type, text)
        origin_flags = _verified_origin_flags(raw, sender, msg_type)
        projection = {
            "id": message_id,
            "conversation_id": conversation_id,
            "conversation_title": title,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "create_time": created.isoformat(),
            "type": msg_type,
            "text": text,
            "content": content,
            "status": status,
            **origin_flags,
        }
        content_hash = _sha256_text(_canonical_json(projection))
        key = f"{conversation_id}\0{message_id}"
        message = {
            **projection,
            "key": _sha256_text(key),
            "content_hash": content_hash,
            "sources": [source],
            "raw": dict(raw),
        }
        existing = messages.get(key)
        if existing is not None:
            if existing["content_hash"] != content_hash:
                raise DwsAdapterError(
                    f"{source} returned conflicting content for one message id",
                    code="MESSAGE_ID_COLLISION",
                    details={"conversation_id": conversation_id, "message_id": message_id},
                )
            counters["duplicates"] += 1
            continue
        messages[key] = message
    return messages, counters


def _merge_sources(
    primary: dict[str, dict[str, Any]], targeted: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    primary_keys = set(primary)
    targeted_keys = set(targeted)
    only_primary = sorted(primary_keys - targeted_keys)
    only_targeted = sorted(targeted_keys - primary_keys)
    merged: dict[str, dict[str, Any]] = {}
    for key in sorted(primary_keys | targeted_keys):
        left = primary.get(key)
        right = targeted.get(key)
        if left is not None and right is not None and left["content_hash"] != right["content_hash"]:
            raise DwsAdapterError(
                "list-all and targeted search disagree on message content",
                code="SOURCE_CONTENT_CONFLICT",
                details={
                    "conversation_id": left["conversation_id"],
                    "message_id": left["id"],
                    "list_all_hash": left["content_hash"],
                    "targeted_hash": right["content_hash"],
                },
            )
        chosen = dict(left or right or {})
        chosen["sources"] = [
            source_name
            for source_name, candidate in (("list_all", left), ("targeted", right))
            if candidate is not None
        ]
        merged[key] = chosen
    difference = {
        "only_list_all_count": len(only_primary),
        "only_targeted_count": len(only_targeted),
        "only_list_all": [
            {"conversation_id": primary[key]["conversation_id"], "message_id": primary[key]["id"]}
            for key in only_primary
        ],
        "only_targeted": [
            {"conversation_id": targeted[key]["conversation_id"], "message_id": targeted[key]["id"]}
            for key in only_targeted
        ],
    }
    return merged, difference


def _projection_digest(messages: Mapping[str, Mapping[str, Any]]) -> str:
    rows = [
        (
            key,
            value["content_hash"],
            value["status"],
            tuple(sorted(value.get("sources") or [])),
        )
        for key, value in sorted(messages.items())
    ]
    return _sha256_text(_canonical_json(rows))


class DwsAdapter:
    """Real DWS v1.0.59 connector used by the configuration panel."""

    def __init__(
        self,
        dws_path: str | os.PathLike[str] | None = None,
        timeout: int = 60,
        *,
        page_limit: int = 500,
        max_passes: int = 4,
        settle_seconds: float = 1.0,
        targeted_crosscheck: bool = True,
        conversation_workers: int = 4,
        command_retries: int = 2,
        command_retry_delay: float = 0.5,
    ) -> None:
        configured = dws_path or os.environ.get("DWS_PATH")
        if configured:
            self.dws_path = pathlib.Path(configured).expanduser().resolve()
        else:
            self.dws_path = (
                pathlib.Path(__file__).resolve().parents[1]
                / ".tools" / "dws" / EXPECTED_DWS_VERSION / "dws.exe"
            )
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not 1 <= page_limit <= 500:
            raise ValueError("page_limit must be between 1 and 500")
        if max_passes < 2:
            raise ValueError("max_passes must be at least 2")
        if settle_seconds < 0:
            raise ValueError("settle_seconds must be non-negative")
        if not 1 <= conversation_workers <= 16:
            raise ValueError("conversation_workers must be between 1 and 16")
        if not 0 <= command_retries <= 5:
            raise ValueError("command_retries must be between 0 and 5")
        if command_retry_delay < 0:
            raise ValueError("command_retry_delay must be non-negative")
        self.timeout = int(timeout)
        self.page_limit = int(page_limit)
        self.max_passes = int(max_passes)
        self.settle_seconds = float(settle_seconds)
        self.targeted_crosscheck = bool(targeted_crosscheck)
        self.conversation_workers = int(conversation_workers)
        self.command_retries = int(command_retries)
        self.command_retry_delay = float(command_retry_delay)
        self.use_normal_chat_reader = True
        self._conversation_kinds: dict[str, str] = {}
        self._conversation_titles: dict[str, str] = {}

    def _run(self, arguments: Sequence[str], *, operation: str) -> Mapping[str, Any]:
        if not self.dws_path.is_file():
            raise DwsAdapterError(
                f"DWS executable not found: {self.dws_path}", code="DWS_NOT_FOUND"
            )
        environment = os.environ.copy()
        environment["DO_NOT_TRACK"] = "1"
        for attempt in range(self.command_retries + 1):
            try:
                completed = subprocess.run(
                    [str(self.dws_path), *arguments],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=environment,
                    check=False,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise DwsAdapterError(
                    f"{operation} timed out after {self.timeout}s", code="DWS_TIMEOUT"
                ) from exc
            except OSError as exc:
                raise DwsAdapterError(
                    f"cannot start DWS for {operation}: {exc}",
                    code="DWS_EXECUTION_FAILED",
                ) from exc
            if completed.returncode == 0:
                return _decode_single_json(completed.stdout, operation=operation)
            details: Any = {
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip(),
                "stdout": completed.stdout.strip(),
                "attempt": attempt + 1,
                "maximum_attempts": self.command_retries + 1,
            }
            for stream_name, stream_value in (
                ("stdout_json", completed.stdout),
                ("stderr_json", completed.stderr),
            ):
                try:
                    details[stream_name] = dict(
                        _decode_single_json(stream_value, operation=operation)
                    )
                except DwsAdapterError:
                    pass
            stderr_json = details.get("stderr_json")
            error_payload = (
                stderr_json.get("error") if isinstance(stderr_json, Mapping) else None
            )
            retryable = bool(
                isinstance(error_payload, Mapping)
                and error_payload.get("retryable") is True
            )
            if retryable and attempt < self.command_retries:
                if self.command_retry_delay:
                    time.sleep(self.command_retry_delay * (2**attempt))
                continue
            raise DwsAdapterError(
                f"{operation} failed with exit code {completed.returncode}",
                code="DWS_COMMAND_FAILED",
                details=details,
            )
        raise AssertionError("unreachable DWS retry loop")  # pragma: no cover

    def preflight(self) -> dict[str, Any]:
        """Return an exact readiness ledger; never treats auth failure as empty data."""

        result: dict[str, Any] = {
            "ok": False,
            "source": "dws",
            "dws_path": str(self.dws_path),
            "expected_version": EXPECTED_DWS_VERSION,
            "version_ok": False,
            "version_pinned": False,
            "auth_command_ok": False,
            "authenticated": False,
            "errors": [],
        }
        if not self.dws_path.is_file():
            result["errors"].append(
                {"code": "DWS_NOT_FOUND", "message": f"DWS executable not found: {self.dws_path}"}
            )
            return result
        environment = os.environ.copy()
        environment["DO_NOT_TRACK"] = "1"

        def invoke(arguments: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
            try:
                return subprocess.run(
                    [str(self.dws_path), *arguments],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=environment,
                    check=False,
                    timeout=self.timeout,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                result["errors"].append({"code": "DWS_EXECUTION_FAILED", "message": str(exc)})
                return None

        version = invoke(["version"])
        if version is not None:
            result["version_ok"] = version.returncode == 0
            result["version"] = version.stdout.strip()
            result["version_stderr"] = version.stderr.strip()
            result["version_pinned"] = bool(
                version.returncode == 0
                and re.search(r"(?m)^Version:\s*v1\.0\.59\s*$", version.stdout)
            )
            if not result["version_pinned"]:
                result["errors"].append(
                    {
                        "code": "VERSION_MISMATCH",
                        "message": f"DWS must be pinned to {EXPECTED_DWS_VERSION}",
                    }
                )
        auth = invoke(["auth", "status", "--format", "json"])
        if auth is not None:
            result["auth_command_ok"] = auth.returncode == 0
            result["auth_stderr"] = auth.stderr.strip()
            try:
                auth_json = _decode_single_json(auth.stdout, operation="auth status")
                result["auth"] = dict(auth_json)
                result["authenticated"] = auth_json.get("authenticated") is True
                if "success" in auth_json and auth_json.get("success") is not True:
                    result["auth_command_ok"] = False
            except DwsAdapterError as exc:
                result["auth"] = {"raw": auth.stdout.strip()}
                result["errors"].append(exc.as_dict())
            if not result["auth_command_ok"]:
                result["errors"].append(
                    {"code": "AUTH_STATUS_FAILED", "message": "DWS auth status command failed"}
                )
            elif not result["authenticated"]:
                result["errors"].append(
                    {"code": "NOT_AUTHENTICATED", "message": "DWS OAuth is not authenticated"}
                )
        result["ok"] = bool(
            result["version_ok"]
            and result["version_pinned"]
            and result["auth_command_ok"]
            and result["authenticated"]
        )
        return result

    def _require_ready(self) -> dict[str, Any]:
        ready = self.preflight()
        if not ready["ok"]:
            raise DwsAdapterError(
                "DWS preflight failed; refusing to read chat data",
                code="PREFLIGHT_FAILED",
                details=ready,
            )
        return ready

    def refresh_conversations(self) -> dict[str, Any]:
        """Return the current user's ordinary direct/group conversation directory."""

        readiness = self._require_ready()
        common = [
            "--page-all", "--page-limit", str(self.page_limit),
            "--max-items", "0", "--page-delay", "0", "--format", "json",
        ]
        conversation_payload = self._run(
            [
                "chat", "+conversation-list", "--limit", "100", *common,
            ],
            operation="complete conversation directory",
        )
        group_payload = self._run(
            [
                "chat", "+chat-list-all", "--limit", "200", *common,
            ],
            operation="complete group directory",
        )
        conversation_rows, conversation_ledger = _validate_conversation_page(
            conversation_payload
        )
        group_rows, group_ledger = _validate_group_page(group_payload)

        groups: dict[str, Mapping[str, Any]] = {}
        for raw in group_rows:
            conversation_id = str(raw["openConversationId"]).strip()
            if conversation_id in groups:
                raise DwsAdapterError(
                    "complete group directory returned a duplicate id",
                    code="CONVERSATION_ID_COLLISION",
                    details=conversation_id,
                )
            groups[conversation_id] = raw

        items: dict[str, dict[str, Any]] = {}
        for raw in conversation_rows:
            conversation_id = str(raw["openConversationId"]).strip()
            if conversation_id in items:
                raise DwsAdapterError(
                    "complete conversation directory returned a duplicate id",
                    code="CONVERSATION_ID_COLLISION",
                    details=conversation_id,
                )
            enriched = dict(raw)
            enriched["conversationType"] = (
                "group" if conversation_id in groups else "direct"
            )
            items[conversation_id] = _normalize_conversation(enriched, source="dws")

        additional_group_count = 0
        for conversation_id, raw in groups.items():
            if conversation_id in items:
                continue
            enriched = dict(raw)
            enriched["conversationType"] = "group"
            items[conversation_id] = _normalize_conversation(enriched, source="dws")
            additional_group_count += 1

        normalized = sorted(
            items.values(), key=lambda item: ((item["title"] or "").casefold(), item["id"])
        )
        self._conversation_kinds = {item["id"]: item["kind"] for item in normalized}
        self._conversation_titles = {item["id"]: item["title"] for item in normalized}
        direct_count = sum(item["kind"] == "direct" for item in normalized)
        group_count = sum(item["kind"] == "group" for item in normalized)
        return {
            "items": normalized,
            "warnings": [],
            "integrity": {
                "complete": True,
                "data_complete": True,
                "pagination_known": True,
                "source_limited": False,
                "directory_incomplete": False,
                "has_more": False,
                "truncated": False,
                "partial": False,
                "stop_reason": "complete_conversation_and_group_union",
                "count": len(normalized),
                "normalized_count": len(normalized),
                "duplicate_count": 0,
                "direct_count": direct_count,
                "group_count": group_count,
                "active_conversation_count": len(conversation_rows),
                "joined_group_count": len(group_rows),
                "additional_joined_group_count": additional_group_count,
                "sources": {
                    "conversation_list": conversation_ledger,
                    "chat_list_all": group_ledger,
                },
                "preflight": {
                    "version": readiness.get("version"),
                    "authenticated": True,
                },
            },
        }

    def _read_normal_conversation(
        self,
        conversation_id: str,
        start: dt.datetime,
        end: dt.datetime,
        pass_number: int,
    ) -> tuple[str, list[Mapping[str, Any]], dict[str, Any]]:
        operation = f"normal conversation messages pass {pass_number}"
        try:
            payload = self._run(
                [
                    "chat",
                    "+chat-messages",
                    "--group",
                    conversation_id,
                    "--start",
                    start.isoformat(timespec="seconds"),
                    "--end",
                    end.isoformat(timespec="seconds"),
                    "--order",
                    "asc",
                    "--limit",
                    "100",
                    "--page-all",
                    "--page-limit",
                    str(self.page_limit),
                    "--max-items",
                    "0",
                    "--page-delay",
                    "0",
                    "--format",
                    "json",
                ],
                operation=operation,
            )
            rows, ledger = _validate_scoped_message_page(
                payload,
                conversation_id=conversation_id,
                start=start,
                end=end,
                source=operation,
            )
        except DwsAdapterError as exc:
            raise DwsAdapterError(
                "读取所选普通会话消息失败",
                code=exc.code,
                details={
                    "conversation_id": conversation_id,
                    "pass": pass_number,
                    "cause": exc.as_dict(),
                },
            ) from exc
        return conversation_id, rows, ledger

    def _read_message_details(
        self, message_ids: Sequence[str], pass_number: int
    ) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
        if not message_ids:
            return {}, {"batch_count": 0, "requested_count": 0, "verified_count": 0}
        batches = [list(message_ids[index : index + 50]) for index in range(0, len(message_ids), 50)]

        def read_batch(
            indexed_batch: tuple[int, list[str]],
        ) -> tuple[int, dict[str, Mapping[str, Any]]]:
            batch_index, batch = indexed_batch
            operation = f"message detail batch pass {pass_number}"
            try:
                payload = self._run(
                    [
                        "chat",
                        "message",
                        "list-by-ids",
                        "--msg-ids",
                        ",".join(batch),
                        "--format",
                        "json",
                    ],
                    operation=operation,
                )
                parsed = _validate_message_detail_batch(
                    payload, batch, source=f"{operation}[{batch_index}]"
                )
            except DwsAdapterError as exc:
                raise DwsAdapterError(
                    "消息详情复核失败",
                    code=exc.code,
                    details={"batch": batch_index, "cause": exc.as_dict()},
                ) from exc
            return batch_index, parsed

        workers = min(self.conversation_workers, len(batches))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(read_batch, enumerate(batches)))
        output: dict[str, Mapping[str, Any]] = {}
        for _, batch_rows in sorted(results):
            overlap = set(output) & set(batch_rows)
            if overlap:
                raise DwsAdapterError(
                    "消息详情批次返回了重复 ID",
                    code="MESSAGE_ID_COLLISION",
                    details={"count": len(overlap)},
                )
            output.update(batch_rows)
        return output, {
            "batch_count": len(batches),
            "requested_count": len(message_ids),
            "verified_count": len(output),
        }

    def _load_group_identity(
        self, conversation_id: str, pass_number: int
    ) -> tuple[set[str], set[str], dict[str, Any]]:
        operation = f"group member identity pass {pass_number}"
        payload = self._run(
            [
                "chat",
                "+chat-members-list",
                "--conversation-id",
                conversation_id,
                "--member-types",
                "user,bot",
                "--page-limit",
                str(self.page_limit),
                "--format",
                "json",
            ],
            operation=operation,
        )
        return _validate_group_identity_page(
            payload, conversation_id=conversation_id, source=operation
        )

    def _load_contact_identity(self, sender_name: str, pass_number: int) -> set[str]:
        operation = f"contact identity pass {pass_number}"
        payload = self._run(
            [
                "contact",
                "user",
                "search",
                "--query",
                sender_name,
                "--format",
                "json",
            ],
            operation=operation,
        )
        return _validate_contact_search_page(payload, source=operation)

    def _fetch_normal_chat_round(
        self,
        conversation_ids: Sequence[str],
        start: dt.datetime,
        end: dt.datetime,
        pass_number: int,
        identity_state: dict[str, Any],
    ) -> tuple[
        list[tuple[Mapping[str, Any], Mapping[str, Any]]],
        dict[str, Any],
        list[str],
    ]:
        """Read and verify one full selected-scope normal-chat round."""

        workers = min(self.conversation_workers, len(conversation_ids))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    lambda conversation_id: self._read_normal_conversation(
                        conversation_id, start, end, pass_number
                    ),
                    conversation_ids,
                )
            )

        rows_by_conversation: dict[str, list[Mapping[str, Any]]] = {}
        conversation_ledgers: list[dict[str, Any]] = []
        all_message_ids: list[str] = []
        observed_message_ids: set[str] = set()
        for conversation_id, rows, ledger in results:
            rows_by_conversation[conversation_id] = rows
            for row in rows:
                message_id = _message_id(row)
                if not message_id:  # already rejected by the typed validator
                    raise DwsAdapterError(
                        "普通会话消息缺少稳定 ID", code="WEAK_MESSAGE_ID"
                    )
                if message_id in observed_message_ids:
                    raise DwsAdapterError(
                        "不同会话返回了同一个消息 ID",
                        code="MESSAGE_ID_COLLISION",
                        details={"message_id": message_id},
                    )
                observed_message_ids.add(message_id)
                all_message_ids.append(message_id)
            conversation_ledgers.append(
                {
                    "conversation_id": conversation_id,
                    "kind": self._conversation_kinds.get(conversation_id, "unknown"),
                    "pages": ledger["pages"],
                    "message_count": ledger["count"],
                    "stop_reason": ledger["stop_reason"],
                    "complete": True,
                }
            )

        details, detail_ledger = self._read_message_details(all_message_ids, pass_number)
        group_cache: dict[str, Any] = identity_state.setdefault("groups", {})
        contact_cache: dict[str, Any] = identity_state.setdefault("contacts", {})
        group_error_codes: dict[str, str] = identity_state.setdefault("group_errors", {})
        contact_error_codes: dict[str, str] = identity_state.setdefault("contact_errors", {})

        group_targets = [
            conversation_id
            for conversation_id, rows in rows_by_conversation.items()
            if rows
            and self._conversation_kinds.get(conversation_id) == "group"
            and conversation_id not in group_cache
        ]

        def load_group(conversation_id: str) -> tuple[str, Any, str | None]:
            try:
                return (
                    conversation_id,
                    self._load_group_identity(conversation_id, pass_number),
                    None,
                )
            except DwsAdapterError as exc:
                return conversation_id, None, exc.code

        if group_targets:
            with ThreadPoolExecutor(
                max_workers=min(self.conversation_workers, len(group_targets))
            ) as executor:
                for conversation_id, identity, error_code in executor.map(
                    load_group, group_targets
                ):
                    if identity is None:
                        group_error_codes[conversation_id] = error_code or "DWS_ERROR"
                    else:
                        users, bots, ledger = identity
                        group_cache[conversation_id] = {
                            "users": users,
                            "bots": bots,
                            "ledger": ledger,
                        }
                        group_error_codes.pop(conversation_id, None)

        unresolved_names: set[str] = set()
        for conversation_id, rows in rows_by_conversation.items():
            group_identity = group_cache.get(conversation_id) or {}
            known_users = group_identity.get("users") or set()
            known_bots = group_identity.get("bots") or set()
            for row in rows:
                sender_id = _sender_id(row) or ""
                sender_name = str(row.get("sender") or "").strip()
                if sender_id not in known_users and sender_id not in known_bots and sender_name:
                    if sender_name not in contact_cache:
                        unresolved_names.add(sender_name)

        def load_contact(sender_name: str) -> tuple[str, set[str] | None, str | None]:
            try:
                return sender_name, self._load_contact_identity(sender_name, pass_number), None
            except DwsAdapterError as exc:
                return sender_name, None, exc.code

        if unresolved_names:
            ordered_names = sorted(unresolved_names)
            with ThreadPoolExecutor(
                max_workers=min(self.conversation_workers, len(ordered_names))
            ) as executor:
                for sender_name, identifiers, error_code in executor.map(
                    load_contact, ordered_names
                ):
                    if identifiers is None:
                        contact_error_codes[sender_name] = error_code or "DWS_ERROR"
                    else:
                        contact_cache[sender_name] = identifiers
                        contact_error_codes.pop(sender_name, None)

        flattened: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        origin_counts = {"verified_human": 0, "verified_bot": 0, "unknown": 0}
        type_counts: dict[str, int] = {}
        for conversation_id in conversation_ids:
            title = self._conversation_titles.get(conversation_id, conversation_id)
            group = {"openConversationId": conversation_id, "title": title}
            group_identity = group_cache.get(conversation_id) or {}
            known_users = group_identity.get("users") or set()
            known_bots = group_identity.get("bots") or set()
            for row in rows_by_conversation.get(conversation_id, []):
                message_id = _message_id(row) or ""
                detail = details[message_id]
                if _conversation_id(detail) != conversation_id:
                    raise DwsAdapterError(
                        "消息详情与会话读取结果的会话 ID 不一致",
                        code="CONVERSATION_ID_CONFLICT",
                        details={"message_id": message_id},
                    )
                sender_id = _sender_id(row) or ""
                if _sender_id(detail) != sender_id:
                    raise DwsAdapterError(
                        "消息详情与会话读取结果的发送者 ID 不一致",
                        code="SENDER_ID_CONFLICT",
                        details={"message_id": message_id},
                    )
                row_time = _parse_time(row.get("createTime"), field="normal_chat.message_time")
                detail_time = _parse_time(
                    detail.get("createTime"), field="message_detail.message_time"
                )
                if row_time != detail_time:
                    raise DwsAdapterError(
                        "消息详情与会话读取结果的时间不一致",
                        code="MESSAGE_TIME_CONFLICT",
                        details={"message_id": message_id},
                    )

                sender_name = str(row.get("sender") or detail.get("sender") or sender_id)
                is_bot: bool | None = None
                identity_source: str | None = None
                if sender_id in known_users:
                    is_bot = False
                    identity_source = "complete_group_user_bucket"
                elif sender_id in known_bots:
                    is_bot = True
                    identity_source = "complete_group_bot_bucket"
                elif sender_id in (contact_cache.get(sender_name) or set()):
                    is_bot = False
                    identity_source = "exact_contact_open_id_match"

                enriched = dict(row)
                enriched["openConversationId"] = conversation_id
                enriched["openMessageId"] = message_id
                enriched["senderOpenDingTalkId"] = sender_id
                enriched["senderName"] = sender_name
                enriched["content"] = detail.get("content", row.get("text", ""))
                enriched["detail"] = dict(detail)
                message_type = _projected_message_type(row, detail)
                enriched["msgType"] = message_type
                type_counts[message_type] = type_counts.get(message_type, 0) + 1
                if is_bot is None:
                    origin_counts["unknown"] += 1
                else:
                    enriched["isBot"] = is_bot
                    enriched["sender_is_bot"] = is_bot
                    # A typed normal-chat row with an exact authored sender
                    # identity is an authored user/bot message, not a system
                    # event.  Unknown identities remain tri-state and are not
                    # silently upgraded to ordinary human messages.
                    enriched["isSystem"] = False
                    enriched["originIdentitySource"] = identity_source
                    origin_counts["verified_bot" if is_bot else "verified_human"] += 1
                flattened.append((group, enriched))

        warnings: list[str] = []
        if origin_counts["unknown"]:
            warnings.append(
                f"{origin_counts['unknown']} 条消息的发送者来源无法精确验证，"
                "已保留为来源未知并交由当前机器人/系统消息筛选策略保守处理。"
            )
        if group_error_codes:
            warnings.append(
                f"{len(group_error_codes)} 个群聊的成员身份目录不可用，"
                "已尝试用发送者通讯录精确匹配补充验证。"
            )
        if contact_error_codes:
            warnings.append(
                f"{len(contact_error_codes)} 个发送者名称无法完成通讯录身份查询，"
                "未匹配消息不会被假定为普通人工消息。"
            )

        return flattened, {
            "complete": True,
            "source": "normal_chat_per_conversation",
            "conversation_count": len(conversation_ids),
            "complete_conversation_count": len(conversation_ledgers),
            "pages": sum(item["pages"] for item in conversation_ledgers),
            "message_count": len(flattened),
            "conversations": conversation_ledgers,
            "details": detail_ledger,
            "origin": {
                **origin_counts,
                "group_identity_complete": len(group_cache),
                "group_identity_errors": len(group_error_codes),
                "contact_queries_complete": len(contact_cache),
                "contact_query_errors": len(contact_error_codes),
            },
            "message_type_counts": type_counts,
        }, warnings

    def _fetch_message_payload(
        self,
        source: str,
        conversation_ids: Sequence[str],
        start: dt.datetime,
        end: dt.datetime,
        pass_number: int,
    ) -> Mapping[str, Any]:
        del pass_number
        common = [
            "--limit", "100", "--page-all", "--page-limit", str(self.page_limit),
            "--max-items", "0", "--page-delay", "0", "--format", "json",
        ]
        if source == "list_all":
            arguments = [
                "chat", "message", "list-all",
                "--start", start.strftime("%Y-%m-%d %H:%M:%S"),
                "--end", end.strftime("%Y-%m-%d %H:%M:%S"),
                *common,
            ]
        elif source == "targeted":
            arguments = [
                "chat", "message", "search-advanced",
                "--conversation-ids", ",".join(conversation_ids),
                "--start", start.isoformat(timespec="seconds"),
                "--end", end.isoformat(timespec="seconds"),
                *common,
            ]
        else:  # pragma: no cover - internal programming guard
            raise ValueError(f"unknown source: {source}")
        return self._run(arguments, operation=f"message {source} pass")

    def collect_selected(
        self,
        conversation_ids: Sequence[str],
        start: str | dt.datetime,
        end: str | dt.datetime,
        passes: int = 2,
        *,
        conversation_metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Collect a frozen ``[start, end)`` scope with stable projections.

        ``passes`` is the number of consecutive identical projections required.
        At most ``max_passes`` source rounds are attempted.  A message that was
        observed and then disappears fails the run even if a smaller tail later
        stabilizes.
        """

        if isinstance(conversation_ids, (str, bytes)):
            raise DwsAdapterError(
                "conversation_ids must be a sequence, not a string", code="INVALID_SELECTION"
            )
        selected_list: list[str] = []
        seen: set[str] = set()
        for value in conversation_ids:
            candidate = str(value).strip()
            if not candidate:
                raise DwsAdapterError("conversation id cannot be empty", code="INVALID_SELECTION")
            if candidate not in seen:
                seen.add(candidate)
                selected_list.append(candidate)
        if not selected_list:
            raise DwsAdapterError(
                "select at least one conversation", code="INVALID_SELECTION"
            )
        if conversation_metadata is not None:
            if isinstance(conversation_metadata, (str, bytes)):
                raise DwsAdapterError(
                    "conversation_metadata must be an object sequence",
                    code="INVALID_SELECTION",
                )
            for index, metadata in enumerate(conversation_metadata):
                if not isinstance(metadata, Mapping):
                    raise DwsAdapterError(
                        "conversation metadata row must be an object",
                        code="INVALID_SELECTION",
                        details={"row": index},
                    )
                conversation_id = _conversation_id(metadata)
                if not conversation_id or conversation_id not in seen:
                    continue
                kind = _conversation_kind(metadata)
                if kind in {"direct", "group"}:
                    self._conversation_kinds[conversation_id] = kind
                title = _first(
                    metadata,
                    ("title", "name", "conversationName", "conversationTitle"),
                )
                if title not in (None, ""):
                    self._conversation_titles[conversation_id] = str(title)
        if type(passes) is not int or passes < 2:
            raise DwsAdapterError(
                "passes must be an integer of at least 2", code="INVALID_STABILITY_POLICY"
            )
        if passes > self.max_passes:
            raise DwsAdapterError(
                "required stable passes exceed max_passes", code="INVALID_STABILITY_POLICY"
            )
        parsed_start, parsed_end = _window(start, end)
        readiness = self._require_ready()
        selected = set(selected_list)
        pass_ledgers: list[dict[str, Any]] = []
        observed_keys: set[str] = set()
        previous_digest: str | None = None
        stable_count = 0
        final_messages: dict[str, dict[str, Any]] = {}
        final_difference: dict[str, Any] = {}
        identity_state: dict[str, Any] = {}
        collection_warnings: list[str] = []
        normal_scope = self.use_normal_chat_reader

        for pass_number in range(1, self.max_passes + 1):
            pass_started_wall = dt.datetime.now(dt.timezone.utc)
            pass_started_monotonic = time.perf_counter_ns()
            if normal_scope:
                primary_rows, primary_paging, round_warnings = self._fetch_normal_chat_round(
                    selected_list,
                    parsed_start,
                    parsed_end,
                    pass_number,
                    identity_state,
                )
                collection_warnings.extend(round_warnings)
                primary, primary_counts = _normalize_selected_messages(
                    primary_rows,
                    selected,
                    parsed_start,
                    parsed_end,
                    source="normal_chat",
                )
                targeted_paging = None
                targeted_counts = None
                merged = primary
                source_difference = {
                    "only_list_all_count": 0,
                    "only_targeted_count": 0,
                    "only_list_all": [],
                    "only_targeted": [],
                    "crosscheck_disabled": True,
                    "reason": "selected_conversations_have_individual_complete_ledgers",
                    "primary_source": "normal_chat_per_conversation",
                    "global_search_entitlement_required": False,
                }
            else:
                primary_payload = self._fetch_message_payload(
                    "list_all", selected_list, parsed_start, parsed_end, pass_number
                )
                primary_rows, primary_paging = _validate_message_page(
                    primary_payload, source="list_all"
                )
                primary, primary_counts = _normalize_selected_messages(
                    primary_rows, selected, parsed_start, parsed_end, source="list_all"
                )
                if self.targeted_crosscheck:
                    targeted_payload = self._fetch_message_payload(
                        "targeted", selected_list, parsed_start, parsed_end, pass_number
                    )
                    targeted_rows, targeted_paging = _validate_message_page(
                        targeted_payload, source="targeted"
                    )
                    targeted, targeted_counts = _normalize_selected_messages(
                        targeted_rows, selected, parsed_start, parsed_end, source="targeted"
                    )
                    merged, source_difference = _merge_sources(primary, targeted)
                else:
                    targeted_paging = None
                    targeted_counts = None
                    merged = primary
                    source_difference = {
                        "only_list_all_count": 0,
                        "only_targeted_count": 0,
                        "only_list_all": [],
                        "only_targeted": [],
                        "crosscheck_disabled": True,
                    }
            digest = _projection_digest(merged)
            keys = set(merged)
            pass_finished_wall = dt.datetime.now(dt.timezone.utc)
            duration_ms = max(
                0, (time.perf_counter_ns() - pass_started_monotonic) // 1_000_000
            )
            selection_digest = _sha256_text(
                _canonical_json(sorted(selected_list))
            )
            pass_ledgers.append(
                {
                    "pass": pass_number,
                    "started_at": _iso_utc(pass_started_wall),
                    "finished_at": _iso_utc(pass_finished_wall),
                    "duration_ms": duration_ms,
                    "window": {
                        "start": parsed_start.isoformat(),
                        "end": parsed_end.isoformat(),
                        "interval": "left_closed_right_open",
                    },
                    "selection": {
                        "conversation_count": len(selected_list),
                        "conversation_ids": list(selected_list),
                        "conversation_ids_digest": selection_digest,
                    },
                    "digest": digest,
                    "message_count": len(merged),
                    "normal_chat_paging": primary_paging if normal_scope else None,
                    "list_all_paging": None if normal_scope else primary_paging,
                    "targeted_paging": targeted_paging,
                    "normal_chat_counts": primary_counts if normal_scope else None,
                    "list_all_counts": None if normal_scope else primary_counts,
                    "targeted_counts": targeted_counts,
                    "source_difference": source_difference,
                }
            )
            observed_keys.update(keys)
            if digest == previous_digest:
                stable_count += 1
            else:
                previous_digest = digest
                stable_count = 1
            final_messages = merged
            final_difference = source_difference
            if stable_count >= passes:
                break
            if self.settle_seconds:
                time.sleep(self.settle_seconds)

        if stable_count < passes:
            raise DwsAdapterError(
                "message collection did not converge",
                code="SOURCE_NOT_STABLE",
                details={"required_stable_passes": passes, "passes": pass_ledgers},
            )
        disappeared = observed_keys - set(final_messages)
        if disappeared:
            raise DwsAdapterError(
                "messages disappeared during collection; refusing a transient subset",
                code="MESSAGES_DISAPPEARED",
                details={"count": len(disappeared)},
            )

        messages = sorted(
            final_messages.values(),
            key=lambda message: (message["create_time"], message["conversation_id"], message["id"]),
        )
        coverage = []
        warnings = list(collection_warnings) if normal_scope else [ISSUE_249]
        if (not normal_scope) and (
            final_difference.get("only_list_all_count")
            or final_difference.get("only_targeted_count")
        ):
            warnings.append(
                "The two complete DWS query paths returned different strong-ID sets; their union is retained and source_difference must be reviewed."
            )
        for conversation_id in selected_list:
            count = sum(1 for message in messages if message["conversation_id"] == conversation_id)
            kind = self._conversation_kinds.get(conversation_id, "unknown")
            uncertainty: str | None = None
            if count == 0 and not normal_scope:
                if kind == "direct":
                    uncertainty = "EMPTY_SELECTED_DIRECT_CONVERSATION_UNCERTAIN"
                    warnings.append(
                        f"Selected direct conversation {conversation_id} returned no in-window messages; this is uncertain, not proof of an empty chat."
                    )
                else:
                    uncertainty = "EMPTY_SELECTED_CONVERSATION_UNCERTAIN"
                    warnings.append(
                        f"Selected conversation {conversation_id} returned no in-window messages; this is not treated as proof that the chat is empty."
                    )
            coverage.append(
                {
                    "conversation_id": conversation_id,
                    "kind": kind,
                    "message_count": count,
                    "uncertainty": uncertainty,
                    "complete": normal_scope,
                }
            )
        dataset_digest = _projection_digest(final_messages)
        return {
            "execution_success": True,
            "data_complete": normal_scope,
            "status": NORMAL_SCOPE_STATUS if normal_scope else "PROVISIONAL_SOURCE_LIMITATION",
            "sealed": normal_scope,
            "source_limitation": None if normal_scope else ISSUE_249,
            "selected_conversation_ids": selected_list,
            "start": parsed_start.isoformat(),
            "end": parsed_end.isoformat(),
            "interval": "left_closed_right_open",
            "messages": messages,
            "warnings": list(dict.fromkeys(warnings)),
            "integrity": {
                "dataset_digest": dataset_digest,
                "message_count": len(messages),
                "required_stable_passes": passes,
                "passes_executed": len(pass_ledgers),
                "stable": True,
                "pagination_complete": True,
                "strong_identity_only": True,
                "source_mode": (
                    "normal_chat_per_conversation" if normal_scope else "global_search_legacy"
                ),
                "source_difference": final_difference,
                "coverage": coverage,
                "passes": pass_ledgers,
                "preflight": {
                    "expected_version": readiness["expected_version"],
                    "version_pinned": readiness["version_pinned"],
                    "authenticated": readiness["authenticated"],
                },
            },
        }


class FixtureDwsAdapter(DwsAdapter):
    """Offline connector with the same public interface as :class:`DwsAdapter`."""

    def __init__(
        self,
        fixture_dir: str | os.PathLike[str] | None = None,
        *,
        pass_paths: Sequence[str | os.PathLike[str]] | None = None,
        max_passes: int = 4,
    ) -> None:
        self.fixture_dir = pathlib.Path(
            fixture_dir
            or pathlib.Path(__file__).resolve().parents[1] / "fixtures"
        ).resolve()
        self.pass_paths = [pathlib.Path(path).resolve() for path in (pass_paths or [])]
        self.timeout = 1
        self.page_limit = 500
        self.max_passes = max_passes
        self.settle_seconds = 0.0
        self.targeted_crosscheck = True
        self.conversation_workers = 1
        self.command_retries = 0
        self.command_retry_delay = 0.0
        self.use_normal_chat_reader = False
        self.dws_path = pathlib.Path("fixture://dws-v1.0.59")
        self._conversation_kinds: dict[str, str] = {}
        self._conversation_titles: dict[str, str] = {}

    @staticmethod
    def _load(path: pathlib.Path) -> Mapping[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DwsAdapterError(
                f"cannot read fixture {path}: {exc}", code="FIXTURE_ERROR"
            ) from exc
        if not isinstance(value, Mapping):
            raise DwsAdapterError(
                f"fixture {path} root must be an object", code="FIXTURE_ERROR"
            )
        return value

    def preflight(self) -> dict[str, Any]:
        return {
            "ok": True,
            "source": "fixture",
            "fixture": True,
            "dws_path": str(self.dws_path),
            "expected_version": EXPECTED_DWS_VERSION,
            "version_ok": True,
            "version_pinned": True,
            "version": "Version:        v1.0.59\nEdition:        fixture",
            "auth_command_ok": True,
            "authenticated": True,
            "errors": [],
        }

    def refresh_conversations(self) -> dict[str, Any]:
        payload = self._load(self.fixture_dir / "conversations.json")
        rows, ledger = _validate_conversation_page(payload)
        items = [_normalize_conversation(row, source="fixture") for row in rows]
        ids = [item["id"] for item in items]
        if len(ids) != len(set(ids)):
            raise DwsAdapterError(
                "fixture conversation ids are not unique", code="CONVERSATION_ID_COLLISION"
            )
        self._conversation_kinds = {item["id"]: item["kind"] for item in items}
        self._conversation_titles = {item["id"]: item["title"] for item in items}
        return {
            "items": items,
            "warnings": [],
            "integrity": {**ledger, "fixture": True, "data_complete": True},
        }

    def _fetch_message_payload(
        self,
        source: str,
        conversation_ids: Sequence[str],
        start: dt.datetime,
        end: dt.datetime,
        pass_number: int,
    ) -> Mapping[str, Any]:
        del conversation_ids, start, end
        if self.pass_paths:
            path = self.pass_paths[min(pass_number - 1, len(self.pass_paths) - 1)]
        else:
            available = sorted(self.fixture_dir.glob("messages_pass_*.json"))
            if not available:
                raise DwsAdapterError(
                    f"no message fixtures in {self.fixture_dir}", code="FIXTURE_ERROR"
                )
            path = available[min(pass_number - 1, len(available) - 1)]
        wrapper = self._load(path)
        candidate = wrapper.get(source)
        if candidate is None and source == "targeted":
            candidate = wrapper.get("list_all")
        if candidate is None and "paging" in wrapper:
            candidate = wrapper
        if not isinstance(candidate, Mapping):
            raise DwsAdapterError(
                f"fixture {path} has no {source} payload", code="FIXTURE_ERROR"
            )
        return candidate
