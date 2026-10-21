"""Grounded, extractive report analysis for normalized DingTalk messages.

This module intentionally does not call an LLM or any external service.  It turns
explicit message text into conservative, evidence-bound findings.  When a value
cannot be read from the message (for example, an owner or a deadline), it remains
unknown instead of being inferred from the sender or surrounding metadata.

Public API
----------
``generate_report(messages, report_config)`` returns both the structured analysis
and the rendered Markdown report.  ``analyze_messages`` and ``render_markdown`` are
also exposed for callers that need the two phases separately.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
import re
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ReportEngineError(ValueError):
    """Raised when the report configuration is internally inconsistent."""


SCHEMA_VERSION = 1

CATEGORY_DEFINITIONS: "OrderedDict[str, dict[str, Any]]" = OrderedDict(
    [
        (
            "completed",
            {
                "title": "已完成事项",
                "patterns": [
                    r"(?<!未)(?:已(?:经)?|现已|成功|顺利)\s*(?:全部)?\s*(?:完成|上线|发布|交付|解决|关闭|通过|验收)",
                    r"(?<!未)(?:完成|上线|发布|交付|解决|关闭|通过|验收)(?:了|完毕|成功)",
                    r"\b(?:done|completed|shipped|released|resolved|closed)\b",
                ],
            },
        ),
        (
            "decision",
            {
                "title": "关键决策",
                "patterns": [
                    r"(?:最终|已经|已)?(?:决定|确定)(?:采用|使用|选择|执行|由|为|将)",
                    r"(?:方案|结论|口径)(?:已经|已)?(?:确认|确定)",
                    r"(?:确认采用|达成一致|一致同意|正式拍板)",
                    r"\b(?:decided|agreed|approved)\b",
                ],
            },
        ),
        (
            "risk",
            {
                "title": "风险与阻塞",
                "patterns": [
                    r"(?:风险|阻塞|卡住|延期|延误|失败|异常|无法|不能|缺少|不稳定|超时)",
                    r"(?:存在|出现|遇到|发现)(?:[^，。；;\n]{0,16})(?:问题|故障|隐患)",
                    r"\b(?:risk|blocker|blocked|issue|failure|failed|delayed|timeout)\b",
                ],
            },
        ),
        (
            "todo",
            {
                "title": "待办事项",
                "patterns": [
                    r"(?:待办|待处理|待跟进|待确认|待补充)",
                    r"(?<!不)(?<!无)(?:需要|需)(?:[^，。；;\n]{0,32})(?:处理|跟进|完成|确认|修复|补充|提交|提供|回复|检查)",
                    r"(?:请|麻烦)(?:[^，。；;\n]{0,32})(?:处理|跟进|完成|确认|修复|补充|提交|提供|回复|检查)",
                    r"(?:记得|请尽快|务必)(?:[^，。；;\n]{0,32})",
                    r"\b(?:todo|to-do|action item|need to|please follow up)\b",
                ],
            },
        ),
        (
            "plan",
            {
                "title": "下一步计划",
                "patterns": [
                    r"(?:下一步|后续|计划|准备|预计|打算|将于|安排在|目标是)",
                    r"\b(?:next step|plan(?:ned)?|will|intend to)\b",
                ],
            },
        ),
        (
            "feedback",
            {
                "title": "反馈",
                "patterns": [
                    r"(?:反馈|建议|意见|希望|诉求|满意|不满意)",
                    r"(?:客户|用户)(?:[^，。；;\n]{0,12})(?:说|表示|要求|认为|反馈|希望)",
                    r"\b(?:feedback|suggestion|requested|customer said)\b",
                ],
            },
        ),
    ]
)

CATEGORY_ALIASES = {
    "completed": "completed",
    "完成": "completed",
    "已完成": "completed",
    "已完成事项": "completed",
    "完成事项": "completed",
    "decision": "decision",
    "决策": "decision",
    "关键决策": "decision",
    "risk": "risk",
    "风险": "risk",
    "风险与阻塞": "risk",
    "阻塞": "risk",
    "todo": "todo",
    "待办": "todo",
    "待办事项": "todo",
    "待跟进": "todo",
    "待跟进事项": "todo",
    "plan": "plan",
    "计划": "plan",
    "下一步": "plan",
    "下一步计划": "plan",
    "feedback": "feedback",
    "反馈": "feedback",
    "客户反馈": "feedback",
    "other": "other",
    "其他": "other",
    "其他重要信息": "other",
    "其他重要信息/待人工审阅": "other",
}

DEFAULT_SECTION_ORDER = [
    "completed",
    "decision",
    "risk",
    "todo",
    "plan",
    "feedback",
    "other",
]

DEFAULT_SECTION_TITLES = {
    **{key: value["title"] for key, value in CATEGORY_DEFINITIONS.items()},
    "other": "其他重要信息/待人工审阅",
}

_CASUAL_ONLY = re.compile(
    r"^(?:你好|您好|早上好|上午好|下午好|晚上好|早|晚安|收到|好的|好|嗯|嗯嗯|"
    r"知道了|明白|了解|可以|行|没问题|谢谢|感谢|辛苦了|ok|okay|thanks?|thx)"
    r"[!！,.，。~～\s]*$",
    re.IGNORECASE,
)

_BENIGN_RISK_PHRASES = re.compile(
    r"(?:没有问题|无问题|问题已解决|问题已经解决|风险已解除|不再阻塞|not an issue|no risk)",
    re.IGNORECASE,
)

_OWNER_PATTERNS = [
    re.compile(r"(?:负责人|责任人|owner)\s*[:：]\s*([^，,。；;\n]{1,40})", re.IGNORECASE),
    re.compile(
        r"(?:明确)?由\s*([@\w.\-\u4e00-\u9fff]{1,30})\s*(?:负责|跟进|处理|完成)",
        re.IGNORECASE,
    ),
    re.compile(r"([@\w.\-\u4e00-\u9fff]{1,30})\s*(?:来负责|负责该事项)", re.IGNORECASE),
]

_DEADLINE_PATTERNS = [
    re.compile(
        r"(?:截止(?:时间|日期)?|deadline)\s*[:：]?\s*"
        r"(\d{4}[年\-/]\d{1,2}[月\-/]\d{1,2}日?(?:\s+\d{1,2}:\d{2})?|"
        r"\d{1,2}月\d{1,2}日(?:\s*\d{1,2}:\d{2})?|"
        r"(?:今天|明天|后天|本周[一二三四五六日天]|下周[一二三四五六日天])(?:\s*\d{1,2}:\d{2})?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"((?:\d{4}[年\-/])?\d{1,2}[月\-/]\d{1,2}日?|"
        r"(?:今天|明天|后天|本周[一二三四五六日天]|下周[一二三四五六日天]))"
        r"(?:之前|以前|前)\s*(?:完成|提交|交付|回复|确认|处理)",
        re.IGNORECASE,
    ),
]


def _clean_string(value: Any, *, limit: int | None = None) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        return ""
    cleaned = re.sub(r"\s+", " ", value).strip()
    if limit is not None and len(cleaned) > limit:
        return cleaned[:limit]
    return cleaned


def _first_string(mapping: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_timezone(value: Any) -> ZoneInfo:
    name = _clean_string(value) or "Asia/Shanghai"
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ReportEngineError(f"未知时区：{name}") from exc


def _parse_datetime(value: Any, default_tz: ZoneInfo) -> tuple[datetime | None, bool]:
    """Return (aware datetime, timezone_was_assumed)."""

    parsed: datetime
    try:
        if isinstance(value, bool) or value is None:
            return None, False
        if isinstance(value, (int, float)):
            epoch = float(value)
            if abs(epoch) > 100_000_000_000:
                epoch /= 1000.0
            parsed = datetime.fromtimestamp(epoch, tz=timezone.utc)
            return parsed, False
        if not isinstance(value, str) or not value.strip():
            return None, False
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    except (OverflowError, OSError, TypeError, ValueError):
        return None, False

    assumed = parsed.tzinfo is None or parsed.utcoffset() is None
    if assumed:
        parsed = parsed.replace(tzinfo=default_tz)
    return parsed, assumed


def _normalise_time(value: Any, default_tz: ZoneInfo) -> tuple[str | None, datetime | None, bool]:
    parsed, assumed = _parse_datetime(value, default_tz)
    if parsed is None:
        return None, None, False
    local = parsed.astimezone(default_tz)
    return local.isoformat(timespec="seconds"), local, assumed


def _extract_text(raw: Mapping[str, Any]) -> str:
    direct = _first_string(raw, ("text", "content_text", "summary", "body"))
    if direct:
        return direct
    content = raw.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Mapping):
        return _first_string(content, ("text", "content", "title", "summary"))
    return ""


def _extract_sender(raw: Mapping[str, Any]) -> str:
    direct = _first_string(raw, ("sender", "sender_name", "sender_display_name", "from_name"))
    if direct:
        return direct
    sender = raw.get("sender")
    if isinstance(sender, Mapping):
        return _first_string(sender, ("name", "display_name", "nick", "id"))
    return ""


def _normalise_message(
    raw: Any,
    index: int,
    default_tz: ZoneInfo,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    if not isinstance(raw, Mapping):
        return None, {"index": index, "reason": "消息不是对象"}, []

    conversation_id = _clean_string(
        _first_string(raw, ("conversation_id", "conversationId", "open_conversation_id"))
    )
    message_id = _clean_string(_first_string(raw, ("message_id", "messageId", "msg_id", "id")))
    sender = _clean_string(_extract_sender(raw))
    text = _extract_text(raw)
    raw_time = next(
        (raw.get(key) for key in ("time", "message_time", "timestamp", "created_at", "create_time") if raw.get(key) is not None),
        None,
    )
    normalised_time, sort_time, assumed_tz = _normalise_time(raw_time, default_tz)

    missing: list[str] = []
    if not conversation_id:
        missing.append("conversation_id")
    elif len(conversation_id) > 512:
        missing.append("conversation_id（过长）")
    if not message_id:
        missing.append("message_id")
    elif len(message_id) > 512:
        missing.append("message_id（过长）")
    if not sender:
        missing.append("sender")
    elif len(sender) > 500:
        missing.append("sender（过长）")
    if not normalised_time:
        missing.append("time")
    if not text:
        missing.append("text")
    if missing:
        return (
            None,
            {
                "index": index,
                "identity": {
                    "conversation_id": conversation_id or None,
                    "message_id": message_id or None,
                },
                "reason": f"缺少或无效字段：{', '.join(missing)}",
            },
            [],
        )

    notes: list[str] = []
    if assumed_tz:
        notes.append(
            f"消息 {conversation_id}/{message_id} 的时间未带时区，按 {default_tz.key} 解释"
        )

    conversation_name = _clean_string(
        _first_string(
            raw,
            ("conversation_name", "conversation_title", "chat_name", "chat_title"),
        ),
        limit=500,
    ) or conversation_id
    message_type = _clean_string(_first_string(raw, ("message_type", "type", "msg_type"))) or "text"
    status = _clean_string(_first_string(raw, ("status", "message_status"))) or "active"

    origin_raw = raw.get("origin_verification")
    if origin_raw is not None and not isinstance(origin_raw, Mapping):
        return (
            None,
            {
                "index": index,
                "identity": {"conversation_id": conversation_id, "message_id": message_id},
                "reason": "origin_verification 必须是对象，来源身份无法验证",
            },
            notes,
        )
    if isinstance(origin_raw, Mapping):
        invalid_origin_fields = [
            field
            for field in ("bot_verified", "system_verified")
            if field in origin_raw and type(origin_raw.get(field)) is not bool
        ]
        if invalid_origin_fields:
            return (
                None,
                {
                    "index": index,
                    "identity": {"conversation_id": conversation_id, "message_id": message_id},
                    "reason": "来源验证字段必须是严格布尔值：" + "、".join(invalid_origin_fields),
                },
                notes,
            )
        bot_origin_verified = origin_raw.get("bot_verified") is True
        system_origin_verified = origin_raw.get("system_verified") is True
    else:
        # Standalone callers can provide explicit strict booleans.  Adapter
        # output always carries ``origin_verification`` and therefore cannot
        # turn an unknown marker into a verified false value here.
        bot_origin_verified = any(
            type(raw.get(field)) is bool for field in ("is_bot", "sender_is_bot")
        )
        system_origin_verified = (
            type(raw.get("is_system")) is bool
            or message_type.casefold() in {"system", "system_message"}
        )

    is_system = raw.get("is_system") is True or message_type.casefold() in {"system", "system_message"}
    is_bot = raw.get("is_bot") is True or raw.get("sender_is_bot") is True

    message = {
        "conversation_id": conversation_id,
        "conversation_name": conversation_name,
        "message_id": message_id,
        "sender": sender,
        "time": normalised_time,
        "_sort_time": sort_time,
        "text": text,
        "message_type": message_type,
        "status": status,
        "is_system": is_system,
        "is_bot": is_bot,
        "bot_origin_verified": bot_origin_verified,
        "system_origin_verified": system_origin_verified,
        "_input_index": index,
    }
    return message, None, notes


def _message_signature(message: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        message["conversation_name"],
        message["sender"],
        message["time"],
        message["text"],
        message["message_type"],
        message["status"],
        message["is_system"],
        message["is_bot"],
        message["bot_origin_verified"],
        message["system_origin_verified"],
    )


def _normalise_string_set(value: Any, field: str, *, casefold: bool = True) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ReportEngineError(f"{field} 必须是字符串数组")
    result = set()
    for item in value:
        text = _clean_string(item)
        if not text:
            raise ReportEngineError(f"{field} 中包含空值或非字符串")
        result.add(text.casefold() if casefold else text)
    return result


def _normalise_filters(raw: Any, default_tz: ZoneInfo) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ReportEngineError("filters 必须是对象")

    include_recalled = raw.get("include_recalled", True)
    if type(include_recalled) is not bool:
        raise ReportEngineError("include_recalled 必须是布尔值")

    result = {
        "include_conversation_ids": _normalise_string_set(
            raw.get("include_conversation_ids"), "include_conversation_ids", casefold=False
        ),
        "exclude_conversation_ids": _normalise_string_set(
            raw.get("exclude_conversation_ids"), "exclude_conversation_ids", casefold=False
        ),
        "include_message_ids": _normalise_string_set(
            raw.get("include_message_ids"), "include_message_ids", casefold=False
        ),
        "exclude_message_ids": _normalise_string_set(
            raw.get("exclude_message_ids"), "exclude_message_ids", casefold=False
        ),
        "include_senders": _normalise_string_set(raw.get("include_senders"), "include_senders"),
        "exclude_senders": _normalise_string_set(raw.get("exclude_senders"), "exclude_senders"),
        "include_keywords": _normalise_string_set(raw.get("include_keywords"), "include_keywords"),
        "exclude_keywords": _normalise_string_set(raw.get("exclude_keywords"), "exclude_keywords"),
        "message_types": _normalise_string_set(raw.get("message_types"), "message_types"),
        "include_system_messages": raw.get("include_system_messages") is True,
        "include_bot_messages": raw.get("include_bot_messages") is True,
        "include_recalled": include_recalled,
        "exclude_casual": raw.get("exclude_casual") is True,
    }

    start_value = raw.get("start") if raw.get("start") is not None else raw.get("start_time")
    end_value = raw.get("end") if raw.get("end") is not None else raw.get("end_time")
    start, _ = _parse_datetime(start_value, default_tz) if start_value is not None else (None, False)
    end, _ = _parse_datetime(end_value, default_tz) if end_value is not None else (None, False)
    if start_value is not None and start is None:
        raise ReportEngineError("filters.start 不是有效时间")
    if end_value is not None and end is None:
        raise ReportEngineError("filters.end 不是有效时间")
    if start is not None:
        start = start.astimezone(default_tz)
    if end is not None:
        end = end.astimezone(default_tz)
    if start is not None and end is not None and start >= end:
        raise ReportEngineError("时间范围必须满足 start < end")
    result["start"] = start
    result["end"] = end
    return result


def _apply_free_instruction(
    instruction: str,
    *,
    grouping: str,
    max_length: int,
    filters: dict[str, Any],
    explicit_include_categories: set[str] | None,
) -> tuple[str, int, set[str] | None, dict[str, Any]]:
    """Apply only a small, documented deterministic grammar.

    Anything outside this grammar is returned as unsupported so the UI never
    claims that an arbitrary natural-language instruction was understood.
    """

    result = {"raw": instruction, "applied": [], "conflicts": [], "unsupported": []}
    if not instruction:
        return grouping, max_length, explicit_include_categories, result

    consumed = [False] * len(instruction)

    def consume(match: re.Match[str]) -> None:
        for pos in range(match.start(), match.end()):
            consumed[pos] = True

    def mark(match: re.Match[str], description: str) -> None:
        consume(match)
        result["applied"].append(description)

    def mark_conflict(match: re.Match[str], description: str) -> None:
        consume(match)
        result["conflicts"].append(description)

    grouping_patterns = [
        (r"按(?:会话|聊天)(?:进行)?分组", "conversation", "按会话分组"),
        (r"按(?:日期|时间)(?:进行)?分组", "time", "按时间分组"),
        (r"按主题(?:进行)?分组", "topic", "按主题分组"),
        (r"按负责人(?:进行)?分组", "owner", "按负责人分组"),
    ]
    for pattern, target, description in grouping_patterns:
        for match in re.finditer(pattern, instruction, re.IGNORECASE):
            if grouping == target:
                mark(match, f"与面板一致：{description}")
            else:
                grouping_labels = {
                    "conversation": "按会话分组",
                    "time": "按日期分组",
                    "topic": "按主题分组",
                    "owner": "按负责人分组",
                }
                mark_conflict(
                    match,
                    f"规则要求“{description}”与面板选择“{grouping_labels.get(grouping, grouping)}”冲突；"
                    "请以分组方式下拉框为准并删除冲突文字",
                )

    for match in re.finditer(r"(?:控制在|不超过|最多)?\s*(\d{3,6})\s*(?:字|字符)(?:以内)?", instruction):
        requested = int(match.group(1))
        requested = min(requested, 200_000)
        if requested < 200:
            mark_conflict(match, "规则文字中的最大长度不能低于 200 字符")
        elif requested == max_length:
            mark(match, f"与面板一致：最大长度 {max_length} 字符")
        else:
            mark_conflict(
                match,
                f"规则要求最大长度 {requested} 字符，与面板选择 {max_length} 字符冲突；"
                "请使用报告长度控件并删除冲突文字",
            )

    for pattern, description in [
        (r"(?:不要|不允许|禁止)(?:做)?推断|只写事实|仅写事实", "仅输出明确事实"),
        (r"(?:关键|重要)?结论(?:附|保留)(?:原始)?(?:消息|聊天)?出处", "结论附消息出处"),
        (r"(?:不要|不得)(?:自动)?(?:编造|补全)(?:[^，。；;\n]{0,32})(?:负责人|截止时间|结论)", "不补全未明确的负责人、截止时间或结论"),
        (r"(?:不明确|未明确)(?:[^，。；;\n]{0,24})标记为待确认", "未明确信息标记为待确认"),
    ]:
        for match in re.finditer(pattern, instruction, re.IGNORECASE):
            mark(match, description)

    for match in re.finditer(r"(?:不要写|排除|过滤)(?:普通)?闲聊", instruction):
        if filters["exclude_casual"]:
            mark(match, "与面板一致：保守过滤纯问候/确认类闲聊")
        else:
            mark_conflict(
                match,
                "规则要求过滤普通闲聊，但面板已关闭闲聊过滤；请以筛选控件为准并删除冲突文字",
            )

    # A category-only instruction is applied only when the entire target phrase
    # consists of known category names and connectors.
    only_pattern = re.compile(r"只(?:写|看|保留|分析|整理)\s*([^。；;\n]+)")
    category_aliases = sorted(
        ((alias, category) for alias, category in CATEGORY_ALIASES.items() if category != "other"),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for match in only_pattern.finditer(instruction):
        segment = match.group(1)
        remainder = segment
        found: set[str] = set()
        for alias, category in category_aliases:
            if alias.casefold() in remainder.casefold():
                remainder = re.sub(re.escape(alias), "", remainder, flags=re.IGNORECASE)
                found.add(category)
        remainder = re.sub(r"(?:相关的?|内容|信息|消息|事项|和|及|与|、|,|，|\s)+", "", remainder)
        if found and not remainder:
            explicit_include_categories = found
            mark(match, "仅保留分类：" + "、".join(sorted(found)))

    residual = "".join(char for idx, char in enumerate(instruction) if not consumed[idx])
    residual = re.sub(r"[\s，,。；;：:、.!！?？]+", "", residual)
    if residual:
        result["unsupported"].append(instruction)
    return grouping, max_length, explicit_include_categories, result


def _normalise_audit(raw: Any) -> dict[str, Any]:
    """Validate collection-layer audit facts for immutable report display."""

    if raw is None:
        return {
            "provided": False,
            "data_complete": None,
            "status": "AUDIT_NOT_PROVIDED",
            "sealed": None,
            "warnings": ["采集审计信息未提供，不能据此断言数据完整"],
            "source_difference": {},
            "zero_message_uncertain_conversations": [],
            "job_id": "AUDIT_NOT_PROVIDED",
            "dataset_digest": "AUDIT_NOT_PROVIDED",
        }
    if not isinstance(raw, Mapping):
        raise ReportEngineError("audit 必须是对象")

    def bounded_string(value: Any, field: str, maximum: int) -> str:
        cleaned = _clean_string(value)
        if not cleaned:
            raise ReportEngineError(f"{field} 不能为空")
        if len(cleaned) > maximum:
            raise ReportEngineError(f"{field} 不能超过 {maximum} 个字符")
        return cleaned

    data_complete = raw.get("data_complete")
    sealed = raw.get("sealed")
    if type(data_complete) is not bool:
        raise ReportEngineError("audit.data_complete 必须是布尔值")
    if type(sealed) is not bool:
        raise ReportEngineError("audit.sealed 必须是布尔值")
    status = bounded_string(raw.get("status"), "audit.status", 160)
    job_id = bounded_string(raw.get("job_id"), "audit.job_id", 512)
    dataset_digest = bounded_string(
        raw.get("dataset_digest"), "audit.dataset_digest", 512
    )

    raw_warnings = raw.get("warnings", raw.get("collection_warnings", []))
    if isinstance(raw_warnings, str) or not isinstance(raw_warnings, Sequence):
        raise ReportEngineError("audit.warnings 必须是字符串数组")
    audit_warnings: list[str] = []
    for index, warning in enumerate(raw_warnings):
        cleaned = _clean_string(warning)
        if not cleaned:
            raise ReportEngineError(f"audit.warnings[{index}] 必须是非空字符串")
        audit_warnings.append(cleaned)

    source_difference = raw.get("source_difference", {})
    if not isinstance(source_difference, Mapping):
        raise ReportEngineError("audit.source_difference 必须是对象")
    try:
        # Round-tripping rejects Python-only objects and produces an immutable,
        # deterministic, JSON-compatible public value.
        source_difference = json.loads(
            json.dumps(
                dict(source_difference),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ReportEngineError("audit.source_difference 必须是有限 JSON 数据") from exc

    uncertain_raw = raw.get("zero_message_uncertain_conversations")
    if uncertain_raw is None:
        coverage = raw.get("coverage", [])
        if isinstance(coverage, str) or not isinstance(coverage, Sequence):
            raise ReportEngineError("audit.coverage 必须是数组")
        uncertain_raw = [
            item
            for item in coverage
            if isinstance(item, Mapping)
            and type(item.get("message_count")) is int
            and item.get("message_count") == 0
            and _clean_string(item.get("uncertainty"))
        ]
    if isinstance(uncertain_raw, str) or not isinstance(uncertain_raw, Sequence):
        raise ReportEngineError("audit.zero_message_uncertain_conversations 必须是数组")
    uncertain: list[dict[str, str]] = []
    for index, item in enumerate(uncertain_raw):
        if isinstance(item, str):
            conversation_id = bounded_string(
                item, f"audit.zero_message_uncertain_conversations[{index}]", 512
            )
            uncertainty = "ZERO_MESSAGE_UNCERTAIN"
            conversation_name = ""
        elif isinstance(item, Mapping):
            conversation_id = bounded_string(
                item.get("conversation_id") or item.get("id"),
                f"audit.zero_message_uncertain_conversations[{index}].conversation_id",
                512,
            )
            conversation_name = _clean_string(
                item.get("conversation_name") or item.get("title")
            )
            if len(conversation_name) > 500:
                raise ReportEngineError(
                    f"audit.zero_message_uncertain_conversations[{index}].conversation_name "
                    "不能超过 500 个字符"
                )
            uncertainty = bounded_string(
                item.get("uncertainty"),
                f"audit.zero_message_uncertain_conversations[{index}].uncertainty",
                300,
            )
        else:
            raise ReportEngineError(
                f"audit.zero_message_uncertain_conversations[{index}] 必须是字符串或对象"
            )
        uncertain.append(
            {
                "conversation_id": conversation_id,
                "conversation_name": conversation_name,
                "uncertainty": uncertainty,
            }
        )

    return {
        "provided": True,
        "data_complete": data_complete,
        "status": status,
        "sealed": sealed,
        "warnings": list(dict.fromkeys(audit_warnings)),
        "source_difference": source_difference,
        "zero_message_uncertain_conversations": uncertain,
        "job_id": job_id,
        "dataset_digest": dataset_digest,
    }


def _normalise_report_config(raw: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ReportEngineError("report_config 必须是对象")

    warnings: list[str] = []
    default_tz = _parse_timezone(raw.get("timezone"))
    title = _clean_string(raw.get("title"), limit=120) or "钉钉聊天报告"

    max_raw = raw.get("max_length_chars", raw.get("max_length", 8_000))
    if isinstance(max_raw, bool) or not isinstance(max_raw, int):
        raise ReportEngineError("max_length_chars 必须是整数")
    if max_raw < 200:
        raise ReportEngineError("max_length_chars 不能小于 200")
    max_length = min(max_raw, 200_000)

    grouping_raw = _clean_string(raw.get("grouping")) or "conversation"
    grouping_aliases = {
        "conversation": "conversation",
        "chat": "conversation",
        "会话": "conversation",
        "topic": "topic",
        "主题": "topic",
        "time": "time",
        "date": "time",
        "时间": "time",
        "日期": "time",
        "owner": "owner",
        "负责人": "owner",
    }
    grouping = grouping_aliases.get(grouping_raw.casefold())
    if grouping is None:
        grouping = "conversation"
        warnings.append(f"不支持分组方式“{grouping_raw}”，已安全降级为按会话分组")

    filters = _normalise_filters(raw.get("filters"), default_tz)
    include_categories_raw = raw.get("include_categories")
    include_categories: set[str] | None = None
    if include_categories_raw is not None:
        aliases = _normalise_string_set(include_categories_raw, "include_categories")
        include_categories = set()
        for alias in aliases:
            category = CATEGORY_ALIASES.get(alias)
            if category is None:
                raise ReportEngineError(f"未知报告分类：{alias}")
            include_categories.add(category)

    instruction = _clean_string(raw.get("custom_instruction"))
    grouping, max_length, include_categories, instruction_result = _apply_free_instruction(
        instruction,
        grouping=grouping,
        max_length=max_length,
        filters=filters,
        explicit_include_categories=include_categories,
    )
    if instruction_result["unsupported"]:
        warnings.append("自由要求中包含无法由确定性规则理解的内容，已保留供人工审阅，未假装执行")
    if instruction_result["conflicts"]:
        warnings.append("自由要求与结构化面板选项存在冲突；结构化选项未被文字静默覆盖")

    allow_inferences = raw.get("allow_inferences") is True
    if allow_inferences:
        warnings.append("当前为准确性优先的提取式引擎；即使允许推断，也不会自动生成无直接证据的推断")

    config = {
        "title": title,
        "timezone": default_tz.key,
        "max_length_chars": max_length,
        "requested_grouping": grouping_raw,
        "grouping": grouping,
        "sections": deepcopy(raw.get("sections")),
        "topic_rules": deepcopy(raw.get("topic_rules")),
        "filters": filters,
        "include_categories": include_categories,
        "custom_instruction": instruction_result,
        "allow_inferences": allow_inferences,
        "audit": _normalise_audit(raw.get("audit")),
    }
    return config, warnings


def _filter_reason(message: Mapping[str, Any], filters: Mapping[str, Any]) -> str | None:
    conversation_id = message["conversation_id"]
    message_id = message["message_id"]
    sender = message["sender"].casefold()
    message_type = message["message_type"].casefold()
    status = message["status"].casefold()
    text = message["text"].casefold()

    if filters["include_conversation_ids"] and conversation_id not in filters["include_conversation_ids"]:
        return "会话不在包含列表"
    if conversation_id in filters["exclude_conversation_ids"]:
        return "会话在排除列表"
    if filters["include_message_ids"] and message_id not in filters["include_message_ids"]:
        return "消息不在包含列表"
    if message_id in filters["exclude_message_ids"]:
        return "消息在排除列表"
    if filters["include_senders"] and sender not in filters["include_senders"]:
        return "发送人不在包含列表"
    if sender in filters["exclude_senders"]:
        return "发送人在排除列表"
    if message["is_system"] and not filters["include_system_messages"]:
        return "系统消息已排除"
    if message["is_bot"] and not filters["include_bot_messages"]:
        return "机器人消息已排除"
    unknown_origin_kinds: list[str] = []
    if not filters["include_bot_messages"] and not message["bot_origin_verified"]:
        unknown_origin_kinds.append("机器人")
    if not filters["include_system_messages"] and not message["system_origin_verified"]:
        unknown_origin_kinds.append("系统")
    if unknown_origin_kinds:
        return (
            "消息来源未验证（"
            + "/".join(unknown_origin_kinds)
            + "），按排除策略未纳入"
        )
    recalled_statuses = {
        "recall",
        "recalled",
        "message_recalled",
        "withdraw",
        "withdrawn",
        "message_withdrawn",
        "recall_notice_unresolved",
        "撤回",
        "已撤回",
    }
    if not filters["include_recalled"] and (
        status in recalled_statuses or "撤回" in status
    ):
        return "撤回消息已排除"
    if filters["message_types"] and message_type not in filters["message_types"]:
        return "消息类型未包含"
    if filters["include_keywords"] and not any(keyword in text for keyword in filters["include_keywords"]):
        return "未命中包含关键词"
    if any(keyword in text for keyword in filters["exclude_keywords"]):
        return "命中排除关键词"
    if filters["exclude_casual"] and _CASUAL_ONLY.fullmatch(message["text"].strip()):
        return "保守识别为纯闲聊"
    if filters["start"] is not None and message["_sort_time"] < filters["start"]:
        return "早于开始时间"
    if filters["end"] is not None and message["_sort_time"] >= filters["end"]:
        return "达到或晚于结束时间"
    return None


def _matching_categories(text: str) -> tuple[list[str], list[str]]:
    categories: list[str] = []
    bases: list[str] = []
    risk_text = _BENIGN_RISK_PHRASES.sub("", text)
    for category, definition in CATEGORY_DEFINITIONS.items():
        target = risk_text if category == "risk" else text
        matched_pattern = next(
            (pattern for pattern in definition["patterns"] if re.search(pattern, target, re.IGNORECASE)),
            None,
        )
        if matched_pattern is not None:
            categories.append(category)
            bases.append(f"{definition['title']}：命中明确规则")
    return categories, bases


def _extract_owner(text: str, message_id: str) -> dict[str, Any]:
    for pattern in _OWNER_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip(" @，,。；;：:")
            if value:
                return {
                    "value": value,
                    "certainty": "explicit",
                    "source_message_id": message_id,
                }
    return {"value": None, "certainty": "unknown", "source_message_id": None}


def _extract_deadline(text: str, message_id: str) -> dict[str, Any]:
    for pattern in _DEADLINE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            return {
                "value": value,
                "certainty": "explicit_text",
                "source_message_id": message_id,
                "normalised": None,
            }
    return {
        "value": None,
        "certainty": "unknown",
        "source_message_id": None,
        "normalised": None,
    }


def _finding_id(message: Mapping[str, Any], category: str) -> str:
    material = "\x1f".join((message["conversation_id"], message["message_id"], category))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _make_finding(
    message: Mapping[str, Any],
    *,
    category: str,
    all_categories: Sequence[str],
    classification_basis: str,
) -> dict[str, Any]:
    excerpt_limit = 280
    excerpt = message["text"]
    excerpt_truncated = len(excerpt) > excerpt_limit
    if excerpt_truncated:
        excerpt = excerpt[: excerpt_limit - 1].rstrip() + "…"

    evidence = {
        "conversation_id": message["conversation_id"],
        "conversation_name": message["conversation_name"],
        "message_id": message["message_id"],
        "sender": message["sender"],
        "time": message["time"],
        "excerpt": excerpt,
        "excerpt_truncated": excerpt_truncated,
    }
    return {
        "id": _finding_id(message, category),
        "section_id": category,
        "category": category,
        "categories": list(all_categories),
        "statement": message["text"],
        "statement_mode": "extractive",
        "classification_basis": [classification_basis],
        "fact_or_inference": "fact",
        "owner": _extract_owner(message["text"], message["message_id"]),
        "deadline": _extract_deadline(message["text"], message["message_id"]),
        "evidence": [evidence],
    }


def _normalise_sections(raw_sections: Any, warnings: list[str]) -> list[dict[str, Any]]:
    if raw_sections is None:
        return [
            {"id": category, "title": DEFAULT_SECTION_TITLES[category], "categories": [category]}
            for category in DEFAULT_SECTION_ORDER
        ]
    if isinstance(raw_sections, str) or not isinstance(raw_sections, Sequence):
        raise ReportEngineError("sections 必须是数组")

    sections: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    assigned_categories: set[str] = set()

    for index, item in enumerate(raw_sections):
        if isinstance(item, str):
            alias = item.strip().casefold()
            category = CATEGORY_ALIASES.get(alias)
            if category is None:
                section_id = f"custom_{index + 1}"
                title = item.strip() or f"自定义章节 {index + 1}"
                categories: list[str] = []
                warnings.append(f"自定义章节“{title}”没有显式分类映射，将保持为空")
            else:
                section_id = category
                title = DEFAULT_SECTION_TITLES[category]
                categories = [category]
        elif isinstance(item, Mapping):
            title = _clean_string(item.get("title"), limit=100) or f"自定义章节 {index + 1}"
            requested_id = _clean_string(item.get("id"))
            inferred = CATEGORY_ALIASES.get(requested_id.casefold()) if requested_id else None
            section_id = inferred or re.sub(r"[^a-zA-Z0-9_-]+", "_", requested_id).strip("_") or f"custom_{index + 1}"
            raw_categories = item.get("categories")
            if raw_categories is None and inferred:
                categories = [inferred]
            elif raw_categories is None:
                categories = []
                warnings.append(f"自定义章节“{title}”没有显式 categories，将保持为空")
            else:
                aliases = _normalise_string_set(raw_categories, f"sections[{index}].categories")
                categories = []
                for alias in sorted(aliases):
                    category = CATEGORY_ALIASES.get(alias)
                    if category is None:
                        raise ReportEngineError(f"章节“{title}”包含未知分类：{alias}")
                    categories.append(category)
        else:
            raise ReportEngineError(f"sections[{index}] 必须是字符串或对象")

        base_id = section_id
        suffix = 2
        while section_id in used_ids:
            section_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(section_id)

        unique_categories = []
        for category in categories:
            if category in assigned_categories:
                warnings.append(f"分类“{category}”重复映射，仅使用首次出现的章节")
                continue
            assigned_categories.add(category)
            unique_categories.append(category)
        sections.append({"id": section_id, "title": title, "categories": unique_categories})

    if "other" not in assigned_categories:
        section_id = "other"
        suffix = 2
        while section_id in used_ids:
            section_id = f"other_{suffix}"
            suffix += 1
        sections.append(
            {
                "id": section_id,
                "title": DEFAULT_SECTION_TITLES["other"],
                "categories": ["other"],
            }
        )
    return sections


def _normalise_topic_rules(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    rules: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        iterable = [{"title": title, "keywords": keywords} for title, keywords in raw.items()]
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        iterable = raw
    else:
        raise ReportEngineError("topic_rules 必须是对象或数组")

    for index, rule in enumerate(iterable):
        if not isinstance(rule, Mapping):
            raise ReportEngineError(f"topic_rules[{index}] 必须是对象")
        title = _clean_string(rule.get("title"), limit=80)
        if not title:
            raise ReportEngineError(f"topic_rules[{index}] 缺少 title")
        keywords = _normalise_string_set(rule.get("keywords"), f"topic_rules[{index}].keywords")
        if not keywords:
            raise ReportEngineError(f"topic_rules[{index}] 至少需要一个关键词")
        rules.append({"key": f"topic_{index + 1}", "title": title, "keywords": sorted(keywords)})
    return rules


def _group_findings(
    findings: list[dict[str, Any]],
    grouping: str,
    topic_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for finding in findings:
        evidence = finding["evidence"][0]
        if grouping == "time":
            key = evidence["time"][:10]
            title = key
        elif grouping == "topic":
            if topic_rules:
                folded = finding["statement"].casefold()
                matched_rule = next(
                    (rule for rule in topic_rules if any(keyword in folded for keyword in rule["keywords"])),
                    None,
                )
                if matched_rule is None:
                    key = "topic_review"
                    title = "其他主题/待人工审阅"
                else:
                    key = matched_rule["key"]
                    title = matched_rule["title"]
            else:
                key = f"category_{finding['category']}"
                title = DEFAULT_SECTION_TITLES.get(
                    finding["category"], DEFAULT_SECTION_TITLES["other"]
                )
        elif grouping == "owner":
            owner = finding["owner"]["value"]
            if owner:
                key = f"owner_{owner}"
                title = f"负责人：{owner}"
            else:
                key = "owner_unknown"
                title = "待确认负责人（消息中未明确）"
        else:
            key = evidence["conversation_id"]
            title = f"{evidence['conversation_name']}（会话 {evidence['conversation_id']}）"
        if key not in groups:
            groups[key] = {"key": key, "title": title, "findings": []}
        groups[key]["findings"].append(finding)
    return list(groups.values())


def _public_filter_config(filters: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in filters.items():
        if isinstance(value, set):
            result[key] = sorted(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat(timespec="seconds")
        else:
            result[key] = value
    return result


def analyze_messages(
    messages: Sequence[Mapping[str, Any]],
    report_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a grounded structured analysis from normalized messages.

    Invalid or conflicting messages are excluded from conclusions and recorded in
    ``review_items``.  All entries in ``facts`` and all section findings contain a
    complete evidence tuple: conversation ID, message ID, sender and timestamp.
    """

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ReportEngineError("messages 必须是消息对象数组")

    config, warnings = _normalise_report_config(report_config)
    default_tz = _parse_timezone(config["timezone"])
    sections_spec = _normalise_sections(config["sections"], warnings)
    topic_rules = _normalise_topic_rules(config["topic_rules"])

    supported_explicit_categories = set(CATEGORY_DEFINITIONS)
    if config["sections"] is None:
        section_selected_categories = set(supported_explicit_categories)
    else:
        section_selected_categories = {
            category
            for section in sections_spec
            for category in section["categories"]
            if category != "other"
        }
    if config["include_categories"] is None:
        selected_explicit_categories = section_selected_categories
    else:
        selected_explicit_categories = section_selected_categories.intersection(
            config["include_categories"]
        )

    effective_grouping = config["grouping"]
    if effective_grouping == "topic" and not topic_rules:
        warnings.append(
            "未提供显式 topic_rules；按明确关键词分类作为保守主题，不进行开放式主题推断"
        )

    valid: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    normalisation_notes: list[str] = []
    for index, raw in enumerate(messages):
        message, issue, notes = _normalise_message(raw, index, default_tz)
        normalisation_notes.extend(notes)
        if issue is not None:
            review_items.append(issue)
        elif message is not None:
            valid.append(message)

    by_identity: "OrderedDict[tuple[str, str], list[dict[str, Any]]]" = OrderedDict()
    for message in valid:
        key = (message["conversation_id"], message["message_id"])
        by_identity.setdefault(key, []).append(message)

    deduplicated: list[dict[str, Any]] = []
    duplicate_count = 0
    conflict_count = 0
    for (conversation_id, message_id), copies in by_identity.items():
        signatures = {_message_signature(copy) for copy in copies}
        duplicate_count += max(0, len(copies) - 1)
        if len(signatures) > 1:
            conflict_count += 1
            review_items.append(
                {
                    "identity": {
                        "conversation_id": conversation_id,
                        "message_id": message_id,
                    },
                    "reason": "同一消息身份存在冲突内容，无法确定有效版本，未进入分析",
                    "variant_count": len(signatures),
                }
            )
            continue
        deduplicated.append(copies[0])

    deduplicated.sort(
        key=lambda message: (
            message["_sort_time"],
            message["conversation_id"],
            message["message_id"],
        )
    )

    included_candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    exclusion_counts: dict[str, int] = {}
    excluded_message_identities: set[tuple[str, str]] = set()
    for message in deduplicated:
        reason = _filter_reason(message, config["filters"])
        if reason:
            identity = (message["conversation_id"], message["message_id"])
            excluded_message_identities.add(identity)
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            exclusions.append(
                {
                    "conversation_id": message["conversation_id"],
                    "message_id": message["message_id"],
                    "reason": reason,
                }
            )
        else:
            included_candidates.append(message)

    unknown_origin_excluded = sum(
        count
        for reason, count in exclusion_counts.items()
        if reason.startswith("消息来源未验证（")
    )
    if unknown_origin_excluded:
        warnings.append(
            f"{unknown_origin_excluded} 条消息缺少可验证的机器人或系统来源标记，"
            "已按当前排除策略保守排除；未将来源未知静默视为普通人工消息"
        )
    unknown_origin_included = sum(
        1
        for message in included_candidates
        if not message["bot_origin_verified"] or not message["system_origin_verified"]
    )
    if unknown_origin_included:
        warnings.append(
            f"{unknown_origin_included} 条进入分析的消息缺少部分机器人或系统来源验证；"
            "本次配置已包含相应类型，报告不据此断言发送来源为人工或非系统"
        )

    findings: list[dict[str, Any]] = []
    included_message_identities: set[tuple[str, str]] = set()
    filtered_by_category = 0
    for message in included_candidates:
        identity = (message["conversation_id"], message["message_id"])
        matched_categories, matched_bases = _matching_categories(message["text"])
        if not matched_categories:
            findings.append(
                _make_finding(
                    message,
                    category="other",
                    all_categories=["other"],
                    classification_basis="未命中受支持的明确分类规则，保留供人工审阅",
                )
            )
            included_message_identities.add(identity)
            continue

        selected_for_message = 0
        for category, basis in zip(matched_categories, matched_bases, strict=True):
            if category not in selected_explicit_categories:
                filtered_by_category += 1
                exclusions.append(
                    {
                        "conversation_id": message["conversation_id"],
                        "message_id": message["message_id"],
                        "reason": "分类不在报告包含范围",
                        "category": category,
                    }
                )
                exclusion_counts["分类不在报告包含范围"] = exclusion_counts.get(
                    "分类不在报告包含范围", 0
                ) + 1
                continue
            findings.append(
                _make_finding(
                    message,
                    category=category,
                    all_categories=matched_categories,
                    classification_basis=basis,
                )
            )
            selected_for_message += 1
        if selected_for_message:
            included_message_identities.add(identity)
        else:
            excluded_message_identities.add(identity)

    category_to_section: dict[str, str] = {}
    for section in sections_spec:
        for category in section["categories"]:
            category_to_section.setdefault(category, section["id"])
    other_section_id = category_to_section["other"]

    section_findings: dict[str, list[dict[str, Any]]] = {section["id"]: [] for section in sections_spec}
    for finding in findings:
        target = category_to_section.get(finding["category"], other_section_id)
        finding["section_id"] = target
        section_findings[target].append(finding)

    sections: list[dict[str, Any]] = []
    for section in sections_spec:
        assigned = section_findings[section["id"]]
        sections.append(
            {
                "id": section["id"],
                "title": section["title"],
                "categories": section["categories"],
                "finding_count": len(assigned),
                "findings": assigned,
                "groups": _group_findings(assigned, effective_grouping, topic_rules),
            }
        )

    public_config = {
        "title": config["title"],
        "timezone": config["timezone"],
        "max_length_chars": config["max_length_chars"],
        "requested_grouping": config["requested_grouping"],
        "effective_grouping": effective_grouping,
        "filters": _public_filter_config(config["filters"]),
        "include_categories": sorted(config["include_categories"]) if config["include_categories"] is not None else None,
        "section_selected_categories": sorted(section_selected_categories),
        "selected_explicit_categories": sorted(selected_explicit_categories),
        "custom_instruction": config["custom_instruction"],
        "allow_inferences": config["allow_inferences"],
        "topic_rules": topic_rules,
        "audit": config["audit"],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "engine": "grounded-extractive-v1",
        "config": public_config,
        "summary": {
            "input_message_count": len(messages),
            "valid_message_count": len(valid),
            "invalid_message_count": len(messages) - len(valid),
            "duplicate_observation_count": duplicate_count,
            "conflicting_identity_count": conflict_count,
            "deduplicated_message_count": len(deduplicated),
            "excluded_message_count": len(excluded_message_identities),
            "exclusion_record_count": len(exclusions),
            "included_message_count": len(included_message_identities),
            "finding_count": len(findings),
            "inference_count": 0,
            "category_filtered_count": filtered_by_category,
            "report_truncated": False,
        },
        "exclusion_counts": exclusion_counts,
        "exclusions": exclusions,
        "review_items": review_items,
        "normalisation_notes": normalisation_notes,
        "warnings": warnings,
        "audit": config["audit"],
        "facts": findings,
        "inferences": [],
        "sections": sections,
        "rendering": {
            "rendered_finding_count": None,
            "omitted_finding_count": None,
            "truncated": None,
        },
    }


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…"[:limit]
    return text[: limit - 1].rstrip() + "…"


def _markdown_inline(value: Any) -> str:
    """Keep untrusted message text on one Markdown line without raw HTML."""

    text = re.sub(r"\s+", " ", str(value)).strip()
    text = html.escape(text, quote=False)
    return re.sub(r"([\\`*_[\]{}()#+.!|>-])", r"\\\1", text)


def render_full_review_appendix(
    messages: Sequence[Mapping[str, Any]],
    *,
    timezone_name: str = "Asia/Shanghai",
) -> tuple[str, dict[str, Any]]:
    """Render every unique input message for deterministic human review.

    This path deliberately performs no relevance or work-value classification.
    Callers must pass the already-normalised scope produced by the user's
    conversation, time and message-filter configuration. Text is never clipped;
    only whitespace is normalised so untrusted content cannot create Markdown or
    raw-HTML structure.
    """

    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ReportEngineError("人工核对清单的 messages 必须是消息对象数组")

    default_tz = _parse_timezone(timezone_name)
    normalised: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, raw in enumerate(messages):
        message, issue, _notes = _normalise_message(raw, index, default_tz)
        if issue is not None:
            issues.append(issue)
        elif message is not None:
            normalised.append(message)
    if issues:
        first = issues[0]
        raise ReportEngineError(
            "人工核对清单发现无法确认身份或原文的消息，已停止生成："
            + str(first.get("reason") or "消息字段不完整")
        )

    by_identity: "OrderedDict[tuple[str, str], list[dict[str, Any]]]" = OrderedDict()
    for message in normalised:
        identity = (message["conversation_id"], message["message_id"])
        by_identity.setdefault(identity, []).append(message)

    unique_messages: list[dict[str, Any]] = []
    duplicate_observation_count = 0
    for (conversation_id, message_id), copies in by_identity.items():
        signatures = {_message_signature(copy) for copy in copies}
        if len(signatures) != 1:
            raise ReportEngineError(
                "人工核对清单发现同一消息 ID 存在冲突内容，已停止生成："
                f"{conversation_id}/{message_id}"
            )
        duplicate_observation_count += max(0, len(copies) - 1)
        unique_messages.append(copies[0])

    unique_messages.sort(
        key=lambda message: (
            message["conversation_name"].casefold(),
            message["conversation_id"].casefold(),
            message["_sort_time"],
            message["message_id"],
        )
    )

    grouped: "OrderedDict[tuple[str, str], list[dict[str, Any]]]" = OrderedDict()
    for message in unique_messages:
        key = (message["conversation_id"], message["conversation_name"])
        grouped.setdefault(key, []).append(message)

    lines = [
        "---",
        "",
        "## 全量人工核对清单（不经 AI 取舍）",
        "",
        (
            f"> 共 {len(unique_messages)} 条唯一消息。清单只服从你选择的会话、时间范围和消息筛选；"
            "AI 不参与相关性判断，也不会删除或截断文字内容。为便于逐条查看，原文换行统一显示为空格。"
        ),
        "",
    ]
    review_number = 0
    canonical_rows: list[dict[str, str]] = []
    identities: list[dict[str, str]] = []
    conversation_counts: list[dict[str, Any]] = []
    for (conversation_id, conversation_name), rows in grouped.items():
        conversation_counts.append(
            {
                "conversation_id": conversation_id,
                "conversation_name": conversation_name,
                "message_count": len(rows),
            }
        )
        lines.extend(
            [
                (
                    f"### {_markdown_inline(conversation_name)}"
                    f"（会话 ID：{_markdown_inline(conversation_id)}，{len(rows)} 条）"
                ),
                "",
            ]
        )
        for message in rows:
            review_number += 1
            lines.extend(
                [
                    (
                        f"{review_number}. {_markdown_inline(message['time'])}｜"
                        f"发送人：{_markdown_inline(message['sender'])}｜"
                        f"类型：{_markdown_inline(message['message_type'])}"
                    ),
                    f"   - 原文：{_markdown_inline(message['text'])}",
                    f"   - 消息 ID：{_markdown_inline(message['message_id'])}",
                    "",
                ]
            )
            identity = {
                "conversation_id": message["conversation_id"],
                "message_id": message["message_id"],
            }
            identities.append(identity)
            canonical_rows.append(
                {
                    **identity,
                    "conversation_name": message["conversation_name"],
                    "time": message["time"],
                    "sender": message["sender"],
                    "message_type": message["message_type"],
                    "text": message["text"],
                }
            )

    digest = hashlib.sha256(
        json.dumps(
            canonical_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    unique_identity_count = len(by_identity)
    audit = {
        "enabled": True,
        "mode": "deterministic-full-message-ledger-v1",
        "source_scope": "after_user_configured_filters",
        "ai_selection_applied": False,
        "text_truncated": False,
        "line_breaks_normalized": True,
        "input_observation_count": len(messages),
        "duplicate_observation_count": duplicate_observation_count,
        "unique_identity_count": unique_identity_count,
        "rendered_message_count": review_number,
        "conversation_count": len(grouped),
        "coverage_validated": review_number == unique_identity_count,
        "dataset_digest": digest,
        "conversation_counts": conversation_counts,
        "message_identities": identities,
    }
    if audit["coverage_validated"] is not True:
        raise ReportEngineError("人工核对清单未覆盖全部唯一消息，已停止生成")
    return "\n".join(lines).rstrip(), audit


def _finding_markdown(finding: Mapping[str, Any], statement_limit: int) -> str:
    statement = _markdown_inline(_clip(str(finding["statement"]), statement_limit))
    evidence = finding["evidence"][0]
    lines = [f"- {statement}"]
    owner = finding["owner"]
    deadline = finding["deadline"]
    if owner["value"] is not None:
        lines.append(f"  - 明确负责人：{_markdown_inline(owner['value'])}")
    elif finding["category"] == "todo":
        lines.append("  - 负责人：待确认（消息中未明确）")
    if deadline["value"] is not None:
        lines.append(f"  - 明确截止表达：{_markdown_inline(deadline['value'])}")
    elif finding["category"] == "todo":
        lines.append("  - 截止时间：待确认（消息中未明确）")
    lines.append(
        "  - 来源："
        f"{_markdown_inline(evidence['conversation_name'])}｜"
        f"conversation_id={_markdown_inline(evidence['conversation_id'])}｜"
        f"{_markdown_inline(evidence['sender'])}｜{_markdown_inline(evidence['time'])}｜"
        f"message_id={_markdown_inline(evidence['message_id'])}"
    )
    return "\n".join(lines)


def _source_difference_summary(source_difference: Mapping[str, Any]) -> str:
    if not source_difference:
        return "无明细"
    parts: list[str] = []
    for key in sorted(source_difference):
        value = source_difference[key]
        if isinstance(value, list):
            rendered = f"{len(value)} 项"
        elif isinstance(value, Mapping):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        elif value is True:
            rendered = "true"
        elif value is False:
            rendered = "false"
        elif value is None:
            rendered = "null"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return "；".join(parts)


def _audit_header(analysis: Mapping[str, Any]) -> str:
    audit = analysis["audit"]

    def boolean_text(value: Any) -> str:
        if value is True:
            return "true"
        if value is False:
            return "false"
        return "unknown"

    lines = [
        "## 数据完整性审计（固定，不可裁剪）",
        f"- data_complete：{boolean_text(audit['data_complete'])}",
        f"- status：{_markdown_inline(audit['status'])}",
        f"- sealed：{boolean_text(audit['sealed'])}",
        f"- job_id：{_markdown_inline(audit['job_id'])}",
        f"- dataset_digest：{_markdown_inline(audit['dataset_digest'])}",
        "- source_difference："
        + _markdown_inline(_source_difference_summary(audit["source_difference"])),
    ]
    if audit["warnings"]:
        lines.append("- 采集警告：")
        lines.extend(f"  - {_markdown_inline(warning)}" for warning in audit["warnings"])
    else:
        lines.append("- 采集警告：无")
    uncertain = audit["zero_message_uncertain_conversations"]
    if uncertain:
        lines.append("- 零消息不确定会话：")
        for item in uncertain:
            name = f"{item['conversation_name']}｜" if item["conversation_name"] else ""
            lines.append(
                "  - "
                f"{_markdown_inline(name)}conversation_id={_markdown_inline(item['conversation_id'])}｜"
                f"{_markdown_inline(item['uncertainty'])}"
            )
    else:
        lines.append("- 零消息不确定会话：无")
    if analysis["warnings"]:
        lines.append("- 分析警告：")
        lines.extend(f"  - {_markdown_inline(warning)}" for warning in analysis["warnings"])
    else:
        lines.append("- 分析警告：无")
    return "\n".join(lines)


def _render_markdown_with_meta(analysis: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    config = analysis["config"]
    max_length = int(config["max_length_chars"])
    title = _markdown_inline(_clip(str(config["title"]), 120))
    summary = analysis["summary"]

    header = (
        f"# {title}\n\n"
        "> 准确性优先的提取式报告：结论只来自列明的消息；未明确的信息不会自动补全。\n\n"
        f"{_audit_header(analysis)}\n\n"
        "## 数据范围与分析计数\n"
        f"- 输入消息：{summary['input_message_count']}\n"
        f"- 进入分析：{summary['included_message_count']}\n"
        f"- 事实条目：{summary['finding_count']}\n"
        f"- 排除消息：{summary['excluded_message_count']}\n"
        f"- 排除记录：{summary['exclusion_record_count']}\n"
        f"- 待人工审阅：{len(analysis['review_items'])}\n"
        f"- 分组方式：{config['effective_grouping']}\n"
    )
    filters = config.get("filters", {})
    start = filters.get("start")
    end = filters.get("end")
    if start is not None and end is not None:
        header += (
            f"- 数据范围：[{_markdown_inline(str(start))}, {_markdown_inline(str(end))})"
            f"（{_markdown_inline(str(config['timezone']))}）\n"
        )
    conversation_ids = filters.get("include_conversation_ids") or []
    if conversation_ids:
        rendered_ids = "、".join(_markdown_inline(str(value)) for value in conversation_ids)
        header += f"- 会话范围：{rendered_ids}\n"
    if config["custom_instruction"]["raw"]:
        header += f"- 用户要求：{_markdown_inline(_clip(config['custom_instruction']['raw'], 180))}\n"
        if config["custom_instruction"]["unsupported"]:
            header += "- 要求处理：存在未被确定性规则理解的部分，未自动执行\n"

    total = int(summary["finding_count"])
    truncation_note = "\n\n> 受最大长度限制，部分事实未写入 Markdown；结构化 analysis 保留全部事实。"
    required_length = len(header.rstrip()) + (len(truncation_note) if total else 0)
    if required_length > max_length:
        raise ReportEngineError(
            "max_length_chars 无法容纳不可裁剪的数据审计头和必要的截断警告；"
            f"至少需要 {required_length} 个字符"
        )

    output = header.rstrip()
    rendered = 0

    current_section: str | None = None
    current_group: tuple[str, str] | None = None
    stop = False
    for section in analysis["sections"]:
        if stop or not section["findings"]:
            continue
        for group in section["groups"]:
            if stop:
                break
            for finding in group["findings"]:
                prefix = ""
                if current_section != section["id"]:
                    prefix += f"\n\n## {_markdown_inline(section['title'])}"
                group_key = (section["id"], group["key"])
                if current_group != group_key:
                    prefix += f"\n\n### {_markdown_inline(group['title'])}"

                added = False
                for statement_limit in (320, 220, 140, 90, 60, 36):
                    block = prefix + "\n\n" + _finding_markdown(finding, statement_limit)
                    reserve = len(truncation_note) if rendered + 1 < total else 0
                    if len(output) + len(block) + reserve <= max_length:
                        output += block
                        rendered += 1
                        current_section = section["id"]
                        current_group = group_key
                        added = True
                        break
                if not added:
                    stop = True
                    break

    omitted = total - rendered
    if omitted:
        if len(output) + len(truncation_note) > max_length:
            raise ReportEngineError("内部长度预算错误：无法保留必要的报告截断警告")
        output += truncation_note

    return output.rstrip(), {
        "rendered_finding_count": rendered,
        "omitted_finding_count": omitted,
        "truncated": omitted > 0,
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    """Render structured analysis as bounded Markdown.

    A conclusion is either emitted together with its full evidence identity or not
    emitted at all; output is never cut through an evidence line.
    """

    markdown, _ = _render_markdown_with_meta(analysis)
    return markdown


def generate_report(
    messages: Sequence[Mapping[str, Any]],
    report_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze normalized messages and return structured analysis plus Markdown."""

    analysis = analyze_messages(messages, report_config)
    markdown, rendering = _render_markdown_with_meta(analysis)
    analysis["rendering"] = rendering
    analysis["summary"]["report_truncated"] = rendering["truncated"]
    return {"analysis": analysis, "markdown": markdown}


__all__ = [
    "ReportEngineError",
    "analyze_messages",
    "generate_report",
    "render_full_review_appendix",
    "render_markdown",
]
