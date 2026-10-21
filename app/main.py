"""FastAPI application for the manual DingTalk report panel.

The application deliberately has no scheduler.  A report job can only be
created by an explicit POST carrying ``explicit_confirm=true``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import pathlib
import re
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import __version__
from .dws_adapter import (
    DwsAdapter,
    DwsAdapterError,
    FixtureDwsAdapter,
)
from .llm_config_store import LLMConfigStore, LLMConfigStoreError
from .llm_engine import LLMConfig, LLMEngineError, LLMReportEngine
from .report_engine import (
    ReportEngineError,
    analyze_messages,
)
from .storage import Storage, StorageError


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
STATIC_ROOT = pathlib.Path(__file__).resolve().parent / "static"
DEFAULT_TIMEZONE = "Asia/Shanghai"
CONVERSATION_DIRECTORY_STATE_SCHEMA = 2
CONVERSATION_DIRECTORY_REFRESH_REQUIRED = (
    "会话目录尚未按完整分页方式验证，请点击“刷新聊天列表”。刷新会读取全部普通会话和"
    "已加入群；任一分页失败都会停止，不会发布部分结果。"
)


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _csv_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        source: Sequence[Any] = re.split(r"[,，\n]", value)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        source = value
    else:
        raise ValueError("必须是字符串数组或逗号分隔文本")
    output: list[str] = []
    seen: set[str] = set()
    for item in source:
        text = str(item).strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            output.append(text)
    return output


class MessageFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_senders: list[str] = Field(default_factory=list)
    exclude_senders: list[str] = Field(default_factory=list)
    include_keywords: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(default_factory=list)
    message_types: list[str] = Field(
        default_factory=lambda: ["text", "image", "file", "link", "audio", "video"]
    )
    include_system_messages: bool = False
    include_bot_messages: bool = False
    include_recalled: bool = True
    expand_replies: bool = False
    expand_forwarded: bool = False
    only_related_to_me: bool = False
    casual_filter: str = "off"

    @field_validator(
        "include_senders",
        "exclude_senders",
        "include_keywords",
        "exclude_keywords",
        "message_types",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: Any) -> list[str]:
        return _csv_strings(value)

    @field_validator("casual_filter")
    @classmethod
    def validate_casual_filter(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"off", "conservative"}:
            raise ValueError("当前准确性版本只支持 off/conservative；aggressive 可能误删工作消息")
        return normalized

    @model_validator(mode="after")
    def reject_unverifiable_filters(self) -> "MessageFilters":
        unsupported = []
        if self.expand_replies:
            unsupported.append("expand_replies")
        if self.expand_forwarded:
            unsupported.append("expand_forwarded")
        if unsupported:
            raise ValueError(
                "当前 DWS 数据契约不能可靠执行以下筛选，必须保持 false："
                + "、".join(unsupported)
            )
        return self


class ReportOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_name: str
    title: str = "钉钉聊天工作报告"
    grouping: str = "conversation"
    sections: list[str] = Field(default_factory=list)
    length: str = "standard"
    max_length_chars: int = 8000
    language: str = "zh-CN"
    inference_policy: str = "facts_only"
    custom_instruction: str = ""
    evidence_enabled: bool = True
    evidence_level: str = "all_items"
    include_full_review_appendix: bool = False

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("报告标题不能为空")
        return normalized[:120]

    @field_validator("subject_name")
    @classmethod
    def validate_subject_name(cls, value: str) -> str:
        if any(character in value for character in ("\r", "\n", "\x00")):
            raise ValueError("报告本人姓名不能包含换行或控制字符")
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            raise ValueError("必须填写报告本人姓名")
        if normalized.casefold() in {"我", "本人", "自己", "me", "myself"}:
            raise ValueError("报告本人必须填写钉钉中可核对的具体姓名")
        if len(normalized) > 80:
            raise ValueError("报告本人姓名不能超过 80 个字符")
        return normalized

    @field_validator("max_length_chars")
    @classmethod
    def validate_length(cls, value: int) -> int:
        if isinstance(value, bool) or not 200 <= value <= 200_000:
            raise ValueError("报告长度必须在 200 到 200000 字符之间")
        return value

    @field_validator("include_full_review_appendix", mode="before")
    @classmethod
    def disable_visible_source_appendix(cls, value: object) -> bool:
        # Accept the legacy browser field without allowing old saved settings to
        # expose raw messages or source metadata in newly generated reports.
        del value
        return False

    @model_validator(mode="after")
    def validate_supported_report_options(self) -> "ReportOptions":
        if self.grouping not in {"conversation", "date", "owner", "topic"}:
            raise ValueError("grouping 只支持 conversation/date/owner/topic")
        if self.length not in {"brief", "standard", "detailed"}:
            raise ValueError("length 只支持 brief/standard/detailed")
        if self.language.lower() not in {"zh-cn", "zh", "中文"}:
            raise ValueError("当前准确性版本只支持中文报告")
        if self.inference_policy not in {"facts_only", "none"}:
            raise ValueError("当前准确性版本只支持 facts_only")
        if self.evidence_enabled is not True or self.evidence_level != "all_items":
            raise ValueError("准确性模式要求后台保留完整消息证据校验")
        if not self.sections:
            raise ValueError("至少选择一个事实章节")
        if len(self.custom_instruction) > 4000:
            raise ValueError("报告要求不能超过 4000 字符")
        if not self.custom_instruction.strip():
            raise ValueError("必须填写受支持的规则要求")
        return self


class ScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_ids: list[str]
    start: str
    end: str
    timezone: str = DEFAULT_TIMEZONE
    filters: MessageFilters = Field(default_factory=MessageFilters)
    report: ReportOptions = Field(default_factory=ReportOptions)
    explicit_confirm: bool = False
    uncertainty_ack: bool = False

    @field_validator("conversation_ids", mode="before")
    @classmethod
    def normalize_conversations(cls, value: Any) -> list[str]:
        result = _csv_strings(value)
        if not result:
            raise ValueError("至少选择一个聊天会话")
        if len(result) > 200:
            raise ValueError("单次最多选择 200 个会话")
        return result

    @model_validator(mode="after")
    def validate_window(self) -> "ScopeRequest":
        try:
            timezone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"未知时区：{self.timezone}") from exc
        start = _parse_local_time(self.start, timezone, "开始时间")
        end = _parse_local_time(self.end, timezone, "结束时间")
        if start >= end:
            raise ValueError("时间范围必须满足开始时间早于结束时间")
        if end > dt.datetime.now(timezone):
            raise ValueError("结束时间不能晚于当前时间")
        self.start = start.isoformat(timespec="seconds")
        self.end = end.isoformat(timespec="seconds")
        self.timezone = timezone.key
        return self


def _parse_local_time(value: str, timezone: ZoneInfo, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label}格式无效") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone)
    return parsed.astimezone(timezone)


def _adapter_message(readiness: Mapping[str, Any]) -> str:
    if readiness.get("ok") is True:
        return "钉钉数据连接已就绪"
    errors = readiness.get("errors")
    if isinstance(errors, list) and errors:
        first = errors[0]
        if isinstance(first, Mapping):
            return str(first.get("message") or first.get("code") or "连接未就绪")
    return "钉钉尚未登录，或数据读取能力尚未启用"


class ApplicationContext:
    def __init__(
        self,
        *,
        adapter: DwsAdapter | FixtureDwsAdapter | None = None,
        storage: Storage | None = None,
        mode: str | None = None,
        data_root: str | os.PathLike[str] | None = None,
        llm_config_store: LLMConfigStore | None = None,
        llm_engine_factory: Callable[[LLMConfig], LLMReportEngine] | None = None,
    ) -> None:
        configured_mode = (mode or os.environ.get("DINGTALK_REPORT_MODE") or "real").lower()
        if configured_mode not in {"real", "fixture"}:
            raise ValueError("DINGTALK_REPORT_MODE must be real or fixture")
        self.mode = configured_mode
        self.storage = storage or Storage(
            data_root
            or os.environ.get("DINGTALK_REPORT_DATA_DIR")
            or PROJECT_ROOT / "data" / configured_mode
        )
        if adapter is not None:
            self.adapter = adapter
        elif self.mode == "fixture":
            self.adapter = FixtureDwsAdapter(PROJECT_ROOT / "fixtures")
        else:
            self.adapter = DwsAdapter()
        config_path = (
            os.environ.get("DINGTALK_LLM_CONFIG_PATH")
            or self.storage.root.parent / "llm-config.json"
        )
        self.llm_config_store = llm_config_store or LLMConfigStore(config_path)
        self.llm_engine_factory = llm_engine_factory or LLMReportEngine
        self.worker_lock = threading.Lock()
        self._job_threads: set[threading.Thread] = set()
        self._job_threads_lock = threading.Lock()

    def _llm_engine(self, config: LLMConfig) -> LLMReportEngine:
        engine = self.llm_engine_factory(config)
        if not hasattr(engine, "test_configuration") or not hasattr(engine, "generate"):
            raise LLMConfigStoreError("大模型服务实现缺少测试或生成功能")
        return engine

    def llm_state(self) -> dict[str, Any]:
        return self.llm_config_store.public_state()

    def _conversation_visible_in_mode(self, item: Mapping[str, Any]) -> bool:
        source = str(item.get("source") or "").strip().casefold()
        if self.mode == "real":
            return source not in {"fixture", "test", "demo", "mock"}
        return source == "fixture"

    def visible_conversations(
        self,
        *,
        search: str | None = None,
        kind: str | None = None,
        available: bool | None = True,
    ) -> list[dict[str, Any]]:
        return [
            item
            for item in self.storage.list_conversations(
                search=search, kind=kind, available=available
            )
            if self._conversation_visible_in_mode(item)
        ]

    def conversation_directory_state(self) -> dict[str, Any]:
        key = f"conversation_directory_state.v{CONVERSATION_DIRECTORY_STATE_SCHEMA}.{self.mode}"
        stored = self.storage.get_metadata(key, {})
        if not isinstance(stored, Mapping):
            raise StorageError("conversation directory state is corrupted")
        integrity = stored.get("integrity")
        if integrity is not None and not isinstance(integrity, Mapping):
            raise StorageError("conversation directory integrity state is corrupted")
        stored_warnings = stored.get("warnings", [])
        if not isinstance(stored_warnings, list) or any(
            not isinstance(item, str) for item in stored_warnings
        ):
            raise StorageError("conversation directory warnings state is corrupted")
        verified = bool(
            stored.get("schema") == CONVERSATION_DIRECTORY_STATE_SCHEMA
            and stored.get("mode") == self.mode
            and stored.get("verified") is True
        )
        warnings = list(stored_warnings)
        if self.mode == "real" and not verified:
            warnings.insert(0, CONVERSATION_DIRECTORY_REFRESH_REQUIRED)
        return {
            "verified": verified,
            "warnings": list(dict.fromkeys(warnings)),
            "integrity": dict(integrity or {}),
            "refreshed_at": stored.get("refreshed_at"),
        }

    def save_conversation_directory_state(
        self,
        *,
        integrity: Mapping[str, Any],
        warnings: Sequence[str],
        refreshed_at: str,
    ) -> dict[str, Any]:
        verified = bool(
            integrity.get("complete") is True
            and integrity.get("data_complete") is True
            and integrity.get("pagination_known") is not False
            and integrity.get("source_limited") is not True
            and integrity.get("directory_incomplete") is not True
        )
        state = {
            "schema": CONVERSATION_DIRECTORY_STATE_SCHEMA,
            "mode": self.mode,
            "verified": verified,
            "warnings": [str(item) for item in warnings],
            "integrity": dict(integrity),
            "refreshed_at": refreshed_at,
        }
        key = f"conversation_directory_state.v{CONVERSATION_DIRECTORY_STATE_SCHEMA}.{self.mode}"
        self.storage.set_metadata(key, state)
        return self.conversation_directory_state()

    def job_visible_in_mode(self, job: Mapping[str, Any]) -> bool:
        return str((job.get("config") or {}).get("data_mode") or "") == self.mode

    def status(self) -> dict[str, Any]:
        with self.worker_lock:
            readiness = self.adapter.preflight()
        try:
            llm_state = self.llm_state()
        except LLMConfigStoreError as exc:
            llm_state = {
                "config": None,
                "test_status": {
                    "ok": False,
                    "connection_ok": False,
                    "model_ok": False,
                    "structured_output_ok": False,
                    "message": str(exc),
                },
            }
        version_text = str(readiness.get("version") or "")
        match = re.search(r"v\d+\.\d+\.\d+", version_text)
        conversations = self.visible_conversations(available=True)
        return {
            "app_version": __version__,
            "data_mode": self.mode,
            "generation_mode": "manual",
            "scheduled": False,
            "automatic_generation": False,
            "conversation_count": len(conversations),
            "analysis_engine": "large_model",
            "llm": llm_state,
            "dws": {
                "installed": readiness.get("version_ok") is True,
                "ready": readiness.get("ok") is True,
                "version": match.group(0) if match else version_text.strip() or None,
                "version_pinned": readiness.get("version_pinned") is True,
                "auth_command_ok": readiness.get("auth_command_ok") is True,
                "authenticated": readiness.get("authenticated") is True,
                "message": _adapter_message(readiness),
                "errors": readiness.get("errors", []),
            },
        }

    def preview(self, request: ScopeRequest) -> dict[str, Any]:
        directory_state = self.conversation_directory_state()
        by_id = {
            item["id"]: item
            for item in self.visible_conversations(available=None)
        }
        missing = [cid for cid in request.conversation_ids if cid not in by_id]
        unavailable = [
            cid for cid in request.conversation_ids
            if cid in by_id and not by_id[cid]["available"]
        ]
        resolved = [
            by_id.get(cid) or {
                "id": cid,
                "title": cid,
                "kind": "unknown",
                "last_active": None,
                "participants": [],
                "available": True,
                "source": "manual",
                "raw": {},
            }
            for cid in request.conversation_ids
        ]
        warnings: list[str] = []
        blocking: list[str] = []
        if self.mode == "real":
            try:
                with self.worker_lock:
                    readiness = self.adapter.preflight()
            except DwsAdapterError as exc:
                blocking.append(f"钉钉数据连接预检失败：{exc}")
            else:
                if readiness.get("ok") is not True:
                    blocking.append("钉钉数据连接未就绪：" + _adapter_message(readiness))
        if missing:
            directory_description = (
                "已验证的会话目录" if directory_state["verified"] else "当前尚未完整验证的会话目录"
            )
            warnings.append(
                f"以下手动会话 ID 不在{directory_description}中；系统不会把它们视为已验证，"
                "采集时会执行 DWS 定向查询并严格核对所有返回消息的会话 ID；命令失败会停止，"
                "若范围内返回零消息则仍保持未验证状态："
                + "、".join(missing)
            )
        if unavailable:
            blocking.append("以下会话当前不可用：" + "、".join(unavailable))
        warnings.extend(directory_state["warnings"])
        request_end = dt.datetime.fromisoformat(request.end)
        if request_end > dt.datetime.now(request_end.tzinfo) - dt.timedelta(seconds=60):
            warnings.append(
                "结束时间距当前不足 60 秒；钉钉可能仍在索引刚发送的消息。系统会执行连续三轮稳定核对，"
                "但建议将结束时间回退至少一分钟后重新预览。"
            )
        # The deterministic engine remains the trusted normalisation/filtering
        # stage.  Free-form report instructions are intentionally not parsed by
        # its old rule grammar; they are passed to the tested LLM after the
        # immutable message snapshot is built.
        preview_config = self._report_config(request.model_dump(mode="json"))
        preview_config["custom_instruction"] = ""
        preview_config["sections"] = None
        preview_analysis = analyze_messages([], preview_config)
        instruction_interpretation = {
            "applied": [request.report.custom_instruction.strip()],
            "conflicts": [],
            "unsupported": [],
            "mode": "large_model",
        }
        try:
            llm_config = self.llm_config_store.load_config()
            llm_test = self.llm_config_store.test_status(llm_config)
        except LLMConfigStoreError as exc:
            llm_test = {"ok": False, "message": str(exc)}
        if llm_test.get("ok") is not True:
            blocking.append(
                "大模型配置尚未通过接口连接、模型响应、结构化输出和本人识别测试："
                + str(llm_test.get("message") or "请先保存并测试模型")
            )
        return {
            "resolved_conversations": resolved,
            "window": {
                "start": request.start,
                "end": request.end,
                "timezone": request.timezone,
                "interval": "left_closed_right_open",
            },
            "filters": request.filters.model_dump(mode="json"),
            "report": request.report.model_dump(mode="json"),
            "effective_filters": {
                **preview_analysis["config"]["filters"],
                "only_related_to_me": request.filters.only_related_to_me,
            },
            "effective_report": {
                "subject_name": request.report.subject_name,
                "only_subject_items": request.filters.only_related_to_me,
                "title": request.report.title,
                "grouping": request.report.grouping,
                "max_length_chars": preview_analysis["config"]["max_length_chars"],
                "include_full_review_appendix": False,
                "source_details_visible": False,
                "sections": request.report.sections,
                "selected_explicit_categories": [],
                "documents": ["completed_items", "weekly_report"],
            },
            "instruction_interpretation": instruction_interpretation,
            "llm_test": llm_test,
            "warnings": warnings,
            "blocking_errors": blocking,
            "requires_uncertainty_ack": bool(warnings),
            "can_generate": not blocking,
            "previewed_at": _now_utc(),
        }

    def start_job(self, request: ScopeRequest) -> dict[str, Any]:
        if request.explicit_confirm is not True:
            raise HTTPException(409, "必须明确确认后才能生成报告")
        try:
            llm_config = self.llm_config_store.require_tested_config()
        except LLMConfigStoreError as exc:
            raise HTTPException(409, str(exc)) from exc
        preview = self.preview(request)
        if not preview["can_generate"]:
            raise HTTPException(422, {"message": "数据范围不可生成", "preview": preview})
        if preview["requires_uncertainty_ack"] and request.uncertainty_ack is not True:
            raise HTTPException(409, {"message": "请先确认已阅读完整性警告", "preview": preview})
        manifest = request.model_dump(mode="json")
        manifest.update(
            {
                "data_mode": self.mode,
                "generation_mode": "manual",
                "analysis_engine": "large_model",
                "scheduled": False,
                "confirmed_at": _now_utc(),
                "preview": preview,
                "llm": {
                    key: value
                    for key, value in llm_config.public_dict().items()
                    if key not in {"api_key_masked"}
                },
            }
        )
        job = self.storage.create_job(manifest, status="QUEUED")
        self.storage.write_job_artifact(job["id"], "manifest", manifest)
        thread = threading.Thread(
            target=self._run_tracked_job,
            args=(job["id"], manifest, llm_config),
            name=f"dingtalk-report-{job['id'][:8]}",
            daemon=True,
        )
        with self._job_threads_lock:
            self._job_threads.add(thread)
        thread.start()
        return self.public_job(self.storage.get_job(job["id"]))

    def _run_tracked_job(
        self,
        job_id: str,
        manifest: Mapping[str, Any],
        llm_config: LLMConfig,
    ) -> None:
        try:
            self._run_job(job_id, manifest, llm_config)
        finally:
            with self._job_threads_lock:
                self._job_threads.discard(threading.current_thread())

    def wait_for_workers(self, timeout_seconds: float = 30.0) -> bool:
        """Wait for report workers to release files during an orderly shutdown."""

        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            with self._job_threads_lock:
                active = [thread for thread in self._job_threads if thread.is_alive()]
            if not active:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            active[0].join(timeout=min(remaining, 0.25))

    def _run_job(
        self,
        job_id: str,
        manifest: Mapping[str, Any],
        llm_config: LLMConfig,
    ) -> None:
        try:
            self.storage.update_job(
                job_id,
                status="RUNNING",
                started_at=_now_utc(),
                progress={"stage": "collecting", "percent": 15, "message": "正在读取所选会话"},
            )
            # DWS uses one local OAuth/profile state.  Serializing collection
            # prevents concurrent jobs from racing token refreshes or mixing
            # command output while still allowing completed snapshots to be
            # analyzed independently.
            with self.worker_lock:
                collection = self.adapter.collect_selected(
                    manifest["conversation_ids"],
                    manifest["start"],
                    manifest["end"],
                    passes=3,
                    conversation_metadata=(manifest.get("preview") or {}).get(
                        "resolved_conversations"
                    ),
                )
            self.storage.update_job(
                job_id,
                progress={"stage": "cleaning", "percent": 52, "message": "正在核对去重、时间和来源"},
            )
            self.storage.write_job_artifact(
                job_id, "raw_pass", collection.get("integrity", {}).get("passes", []),
                name="collection_ledger"
            )
            self.storage.write_job_artifact(
                job_id,
                "snapshot",
                {
                    "window": {
                        "start": manifest["start"],
                        "end": manifest["end"],
                        "timezone": manifest["timezone"],
                    },
                    "selected_conversation_ids": manifest["conversation_ids"],
                    "messages": collection["messages"],
                },
            )
            self.storage.update_job(
                job_id,
                progress={
                    "stage": "analyzing",
                    "percent": 68,
                    "message": "正在筛选消息并调用已测试的大模型",
                },
            )
            report_config = self._report_config(manifest, collection=collection, job_id=job_id)
            filtered_messages, preprocessing = self._prepare_messages_for_llm(
                collection["messages"], report_config
            )
            report_options = dict(manifest.get("report") or {})
            report_subject_name = str(report_options.get("subject_name") or "").strip()
            only_subject_items = bool(
                (manifest.get("filters") or {}).get("only_related_to_me")
            )
            subject_scope_instruction = (
                "AI 摘要只汇总报告本人亲自完成、执行、推进的事项，以及明确分配给报告本人但尚未闭环的事项；其他人的工作不得算作报告本人成果。"
                if only_subject_items
                else "保留团队事项，但必须明确区分报告本人和其他参与人，不得把他人的工作算作报告本人成果。"
            )
            per_job_instructions = (
                "本次报告配置：\n"
                f"- 报告本人：{report_subject_name}\n"
                f"- 本人范围：{subject_scope_instruction}\n"
                f"- 标题：{report_options.get('title') or '钉钉聊天工作报告'}\n"
                f"- 归并偏好：{report_options.get('grouping') or 'conversation'}\n"
                f"- 选择的关注章节：{'、'.join(report_options.get('sections') or []) or '全部'}\n"
                f"- 本次补充要求：{report_options.get('custom_instruction') or '只写有证据的事实'}\n"
                "- 固定展示要求：报告只写整理后的业务事项，不显示消息出处、引用原文、消息 ID、会话或发送人。"
            )
            runtime_config = llm_config.model_copy(
                update={"custom_instructions": per_job_instructions}
            )
            def update_model_progress(event: Mapping[str, Any]) -> None:
                stage = str(event.get("stage") or "map")
                completed = int(event.get("completed") or 0)
                total = max(1, int(event.get("total") or 1))
                if stage == "reduce":
                    percent = 83 + round(4 * completed / total)
                    message = f"正在归并模型候选结果（{completed}/{total}）"
                else:
                    percent = 68 + round(14 * completed / total)
                    message = f"正在分批分析消息（{completed}/{total}）"
                self.storage.update_job(
                    job_id,
                    progress={
                        "stage": "analyzing",
                        "percent": min(87, percent),
                        "message": message,
                        "model_stage": stage,
                        "model_completed": completed,
                        "model_total": total,
                    },
                )
            model_result = asyncio.run(
                self._llm_engine(runtime_config).generate(
                    filtered_messages,
                    report_subject_name=report_subject_name,
                    only_subject_items=only_subject_items,
                    progress_callback=update_model_progress,
                    checkpoint_dir=self.storage.root / "model-cache",
                )
            )
            documents = dict(model_result.get("documents") or {})
            completed_markdown = str(documents.get("completed_items_markdown") or "")
            weekly_markdown = str(documents.get("weekly_report_markdown") or "")
            if not completed_markdown.strip() or not weekly_markdown.strip():
                raise LLMEngineError("大模型没有生成完整的两份报告", code="LLM_DOCUMENTS_MISSING")
            maximum_length = int(report_options.get("max_length_chars") or 8000)
            if len(completed_markdown) > maximum_length or len(weekly_markdown) > maximum_length:
                raise LLMEngineError(
                    "大模型摘要超过当前长度上限，请在面板选择更长摘要后重试",
                    code="LLM_DOCUMENT_TOO_LONG",
                )
            manual_review = {
                "enabled": False,
                "mode": "disabled_by_report_display_policy",
                "source_scope": "after_user_configured_filters",
                "ai_selection_applied": False,
                "coverage_validated": None,
                "rendered_message_count": 0,
            }
            manual_review_summary = {
                key: value
                for key, value in manual_review.items()
                if key not in {"message_identities", "conversation_counts"}
            }
            self.storage.update_job(
                job_id,
                progress={
                    "stage": "reporting",
                    "percent": 88,
                    "message": "正在发布本周已完成事项、周报和完整性记录",
                },
            )
            analysis_artifact = self.storage.write_job_artifact(
                job_id,
                "analysis",
                {
                    "schema_version": model_result.get("schema_version"),
                    "engine": model_result.get("engine"),
                    "model": model_result.get("model"),
                    "report_subject": {
                        "name": report_subject_name,
                        "only_subject_items": only_subject_items,
                    },
                    "preprocessing": preprocessing,
                    "structured": model_result.get("structured"),
                    "manual_review_appendix": manual_review,
                },
                name="llm_analysis",
                media_type="application/json",
            )
            completed_artifact = self.storage.write_job_artifact(
                job_id,
                "report",
                completed_markdown,
                name="completed_items.md",
                media_type="text/markdown",
            )
            weekly_artifact = self.storage.write_job_artifact(
                job_id,
                "report",
                weekly_markdown,
                name="weekly_report.md",
                media_type="text/markdown",
            )
            structured = dict(model_result.get("structured") or {})
            weekly = dict(structured.get("weekly_report") or {})
            output_item_count = (
                len(structured.get("completed_items") or [])
                + len(weekly.get("this_week") or [])
                + len(weekly.get("next_week") or [])
            )
            integrity = {
                "execution_success": True,
                "data_complete": collection.get("data_complete") is True,
                "status": collection.get("status"),
                "sealed": collection.get("sealed") is True,
                "collection": collection.get("integrity", {}),
                "analysis": {
                    "engine": model_result.get("engine"),
                    "preprocessing": preprocessing.get("summary", {}),
                    "model": model_result.get("model", {}),
                    "output_item_count": output_item_count,
                    "evidence_scope_validated": True,
                    "manual_review_appendix": manual_review_summary,
                },
                "warnings": list(
                    dict.fromkeys(
                        [
                            *collection.get("warnings", []),
                            *preprocessing.get("warnings", []),
                        ]
                    )
                ),
            }
            self.storage.write_job_artifact(job_id, "integrity", integrity)
            counts = {
                "原始快照消息": len(collection["messages"]),
                "进入分析": preprocessing["summary"].get("included_message_count", 0),
                "被筛选排除": preprocessing["summary"].get("excluded_message_count", 0),
                "证据结论": output_item_count,
                "模型请求": (model_result.get("model") or {}).get("request_count", 0),
                "选定会话": len(manifest["conversation_ids"]),
            }
            result = {
                "execution_success": True,
                "data_complete": integrity["data_complete"],
                "snapshot_status": integrity["status"],
                "counts": counts,
                "warnings": integrity["warnings"],
                "integrity": integrity,
                "documents": ["completed_items", "weekly_report"],
                "report_subject": {
                    "name": report_subject_name,
                    "only_subject_items": only_subject_items,
                },
                "document_urls": {
                    "completed_items": f"/api/jobs/{job_id}/report?document=completed_items",
                    "weekly_report": f"/api/jobs/{job_id}/report?document=weekly_report",
                },
                "report_url": f"/api/jobs/{job_id}/report?document=completed_items",
                "download_url": f"/api/jobs/{job_id}/download?document=completed_items",
                "report_artifact": completed_artifact,
                "report_artifacts": {
                    "completed_items": completed_artifact,
                    "weekly_report": weekly_artifact,
                },
                "analysis_artifact": analysis_artifact,
            }
            self.storage.update_job(
                job_id,
                status="COMPLETED",
                progress={
                    "stage": "completed",
                    "percent": 100,
                    "message": "本周已完成事项和周报已生成",
                },
                result=result,
                finished_at=_now_utc(),
            )
        except (
            DwsAdapterError,
            LLMConfigStoreError,
            LLMEngineError,
            ReportEngineError,
            StorageError,
            ValueError,
        ) as exc:
            error: dict[str, Any] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            if isinstance(exc, DwsAdapterError):
                error.update(exc.as_dict())
            elif isinstance(exc, LLMEngineError):
                error["code"] = exc.code
            self._fail_job(job_id, error)
        except Exception as exc:  # pragma: no cover - defensive job boundary
            self._fail_job(
                job_id,
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "trace": traceback.format_exc(limit=8),
                },
            )

    @staticmethod
    def _prepare_messages_for_llm(
        messages: Sequence[Mapping[str, Any]],
        report_config: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply the existing deterministic filters before any model request."""

        preprocessing_config = dict(report_config)
        preprocessing_config["sections"] = None
        preprocessing_config["custom_instruction"] = ""
        preprocessing_config["grouping"] = "conversation"
        analysis = analyze_messages(messages, preprocessing_config)
        included: set[tuple[str, str]] = set()
        for finding in analysis.get("facts") or []:
            if not isinstance(finding, Mapping):
                continue
            for evidence in finding.get("evidence") or []:
                if not isinstance(evidence, Mapping):
                    continue
                conversation_id = str(evidence.get("conversation_id") or "").strip()
                message_id = str(evidence.get("message_id") or "").strip()
                if conversation_id and message_id:
                    included.add((conversation_id, message_id))

        filtered: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw in messages:
            conversation_id = str(raw.get("conversation_id") or "").strip()
            message_id = str(raw.get("id") or raw.get("message_id") or "").strip()
            identity = (conversation_id, message_id)
            if identity not in included or identity in seen:
                continue
            seen.add(identity)
            filtered.append(dict(raw))

        expected = int(analysis.get("summary", {}).get("included_message_count") or 0)
        if len(filtered) != expected:
            raise ReportEngineError(
                "本地筛选结果无法与规范化证据身份一一对应，已停止调用大模型"
            )
        return filtered, analysis

    def _fail_job(self, job_id: str, error: Mapping[str, Any]) -> None:
        try:
            self.storage.update_job(
                job_id,
                status="FAILED",
                progress={"stage": "failed", "percent": 100, "message": "任务失败，未生成可信报告"},
                error=error,
                finished_at=_now_utc(),
            )
        except StorageError:
            pass

    @staticmethod
    def _report_config(
        manifest: Mapping[str, Any],
        *,
        collection: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        filters = dict(manifest.get("filters") or {})
        report = dict(manifest.get("report") or {})
        config = {
            "title": report.get("title"),
            "timezone": manifest.get("timezone", DEFAULT_TIMEZONE),
            "grouping": report.get("grouping", "conversation"),
            "sections": report.get("sections") or None,
            "max_length_chars": report.get("max_length_chars", 8000),
            "include_full_review_appendix": False,
            "custom_instruction": report.get("custom_instruction", ""),
            "allow_inferences": report.get("inference_policy") not in {"facts_only", "none"},
            "filters": {
                "include_conversation_ids": list(manifest.get("conversation_ids") or []),
                "include_senders": filters.get("include_senders", []),
                "exclude_senders": filters.get("exclude_senders", []),
                "include_keywords": filters.get("include_keywords", []),
                "exclude_keywords": filters.get("exclude_keywords", []),
                "message_types": filters.get("message_types", []),
                "include_system_messages": filters.get("include_system_messages") is True,
                "include_bot_messages": filters.get("include_bot_messages") is True,
                "include_recalled": filters.get("include_recalled") is not False,
                "exclude_casual": filters.get("casual_filter") in {"conservative", "aggressive"},
                "start": manifest.get("start"),
                "end": manifest.get("end"),
            },
        }
        if collection is not None:
            collection_integrity = dict(collection.get("integrity") or {})
            config["audit"] = {
                "job_id": job_id,
                "execution_success": collection.get("execution_success") is True,
                "data_complete": collection.get("data_complete") is True,
                "status": collection.get("status"),
                "sealed": collection.get("sealed") is True,
                "warnings": list(collection.get("warnings") or []),
                "dataset_digest": collection_integrity.get("dataset_digest"),
                "source_difference": collection_integrity.get("source_difference") or {},
                "coverage": collection_integrity.get("coverage") or [],
                "required_stable_passes": collection_integrity.get("required_stable_passes"),
                "passes_executed": collection_integrity.get("passes_executed"),
            }
        return config

    @staticmethod
    def public_job(job: Mapping[str, Any]) -> dict[str, Any]:
        progress = dict(job.get("progress") or {})
        result = dict(job.get("result") or {})
        return {
            "id": job["id"],
            "status": job["status"],
            "stage": progress.get("stage", job["status"].lower()),
            "progress": progress.get("percent", 0),
            "progress_detail": progress,
            "error": job.get("error"),
            "counts": result.get("counts", {}),
            "warnings": result.get("warnings", []),
            "integrity": result.get("integrity", {}),
            "documents": result.get("documents", []),
            "document_urls": result.get("document_urls", {}),
            "report_url": result.get("report_url"),
            "download_url": result.get("download_url"),
            "data_complete": result.get("data_complete"),
            "snapshot_status": result.get("snapshot_status"),
            "config": job.get("config", {}),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "finished_at": job.get("finished_at"),
        }


def create_app(
    *,
    adapter: DwsAdapter | FixtureDwsAdapter | None = None,
    storage: Storage | None = None,
    mode: str | None = None,
    data_root: str | os.PathLike[str] | None = None,
    llm_config_store: LLMConfigStore | None = None,
    llm_engine_factory: Callable[[LLMConfig], LLMReportEngine] | None = None,
) -> FastAPI:
    application = FastAPI(
        title="钉钉会话报告台",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )
    context = ApplicationContext(
        adapter=adapter,
        storage=storage,
        mode=mode,
        data_root=data_root,
        llm_config_store=llm_config_store,
        llm_engine_factory=llm_engine_factory,
    )
    application.state.context = context
    application.router.add_event_handler("shutdown", context.wait_for_workers)
    application.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @application.get("/healthz")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "scheduled": False}

    @application.get("/api/status")
    def status() -> dict[str, Any]:
        return context.status()

    @application.get("/api/llm/config")
    def get_llm_config() -> dict[str, Any]:
        try:
            return context.llm_state()
        except LLMConfigStoreError as exc:
            raise HTTPException(500, str(exc)) from exc

    @application.put("/api/llm/config")
    def put_llm_config(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            context.llm_config_store.save_config(payload)
            return context.llm_state()
        except LLMConfigStoreError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.post("/api/llm/test")
    async def test_llm_config() -> dict[str, Any]:
        try:
            config = context.llm_config_store.load_config()
            result = await context._llm_engine(config).test_configuration()
            return context.llm_config_store.record_test(result, config)
        except LLMEngineError as exc:
            connection_ok = exc.code not in {"LLM_CONNECTION_FAILED"}
            model_ok = connection_ok and exc.code not in {"LLM_HTTP_ERROR"}
            try:
                config
            except UnboundLocalError:
                raise HTTPException(422, str(exc)) from exc
            return context.llm_config_store.record_failed_test(
                config=config,
                message=str(exc),
                code=exc.code,
                connection_ok=connection_ok,
                model_ok=model_ok,
                structured_output_ok=False,
            )
        except LLMConfigStoreError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.post("/api/conversations/refresh")
    def refresh_conversations() -> dict[str, Any]:
        try:
            with context.worker_lock:
                refreshed = context.adapter.refresh_conversations()
            integrity = refreshed.get("integrity")
            if not isinstance(integrity, Mapping):
                raise DwsAdapterError(
                    "conversation refresh has no integrity ledger",
                    code="INVALID_OUTPUT_SHAPE",
                )
            warnings = refreshed.get("warnings", [])
            if not isinstance(warnings, list) or any(
                not isinstance(item, str) for item in warnings
            ):
                raise DwsAdapterError(
                    "conversation refresh has invalid warnings",
                    code="INVALID_OUTPUT_SHAPE",
                )
            directory_complete = bool(
                integrity.get("complete") is True
                and integrity.get("data_complete") is True
                and integrity.get("source_limited") is not True
                and integrity.get("directory_incomplete") is not True
            )
            persistence = context.storage.upsert_conversations(
                refreshed["items"], complete=directory_complete
            )
            refreshed_at = _now_utc()
            directory_state = context.save_conversation_directory_state(
                integrity=integrity,
                warnings=warnings,
                refreshed_at=refreshed_at,
            )
            return {
                **refreshed,
                "persistence": persistence,
                "directory": directory_state,
                "refreshed_at": refreshed_at,
            }
        except DwsAdapterError as exc:
            raise HTTPException(409, exc.as_dict()) from exc
        except StorageError as exc:
            raise HTTPException(500, str(exc)) from exc

    @application.get("/api/conversations")
    def conversations(
        query: str | None = Query(default=None, max_length=200),
        kind: str | None = Query(default=None, pattern="^(direct|group|unknown)$"),
    ) -> dict[str, Any]:
        try:
            items = context.visible_conversations(
                search=query, kind=kind, available=True
            )
            directory_state = context.conversation_directory_state()
            return {
                "items": items,
                "total": len(items),
                "warnings": directory_state["warnings"],
                "directory": directory_state,
            }
        except StorageError as exc:
            raise HTTPException(400, str(exc)) from exc

    @application.post("/api/preview")
    def preview(request: ScopeRequest) -> dict[str, Any]:
        return context.preview(request)

    @application.post("/api/jobs", status_code=202)
    def create_job(request: ScopeRequest) -> dict[str, Any]:
        return context.start_job(request)

    @application.get("/api/jobs")
    def jobs(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        try:
            items = [
                context.public_job(job)
                for job in context.storage.list_jobs(limit=limit)
                if context.job_visible_in_mode(job)
            ]
            return {"items": items, "total": len(items)}
        except StorageError as exc:
            raise HTTPException(500, str(exc)) from exc

    @application.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        try:
            job_row = context.storage.get_job(job_id)
            if not context.job_visible_in_mode(job_row):
                raise HTTPException(404, "任务不存在于当前数据模式")
            return context.public_job(job_row)
        except StorageError as exc:
            raise HTTPException(404, str(exc)) from exc

    def report_document_path(job_id: str, document: str) -> pathlib.Path:
        names = {
            "completed_items": "completed_items.md",
            "weekly_report": "weekly_report.md",
        }
        path = context.storage.artifact_path(job_id, "report", names[document])
        if not path.is_file() and document == "weekly_report":
            legacy = context.storage.artifact_path(job_id, "report", "report.md")
            if legacy.is_file():
                return legacy
        return path

    @application.get("/api/jobs/{job_id}/report")
    def report(
        job_id: str,
        document: str = Query(
            default="completed_items", pattern="^(completed_items|weekly_report)$"
        ),
    ) -> JSONResponse:
        try:
            job_row = context.storage.get_job(job_id)
            if not context.job_visible_in_mode(job_row):
                raise HTTPException(404, "任务不存在于当前数据模式")
            if job_row["status"] != "COMPLETED":
                raise HTTPException(409, "报告尚未生成")
            path = report_document_path(job_id, document)
            if not path.is_file():
                raise HTTPException(404, "所选报告文件不存在")
            return JSONResponse(
                {"document": document, "markdown": path.read_text(encoding="utf-8")}
            )
        except StorageError as exc:
            raise HTTPException(404, str(exc)) from exc

    @application.get("/api/jobs/{job_id}/download")
    def download(
        job_id: str,
        document: str = Query(
            default="completed_items", pattern="^(completed_items|weekly_report)$"
        ),
    ) -> FileResponse:
        try:
            job_row = context.storage.get_job(job_id)
            if not context.job_visible_in_mode(job_row):
                raise HTTPException(404, "任务不存在于当前数据模式")
            if job_row["status"] != "COMPLETED":
                raise HTTPException(409, "报告尚未生成")
            path = report_document_path(job_id, document)
            if not path.is_file():
                raise HTTPException(404, "所选报告文件不存在")
            suffix = "weekly-report" if document == "weekly_report" else "completed-items"
            return FileResponse(
                path,
                media_type="text/markdown",
                filename=f"dingtalk-{suffix}-{job_id[:8]}.md",
            )
        except StorageError as exc:
            raise HTTPException(404, str(exc)) from exc

    return application


app = create_app()
