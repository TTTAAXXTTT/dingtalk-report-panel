"""Atomic local persistence for editable LLM settings and test state."""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import threading
import uuid
from collections.abc import Mapping
from typing import Any

from .llm_engine import ENGINE_VERSION, LLMConfig


class LLMConfigStoreError(RuntimeError):
    """Raised when the local model configuration cannot be read or written."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


class LLMConfigStore:
    """Persist one model configuration without ever returning its credential.

    The file is intentionally separate from the job database.  A single model
    connection can therefore be used in both real and fixture modes, while job
    snapshots continue to remain mode-isolated.  Tests can inject a temporary
    path.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = pathlib.Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _environment_api_key() -> str:
        return (
            os.environ.get("DINGTALK_LLM_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LLMConfigStoreError("本地大模型配置文件无法读取或格式已损坏") from exc
        if not isinstance(payload, dict):
            raise LLMConfigStoreError("本地大模型配置必须是 JSON 对象")
        return payload

    def _write_payload(self, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8")
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise LLMConfigStoreError("本地大模型配置无法保存") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _effective_config_from_payload(self, payload: Mapping[str, Any]) -> LLMConfig:
        raw_config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
        values = dict(raw_config)
        if not str(values.get("api_key") or "").strip():
            environment_key = self._environment_api_key()
            if environment_key:
                values["api_key"] = environment_key
        try:
            return LLMConfig.model_validate(values)
        except Exception as exc:
            raise LLMConfigStoreError("本地大模型配置无效，请重新保存") from exc

    def load_config(self) -> LLMConfig:
        with self._lock:
            return self._effective_config_from_payload(self._read_payload())

    def save_config(self, values: Mapping[str, Any]) -> LLMConfig:
        """Validate and save settings; an omitted/blank key preserves the old key."""

        if not isinstance(values, Mapping):
            raise LLMConfigStoreError("大模型配置必须是对象")
        with self._lock:
            payload = self._read_payload()
            previous_raw = (
                dict(payload.get("config"))
                if isinstance(payload.get("config"), Mapping)
                else {}
            )
            submitted = dict(values)
            # PUT behaves as a safe partial update for advanced provider fields
            # that are not shown in the business UI (for example X-API-Key or
            # max_input_chars).  Visible fields still replace their old value.
            incoming = {**previous_raw, **submitted}
            incoming_key = submitted.get("api_key")
            if incoming_key is None or not str(incoming_key).strip():
                if str(previous_raw.get("api_key") or "").strip():
                    incoming["api_key"] = previous_raw["api_key"]
                else:
                    incoming.pop("api_key", None)

            try:
                config = LLMConfig.model_validate(incoming)
            except Exception as exc:
                raise LLMConfigStoreError("大模型配置校验失败") from exc

            previous_fingerprint = None
            try:
                if previous_raw:
                    previous_fingerprint = self._effective_config_from_payload(payload).fingerprint()
            except LLMConfigStoreError:
                previous_fingerprint = None

            stored = {
                "schema_version": self.SCHEMA_VERSION,
                "updated_at": _utc_now(),
                "config": config.storage_dict(),
                "last_test": payload.get("last_test")
                if previous_fingerprint == config.fingerprint()
                else None,
            }
            self._write_payload(stored)
            return self._effective_config_from_payload(stored)

    def record_test(self, result: Mapping[str, Any], config: LLMConfig) -> dict[str, Any]:
        with self._lock:
            payload = self._read_payload()
            current = self._effective_config_from_payload(payload)
            if current.fingerprint() != config.fingerprint():
                raise LLMConfigStoreError("测试期间模型配置已变化，请重新测试")
            status = dict(result)
            status["config_fingerprint"] = config.fingerprint()
            status.setdefault("engine_version", ENGINE_VERSION)
            status.setdefault("tested_at", _utc_now())
            status["ok"] = status.get("ok") is True
            payload["last_test"] = status
            payload["updated_at"] = _utc_now()
            payload.setdefault("schema_version", self.SCHEMA_VERSION)
            payload.setdefault("config", config.storage_dict())
            self._write_payload(payload)
            return status

    def record_failed_test(
        self,
        *,
        config: LLMConfig,
        message: str,
        code: str,
        connection_ok: bool = False,
        model_ok: bool = False,
        structured_output_ok: bool = False,
    ) -> dict[str, Any]:
        status = {
            "ok": False,
            "connection_ok": connection_ok,
            "model_ok": model_ok,
            "structured_output_ok": structured_output_ok,
            "message": message,
            "code": code,
            "tested_at": _utc_now(),
        }
        return self.record_test(status, config)

    def test_status(self, config: LLMConfig | None = None) -> dict[str, Any]:
        with self._lock:
            payload = self._read_payload()
            effective = config or self._effective_config_from_payload(payload)
            raw = payload.get("last_test")
            if not isinstance(raw, Mapping):
                return {
                    "ok": False,
                    "connection_ok": None,
                    "model_ok": None,
                    "structured_output_ok": None,
                    "message": "模型配置尚未完成测试",
                    "tested_at": None,
                }
            status = dict(raw)
            matches = status.get("config_fingerprint") == effective.fingerprint()
            if not matches:
                status.update(
                    {
                        "ok": False,
                        "connection_ok": None,
                        "model_ok": None,
                        "structured_output_ok": None,
                        "message": "模型配置已变化，请重新测试",
                    }
                )
            elif status.get("engine_version") != ENGINE_VERSION:
                status.update(
                    {
                        "ok": False,
                        "connection_ok": None,
                        "model_ok": None,
                        "structured_output_ok": None,
                        "message": "分析引擎已升级，请重新测试模型配置",
                    }
                )
            return status

    def require_tested_config(self) -> LLMConfig:
        config = self.load_config()
        status = self.test_status(config)
        if status.get("ok") is not True:
            raise LLMConfigStoreError("大模型配置尚未通过完整测试")
        if status.get("config_fingerprint") != config.fingerprint():
            raise LLMConfigStoreError("大模型配置与最近一次测试不一致，请重新测试")
        if status.get("engine_version") != ENGINE_VERSION:
            raise LLMConfigStoreError("分析引擎已升级，请重新测试模型配置")
        return config

    def public_state(self) -> dict[str, Any]:
        config = self.load_config()
        public_config = config.public_dict()
        public_config.pop("api_key_masked", None)
        return {
            "config": public_config,
            "test_status": self.test_status(config),
        }


__all__ = ["LLMConfigStore", "LLMConfigStoreError"]
