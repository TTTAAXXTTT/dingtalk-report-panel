from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from app.llm_config_store import LLMConfigStore, LLMConfigStoreError


class LLMConfigStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temporary.name) / "llm-config.json"
        self.store = LLMConfigStore(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def values(**overrides):
        result = {
            "protocol": "openai_responses",
            "base_url": "https://model.test/v1",
            "model": "test-model",
            "api_key": "secret-one",
            "timeout_seconds": 120,
            "temperature": 0.2,
            "max_output_tokens": 12000,
            "analysis_instruction": "只根据选中消息整理",
            "weekly_report_template": "本周工作：\n\n1，事项：内容\n\n下周工作计划：\n\n1，事项：计划",
        }
        result.update(overrides)
        return result

    def test_public_state_never_returns_key_and_blank_key_preserves_it(self) -> None:
        first = self.store.save_config(
            self.values(
                api_key_header="X-API-Key",
                api_key_prefix="",
                max_input_chars=88000,
                max_messages_per_batch=7,
            )
        )
        original_fingerprint = first.fingerprint()
        public = self.store.public_state()
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn("secret-one", serialized)
        self.assertNotIn("api_key_masked", public["config"])
        self.assertTrue(public["config"]["api_key_configured"])

        values = self.values(api_key="", temperature=0.4)
        second = self.store.save_config(values)
        self.assertNotEqual(original_fingerprint, second.fingerprint())
        self.assertEqual("secret-one", second.storage_dict()["api_key"])
        self.assertEqual("X-API-Key", second.api_key_header)
        self.assertEqual("", second.api_key_prefix)
        self.assertEqual(88000, second.max_input_chars)
        self.assertEqual(7, second.max_messages_per_batch)

    def test_successful_test_is_bound_to_exact_fingerprint(self) -> None:
        config = self.store.save_config(self.values())
        self.store.record_test(
            {
                "ok": True,
                "connection_ok": True,
                "model_ok": True,
                "structured_output_ok": True,
                "message": "通过",
            },
            config,
        )
        self.assertTrue(self.store.require_tested_config().fingerprint())
        self.assertTrue(self.store.test_status()["ok"])

        self.store.save_config(self.values(model="changed-model", api_key=""))
        self.assertFalse(self.store.test_status()["ok"])
        with self.assertRaisesRegex(LLMConfigStoreError, "尚未通过"):
            self.store.require_tested_config()

    def test_null_temperature_round_trips_as_automatic_mode(self) -> None:
        config = self.store.save_config(self.values(temperature=None))

        self.assertIsNone(config.temperature)
        self.assertIsNone(self.store.public_state()["config"]["temperature"])
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsNone(stored["config"]["temperature"])

    def test_failed_test_is_persisted_but_never_allows_generation(self) -> None:
        config = self.store.save_config(self.values())
        status = self.store.record_failed_test(
            config=config,
            message="HTTP 401",
            code="LLM_HTTP_ERROR",
            connection_ok=True,
            model_ok=False,
        )
        self.assertFalse(status["ok"])
        self.assertTrue(self.store.test_status()["connection_ok"])
        self.assertFalse(self.store.test_status()["model_ok"])
        with self.assertRaises(LLMConfigStoreError):
            self.store.require_tested_config()

    def test_corrupt_file_fails_closed(self) -> None:
        self.path.write_text("not-json", encoding="utf-8")
        with self.assertRaisesRegex(LLMConfigStoreError, "损坏"):
            self.store.public_state()


if __name__ == "__main__":
    unittest.main()
