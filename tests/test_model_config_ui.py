from __future__ import annotations

import pathlib
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class ModelConfigurationUiTests(unittest.TestCase):
    def test_blank_temperature_is_presented_and_serialised_as_automatic_mode(self) -> None:
        html = (PROJECT_ROOT / "app" / "static" / "index.html").read_text(
            encoding="utf-8"
        )
        javascript = (PROJECT_ROOT / "app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('placeholder="自动（不发送）"', html)
        self.assertIn('temperatureText === "" ? null : Number(temperatureText)', javascript)
        self.assertIn(
            'el.llmTemperature.value = config.temperature == null ? "" : String(config.temperature)',
            javascript,
        )
        self.assertNotIn(
            "temperature: Number(el.llmTemperature.value)",
            javascript,
        )
        self.assertIn('id="llmMaxInputChars"', html)
        self.assertIn("max_input_chars: Number(el.llmMaxInputChars.value)", javascript)
        self.assertIn('id="llmMaxMessagesPerBatch"', html)
        self.assertIn(
            "max_messages_per_batch: Number(el.llmMaxMessagesPerBatch.value)",
            javascript,
        )

    def test_source_details_are_hidden_while_internal_validation_remains_enabled(self) -> None:
        html = (PROJECT_ROOT / "app" / "static" / "index.html").read_text(
            encoding="utf-8"
        )
        javascript = (PROJECT_ROOT / "app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="includeFullReviewAppendix"', html)
        self.assertIn('id="includeFullReviewAppendix" type="checkbox" hidden', html)
        self.assertNotIn("全量人工核对清单（默认开启）", html)
        self.assertNotIn("所有报告事项固定附消息出处", html)
        self.assertIn("只展示整理结果", html)
        self.assertIn(
            "include_full_review_appendix: false",
            javascript,
        )
        self.assertIn(
            "setChecked(el.includeFullReviewAppendix, false)",
            javascript,
        )
        self.assertIn('<option value="off" selected>', html)
        self.assertIn('value="audio" checked', html)
        self.assertIn('value="video" checked', html)

    def test_report_subject_is_editable_and_subject_only_mode_defaults_on(self) -> None:
        html = (PROJECT_ROOT / "app" / "static" / "index.html").read_text(
            encoding="utf-8"
        )
        javascript = (PROJECT_ROOT / "app" / "static" / "app.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('id="reportSubjectName"', html)
        self.assertIn('value="张三"', html)
        self.assertIn(
            'id="onlyRelatedToMe" type="checkbox" role="switch" checked',
            html,
        )
        self.assertNotIn('id="onlyRelatedToMe" type="checkbox" role="switch" disabled', html)
        self.assertIn("subject_name: el.reportSubjectName.value.trim()", javascript)
        self.assertIn("only_related_to_me: el.onlyRelatedToMe.checked", javascript)
        self.assertIn(
            "setChecked(el.onlyRelatedToMe, filters.only_related_to_me, true)",
            javascript,
        )
        self.assertNotIn("only_related_to_me: false", javascript)


if __name__ == "__main__":
    unittest.main()
