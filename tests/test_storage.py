from __future__ import annotations

import contextlib
import pathlib
import sqlite3
import tempfile
import threading
import unittest

from app.storage import Storage, StorageError


def conversations() -> list[dict]:
    return [
        {
            "id": "cid-group",
            "title": "项目群",
            "kind": "group",
            "last_active": "2026-08-23T10:00:00+08:00",
            "participants": [{"id": "u-a", "name": "张三"}],
            "available": True,
            "source": "fixture",
            "raw": {"openConversationId": "cid-group"},
        },
        {
            "id": "cid-direct",
            "title": "李四",
            "kind": "direct",
            "last_active": "2026-08-22T10:00:00+08:00",
            "participants": [{"id": "u-b", "name": "李四"}],
            "available": True,
            "source": "fixture",
            "raw": {"openConversationId": "cid-direct"},
        },
    ]


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.storage = Storage(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_database_uses_wal_and_directory_search_uses_participants(self) -> None:
        with contextlib.closing(sqlite3.connect(self.storage.db_path)) as connection:
            self.assertEqual("wal", connection.execute("PRAGMA journal_mode").fetchone()[0])
        result = self.storage.upsert_conversations(conversations())
        self.assertEqual({"observed": 2, "inserted": 2, "updated": 0, "marked_unavailable": 0}, result)
        self.assertEqual("cid-group", self.storage.list_conversations(search="张三")[0]["id"])
        self.assertEqual("cid-direct", self.storage.list_conversations(kind="direct")[0]["id"])

    def test_refresh_retains_missing_conversations_as_unavailable(self) -> None:
        self.storage.upsert_conversations(conversations())
        result = self.storage.upsert_conversations(conversations()[:1], complete=True)
        self.assertEqual(1, result["marked_unavailable"])
        available = self.storage.list_conversations()
        self.assertEqual(["cid-group"], [item["id"] for item in available])
        unavailable = self.storage.list_conversations(available=False)
        self.assertEqual(["cid-direct"], [item["id"] for item in unavailable])

    def test_incomplete_refresh_does_not_invent_unavailable_state(self) -> None:
        self.storage.upsert_conversations(conversations())
        result = self.storage.upsert_conversations(conversations()[:1])
        self.assertEqual(0, result["marked_unavailable"])
        self.assertEqual(
            {"cid-group", "cid-direct"},
            {item["id"] for item in self.storage.list_conversations()},
        )

    def test_metadata_round_trip_and_update(self) -> None:
        self.assertIsNone(self.storage.get_metadata("directory.state"))
        first = self.storage.set_metadata(
            "directory.state", {"verified": False, "warnings": ["请刷新"]}
        )
        self.assertEqual("directory.state", first["key"])
        self.assertEqual(
            {"verified": False, "warnings": ["请刷新"]},
            self.storage.get_metadata("directory.state"),
        )
        self.storage.set_metadata(
            "directory.state", {"verified": True, "warnings": []}
        )
        self.assertEqual(
            {"verified": True, "warnings": []},
            self.storage.get_metadata("directory.state"),
        )
        with self.assertRaisesRegex(StorageError, "metadata key"):
            self.storage.set_metadata("../bad", {})

    def test_job_lifecycle_and_listing(self) -> None:
        job = self.storage.create_job({"conversation_ids": ["cid-group"]}, job_id="job-1")
        self.assertEqual("DRAFT", job["status"])
        updated = self.storage.update_job(
            "job-1",
            status="collecting",
            progress={"stage": "data", "percent": 20},
            started_at="2026-08-24T10:00:00+00:00",
        )
        self.assertEqual("COLLECTING", updated["status"])
        self.assertEqual(20, updated["progress"]["percent"])
        self.assertEqual("job-1", self.storage.list_jobs(status="COLLECTING")[0]["id"])
        with self.assertRaisesRegex(StorageError, "job already exists"):
            self.storage.create_job({}, job_id="job-1")
        with self.assertRaisesRegex(StorageError, "job not found"):
            self.storage.get_job("missing")

    def test_all_required_artifact_paths_are_immutable_and_atomic(self) -> None:
        self.storage.create_job({}, job_id="job-artifacts")
        artifacts = [
            self.storage.write_job_artifact("job-artifacts", "manifest", {"version": 1}),
            self.storage.write_job_artifact(
                "job-artifacts", "raw_pass", {"pass": 1}, name="pass_001_list_all"
            ),
            self.storage.write_job_artifact("job-artifacts", "snapshot", {"messages": []}),
            self.storage.write_job_artifact(
                "job-artifacts", "analysis", {"structured": {}}, name="llm_analysis"
            ),
            self.storage.write_job_artifact("job-artifacts", "integrity", {"complete": False}),
            self.storage.write_job_artifact(
                "job-artifacts", "report", "本周工作：\n", name="weekly_report.md"
            ),
            self.storage.write_job_artifact(
                "job-artifacts", "report", "本周已完成事项：\n", name="completed_items.md"
            ),
        ]
        for artifact in artifacts:
            path = pathlib.Path(artifact["path"])
            self.assertTrue(path.is_file())
            self.assertTrue(path.resolve().is_relative_to(self.root.resolve()))
            self.assertEqual(64, len(artifact["sha256"]))
        repeated = self.storage.write_job_artifact(
            "job-artifacts", "manifest", {"version": 1}
        )
        self.assertEqual(artifacts[0]["sha256"], repeated["sha256"])
        with self.assertRaisesRegex(StorageError, "different bytes"):
            self.storage.write_job_artifact(
                "job-artifacts", "manifest", {"version": 2}
            )
        job = self.storage.get_job("job-artifacts")
        self.assertEqual(7, len(job["artifacts"]))

    def test_paths_and_values_are_validated(self) -> None:
        self.storage.create_job({}, job_id="safe-job")
        with self.assertRaises(StorageError):
            self.storage.artifact_path("../escape", "report")
        with self.assertRaises(StorageError):
            self.storage.artifact_path("safe-job", "other")
        bad = conversations()[0]
        bad["available"] = 1
        with self.assertRaisesRegex(StorageError, "boolean"):
            self.storage.upsert_conversations([bad])

    def test_concurrent_writers_cannot_replace_immutable_artifact(self) -> None:
        self.storage.create_job({}, job_id="job-race")
        barrier = threading.Barrier(2)
        outcomes: list[tuple[str, str]] = []

        def writer(content: str) -> None:
            barrier.wait()
            try:
                artifact = self.storage.write_job_artifact(
                    "job-race", "report", content, name="race.md"
                )
                outcomes.append(("ok", artifact["sha256"]))
            except StorageError as exc:
                outcomes.append(("error", str(exc)))

        threads = [
            threading.Thread(target=writer, args=("first",)),
            threading.Thread(target=writer, args=("second",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(1, sum(1 for status, _ in outcomes if status == "ok"))
        self.assertEqual(1, sum(1 for status, _ in outcomes if status == "error"))
        published = self.storage.artifact_path("job-race", "report", "race.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(published, {"first", "second"})


if __name__ == "__main__":
    unittest.main()
