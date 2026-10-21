"""SQLite state and immutable artifact storage for on-demand report jobs."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import Any


class StorageError(RuntimeError):
    """A storage invariant was violated."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def _canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parse_json(value: str | None, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # corruption must not be hidden
        raise StorageError("stored JSON is corrupted") from exc


def _safe_identifier(value: str, *, field: str) -> str:
    candidate = str(value).strip()
    if not candidate or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", candidate):
        raise StorageError(f"invalid {field}: {value!r}")
    return candidate


SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('direct', 'group', 'unknown')),
    last_active TEXT,
    participants_json TEXT NOT NULL,
    available INTEGER NOT NULL CHECK(available IN (0, 1)),
    source TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_title ON conversations(title);
CREATE INDEX IF NOT EXISTS idx_conversations_kind ON conversations(kind);
CREATE INDEX IF NOT EXISTS idx_conversations_last_active ON conversations(last_active DESC);

CREATE TABLE IF NOT EXISTS app_metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    progress_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS job_artifacts (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE RESTRICT,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    relative_path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(job_id, kind, name)
);
"""


class Storage:
    """Small state store with immutable, atomically published job evidence."""

    ARTIFACT_KINDS = {
        "manifest",
        "raw_pass",
        "snapshot",
        "analysis",
        "integrity",
        "report",
    }
    _artifact_lock = threading.Lock()

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = pathlib.Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_root = self.root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "panel.sqlite3"
        with contextlib.closing(self._connect()) as connection:
            connection.executescript(SCHEMA)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def get_metadata(self, key: str, fallback: Any = None) -> Any:
        metadata_key = _safe_identifier(key, field="metadata key")
        with contextlib.closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value_json FROM app_metadata WHERE key=?",
                (metadata_key,),
            ).fetchone()
        if row is None:
            return fallback
        return _parse_json(row["value_json"], fallback)

    def set_metadata(self, key: str, value: Any) -> dict[str, str]:
        metadata_key = _safe_identifier(key, field="metadata key")
        try:
            serialized = _canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise StorageError("metadata value is not JSON serializable") from exc
        updated_at = _utc_now()
        with contextlib.closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO app_metadata(key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (metadata_key, serialized, updated_at),
            )
            connection.commit()
        return {"key": metadata_key, "updated_at": updated_at}

    @staticmethod
    def _conversation_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "kind": row["kind"],
            "last_active": row["last_active"],
            "participants": _parse_json(row["participants_json"], []),
            "available": bool(row["available"]),
            "source": row["source"],
            "raw": _parse_json(row["raw_json"], {}),
            "first_seen_at": row["first_seen_at"],
            "last_seen_at": row["last_seen_at"],
        }

    @staticmethod
    def _job_row(row: sqlite3.Row, artifacts: Sequence[sqlite3.Row] = ()) -> dict[str, Any]:
        return {
            "id": row["id"],
            "status": row["status"],
            "config": _parse_json(row["config_json"], {}),
            "progress": _parse_json(row["progress_json"], {}),
            "result": _parse_json(row["result_json"], {}),
            "error": _parse_json(row["error_json"], None),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "artifacts": [
                {
                    "kind": artifact["kind"],
                    "name": artifact["name"],
                    "relative_path": artifact["relative_path"],
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                    "media_type": artifact["media_type"],
                    "created_at": artifact["created_at"],
                }
                for artifact in artifacts
            ],
        }

    def upsert_conversations(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        complete: bool = False,
    ) -> dict[str, int]:
        """Insert/update one directory observation.

        Missing rows are marked unavailable only when the caller can prove
        ``complete=True``.  The safe default preserves previously observed
        conversations instead of inventing an unavailable state.
        """

        if isinstance(items, (str, bytes)):
            raise StorageError("conversation items must be a sequence")
        if type(complete) is not bool:
            raise StorageError("complete must be a boolean")
        now = _utc_now()
        prepared: list[tuple[Any, ...]] = []
        observed_ids: set[str] = set()
        for item in items:
            if not isinstance(item, Mapping):
                raise StorageError("each conversation must be an object")
            conversation_id = str(item.get("id") or "").strip()
            if not conversation_id:
                raise StorageError("conversation id is required")
            if conversation_id in observed_ids:
                raise StorageError(f"duplicate conversation id in one refresh: {conversation_id}")
            observed_ids.add(conversation_id)
            title = str(item.get("title") or conversation_id).strip()
            kind = str(item.get("kind") or "unknown").lower()
            if kind not in {"direct", "group", "unknown"}:
                raise StorageError(f"invalid conversation kind: {kind}")
            available = item.get("available", True)
            if type(available) is not bool:
                raise StorageError("conversation available must be a boolean")
            participants = item.get("participants", [])
            if not isinstance(participants, list):
                raise StorageError("conversation participants must be a list")
            source = str(item.get("source") or "unknown").strip()
            raw = item.get("raw", {})
            if not isinstance(raw, Mapping):
                raise StorageError("conversation raw value must be an object")
            prepared.append(
                (
                    conversation_id,
                    title,
                    kind,
                    item.get("last_active"),
                    _canonical_json(participants),
                    1 if available else 0,
                    source,
                    _canonical_json(raw),
                    now,
                    now,
                )
            )
        with contextlib.closing(self._connect()) as connection:
            before = {
                row["id"]
                for row in connection.execute("SELECT id FROM conversations").fetchall()
            }
            connection.execute("BEGIN IMMEDIATE")
            if complete:
                if observed_ids:
                    placeholders = ",".join("?" for _ in observed_ids)
                    connection.execute(
                        f"UPDATE conversations SET available=0 WHERE id NOT IN ({placeholders})",
                        tuple(sorted(observed_ids)),
                    )
                else:
                    connection.execute("UPDATE conversations SET available=0")
            connection.executemany(
                """
                INSERT INTO conversations(
                    id, title, kind, last_active, participants_json, available,
                    source, raw_json, first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    kind=excluded.kind,
                    last_active=excluded.last_active,
                    participants_json=excluded.participants_json,
                    available=excluded.available,
                    source=excluded.source,
                    raw_json=excluded.raw_json,
                    last_seen_at=excluded.last_seen_at
                """,
                prepared,
            )
            connection.commit()
        return {
            "observed": len(prepared),
            "inserted": len(observed_ids - before),
            "updated": len(observed_ids & before),
            "marked_unavailable": len(before - observed_ids) if complete else 0,
        }

    def list_conversations(
        self,
        search: str | None = None,
        kind: str | None = None,
        available: bool | None = True,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if kind is not None:
            normalized_kind = kind.lower()
            if normalized_kind not in {"direct", "group", "unknown"}:
                raise StorageError(f"invalid conversation kind: {kind}")
            clauses.append("kind=?")
            parameters.append(normalized_kind)
        if available is not None:
            if type(available) is not bool:
                raise StorageError("available must be boolean or None")
            clauses.append("available=?")
            parameters.append(1 if available else 0)
        sql = "SELECT * FROM conversations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY CASE WHEN last_active IS NULL THEN 1 ELSE 0 END, last_active DESC, title COLLATE NOCASE, id"
        with contextlib.closing(self._connect()) as connection:
            rows = [self._conversation_row(row) for row in connection.execute(sql, parameters)]
        if search:
            needle = search.casefold().strip()
            if needle:
                rows = [
                    row
                    for row in rows
                    if needle in row["title"].casefold()
                    or needle in row["id"].casefold()
                    or any(
                        needle in str(participant.get("name") or "").casefold()
                        or needle in str(participant.get("id") or "").casefold()
                        for participant in row["participants"]
                        if isinstance(participant, Mapping)
                    )
                ]
        return rows

    def create_job(
        self,
        config: Mapping[str, Any],
        job_id: str | None = None,
        *,
        status: str = "DRAFT",
    ) -> dict[str, Any]:
        if not isinstance(config, Mapping):
            raise StorageError("job config must be an object")
        identifier = _safe_identifier(job_id or uuid.uuid4().hex, field="job id")
        normalized_status = str(status).strip().upper()
        if not normalized_status:
            raise StorageError("job status is required")
        now = _utc_now()
        with contextlib.closing(self._connect()) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id, status, config_json, progress_json, result_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, '{}', '{}', ?, ?)
                    """,
                    (identifier, normalized_status, _canonical_json(dict(config)), now, now),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                raise StorageError(f"job already exists: {identifier}") from exc
        return self.get_job(identifier)

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> dict[str, Any]:
        identifier = _safe_identifier(job_id, field="job id")
        assignments = ["updated_at=?"]
        parameters: list[Any] = [_utc_now()]
        if status is not None:
            normalized_status = str(status).strip().upper()
            if not normalized_status:
                raise StorageError("job status cannot be empty")
            assignments.append("status=?")
            parameters.append(normalized_status)
        if progress is not None:
            if not isinstance(progress, Mapping):
                raise StorageError("job progress must be an object")
            assignments.append("progress_json=?")
            parameters.append(_canonical_json(dict(progress)))
        if result is not None:
            if not isinstance(result, Mapping):
                raise StorageError("job result must be an object")
            assignments.append("result_json=?")
            parameters.append(_canonical_json(dict(result)))
        if error is not None:
            error_value = dict(error) if isinstance(error, Mapping) else {"message": str(error)}
            assignments.append("error_json=?")
            parameters.append(_canonical_json(error_value))
        if started_at is not None:
            assignments.append("started_at=?")
            parameters.append(started_at)
        if finished_at is not None:
            assignments.append("finished_at=?")
            parameters.append(finished_at)
        parameters.append(identifier)
        with contextlib.closing(self._connect()) as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id=?", parameters
            )
            if cursor.rowcount != 1:
                raise StorageError(f"job not found: {identifier}")
            connection.commit()
        return self.get_job(identifier)

    def get_job(self, job_id: str) -> dict[str, Any]:
        identifier = _safe_identifier(job_id, field="job id")
        with contextlib.closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise StorageError(f"job not found: {identifier}")
            artifacts = connection.execute(
                "SELECT * FROM job_artifacts WHERE job_id=? ORDER BY kind, name", (identifier,)
            ).fetchall()
            return self._job_row(row, artifacts)

    def list_jobs(self, *, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise StorageError("limit must be an integer between 1 and 1000")
        parameters: list[Any] = []
        sql = "SELECT * FROM jobs"
        if status is not None:
            sql += " WHERE status=?"
            parameters.append(str(status).strip().upper())
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        parameters.append(limit)
        with contextlib.closing(self._connect()) as connection:
            return [self._job_row(row) for row in connection.execute(sql, parameters)]

    def artifact_path(self, job_id: str, kind: str, name: str | None = None) -> pathlib.Path:
        identifier = _safe_identifier(job_id, field="job id")
        normalized_kind = str(kind).strip().lower()
        if normalized_kind not in self.ARTIFACT_KINDS:
            raise StorageError(f"invalid artifact kind: {kind}")
        if normalized_kind == "raw_pass":
            artifact_name = _safe_identifier(name or "pass_001", field="artifact name")
            relative = pathlib.Path("jobs") / identifier / "raw" / f"{artifact_name}.json"
        elif normalized_kind == "manifest":
            artifact_name = _safe_identifier(name or "manifest", field="artifact name")
            relative = pathlib.Path("jobs") / identifier / f"{artifact_name}.json"
        elif normalized_kind == "snapshot":
            artifact_name = _safe_identifier(name or "snapshot", field="artifact name")
            relative = pathlib.Path("jobs") / identifier / f"{artifact_name}.json"
        elif normalized_kind == "analysis":
            artifact_name = _safe_identifier(name or "llm_analysis", field="artifact name")
            if pathlib.Path(artifact_name).suffix.lower() != ".json":
                artifact_name += ".json"
            relative = pathlib.Path("jobs") / identifier / "analysis" / artifact_name
        elif normalized_kind == "integrity":
            artifact_name = _safe_identifier(name or "integrity", field="artifact name")
            relative = pathlib.Path("jobs") / identifier / f"{artifact_name}.json"
        else:
            artifact_name = _safe_identifier(name or "report", field="artifact name")
            suffix = pathlib.Path(artifact_name).suffix.lower()
            if suffix not in {".md", ".docx", ".pdf"}:
                artifact_name += ".md"
            relative = pathlib.Path("jobs") / identifier / "reports" / artifact_name
        resolved = (self.root / relative).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:  # defensive; safe identifiers already prevent traversal
            raise StorageError("artifact path escapes storage root") from exc
        return resolved

    def write_job_artifact(
        self,
        job_id: str,
        kind: str,
        content: Any,
        *,
        name: str | None = None,
        media_type: str | None = None,
    ) -> dict[str, Any]:
        """Atomically publish an immutable artifact.

        Repeating the same write is idempotent.  Reusing a path with different
        bytes is rejected; callers must choose a new versioned ``name``.
        """

        identifier = _safe_identifier(job_id, field="job id")
        normalized_kind = str(kind).strip().lower()
        path = self.artifact_path(identifier, normalized_kind, name)
        # Prove that the owning job exists before touching the filesystem.
        self.get_job(identifier)
        if isinstance(content, bytes):
            encoded = content
        elif isinstance(content, str):
            encoded = content.encode("utf-8")
        else:
            encoded = _canonical_json(content, pretty=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        # The process lock makes the common threaded panel path deterministic;
        # the hard-link publication also provides cross-process no-overwrite
        # semantics.  Unlike os.replace(), it can never replace an existing
        # immutable artifact during a check/publish race.
        with self._artifact_lock:
            if path.exists():
                try:
                    existing = path.read_bytes()
                except OSError as exc:
                    raise StorageError(f"cannot verify existing artifact: {path}") from exc
                if existing != encoded:
                    raise StorageError(
                        f"immutable artifact already exists with different bytes: {path}"
                    )
            else:
                temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("xb") as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                    try:
                        os.link(temporary, path)
                    except FileExistsError:
                        existing = path.read_bytes()
                        if existing != encoded:
                            raise StorageError(
                                f"immutable artifact already exists with different bytes: {path}"
                            )
                except StorageError:
                    raise
                except OSError as exc:
                    raise StorageError(f"cannot publish artifact: {path}") from exc
                finally:
                    with contextlib.suppress(OSError):
                        temporary.unlink()
        default_media = {
            ".json": "application/json",
            ".md": "text/markdown; charset=utf-8",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }.get(path.suffix.lower(), "application/octet-stream")
        created_at = _utc_now()
        relative = path.relative_to(self.root).as_posix()
        artifact_name = path.stem if normalized_kind != "report" else path.name
        with contextlib.closing(self._connect()) as connection:
            existing = connection.execute(
                "SELECT * FROM job_artifacts WHERE job_id=? AND kind=? AND name=?",
                (identifier, normalized_kind, artifact_name),
            ).fetchone()
            if existing is not None and (
                existing["sha256"] != digest or existing["relative_path"] != relative
            ):
                raise StorageError("artifact ledger conflicts with immutable file")
            connection.execute(
                """
                INSERT OR IGNORE INTO job_artifacts(
                    job_id, kind, name, relative_path, sha256, size_bytes,
                    media_type, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    normalized_kind,
                    artifact_name,
                    relative,
                    digest,
                    len(encoded),
                    media_type or default_media,
                    created_at,
                ),
            )
            connection.commit()
        return {
            "job_id": identifier,
            "kind": normalized_kind,
            "name": artifact_name,
            "path": str(path),
            "relative_path": relative,
            "sha256": digest,
            "size_bytes": len(encoded),
            "media_type": media_type or default_media,
        }
