import json
import sqlite3
from pathlib import Path
from typing import Any

from ..exceptions import RunNotFoundError, RunStateError
from ..schemas import RunSnapshot, RunStatus, utc_now


class RunRepository:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    target_stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    user_request TEXT NOT NULL,
                    interruption_json TEXT,
                    result_json TEXT,
                    metadata_json TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS runs_updated_at_idx ON runs(updated_at DESC)")

    def create(
        self,
        run_id: str,
        thread_id: str,
        target_stage: str,
        user_request: str,
        metadata: dict[str, Any] | None = None,
    ) -> RunSnapshot:
        now = utc_now()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        thread_id,
                        target_stage,
                        RunStatus.created.value,
                        "created",
                        user_request,
                        None,
                        None,
                        json.dumps(metadata or {}, ensure_ascii=False),
                        0,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise RunStateError(f"运行已存在: {run_id}") from exc
        return self.get(run_id)

    def update(
        self,
        run_id: str,
        status: str | RunStatus | None = None,
        current_stage: str | None = None,
        interruption: dict[str, Any] | None = None,
        clear_interruption: bool = False,
        result: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunSnapshot:
        snapshot = self.get(run_id)
        next_status = RunStatus(status).value if status is not None else snapshot.status.value
        next_stage = current_stage or snapshot.current_stage
        next_interruption = None if clear_interruption else snapshot.interruption
        if interruption is not None:
            next_interruption = interruption
        next_result = result if result is not None else snapshot.result
        next_metadata = {**snapshot.metadata, **(metadata or {})}
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, current_stage = ?, interruption_json = ?, result_json = ?, metadata_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    next_status,
                    next_stage,
                    self._dump(next_interruption),
                    self._dump(next_result),
                    self._dump(next_metadata) or "{}",
                    utc_now(),
                    run_id,
                ),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> RunSnapshot:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(f"运行不存在: {run_id}")
        return self._snapshot(row)

    def list(self, limit: int = 50, status: str | RunStatus | None = None) -> list[RunSnapshot]:
        safe_limit = min(max(int(limit), 1), 500)
        with self._connect() as connection:
            if status is None:
                rows = connection.execute("SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?", (safe_limit,)).fetchall()
            else:
                status_value = RunStatus(status).value
                rows = connection.execute(
                    "SELECT * FROM runs WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status_value, safe_limit),
                ).fetchall()
        return [self._snapshot(row) for row in rows]

    def request_cancel(self, run_id: str) -> RunSnapshot:
        snapshot = self.get(run_id)
        if snapshot.status in {RunStatus.completed, RunStatus.failed, RunStatus.cancelled}:
            raise RunStateError(f"终态运行不可取消: {run_id}")
        with self._connect() as connection:
            connection.execute("UPDATE runs SET cancel_requested = 1, updated_at = ? WHERE run_id = ?", (utc_now(), run_id))
        return self.get(run_id)

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT cancel_requested FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return bool(row and row[0])

    def latest_plan_run(self) -> RunSnapshot | None:
        for snapshot in self.list(500):
            if snapshot.result and snapshot.result.get("plan"):
                return snapshot
        return None

    def _snapshot(self, row: sqlite3.Row) -> RunSnapshot:
        return RunSnapshot(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            target_stage=row["target_stage"],
            status=RunStatus(row["status"]),
            current_stage=row["current_stage"],
            user_request=row["user_request"],
            interruption=self._load(row["interruption_json"]),
            result=self._load(row["result_json"]),
            metadata=self._load(row["metadata_json"]) or {},
            cancel_requested=bool(row["cancel_requested"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _dump(self, value: Any) -> str | None:
        return None if value is None else json.dumps(value, ensure_ascii=False, default=str)

    def _load(self, value: str | None) -> Any:
        return None if value is None else json.loads(value)
