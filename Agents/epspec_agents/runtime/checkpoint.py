import sqlite3
from pathlib import Path
from typing import Any

from ..exceptions import DependencyError


class CheckpointManager:
    def __init__(self, path: Path):
        self.path = path
        self.connection: sqlite3.Connection | None = None
        self.checkpointer: Any = None

    def open(self):
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise DependencyError("缺少 langgraph-checkpoint-sqlite，请安装 Agents/requirements-agent.txt。") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        if hasattr(self.checkpointer, "setup"):
            self.checkpointer.setup()
        return self.checkpointer

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
