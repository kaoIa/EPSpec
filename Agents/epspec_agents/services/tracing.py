import json
import re
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_KEYS = {"api_key", "authorization", "token", "access_token", "refresh_token", "secret", "password", "client_secret", "private_key", "cookie", "set_cookie"}
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            secret = normalized in SECRET_KEYS or normalized.endswith(("_api_key", "_access_token", "_refresh_token", "_secret", "_password"))
            output[key] = "[REDACTED]" if secret else redact(item)
        return output
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in ("bearer ", "sk-", "api_key=")) or re.search(r"://[^/\s:@]+:[^/\s@]+@", value):
            return "[REDACTED]"
    return value


def _lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


class LocalTracer:
    def __init__(self, path: Path, run_id: str):
        self.path = path.resolve()
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _lock(self.path)

    def emit(self, stage: str, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "stage": stage,
            "event_type": event_type,
            **redact(payload),
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                stream.flush()
        return event

    def read(self, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        if not self.path.is_file():
            return [], offset
        events = []
        with self._lock:
            with self.path.open("rb") as stream:
                stream.seek(offset)
                for line in stream:
                    try:
                        events.append(json.loads(line.decode("utf-8")))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                position = stream.tell()
        return events, position

    def iter_from(self, offset: int = 0) -> Iterator[dict[str, Any]]:
        events, _ = self.read(offset)
        yield from events
