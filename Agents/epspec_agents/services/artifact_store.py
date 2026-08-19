import hashlib
import json
import mimetypes
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ..exceptions import ArtifactError
from ..schemas import ArtifactRef, ExperimentPlan


class ArtifactStore:
    def __init__(self, agents_dir: Path, run_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise ArtifactError("run_id 格式不合法")
        self.agents_dir = agents_dir.resolve()
        self.run_id = run_id
        self.run_dir = (self.agents_dir / "runs" / run_id).resolve()
        try:
            self.run_dir.relative_to((self.agents_dir / "runs").resolve())
        except ValueError as exc:
            raise ArtifactError(f"运行目录越界: {self.run_dir}") from exc
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def work_dir(self) -> Path:
        return self.directory("work")

    @property
    def results_dir(self) -> Path:
        return self.directory("results")

    @property
    def logs_dir(self) -> Path:
        return self.directory("logs")

    def directory(self, value: str | Path) -> Path:
        path = self._resolve(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve(self, value: str | Path) -> Path:
        candidate = Path(value)
        path = candidate.resolve() if candidate.is_absolute() else (self.run_dir / candidate).resolve()
        self._assert_within(path)
        return path

    def _assert_within(self, path: Path) -> None:
        try:
            path.relative_to(self.run_dir)
        except ValueError as exc:
            raise ArtifactError(f"运行产物路径越界: {path}") from exc

    def _atomic_bytes(self, value: str | Path, data: bytes, role: str = "artifact") -> ArtifactRef:
        path = self._resolve(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return self.reference(path, role)

    def write_json(self, value: str | Path, payload: Any, role: str = "artifact") -> ArtifactRef:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        return self._atomic_bytes(value, data, role)

    def write_text(self, value: str | Path, payload: str, role: str = "artifact") -> ArtifactRef:
        return self._atomic_bytes(value, payload.encode("utf-8"), role)

    def write_plan(self, plan: ExperimentPlan) -> ArtifactRef:
        return self.write_json("plan.json", plan, "plan")

    def write_request(self, request: str) -> ArtifactRef:
        return self.write_json("request.json", {"request": request}, "request")

    def write_report_artifacts(self, response: dict[str, Any], markdown: str, prompt: dict[str, Any] | None = None) -> list[ArtifactRef]:
        artifacts = [
            self.write_json("report/response.json", response, "report"),
            self.write_text("report/summary.md", markdown, "report"),
        ]
        if prompt is not None:
            artifacts.append(self.write_json("prompts/interpretation.json", prompt, "prompt"))
        return artifacts

    def reference(self, value: str | Path, role: str = "artifact") -> ArtifactRef:
        path = self._resolve(value)
        if not path.is_file():
            raise ArtifactError(f"产物不存在: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        media_type = {
            ".jsonl": "application/x-ndjson",
            ".md": "text/markdown",
            ".csv": "text/csv",
        }.get(path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        return ArtifactRef(
            name=str(path.relative_to(self.run_dir)).replace("\\", "/"),
            path=path,
            media_type=media_type,
            role=role,
            sha256=digest.hexdigest(),
            size_bytes=path.stat().st_size,
        )

    def collect(self) -> list[ArtifactRef]:
        references = []
        for path in sorted(item for item in self.run_dir.rglob("*") if item.is_file() and not item.name.startswith(".")):
            relative = path.relative_to(self.run_dir)
            role = relative.parts[0] if len(relative.parts) > 1 else relative.stem
            references.append(self.reference(path, role))
        return references
