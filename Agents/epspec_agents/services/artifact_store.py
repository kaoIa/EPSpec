import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from ..schemas import ArtifactRef, ExperimentPlan


class ArtifactStore:
    def __init__(self, agents_dir: Path, run_id: str):
        self.agents_dir = agents_dir.resolve()
        self.run_dir = self.agents_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_bytes(self, path: Path, data: bytes) -> ArtifactRef:
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
        media_type = "application/json" if path.suffix == ".json" else "text/markdown" if path.suffix == ".md" else "application/octet-stream"
        return ArtifactRef(
            name=path.name,
            path=path,
            media_type=media_type,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    def write_json(self, path: Path, value: Any) -> ArtifactRef:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        data = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
        return self._atomic_bytes(path, data)

    def write_text(self, path: Path, value: str) -> ArtifactRef:
        return self._atomic_bytes(path, value.encode("utf-8"))

    def write_plan(self, plan: ExperimentPlan) -> list[ArtifactRef]:
        return [
            self.write_json(self.agents_dir / "plan.json", plan.to_legacy_dict()),
            self.write_json(self.run_dir / "plan.json", plan),
        ]

    def write_run_plan(self, plan: ExperimentPlan) -> ArtifactRef:
        return self.write_json(self.run_dir / "plan.json", plan)

    def write_legacy_plan(self, plan: ExperimentPlan) -> ArtifactRef:
        return self.write_json(self.agents_dir / "plan.json", plan.to_legacy_dict())

    def write_report_artifacts(self, send: dict[str, Any], response: dict[str, Any], markdown: str) -> list[ArtifactRef]:
        return [
            self.write_json(self.agents_dir / "send.json", send),
            self.write_json(self.agents_dir / "response.json", response),
            self.write_text(self.agents_dir / "summary_report.md", markdown),
            self.write_json(self.run_dir / "send.json", send),
            self.write_json(self.run_dir / "response.json", response),
            self.write_text(self.run_dir / "summary_report.md", markdown),
        ]
