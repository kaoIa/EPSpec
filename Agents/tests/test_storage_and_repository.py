from pathlib import Path

import pytest

from epspec_agents.exceptions import ArtifactError, RunStateError
from epspec_agents.runtime.repository import RunRepository
from epspec_agents.schemas import RunStatus
from epspec_agents.services.artifact_store import ArtifactStore


def test_artifact_store_hashes_and_contains_paths(runtime_config) -> None:
    store = ArtifactStore(runtime_config.agents_dir, "artifact_run")
    reference = store.write_json("nested/value.json", {"value": 1}, "test")
    assert reference.sha256
    assert reference.size_bytes
    assert reference.path.is_relative_to(store.run_dir)
    with pytest.raises(ArtifactError):
        store.write_text(store.run_dir.parent / "other_run" / "escape.txt", "blocked")


def test_run_repository_lifecycle(tmp_path: Path) -> None:
    repository = RunRepository(tmp_path / "runtime" / "runs.sqlite")
    created = repository.create("run_1", "thread_1", "full", "request")
    assert created.status == RunStatus.created
    updated = repository.update("run_1", status="planning", current_stage="planning", result={"value": 1})
    assert updated.result == {"value": 1}
    cancelled = repository.request_cancel("run_1")
    assert cancelled.cancel_requested
    with pytest.raises(RunStateError):
        repository.create("run_1", "thread_1", "full", "request")
