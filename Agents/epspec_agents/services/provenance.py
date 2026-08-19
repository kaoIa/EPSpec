import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..config import RuntimeConfig
from ..schemas import ExecutionError, ExperimentPlan, RunManifest, RunStatus
from ..tools.registry import ScientificToolRegistry
from .artifact_store import ArtifactStore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _packages() -> dict[str, str | None]:
    packages = [
        "openai-agents",
        "openai",
        "langgraph",
        "langgraph-checkpoint-sqlite",
        "mcp",
        "fastapi",
        "pydantic",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
    ]
    output: dict[str, str | None] = {}
    for package in packages:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = None
    return output


def _sources(plan: ExperimentPlan | None, registry: ScientificToolRegistry) -> dict[str, list[dict[str, str | None]]]:
    if plan is None:
        return {}
    identifiers = []
    if plan.step_preprocess.enabled and plan.step_preprocess.method:
        identifiers.append(str(plan.step_preprocess.method))
    identifiers.extend([plan.step_model_main.method, *[step.method for step in plan.step_model_compare.models]])
    output: dict[str, list[dict[str, str | None]]] = {}
    for identifier in identifiers:
        spec = registry.resolve(identifier)
        output[identifier] = [{"path": str(source), "sha256": file_sha256(source)} for source in spec.source_files()]
    return output


def build_manifest(
    state: dict[str, Any],
    config: RuntimeConfig,
    registry: ScientificToolRegistry,
    store: ArtifactStore,
) -> RunManifest:
    plan_value = state.get("plan")
    plan = ExperimentPlan.from_legacy_dict(plan_value) if plan_value else None
    prompt_hashes = {path.name: file_sha256(path) for path in sorted(config.prompts_dir.glob("*.txt")) if path.is_file()}
    status = RunStatus(state.get("status", "initialized"))
    git_status = _git(config.project_root, "status", "--porcelain", "--untracked-files=no")
    return RunManifest(
        run_id=state["run_id"],
        thread_id=state["thread_id"],
        status=status,
        current_stage=state.get("current_stage", "unknown"),
        created_at=state.get("created_at", utc_now()),
        updated_at=state.get("updated_at", utc_now()),
        plan=plan,
        artifacts=store.collect(),
        errors=[ExecutionError.model_validate(item) for item in state.get("errors", [])],
        runtime={
            **config.public(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _packages(),
        },
        provenance={
            "git_commit": _git(config.project_root, "rev-parse", "HEAD"),
            "git_branch": _git(config.project_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "working_tree_dirty": None if git_status is None else bool(git_status),
            "input": {
                "path": str(plan.step_preprocess.input_path) if plan else None,
                "sha256": file_sha256(plan.step_preprocess.input_path) if plan else None,
            },
            "tool_sources": _sources(plan, registry),
            "prompt_hashes": prompt_hashes,
            "prior_knowledge": {"path": str(config.prior_path), "sha256": file_sha256(config.prior_path)},
        },
    )
