from datetime import datetime, timezone
import platform
import sys
from typing import Any

from ..schemas import ExecutionError, ExperimentPlan, RunManifest, RunStatus


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_manifest(state: dict[str, Any]) -> RunManifest:
    plan_value = state.get("plan")
    return RunManifest(
        run_id=state["run_id"],
        thread_id=state["thread_id"],
        status=RunStatus(state.get("status", "initialized")),
        current_stage=state.get("current_stage", "unknown"),
        created_at=state["created_at"],
        updated_at=state.get("updated_at", utc_now()),
        plan=ExperimentPlan.from_legacy_dict(plan_value) if plan_value else None,
        artifacts=state.get("artifacts", []),
        errors=[ExecutionError.model_validate(item) for item in state.get("errors", [])],
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "max_concurrency": state.get("max_concurrency", 1),
        },
    )
