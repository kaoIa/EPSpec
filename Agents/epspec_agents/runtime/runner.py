import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import RuntimeConfig
from ..graph.workflow import build_workflow
from ..schemas import ExperimentPlan
from ..services.model_factory import ModelFactory, StructuredModelAdapter
from .checkpoint import CheckpointManager


class RuntimeRunner:
    def __init__(self, config: RuntimeConfig | None = None, adapter: StructuredModelAdapter | None = None):
        self.config = config or RuntimeConfig.from_env()
        self.adapter = adapter or ModelFactory(self.config).create()

    def run(self, target_stage: str = "full", user_request: str = "", plan: ExperimentPlan | None = None) -> dict[str, Any]:
        run_id = uuid4().hex
        thread_id = run_id
        manager = CheckpointManager(self.config.checkpoint_path)
        checkpointer = manager.open()
        try:
            graph = build_workflow(self.config, self.adapter, checkpointer)
            initial: dict[str, Any] = {
                "run_id": run_id,
                "thread_id": thread_id,
                "target_stage": target_stage,
                "user_request": user_request,
                "messages": [{"role": "user", "content": user_request}] if user_request else [],
                "plan": plan.to_legacy_dict() if plan else None,
                "artifacts": [],
                "errors": [],
                "comparison_results": [],
            }
            invocation_config = {"configurable": {"thread_id": thread_id}}
            result = graph.invoke(initial, invocation_config)
            while result.get("__interrupt__"):
                payload = self._interrupt_payload(result["__interrupt__"])
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                answer = input("输入：").strip()
                from langgraph.types import Command
                result = graph.invoke(Command(resume=answer), invocation_config)
            return result
        finally:
            manager.close()

    def _interrupt_payload(self, value: Any) -> Any:
        item = value[0] if isinstance(value, (list, tuple)) and value else value
        return getattr(item, "value", item)

    def load_plan(self, path: Path | None = None) -> ExperimentPlan:
        plan_path = path or self.config.plan_path
        value = json.loads(plan_path.read_text(encoding="utf-8"))
        return ExperimentPlan.from_legacy_dict(value)
