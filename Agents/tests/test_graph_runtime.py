from dataclasses import dataclass
from pathlib import Path

from langgraph.types import Command

from epspec_agents.config import RuntimeConfig
from epspec_agents.graph.workflow import build_workflow
from epspec_agents.runtime.checkpoint import CheckpointManager
from epspec_agents.schemas import PlanningOutput
from epspec_agents.services.model_factory import GenerationResult


@dataclass
class SequenceAdapter:
    outputs: list[PlanningOutput]

    def generate(self, name, instructions, input_text, output_type):
        value = self.outputs.pop(0)
        return GenerationResult(value=value, raw=value.model_dump_json())


def config_for(tmp_path: Path) -> RuntimeConfig:
    agents = tmp_path / "Agents"
    (agents / "epspec_agents" / "prompts").mkdir(parents=True)
    source_prompts = Path(__file__).resolve().parents[1] / "epspec_agents" / "prompts"
    for name in ["planning.txt", "interpretation.txt"]:
        (agents / "epspec_agents" / "prompts" / name).write_text((source_prompts / name).read_text(encoding="utf-8"), encoding="utf-8")
    raw = tmp_path / "Data" / "Raw Data"
    raw.mkdir(parents=True)
    (raw / "corn.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    return RuntimeConfig(
        agents_dir=agents,
        project_root=tmp_path,
        api_key="fake",
        base_url=None,
        model_name="fake",
        timeout_seconds=1,
        max_retries=0,
        max_concurrency=1,
        allow_text_fallback=False,
    )


def ready_output():
    return PlanningOutput.model_validate({
        "status": "ready",
        "message": "",
        "intent": {"dataset_name": "corn", "model": "plsr"},
    })


def test_real_graph_checkpoint_and_approval_resume(tmp_path):
    config = config_for(tmp_path)
    manager = CheckpointManager(config.checkpoint_path)
    graph = build_workflow(config, SequenceAdapter([ready_output()]), manager.open())
    invocation = {"configurable": {"thread_id": "approval-test"}}
    try:
        result = graph.invoke({
            "run_id": "approval-run",
            "thread_id": "approval-test",
            "target_stage": "planning",
            "user_request": "corn + plsr",
            "messages": [{"role": "user", "content": "corn + plsr"}],
            "artifacts": [],
            "errors": [],
            "comparison_results": [],
        }, invocation)
        assert result["__interrupt__"]
        completed = graph.invoke(Command(resume="确认"), invocation)
        assert completed["status"] == "completed"
        assert config.plan_path.is_file()
        assert config.checkpoint_path.is_file()
    finally:
        manager.close()


def test_real_graph_clarification_resume(tmp_path):
    config = config_for(tmp_path)
    clarification = PlanningOutput(status="needs_clarification", message="请选择模型", intent=None)
    manager = CheckpointManager(config.checkpoint_path)
    graph = build_workflow(config, SequenceAdapter([clarification, ready_output()]), manager.open())
    invocation = {"configurable": {"thread_id": "clarification-test"}}
    try:
        first = graph.invoke({
            "run_id": "clarification-run",
            "thread_id": "clarification-test",
            "target_stage": "planning",
            "user_request": "corn",
            "messages": [{"role": "user", "content": "corn"}],
            "artifacts": [],
            "errors": [],
            "comparison_results": [],
        }, invocation)
        assert first["__interrupt__"]
        second = graph.invoke(Command(resume="plsr"), invocation)
        assert second["__interrupt__"]
        completed = graph.invoke(Command(resume="confirm"), invocation)
        assert completed["status"] == "completed"
        assert len(completed["messages"]) == 3
    finally:
        manager.close()
