from dataclasses import dataclass

from epspec_agents.agents.planning import ExperimentPlanningAgent
from epspec_agents.schemas import PlanningOutput
from epspec_agents.services.model_factory import GenerationResult


@dataclass
class FakeAdapter:
    output: PlanningOutput

    def generate(self, name, instructions, input_text, output_type):
        return GenerationResult(value=self.output, raw=self.output.model_dump_json())


def test_planning_agent_uses_typed_output(tmp_path):
    prompt = tmp_path / "planning.txt"
    prompt.write_text("rules", encoding="utf-8")
    output = PlanningOutput.model_validate({
        "status": "ready",
        "message": "",
        "intent": {"dataset_name": "tecator", "model": "EPSpec_plsr", "preprocess": {"enabled": True, "method": "snv"}},
    })
    agent = ExperimentPlanningAgent(FakeAdapter(output), prompt)
    actual, raw = agent.plan([{"role": "user", "content": "tecator + EPSpec + SNV"}])
    assert actual.intent.dataset_name == "tecator"
    assert actual.intent.preprocess.method == "snv"
    assert raw
