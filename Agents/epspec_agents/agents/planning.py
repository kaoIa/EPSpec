import json
from pathlib import Path

from ..guardrails.planning import validate_intent
from ..schemas import PlanningOutput
from ..services.model_factory import StructuredModelAdapter


class ExperimentPlanningAgent:
    name = "Experiment Planning Agent"

    def __init__(self, adapter: StructuredModelAdapter, prompt_path: Path):
        self.adapter = adapter
        self.instructions = prompt_path.read_text(encoding="utf-8")

    def plan(self, messages: list[dict[str, str]]) -> tuple[PlanningOutput, str]:
        input_text = json.dumps({"conversation": messages}, ensure_ascii=False, indent=2)
        generated = self.adapter.generate(self.name, self.instructions, input_text, PlanningOutput)
        output = generated.value
        if output.intent is not None:
            validate_intent(output.intent)
        return output, generated.raw
