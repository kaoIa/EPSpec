import json
from pathlib import Path

from ..exceptions import InterpretationError
from ..schemas import ExperimentResult, ScientificReport
from ..services.model_factory import StructuredModelAdapter


class ScientificInterpretationAgent:
    name = "Scientific Interpretation Agent"

    def __init__(self, adapter: StructuredModelAdapter, prompt_path: Path):
        self.adapter = adapter
        self.instructions = prompt_path.read_text(encoding="utf-8")

    def interpret(self, result: ExperimentResult) -> tuple[ScientificReport, str, dict]:
        payload = result.model_dump(mode="json")
        input_text = "下面给出结构化实验摘要：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        generated = self.adapter.generate(self.name, self.instructions, input_text, ScientificReport)
        markdown = generated.value.markdown.strip()
        if markdown.startswith("```") and markdown.endswith("```"):
            lines = markdown.splitlines()
            markdown = "\n".join(lines[1:-1]).strip()
        if not markdown:
            raise InterpretationError("Interpretation Agent 返回空报告")
        return ScientificReport(markdown=markdown), generated.raw, {
            "messages": [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": input_text},
            ],
            "prompt_payload": payload,
        }
