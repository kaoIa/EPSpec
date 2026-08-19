import json
from pathlib import Path
from typing import Any

from .schemas import PlanningOutput
from .services.model_factory import HeuristicModelAdapter


def run_evaluation(path: Path) -> dict[str, Any]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    adapter = HeuristicModelAdapter()
    results = []
    passed = 0
    for case in cases:
        payload = json.dumps({"conversation": [{"role": "user", "content": case["request"]}]}, ensure_ascii=False)
        output = adapter.generate("Evaluation Planning Agent", "", payload, PlanningOutput).value
        actual = output.intent.model_dump(mode="json") if output.intent else None
        success = output.status == case["expected_status"]
        for key, expected in case.get("expected_intent", {}).items():
            success = success and actual is not None and actual.get(key) == expected
        passed += int(success)
        results.append({"id": case["id"], "passed": success, "actual": actual, "status": output.status})
    return {"passed": passed, "total": len(cases), "score": passed / len(cases) if cases else 0.0, "cases": results}
