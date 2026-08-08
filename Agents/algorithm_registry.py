from pathlib import Path

from epspec_agents.tools.registry import ScientificToolRegistry, ToolSpec


REGISTRY = ScientificToolRegistry(Path(__file__).resolve().parent.parent)


def resolve(tool_id: str, kind: str | None = None) -> ToolSpec:
    return REGISTRY.resolve(tool_id, kind)


__all__ = ["REGISTRY", "ScientificToolRegistry", "ToolSpec", "resolve"]
