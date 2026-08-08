import pytest

from epspec_agents.exceptions import ToolResolutionError
from epspec_agents.tools.registry import ScientificToolRegistry


def test_registry_resolves_supported_tools(tmp_path):
    registry = ScientificToolRegistry(tmp_path)
    assert registry.resolve("snv", "preprocessing").tool_id == "snv"
    assert registry.resolve("EPSpec_plsr", "modeling").task_type == "regression"
    assert registry.ids("modeling") == ["EPSpec_plsr", "EPSpec_plsr_sliding", "cars_plsr", "ipls_plsr", "plsr"]


def test_unknown_tool_rejected(tmp_path):
    registry = ScientificToolRegistry(tmp_path)
    with pytest.raises(ToolResolutionError):
        registry.resolve("unknown")


def test_capability_mismatch_rejected(tmp_path):
    registry = ScientificToolRegistry(tmp_path)
    with pytest.raises(ToolResolutionError):
        registry.resolve("plsr", "preprocessing")
