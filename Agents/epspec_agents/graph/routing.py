from typing import Any


def route_start(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failure"
    target = state.get("target_stage", "full")
    if target == "execution":
        return "validate_plan"
    if target == "interpretation":
        return "aggregate_results"
    return "planning"


def route_planning(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failure"
    output = state.get("planning_output") or {}
    return "clarification" if output.get("status") == "needs_clarification" else "compile_plan"


def route_approval(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failure"
    if state.get("approval_status") == "approved":
        return "finalization" if state.get("target_stage") == "planning" else "preprocessing"
    return "planning"


def route_after_plan_validation(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failure"
    if state.get("target_stage") == "execution":
        return "preprocessing"
    return "plan_approval"


def route_after_execution(state: dict[str, Any]) -> str:
    if state.get("status") == "failed":
        return "failure"
    return "finalization" if state.get("target_stage") == "execution" else "interpretation"


def route_success(next_node: str):
    def route(state: dict[str, Any]) -> str:
        return "failure" if state.get("status") == "failed" else next_node
    return route
