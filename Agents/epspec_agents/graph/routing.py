from collections.abc import Callable

from ..state import RuntimeState


def _terminal(state: RuntimeState) -> str | None:
    if state.get("status") == "failed":
        return "failure"
    if state.get("status") == "cancelled":
        return "cancellation"
    return None


def route_start(state: RuntimeState) -> str:
    terminal = _terminal(state)
    if terminal:
        return terminal
    target = state.get("target_stage", "full")
    if target == "execution":
        return "validate_plan"
    if target == "interpretation":
        return "aggregate_results"
    return "planning"


def route_planning(state: RuntimeState) -> str:
    terminal = _terminal(state)
    if terminal:
        return terminal
    output = state.get("planning_output") or {}
    return "clarification" if output.get("status") == "needs_clarification" else "compile_plan"


def route_after_plan_validation(state: RuntimeState) -> str:
    terminal = _terminal(state)
    if terminal:
        return terminal
    target = state.get("target_stage", "full")
    if target == "planning":
        return "finalization"
    if target == "execution":
        return "preprocessing"
    return "plan_approval"


def route_approval(state: RuntimeState) -> str:
    terminal = _terminal(state)
    if terminal:
        return terminal
    return "preprocessing" if state.get("approval_status") == "approved" else "planning"


def route_after_execution(state: RuntimeState) -> str:
    terminal = _terminal(state)
    if terminal:
        return terminal
    return "finalization" if state.get("target_stage") == "execution" else "interpretation"


def route_success(next_node: str) -> Callable[[RuntimeState], str]:
    def route(state: RuntimeState) -> str:
        return _terminal(state) or next_node

    return route
