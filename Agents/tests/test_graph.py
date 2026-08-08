from epspec_agents.graph.routing import route_after_execution, route_after_plan_validation, route_approval, route_planning, route_start


def test_clarification_branch():
    assert route_planning({"planning_output": {"status": "needs_clarification"}}) == "clarification"


def test_ready_branch():
    assert route_planning({"planning_output": {"status": "ready"}}) == "compile_plan"


def test_approval_branch():
    assert route_approval({"approval_status": "approved", "target_stage": "full"}) == "preprocessing"
    assert route_approval({"approval_status": "approved", "target_stage": "planning"}) == "finalization"
    assert route_approval({"approval_status": "rejected"}) == "planning"


def test_stage_entry_and_exit_routes():
    assert route_start({"target_stage": "execution"}) == "validate_plan"
    assert route_start({"target_stage": "interpretation"}) == "aggregate_results"
    assert route_after_plan_validation({"target_stage": "execution"}) == "preprocessing"
    assert route_after_execution({"target_stage": "execution"}) == "finalization"
