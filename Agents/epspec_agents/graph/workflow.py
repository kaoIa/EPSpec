from typing import Any

from ..config import RuntimeConfig
from ..exceptions import DependencyError
from ..services.model_factory import StructuredModelAdapter
from ..state import RuntimeState
from .nodes import WorkflowNodes
from .routing import route_after_execution, route_after_plan_validation, route_approval, route_planning, route_start, route_success


def build_workflow(config: RuntimeConfig, adapter: StructuredModelAdapter, checkpointer: Any):
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise DependencyError("缺少 langgraph，请安装 Agents/requirements-agent.txt。") from exc
    nodes = WorkflowNodes(config, adapter)
    graph = StateGraph(RuntimeState)
    graph.add_node("initialize_run", nodes.guarded("initialize_run", nodes.initialize_run))
    graph.add_node("planning", nodes.guarded("planning", nodes.planning))
    graph.add_node("clarification", nodes.clarification)
    graph.add_node("compile_plan", nodes.guarded("compile_plan", nodes.compile_plan))
    graph.add_node("validate_plan", nodes.guarded("validate_plan", nodes.validate_plan))
    graph.add_node("plan_approval", nodes.plan_approval)
    graph.add_node("preprocessing", nodes.guarded("preprocessing", nodes.preprocessing))
    graph.add_node("primary_model_execution", nodes.guarded("primary_model_execution", nodes.primary_model_execution))
    graph.add_node("comparison_model_execution", nodes.guarded("comparison_model_execution", nodes.comparison_model_execution))
    graph.add_node("aggregate_results", nodes.guarded("aggregate_results", nodes.aggregate_results))
    graph.add_node("validate_results", nodes.guarded("validate_results", nodes.validate_results))
    graph.add_node("interpretation", nodes.guarded("interpretation", nodes.interpretation))
    graph.add_node("validate_report", nodes.guarded("validate_report", nodes.validate_report))
    graph.add_node("finalization", nodes.guarded("finalization", nodes.finalization))
    graph.add_node("failure", nodes.failure)
    graph.add_edge(START, "initialize_run")
    graph.add_conditional_edges("initialize_run", route_start)
    graph.add_conditional_edges("planning", route_planning)
    graph.add_edge("clarification", "planning")
    graph.add_conditional_edges("compile_plan", route_success("validate_plan"))
    graph.add_conditional_edges("validate_plan", route_after_plan_validation)
    graph.add_conditional_edges("plan_approval", route_approval)
    graph.add_conditional_edges("preprocessing", route_success("primary_model_execution"))
    graph.add_conditional_edges("primary_model_execution", route_success("comparison_model_execution"))
    graph.add_conditional_edges("comparison_model_execution", route_success("aggregate_results"))
    graph.add_conditional_edges("aggregate_results", route_success("validate_results"))
    graph.add_conditional_edges("validate_results", route_after_execution)
    graph.add_conditional_edges("interpretation", route_success("validate_report"))
    graph.add_conditional_edges("validate_report", route_success("finalization"))
    graph.add_edge("finalization", END)
    graph.add_edge("failure", END)
    return graph.compile(checkpointer=checkpointer)
