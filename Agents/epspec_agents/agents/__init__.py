__all__ = ["ExperimentPlanningAgent", "ScientificExecutionAgent", "ScientificInterpretationAgent"]


def __getattr__(name: str):
    if name == "ExperimentPlanningAgent":
        from .planning import ExperimentPlanningAgent

        return ExperimentPlanningAgent
    if name == "ScientificExecutionAgent":
        from .execution import ScientificExecutionAgent

        return ScientificExecutionAgent
    if name == "ScientificInterpretationAgent":
        from .interpretation import ScientificInterpretationAgent

        return ScientificInterpretationAgent
    raise AttributeError(name)
