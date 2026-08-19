class AgentRuntimeError(Exception):
    __slots__ = ()


class ConfigurationError(AgentRuntimeError):
    __slots__ = ()


class DependencyError(ConfigurationError):
    __slots__ = ()


class PlanningError(AgentRuntimeError):
    __slots__ = ()


class PlanningValidationError(PlanningError):
    __slots__ = ()


class PlanCompilationError(AgentRuntimeError):
    __slots__ = ()


class PlanValidationError(AgentRuntimeError):
    __slots__ = ()


class ApprovalError(AgentRuntimeError):
    __slots__ = ()


class ToolResolutionError(AgentRuntimeError):
    __slots__ = ()


class ToolExecutionError(AgentRuntimeError):
    __slots__ = ()


class WorkerTimeoutError(ToolExecutionError):
    __slots__ = ()


class ResultParsingError(AgentRuntimeError):
    __slots__ = ()


class InterpretationError(AgentRuntimeError):
    __slots__ = ()


class ReportValidationError(AgentRuntimeError):
    __slots__ = ()


class RunNotFoundError(AgentRuntimeError):
    __slots__ = ()


class RunStateError(AgentRuntimeError):
    __slots__ = ()


class RunCancelledError(AgentRuntimeError):
    __slots__ = ()


class ArtifactError(AgentRuntimeError):
    __slots__ = ()
