from pydantic import ValidationError

from ..exceptions import PlanningValidationError
from ..schemas import ExperimentIntent


def validate_intent(value: ExperimentIntent | dict) -> ExperimentIntent:
    try:
        return value if isinstance(value, ExperimentIntent) else ExperimentIntent.model_validate(value)
    except ValidationError as exc:
        raise PlanningValidationError(str(exc)) from exc
