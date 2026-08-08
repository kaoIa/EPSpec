from typing import Any, TypedDict


class RuntimeState(TypedDict, total=False):
    run_id: str
    thread_id: str
    target_stage: str
    messages: list[dict[str, str]]
    user_request: str
    planning_output: dict[str, Any] | None
    intent: dict[str, Any] | None
    plan: dict[str, Any] | None
    approval_status: str
    current_stage: str
    preprocess_result: dict[str, Any] | None
    main_result: dict[str, Any] | None
    comparison_results: list[dict[str, Any]]
    experiment_result: dict[str, Any] | None
    report: dict[str, Any] | None
    artifacts: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    status: str
    created_at: str
    updated_at: str
