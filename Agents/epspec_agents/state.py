from typing import Any, TypedDict


class RuntimeState(TypedDict, total=False):
    run_id: str
    thread_id: str
    target_stage: str
    user_request: str
    messages: list[dict[str, str]]
    planning_output: dict[str, Any]
    planning_raw: str
    intent: dict[str, Any]
    plan: dict[str, Any]
    approval_status: str
    auto_approve: bool
    current_stage: str
    preprocess_result: dict[str, Any]
    main_result: dict[str, Any]
    comparison_results: list[dict[str, Any]]
    experiment_result: dict[str, Any]
    report: dict[str, Any]
    artifacts: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    status: str
    created_at: str
    updated_at: str
    max_concurrency: int
    offline: bool
    execution_mode: str
