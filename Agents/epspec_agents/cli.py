import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .exceptions import AgentRuntimeError
from .runtime.doctor import run_doctor
from .runtime.runner import RuntimeRunner
from .schemas import ExperimentPlan, RunSnapshot
from .tools.registry import ScientificToolRegistry


def _mode_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--approve", action=argparse.BooleanOptionalAction, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epspec-agents")
    parser.add_argument("--stage", choices=["full", "planning", "execution", "interpretation"])
    parser.add_argument("--request", default="")
    subparsers = parser.add_subparsers(dest="command")

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--json", action="store_true")

    demo = subparsers.add_parser("demo")
    demo.add_argument("request", nargs="?", default="Run corn with EPSpec_plsr using SNV and compare plsr, ipls_plsr, and cars_plsr.")
    demo.add_argument("--json", action="store_true")

    plan = subparsers.add_parser("plan")
    plan.add_argument("request", nargs="?")
    plan.add_argument("--json", action="store_true")
    _mode_options(plan)

    run = subparsers.add_parser("run")
    run.add_argument("request", nargs="?")
    run.add_argument("--plan", type=Path)
    run.add_argument("--stage", choices=["full", "execution", "interpretation"], default="full")
    run.add_argument("--non-interactive", action="store_true")
    run.add_argument("--json", action="store_true")
    _mode_options(run)

    resume = subparsers.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("response", nargs="?")
    resume.add_argument("--non-interactive", action="store_true")
    resume.add_argument("--json", action="store_true")

    status = subparsers.add_parser("status")
    status.add_argument("run_id")

    listing = subparsers.add_parser("list")
    listing.add_argument("--limit", type=int, default=50)
    listing.add_argument("--status")

    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("run_id")

    report = subparsers.add_parser("report")
    report.add_argument("run_id")

    artifacts = subparsers.add_parser("artifacts")
    artifacts.add_argument("run_id")

    events = subparsers.add_parser("events")
    events.add_argument("run_id")
    events.add_argument("--offset", type=int, default=0)

    subparsers.add_parser("capabilities")

    serve = subparsers.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    mcp = subparsers.add_parser("mcp")
    mcp.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    mcp.add_argument("--host", default="127.0.0.1")
    mcp.add_argument("--port", type=int, default=8000)

    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--cases", type=Path)
    return parser


def _effective(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    simulate = getattr(args, "simulate", None)
    return config.with_overrides(
        offline=getattr(args, "offline", None),
        execution_mode="simulate" if simulate is True else "native" if simulate is False else None,
        auto_approve=getattr(args, "approve", None),
    )


def _request(value: str | None) -> str:
    if value and value.strip():
        return value.strip()
    print("请描述实验需求，支持中文、English 或中英混合表达。")
    return input("你：").strip()


def _continue_interactive(runner: RuntimeRunner, snapshot: RunSnapshot, enabled: bool) -> RunSnapshot:
    while enabled and snapshot.interruption is not None:
        print(json.dumps(snapshot.interruption, ensure_ascii=False, indent=2))
        response = input("输入：").strip()
        snapshot = runner.resume(snapshot.run_id, response)
    return snapshot


def _run_snapshot(
    config: RuntimeConfig,
    target_stage: str,
    request: str,
    plan: ExperimentPlan | None,
    interactive: bool,
) -> RunSnapshot:
    runner = RuntimeRunner(config)
    snapshot = runner.create_run(target_stage, request)
    snapshot = runner.execute(snapshot.run_id, plan)
    return _continue_interactive(runner, snapshot, interactive)


def _print_snapshot(snapshot: RunSnapshot, as_json: bool = False) -> None:
    if as_json:
        print(snapshot.model_dump_json(indent=2))
        return
    print(f"run_id: {snapshot.run_id}")
    print(f"status: {snapshot.status.value}")
    print(f"stage: {snapshot.current_stage}")
    if snapshot.interruption:
        print(json.dumps(snapshot.interruption, ensure_ascii=False, indent=2))
    result = snapshot.result or {}
    report = result.get("report")
    if isinstance(report, dict) and report.get("markdown"):
        print(report["markdown"])


def _doctor(config: RuntimeConfig, as_json: bool) -> int:
    report = run_doctor(config)
    if as_json:
        print(report.model_dump_json(indent=2))
    else:
        print(f"ready: {str(report.ready).lower()}")
        for check in report.checks:
            print(f"{check.status.upper():4} {check.name}: {check.message}")
    return 0 if report.ready else 1


def _capabilities(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "datasets": ["shootout", "corn", "soil", "tecator"],
        "preprocessors": ["savitzky_golay", "snv"],
        "models": ["plsr", "ipls_plsr", "cars_plsr", "EPSpec_plsr", "EPSpec_plsr_sliding"],
        "tools": ScientificToolRegistry(config.project_root).capabilities(),
    }


def _legacy(args: argparse.Namespace, config: RuntimeConfig) -> int:
    stage = args.stage or "full"
    request = args.request
    plan = None
    if stage in {"execution", "interpretation"}:
        plan = RuntimeRunner(config).load_plan()
    else:
        request = _request(request)
    snapshot = _run_snapshot(config, stage, request, plan, True)
    _print_snapshot(snapshot)
    return 0 if snapshot.status.value not in {"failed", "cancelled"} else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RuntimeConfig.from_env()
    try:
        if args.command is None:
            return _legacy(args, config)
        if args.command == "doctor":
            return _doctor(config, args.json)
        if args.command == "demo":
            effective = config.with_overrides(offline=True, execution_mode="simulate", auto_approve=True)
            snapshot = _run_snapshot(effective, "full", args.request, None, False)
            _print_snapshot(snapshot, args.json)
            return 0 if snapshot.status.value == "completed" else 1
        if args.command == "plan":
            effective = _effective(config, args)
            snapshot = _run_snapshot(effective, "planning", _request(args.request), None, True)
            _print_snapshot(snapshot, args.json)
            return 0 if snapshot.status.value == "completed" else 1
        if args.command == "run":
            effective = _effective(config, args)
            runner = RuntimeRunner(effective)
            plan = runner.load_plan(args.plan) if args.plan else None
            request = "" if plan and args.stage in {"execution", "interpretation"} else _request(args.request)
            snapshot = _run_snapshot(effective, args.stage, request, plan, not args.non_interactive)
            _print_snapshot(snapshot, args.json)
            return 0 if snapshot.status.value not in {"failed", "cancelled"} else 1
        if args.command == "resume":
            runner = RuntimeRunner(config)
            response = args.response or input("输入：").strip()
            snapshot = runner.resume(args.run_id, response)
            snapshot = _continue_interactive(runner, snapshot, not args.non_interactive)
            _print_snapshot(snapshot, args.json)
            return 0 if snapshot.status.value not in {"failed", "cancelled"} else 1
        if args.command == "status":
            print(RuntimeRunner(config).get(args.run_id).model_dump_json(indent=2))
            return 0
        if args.command == "list":
            print(json.dumps([item.model_dump(mode="json") for item in RuntimeRunner(config).list(args.limit, args.status)], ensure_ascii=False, indent=2))
            return 0
        if args.command == "cancel":
            print(RuntimeRunner(config).cancel(args.run_id).model_dump_json(indent=2))
            return 0
        if args.command == "report":
            print(RuntimeRunner(config).report(args.run_id))
            return 0
        if args.command == "artifacts":
            print(json.dumps(RuntimeRunner(config).artifacts(args.run_id), ensure_ascii=False, indent=2))
            return 0
        if args.command == "events":
            events, offset = RuntimeRunner(config).events(args.run_id, args.offset)
            print(json.dumps({"events": events, "next_offset": offset}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "capabilities":
            print(json.dumps(_capabilities(config), ensure_ascii=False, indent=2))
            return 0
        if args.command == "serve":
            import uvicorn

            if args.reload:
                uvicorn.run("epspec_agents.api:create_app", factory=True, host=args.host, port=args.port, reload=True)
            else:
                from .api import create_app

                uvicorn.run(create_app(config), host=args.host, port=args.port)
            return 0
        if args.command == "mcp":
            from .mcp_server import serve

            serve(args.transport, config, args.host, args.port)
            return 0
        if args.command == "eval":
            from .evaluation import run_evaluation

            cases = args.cases or config.agents_dir / "evals" / "cases.json"
            result = run_evaluation(cases)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["score"] == 1.0 else 1
    except (AgentRuntimeError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print("已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"运行失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
