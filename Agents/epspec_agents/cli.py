import argparse
import json
import sys

from .config import RuntimeConfig
from .exceptions import AgentRuntimeError
from .runtime.runner import RuntimeRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epspec-agents")
    parser.add_argument("--stage", choices=["full", "planning", "execution", "interpretation"], default="full")
    parser.add_argument("--request", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RuntimeConfig.from_env()
    runner = RuntimeRunner(config)
    request = args.request
    if args.stage in {"full", "planning"} and not request:
        print("请描述实验需求，支持中文、English 或中英混合表达。")
        try:
            request = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已中断。")
            return 130
    try:
        plan = runner.load_plan() if args.stage in {"execution", "interpretation"} else None
        result = runner.run(target_stage=args.stage, user_request=request, plan=plan)
    except FileNotFoundError:
        print(f"未找到 plan.json：{config.plan_path}")
        return 1
    except AgentRuntimeError as exc:
        print(f"运行失败：{exc}")
        return 1
    if result.get("status") == "failed":
        print(json.dumps(result.get("errors", []), ensure_ascii=False, indent=2))
        return 1
    if args.stage in {"full", "interpretation"}:
        print(f"总结报告已生成：{config.agents_dir / 'summary_report.md'}")
    elif args.stage == "planning":
        print(f"实验计划已生成：{config.plan_path}")
    else:
        print("确定性实验执行完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
