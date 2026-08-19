import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

sys.dont_write_bytecode = True


def _scientific_call(messages: list[dict[str, str]]) -> str:
    from openai import OpenAI

    api_key = os.getenv("EPSPEC_WORKER_API_KEY", "")
    if not api_key:
        raise RuntimeError("EPSpec scientific ranking 未配置 API key")
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": float(os.getenv("EPSPEC_WORKER_TIMEOUT", "600")),
        "max_retries": int(os.getenv("EPSPEC_WORKER_RETRIES", "2")),
    }
    base_url = os.getenv("EPSPEC_WORKER_BASE_URL", "")
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    typed_messages = [cast(ChatCompletionMessageParam, message) for message in messages]
    response = client.chat.completions.create(
        model=os.getenv("EPSPEC_WORKER_MODEL", "gpt-5.2"),
        messages=typed_messages,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _patch_epspec(search_dir: Path) -> None:
    if str(search_dir) not in sys.path:
        sys.path.insert(0, str(search_dir))
    helper = importlib.import_module("ep_global_prior_and_ranking_regression")
    helper.__dict__["PRIOR_KB_PATH"] = os.getenv("EPSPEC_WORKER_PRIOR_PATH", str(getattr(helper, "PRIOR_KB_PATH", "")))
    helper.__dict__["call_llm_agent35"] = _scientific_call
    helper.__dict__["call_llm"] = _scientific_call


def _patch_sliding(module: ModuleType) -> None:
    module.__dict__["API_KEY"] = os.getenv("EPSPEC_WORKER_API_KEY", "")
    module.__dict__["CLIENT_BASE_URL"] = os.getenv("EPSPEC_WORKER_BASE_URL", "") or None
    module.__dict__["CLIENT_MODEL"] = os.getenv("EPSPEC_WORKER_MODEL", "gpt-5.2")
    module.__dict__["PRIOR_KB_PATH"] = os.getenv("EPSPEC_WORKER_PRIOR_PATH", str(getattr(module, "PRIOR_KB_PATH", "")))
    module.__dict__["_client"] = None


def _load(tool: dict[str, Any]) -> Callable[..., Any]:
    search_dir = Path(tool["search_dir"]).resolve()
    if str(search_dir) not in sys.path:
        sys.path.insert(0, str(search_dir))
    if tool.get("runtime_patch") == "epspec":
        _patch_epspec(search_dir)
    file_path = tool.get("file_path")
    module: ModuleType | None = None
    failures: list[str] = []
    if file_path:
        path = Path(file_path).resolve()
        spec = importlib.util.spec_from_file_location(f"epspec_worker_{os.getpid()}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载算法文件: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        for module_name in tool.get("module_names", []):
            try:
                module = importlib.import_module(module_name)
                break
            except Exception as exc:
                failures.append(f"{module_name}: {exc}")
    if module is None:
        raise RuntimeError("; ".join(failures) if not file_path else "算法模块加载失败")
    if tool.get("runtime_patch") == "sliding":
        _patch_sliding(module)
    function = getattr(module, tool["function_name"], None)
    if not callable(function):
        raise RuntimeError(f"算法入口不可调用: {tool['function_name']}")
    return function


def execute(payload: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(payload["input_path"]).resolve()
    output_path = Path(payload["output_path"]).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if payload["kind"] == "preprocessing":
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        if payload["tool"]["tool_id"].startswith("EPSpec"):
            (output_path / "EP").mkdir(parents=True, exist_ok=True)
    function = _load(payload["tool"])
    returned = function(str(input_path), str(output_path), **payload["tool"].get("call_kwargs", {}))
    return {
        "status": "completed",
        "run_id": payload["run_id"],
        "stage": payload["stage"],
        "tool_id": payload["tool"]["tool_id"],
        "output_path": str(output_path),
        "returned": None if returned is None else repr(returned),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="epspec-worker")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    result_path = Path(args.result).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
        result = execute(payload)
        code = 0
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        code = 1
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
