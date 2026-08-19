import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ..config import RuntimeConfig
from ..exceptions import RunCancelledError, ToolExecutionError, WorkerTimeoutError
from ..runtime.repository import RunRepository
from ..services.artifact_store import ArtifactStore
from ..services.tracing import LocalTracer
from ..tools.registry import ScientificToolRegistry, ToolKind


class SubprocessToolRunner:
    def __init__(
        self,
        config: RuntimeConfig,
        registry: ScientificToolRegistry,
        repository: RunRepository,
        run_id: str,
        tracer: LocalTracer,
    ):
        self.config = config
        self.registry = registry
        self.repository = repository
        self.run_id = run_id
        self.tracer = tracer
        self.store = ArtifactStore(config.agents_dir, run_id)

    def run(self, tool_id: str, kind: ToolKind, input_path: Path, output_path: Path, stage: str) -> dict[str, Any]:
        spec = self.registry.resolve(tool_id, kind)
        resolved_input = input_path.resolve()
        resolved_output = output_path.resolve()
        try:
            resolved_input.relative_to(self.config.project_root.resolve())
        except ValueError as exc:
            raise ToolExecutionError(f"工具输入路径越界: {resolved_input}") from exc
        try:
            resolved_output.relative_to(self.store.run_dir)
        except ValueError as exc:
            raise ToolExecutionError(f"工具输出路径越界: {resolved_output}") from exc
        if not resolved_input.is_file():
            raise ToolExecutionError(f"工具输入不存在: {resolved_input}")
        safe_stage = "".join(character if character.isalnum() or character in "_-" else "_" for character in stage)
        payload_path = self.store.run_dir / "workers" / f"{safe_stage}.request.json"
        result_path = self.store.run_dir / "workers" / f"{safe_stage}.result.json"
        log_path = self.store.run_dir / "logs" / f"{safe_stage}.log"
        payload = {
            "run_id": self.run_id,
            "stage": stage,
            "kind": kind,
            "input_path": str(resolved_input),
            "output_path": str(resolved_output),
            "tool": spec.payload(),
        }
        self.store.write_json(payload_path, payload, "worker-request")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.python_executable,
            "-m",
            "epspec_agents.execution.worker",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
        ]
        environment = self._environment()
        started = time.monotonic()
        self.tracer.emit(stage, "worker_started", tool_id=tool_id, process_command=command[:3], log_path=str(log_path))
        with log_path.open("w", encoding="utf-8") as log_stream:
            process = subprocess.Popen(
                command,
                cwd=str(self.config.agents_dir),
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if self.repository.is_cancel_requested(self.run_id):
                    self._stop(process)
                    self.tracer.emit(stage, "worker_cancelled", tool_id=tool_id, elapsed_seconds=elapsed)
                    raise RunCancelledError(f"运行已取消: {self.run_id}")
                if elapsed > self.config.worker_timeout_seconds:
                    self._stop(process)
                    self.tracer.emit(stage, "worker_timed_out", tool_id=tool_id, elapsed_seconds=elapsed)
                    raise WorkerTimeoutError(f"工具 {tool_id} 超过 {self.config.worker_timeout_seconds:.0f} 秒")
                time.sleep(0.2)
        elapsed = time.monotonic() - started
        if not result_path.is_file():
            raise ToolExecutionError(f"工具 {tool_id} 未生成 worker result，退出码 {process.returncode}，日志 {log_path}")
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if process.returncode != 0 or value.get("status") != "completed":
            error = value.get("error") or f"worker 退出码 {process.returncode}"
            raise ToolExecutionError(f"工具 {tool_id} 执行失败: {error}，日志 {log_path}")
        self.tracer.emit(stage, "worker_completed", tool_id=tool_id, elapsed_seconds=elapsed, output_path=str(output_path))
        return {**value, "elapsed_seconds": elapsed, "log_path": str(log_path)}

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        for name in list(environment):
            if re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD|AUTHORIZATION|COOKIE|CREDENTIAL)", name, flags=re.I):
                environment.pop(name, None)
        package_paths = [str(self.config.agents_dir), str(Path(__file__).resolve().parents[2])]
        current_path = environment.get("PYTHONPATH", "")
        if current_path:
            package_paths.append(current_path)
        environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(package_paths))
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["EPSPEC_WORKER_API_KEY"] = self.config.scientific.api_key
        environment["OPENAI_API_KEY"] = self.config.scientific.api_key
        environment["EPSPEC_WORKER_MODEL"] = self.config.scientific.model
        environment["EPSPEC_WORKER_BASE_URL"] = self.config.scientific.base_url or ""
        environment["OPENAI_BASE_URL"] = self.config.scientific.base_url or ""
        environment["EPSPEC_WORKER_TIMEOUT"] = str(self.config.scientific.timeout_seconds)
        environment["EPSPEC_WORKER_RETRIES"] = str(self.config.scientific.max_retries)
        environment["EPSPEC_WORKER_PRIOR_PATH"] = str(self.config.prior_path)
        return environment

    def _stop(self, process: subprocess.Popen[str]) -> None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
