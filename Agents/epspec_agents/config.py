import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

from .exceptions import ConfigurationError

ExecutionMode = Literal["native", "simulate"]


def _discover_agents_dir() -> Path:
    configured = os.getenv("EPSPEC_AGENTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    package_parent = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd().resolve(), package_parent]
    for candidate in candidates:
        if candidate.name.lower() == "agents" and (candidate / "epspec_agents").is_dir():
            return candidate
        nested = candidate / "Agents"
        if nested.is_dir() and (nested / "epspec_agents").is_dir():
            return nested.resolve()
    return package_parent


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须为整数") from exc


def _number(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError as exc:
        raise ConfigurationError(f"{name} 必须为数值") from exc


@dataclass(frozen=True)
class ModelProfile:
    role: str
    model: str
    api_key: str
    base_url: str | None
    temperature: float
    timeout_seconds: float
    max_retries: int

    @property
    def configured(self) -> bool:
        invalid = {"", "your key", "your api key", "your_key"}
        return self.api_key.strip().lower() not in invalid and bool(self.model.strip())

    def require(self) -> None:
        if not self.configured:
            raise ConfigurationError(f"未配置 {self.role} 模型凭据")

    def public(self) -> dict[str, object]:
        return {
            "role": self.role,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "configured": self.configured,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    agents_dir: Path
    project_root: Path
    planner: ModelProfile
    interpreter: ModelProfile
    scientific: ModelProfile
    offline: bool
    execution_mode: ExecutionMode
    worker_timeout_seconds: float
    max_concurrency: int
    allow_text_fallback: bool
    capture_prompts: bool
    sdk_tracing: bool
    auto_approve: bool
    server_token: str
    python_executable: str

    @classmethod
    def from_env(cls, agents_dir: Path | None = None) -> "RuntimeConfig":
        resolved_agents = (agents_dir or _discover_agents_dir()).resolve()
        try:
            from dotenv import load_dotenv

            load_dotenv(resolved_agents / ".env", override=False)
        except ImportError:
            pass
        legacy_key = os.getenv("EPSPEC_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        legacy_base = os.getenv("EPSPEC_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        timeout = _number("EPSPEC_AGENT_TIMEOUT", 600.0, 1.0)
        retries = _integer("EPSPEC_AGENT_LLM_MAX_RETRIES", 2, 0)
        orchestration_key = os.getenv("EPSPEC_ORCHESTRATION_API_KEY") or os.getenv("GLM_API_KEY") or legacy_key
        orchestration_base = os.getenv("EPSPEC_ORCHESTRATION_BASE_URL") or legacy_base or "https://api.z.ai/api/paas/v4/"
        planner = ModelProfile(
            role="planning",
            model=os.getenv("EPSPEC_PLANNER_MODEL") or os.getenv("EPSPEC_AGENT_MODEL") or "glm-4.7",
            api_key=os.getenv("EPSPEC_PLANNER_API_KEY") or orchestration_key,
            base_url=os.getenv("EPSPEC_PLANNER_BASE_URL") or orchestration_base,
            temperature=_number("EPSPEC_PLANNER_TEMPERATURE", 0.0, 0.0),
            timeout_seconds=timeout,
            max_retries=retries,
        )
        interpreter = ModelProfile(
            role="interpretation",
            model=os.getenv("EPSPEC_INTERPRETER_MODEL") or os.getenv("EPSPEC_AGENT_MODEL") or "glm-4.7",
            api_key=os.getenv("EPSPEC_INTERPRETER_API_KEY") or orchestration_key,
            base_url=os.getenv("EPSPEC_INTERPRETER_BASE_URL") or orchestration_base,
            temperature=_number("EPSPEC_INTERPRETER_TEMPERATURE", 0.0, 0.0),
            timeout_seconds=timeout,
            max_retries=retries,
        )
        scientific_key = os.getenv("EPSPEC_SCIENTIFIC_API_KEY") or os.getenv("OPENAI_API_KEY") or legacy_key
        scientific = ModelProfile(
            role="scientific-ranking",
            model=os.getenv("EPSPEC_SCIENTIFIC_MODEL") or "gpt-5.2",
            api_key=scientific_key,
            base_url=os.getenv("EPSPEC_SCIENTIFIC_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
            temperature=_number("EPSPEC_SCIENTIFIC_TEMPERATURE", 0.0, 0.0),
            timeout_seconds=timeout,
            max_retries=retries,
        )
        mode_value = os.getenv("EPSPEC_EXECUTION_MODE", "native").strip().lower()
        if mode_value not in {"native", "simulate"}:
            raise ConfigurationError("EPSPEC_EXECUTION_MODE 只允许 native 或 simulate")
        mode = cast(ExecutionMode, mode_value)
        configured_project = os.getenv("EPSPEC_PROJECT_ROOT")
        project_root = Path(configured_project).expanduser().resolve() if configured_project else resolved_agents.parent
        return cls(
            agents_dir=resolved_agents,
            project_root=project_root,
            planner=planner,
            interpreter=interpreter,
            scientific=scientific,
            offline=_flag("EPSPEC_OFFLINE", False),
            execution_mode=mode,
            worker_timeout_seconds=_number("EPSPEC_WORKER_TIMEOUT", 7200.0, 1.0),
            max_concurrency=_integer("EPSPEC_AGENT_MAX_CONCURRENCY", 1, 1),
            allow_text_fallback=_flag("EPSPEC_AGENT_ALLOW_TEXT_FALLBACK", False),
            capture_prompts=_flag("EPSPEC_CAPTURE_PROMPTS", False),
            sdk_tracing=_flag("EPSPEC_SDK_TRACING", True),
            auto_approve=_flag("EPSPEC_AUTO_APPROVE", False),
            server_token=os.getenv("EPSPEC_SERVER_TOKEN", ""),
            python_executable=os.getenv("EPSPEC_PYTHON_EXECUTABLE") or sys.executable,
        )

    def with_overrides(
        self,
        offline: bool | None = None,
        execution_mode: ExecutionMode | None = None,
        auto_approve: bool | None = None,
    ) -> "RuntimeConfig":
        return replace(
            self,
            offline=self.offline if offline is None else offline,
            execution_mode=self.execution_mode if execution_mode is None else execution_mode,
            auto_approve=self.auto_approve if auto_approve is None else auto_approve,
        )

    @property
    def runtime_dir(self) -> Path:
        return self.agents_dir / ".runtime"

    @property
    def checkpoint_path(self) -> Path:
        return self.runtime_dir / "checkpoints.sqlite"

    @property
    def repository_path(self) -> Path:
        return self.runtime_dir / "runs.sqlite"

    @property
    def runs_dir(self) -> Path:
        return self.agents_dir / "runs"

    @property
    def prompts_dir(self) -> Path:
        return self.agents_dir / "epspec_agents" / "prompts"

    @property
    def prior_path(self) -> Path:
        return self.project_root / "RAG_Prior knowledge" / "Data" / "Functional Group.xlsx"

    def public(self) -> dict[str, object]:
        return {
            "agents_dir": str(self.agents_dir),
            "project_root": str(self.project_root),
            "offline": self.offline,
            "execution_mode": self.execution_mode,
            "worker_timeout_seconds": self.worker_timeout_seconds,
            "max_concurrency": self.max_concurrency,
            "allow_text_fallback": self.allow_text_fallback,
            "capture_prompts": self.capture_prompts,
            "sdk_tracing": self.sdk_tracing,
            "auto_approve": self.auto_approve,
            "planner": self.planner.public(),
            "interpreter": self.interpreter.public(),
            "scientific": self.scientific.public(),
        }
