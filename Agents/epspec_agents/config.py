from dataclasses import dataclass
import os
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class RuntimeConfig:
    agents_dir: Path
    project_root: Path
    api_key: str
    base_url: str | None
    model_name: str
    timeout_seconds: float
    max_retries: int
    max_concurrency: int
    allow_text_fallback: bool

    @classmethod
    def from_env(cls, agents_dir: Path | None = None) -> "RuntimeConfig":
        resolved_agents = (agents_dir or Path(__file__).resolve().parents[1]).resolve()
        api_key = os.getenv("EPSPEC_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        base_url = os.getenv("EPSPEC_AGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
        model_name = os.getenv("EPSPEC_AGENT_MODEL") or "gpt-4o-mini"
        return cls(
            agents_dir=resolved_agents,
            project_root=resolved_agents.parent,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            timeout_seconds=float(os.getenv("EPSPEC_AGENT_TIMEOUT", "600")),
            max_retries=int(os.getenv("EPSPEC_AGENT_LLM_MAX_RETRIES", "2")),
            max_concurrency=max(1, int(os.getenv("EPSPEC_AGENT_MAX_CONCURRENCY", "1"))),
            allow_text_fallback=os.getenv("EPSPEC_AGENT_ALLOW_TEXT_FALLBACK", "0").lower() in {"1", "true", "yes"},
        )

    @property
    def plan_path(self) -> Path:
        return self.agents_dir / "plan.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.agents_dir / ".runtime" / "checkpoints.sqlite"

    @property
    def runs_dir(self) -> Path:
        return self.agents_dir / "runs"

    @property
    def prompts_dir(self) -> Path:
        return self.agents_dir / "epspec_agents" / "prompts"

    def require_llm(self) -> None:
        invalid = {"", "your key", "your api key"}
        if self.api_key.strip().lower() in invalid:
            raise ConfigurationError("未配置 LLM API key，请设置 EPSPEC_AGENT_API_KEY 或 OPENAI_API_KEY。")
        if not self.model_name.strip():
            raise ConfigurationError("未配置模型名称，请设置 EPSPEC_AGENT_MODEL。")
