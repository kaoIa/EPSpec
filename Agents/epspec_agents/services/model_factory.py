from dataclasses import dataclass
import json
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from ..config import RuntimeConfig
from ..exceptions import ConfigurationError, DependencyError


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class GenerationResult(Generic[T]):
    value: T
    raw: str


class StructuredModelAdapter(Protocol):
    def generate(self, name: str, instructions: str, input_text: str, output_type: type[T]) -> GenerationResult[T]:
        ...


class AgentsSDKAdapter:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def generate(self, name: str, instructions: str, input_text: str, output_type: type[T]) -> GenerationResult[T]:
        self.config.require_llm()
        try:
            from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise DependencyError("缺少 openai-agents，请安装 Agents/requirements-agent.txt。") from exc
        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
        }
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        client = AsyncOpenAI(**client_kwargs)
        model = OpenAIChatCompletionsModel(model=self.config.model_name, openai_client=client)
        agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
            output_type=output_type,
            model_settings=ModelSettings(temperature=0),
        )
        try:
            result = Runner.run_sync(agent, input=input_text)
            value = result.final_output
            if not isinstance(value, output_type):
                value = output_type.model_validate(value)
            raw = value.model_dump_json(indent=2)
            return GenerationResult(value=value, raw=raw)
        except Exception as exc:
            if not self.config.allow_text_fallback:
                raise ConfigurationError(f"OpenAI Agents SDK structured output 调用失败: {exc}") from exc
            return self._fallback(name, instructions, input_text, output_type, exc)

    def _fallback(self, name: str, instructions: str, input_text: str, output_type: type[T], original_error: Exception) -> GenerationResult[T]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DependencyError("缺少 openai Python SDK。") from exc
        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
        }
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        client = OpenAI(**client_kwargs)
        schema = output_type.model_json_schema()
        prompt = instructions + "\n\n必须只输出符合以下 JSON Schema 的单一 JSON 对象：\n" + json.dumps(schema, ensure_ascii=False)
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": input_text}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            value = output_type.model_validate_json(content)
            return GenerationResult(value=value, raw=response.model_dump_json(indent=2))
        except Exception as exc:
            raise ConfigurationError(f"structured output 与兼容 fallback 均失败: {original_error}; {exc}") from exc


class ModelFactory:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def create(self) -> StructuredModelAdapter:
        return AgentsSDKAdapter(self.config)
