"""Pluggable LLM backend for NoLemming.

Supports any OpenAI SDK-compatible provider: OpenAI, Anthropic (via proxy),
Ollama, vLLM, Together, Groq, Fireworks, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMBackend(ABC):
    """Abstract LLM backend. All NoLemming LLM calls go through this."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Generate a completion."""

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""


class OpenAICompatibleBackend(LLMBackend):
    """Backend for any OpenAI SDK-compatible API.

    Works with: OpenAI, Ollama, vLLM, Together, Groq, Fireworks,
    LM Studio, Anthropic (via litellm), and any other provider
    that exposes an OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            from openai import AsyncOpenAI

            kwargs: dict[str, str] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            elif self._base_url is not None:
                kwargs["api_key"] = "ollama"  # Ollama doesn't need a real key
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        import asyncio

        client = self._get_client()
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(  # type: ignore[union-attr]
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                    continue
                raise
        else:
            return LLMResponse(content="[rate limited]", model=self._model)
        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=usage,
        )

    def model_name(self) -> str:
        return self._model


class LLMRegistry:
    """Registry of available LLM backends."""

    def __init__(self) -> None:
        self._backends: dict[str, type[LLMBackend]] = {
            "openai": OpenAICompatibleBackend,
        }

    def register(self, name: str, backend_class: type[LLMBackend]) -> None:
        self._backends[name] = backend_class

    def get(
        self,
        name: str = "openai",
        **kwargs: str | None,
    ) -> LLMBackend:
        if name not in self._backends:
            msg = f"Unknown LLM backend: {name}. Available: {list(self._backends)}"
            raise ValueError(msg)
        return self._backends[name](**kwargs)

    def list_backends(self) -> list[str]:
        return list(self._backends)


llm_registry = LLMRegistry()
