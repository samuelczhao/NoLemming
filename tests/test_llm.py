"""Tests for LLM backend abstraction."""

from __future__ import annotations

import pytest

from nolemming.core.llm import (
    LLMBackend,
    LLMRegistry,
    LLMResponse,
    OpenAICompatibleBackend,
)


class TestLLMResponse:
    def test_creation(self) -> None:
        resp = LLMResponse(content="hello", model="gpt-4o-mini")
        assert resp.content == "hello"
        assert resp.model == "gpt-4o-mini"
        assert resp.usage == {}


class TestOpenAICompatibleBackend:
    def test_model_name(self) -> None:
        backend = OpenAICompatibleBackend(model="llama3")
        assert backend.model_name() == "llama3"

    def test_custom_base_url(self) -> None:
        backend = OpenAICompatibleBackend(
            model="llama3",
            base_url="http://localhost:11434/v1",
        )
        assert backend._base_url == "http://localhost:11434/v1"


class TestLLMRegistry:
    def test_list_backends(self) -> None:
        registry = LLMRegistry()
        assert "openai" in registry.list_backends()

    def test_get_openai(self) -> None:
        registry = LLMRegistry()
        backend = registry.get("openai", model="gpt-4o")
        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend.model_name() == "gpt-4o"

    def test_unknown_backend_raises(self) -> None:
        registry = LLMRegistry()
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            registry.get("nonexistent")

    def test_register_custom(self) -> None:
        registry = LLMRegistry()

        class CustomBackend(LLMBackend):
            async def generate(
                self, system_prompt: str, user_prompt: str,
                temperature: float = 0.7, max_tokens: int = 2000,
            ) -> LLMResponse:
                return LLMResponse(content="custom", model="custom")

            def model_name(self) -> str:
                return "custom"

        registry.register("custom", CustomBackend)
        assert "custom" in registry.list_backends()
