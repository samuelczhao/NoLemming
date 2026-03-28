"""Encoder plugin registry."""

from __future__ import annotations

from nolemming.encoders.base import BrainEncoder


class EncoderRegistry:
    """Registry mapping encoder names to encoder classes."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BrainEncoder]] = {}

    def register(self, name: str, encoder_class: type[BrainEncoder]) -> None:
        self._registry[name] = encoder_class

    def get(self, name: str, **kwargs: object) -> BrainEncoder:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(none)"
            msg = f"Unknown encoder '{name}'. Available: {available}"
            raise KeyError(msg)
        return self._registry[name](**kwargs)

    def list_encoders(self) -> list[str]:
        return sorted(self._registry)


def _create_default_registry() -> EncoderRegistry:
    registry = EncoderRegistry()

    from nolemming.encoders.mock import MockEncoder
    registry.register("mock", MockEncoder)

    try:
        from nolemming.encoders.tribe_v2 import TribeV2Encoder
        registry.register("tribe_v2", TribeV2Encoder)
    except ImportError:
        pass

    return registry


encoder_registry = _create_default_registry()
