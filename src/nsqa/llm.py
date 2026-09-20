"""Accès à un LLM (optionnel). Le projet fonctionne aussi 100 % hors ligne."""
from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str: ...


class AnthropicClient:
    """Client minimal autour du SDK Anthropic.

    Nécessite `pip install anthropic` et la variable d'environnement ANTHROPIC_API_KEY.
    Le modèle se règle via NSQA_MODEL (défaut : claude-sonnet-5).
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Installez le SDK : pip install 'nsqa[llm]'") from exc
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY n'est pas définie.")
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model or os.getenv("NSQA_MODEL", "claude-sonnet-5")

    def complete(self, system: str, user: str, max_tokens: int = 1000) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")
