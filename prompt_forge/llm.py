"""A thin, swappable wrapper around the Anthropic SDK.

The rest of prompt-forge never imports ``anthropic`` directly or talks to
the network. It depends only on the small ``complete(model, system, user)``
surface defined here, so any object with that method -- a real ``LLM``, a
scripted fake in a test, or a different provider's adapter -- can stand in
for it.
"""

from __future__ import annotations

from typing import Any, Protocol


class LLMProtocol(Protocol):
    """The minimal interface the optimizer and scorers depend on."""

    def complete(self, model: str, system: str, user: str) -> str:
        """Return the model's text response to a single-turn request."""
        ...


class LLM:
    """Synchronous wrapper around ``anthropic.Anthropic``.

    Instructions are folded into the user turn instead of passed through
    the API's native ``system`` parameter. This keeps every request a
    single, self-contained user message -- easier to log, easier to replay,
    and unaffected by proxies or gateways that don't forward ``system``
    correctly.

    Pass an already-constructed client (real or fake) via ``client`` to
    avoid touching the network, e.g. in tests.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client

    def complete(self, model: str, system: str, user: str, *, max_tokens: int = 1024) -> str:
        """Send a single-turn completion request and return its text."""
        combined = f"{system}\n\n{user}" if system else user
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": combined}],
        )
        return _extract_text(response)


def _extract_text(response: Any) -> str:
    """Concatenate the text blocks of an Anthropic Messages API response."""
    content = getattr(response, "content", None)
    if content is None:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)
