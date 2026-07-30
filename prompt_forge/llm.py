"""A thin, swappable wrapper around the Anthropic SDK.

The rest of prompt-forge never imports ``anthropic`` directly or talks to
the network. It depends only on the small ``complete(model, system, user)``
surface defined here, so any object with that method -- a real ``LLM``, a
scripted fake in a test, or a different provider's adapter -- can stand in
for it.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class FatalLLMError(RuntimeError):
    """A deterministic, non-transient LLM failure (bad key, unknown model, ...).

    Providers should raise this (or let ``LLM`` translate into it) for errors
    that will fail on *every* call. The optimizer re-raises it immediately
    instead of scoring the example 0.0, so a typo'd model ID or revoked key
    aborts the run loudly rather than producing an all-zero "success".
    """


class LLMProtocol(Protocol):
    """The minimal interface the optimizer and scorers depend on."""

    def complete(self, model: str, system: str, user: str) -> str:
        """Return the model's text response to a single-turn request."""
        ...


class LLM:
    """Synchronous wrapper around ``anthropic.Anthropic``.

    By default, system instructions are sent through the API's native
    ``system`` parameter, which is how the models are trained to receive
    them. ``fold_system_into_user=True`` is a compatibility escape hatch
    for API gateways that strip or drop the ``system`` field: it prepends
    the system text to the user message instead, so no instruction is
    silently lost in transit.

    Pass an already-constructed client (real or fake) via ``client`` to
    avoid touching the network, e.g. in tests.

    ``max_tokens`` (default 4096) bounds every completion this instance makes.
    Size it generously: the optimizer's proposer returns an entire rewritten
    prompt, and on current-generation Opus models thinking is on by default
    and counts against ``max_tokens``. A truncated response is logged as a
    warning (checked via ``stop_reason``) instead of failing silently.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        fold_system_into_user: bool = False,
        max_tokens: int = 4096,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self._fold_system_into_user = fold_system_into_user
        self._max_tokens = max_tokens

    def complete(self, model: str, system: str, user: str, *, max_tokens: int | None = None) -> str:
        """Send a single-turn completion request and return its text."""
        kwargs: dict[str, Any] = {}
        if self._fold_system_into_user:
            user = f"{system}\n\n{user}" if system else user
        elif system:
            kwargs["system"] = system
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
                messages=[{"role": "user", "content": user}],
                **kwargs,
            )
        except _fatal_error_types() as exc:
            raise FatalLLMError(f"non-transient API error for model {model!r}: {exc}") from exc
        if getattr(response, "stop_reason", None) == "max_tokens":
            logger.warning(
                "Response from %s truncated at max_tokens=%d; raise LLM(max_tokens=...) "
                "if prompts are being cut off.",
                model,
                max_tokens if max_tokens is not None else self._max_tokens,
            )
        return _extract_text(response)


def _fatal_error_types() -> tuple[type[Exception], ...]:
    """Anthropic error classes that are deterministic, not transient.

    Imported lazily so fakes and alternate providers never need ``anthropic``.
    """
    try:
        import anthropic
    except ImportError:  # pragma: no cover - anthropic is a declared dependency
        return ()
    return (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.BadRequestError,
    )


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
