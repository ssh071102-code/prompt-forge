"""Tests for the LLM wrapper.

``LLM`` accepts any client object with a ``messages.create(**kwargs)`` method,
so these tests inject a small fake that records the request and returns a
canned response. Nothing here opens a socket, and nothing depends on which
HTTP library the installed ``anthropic`` SDK happens to use internally.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from prompt_forge.llm import LLM, FatalLLMError


class _FakeMessages:
    def __init__(self, text: str, stop_reason: str, error: Exception | None) -> None:
        self._text = text
        self._stop_reason = stop_reason
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
            stop_reason=self._stop_reason,
        )


class _FakeClient:
    def __init__(
        self, text: str = "ok", stop_reason: str = "end_turn", error: Exception | None = None
    ) -> None:
        self.messages = _FakeMessages(text, stop_reason, error)

    @property
    def last_request(self) -> dict[str, Any]:
        return self.messages.calls[-1]


def _api_status_error(cls: type[anthropic.APIStatusError], status: int) -> Exception:
    """Build an SDK status error without a real HTTP response object."""
    response = SimpleNamespace(request=None, status_code=status, headers={})
    body = {"type": "error", "error": {"type": "authentication_error", "message": "bad"}}
    return cls("bad", response=response, body=body)  # type: ignore[arg-type]


def test_llm_complete_extracts_text_from_response() -> None:
    llm = LLM(client=_FakeClient(text="hello world"))

    result = llm.complete(model="claude-haiku-4-5", system="Be terse.", user="Say hi.")

    assert result == "hello world"


def test_llm_complete_uses_native_system_param_by_default() -> None:
    client = _FakeClient()
    llm = LLM(client=client)

    llm.complete(model="claude-haiku-4-5", system="Be terse.", user="Say hi.")

    assert client.last_request["system"] == "Be terse."
    assert client.last_request["messages"] == [{"role": "user", "content": "Say hi."}]


def test_llm_complete_omits_system_param_when_system_is_empty() -> None:
    client = _FakeClient()
    llm = LLM(client=client)

    llm.complete(model="claude-haiku-4-5", system="", user="Just the user turn.")

    assert "system" not in client.last_request
    assert client.last_request["messages"] == [{"role": "user", "content": "Just the user turn."}]


def test_llm_fold_system_into_user_escape_hatch_prepends_to_user_turn() -> None:
    client = _FakeClient()
    llm = LLM(client=client, fold_system_into_user=True)

    llm.complete(model="claude-haiku-4-5", system="Be terse.", user="Say hi.")

    assert "system" not in client.last_request
    assert client.last_request["messages"] == [{"role": "user", "content": "Be terse.\n\nSay hi."}]


def test_llm_default_max_tokens_is_generous() -> None:
    # 1024 silently truncated long proposer outputs (losing the END_PROMPT
    # marker); the default must leave room for a full rewritten prompt, and
    # for current-generation models whose thinking counts against max_tokens.
    client = _FakeClient()
    LLM(client=client).complete(model="claude-haiku-4-5", system="", user="hi")

    assert client.last_request["max_tokens"] == 4096


def test_llm_max_tokens_is_configurable_per_instance() -> None:
    client = _FakeClient()
    LLM(client=client, max_tokens=8192).complete(model="claude-haiku-4-5", system="", user="hi")

    assert client.last_request["max_tokens"] == 8192


def test_llm_warns_when_response_is_truncated(caplog: pytest.LogCaptureFixture) -> None:
    llm = LLM(client=_FakeClient(text="cut off mid-", stop_reason="max_tokens"))

    with caplog.at_level("WARNING"):
        result = llm.complete(model="claude-haiku-4-5", system="", user="hi")

    assert result == "cut off mid-"
    assert any("truncated" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    ("error_cls", "status"),
    [
        (anthropic.AuthenticationError, 401),
        (anthropic.PermissionDeniedError, 403),
        (anthropic.NotFoundError, 404),
        (anthropic.BadRequestError, 400),
    ],
)
def test_llm_translates_deterministic_api_errors_to_fatal(
    error_cls: type[anthropic.APIStatusError], status: int
) -> None:
    # Deterministic API errors must abort the whole optimization run instead
    # of being scored 0.0 per example by the optimizer's transient-error net.
    llm = LLM(client=_FakeClient(error=_api_status_error(error_cls, status)))

    with pytest.raises(FatalLLMError):
        llm.complete(model="claude-haiku-4-5", system="", user="hi")


def test_llm_leaves_transient_errors_alone() -> None:
    # A rate limit or overload is worth retrying on the next example, so it
    # must reach the optimizer unchanged rather than being made fatal.
    llm = LLM(client=_FakeClient(error=_api_status_error(anthropic.RateLimitError, 429)))

    with pytest.raises(anthropic.RateLimitError):
        llm.complete(model="claude-haiku-4-5", system="", user="hi")


def test_llm_treats_missing_credentials_as_fatal() -> None:
    # With no API key configured, recent SDKs raise TypeError from inside
    # messages.create before any request is sent. Every call would fail the
    # same way, so it must abort the run instead of scoring 0.0 per example.
    missing_key = TypeError('"Could not resolve authentication method."')
    llm = LLM(client=_FakeClient(error=missing_key))

    with pytest.raises(FatalLLMError):
        llm.complete(model="claude-haiku-4-5", system="", user="hi")
