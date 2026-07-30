"""Tests for the LLM wrapper.

These use respx to mock the Anthropic Messages endpoint at the HTTP layer,
so the real ``anthropic.Anthropic`` client and request/response handling is
exercised without ever touching the network.
"""

from __future__ import annotations

import json

import anthropic
import httpx
import respx

from prompt_forge.llm import LLM

MESSAGES_URL = "https://api.anthropic.com/v1/messages"


def _mock_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


@respx.mock
def test_llm_complete_extracts_text_from_response() -> None:
    respx.post(MESSAGES_URL).mock(return_value=_mock_response("hello world"))
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client)

    result = llm.complete(model="claude-haiku-4-5", system="Be terse.", user="Say hi.")

    assert result == "hello world"


@respx.mock
def test_llm_complete_uses_native_system_param_by_default() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=_mock_response("ok"))
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client)

    llm.complete(model="claude-haiku-4-5", system="Be terse.", user="Say hi.")

    request_body = json.loads(route.calls.last.request.content)
    assert request_body["system"] == "Be terse."
    assert request_body["messages"] == [{"role": "user", "content": "Say hi."}]


@respx.mock
def test_llm_complete_omits_system_param_when_system_is_empty() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=_mock_response("ok"))
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client)

    llm.complete(model="claude-haiku-4-5", system="", user="Just the user turn.")

    request_body = json.loads(route.calls.last.request.content)
    assert "system" not in request_body
    assert request_body["messages"] == [{"role": "user", "content": "Just the user turn."}]


@respx.mock
def test_llm_fold_system_into_user_escape_hatch_prepends_to_user_turn() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=_mock_response("ok"))
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client, fold_system_into_user=True)

    llm.complete(model="claude-haiku-4-5", system="Be terse.", user="Say hi.")

    request_body = json.loads(route.calls.last.request.content)
    assert "system" not in request_body
    assert request_body["messages"] == [{"role": "user", "content": "Be terse.\n\nSay hi."}]


@respx.mock
def test_llm_default_max_tokens_is_generous() -> None:
    # 1024 silently truncated long proposer outputs (losing the END_PROMPT
    # marker); the default must leave room for a full rewritten prompt, and
    # for current-generation models whose thinking counts against max_tokens.
    route = respx.post(MESSAGES_URL).mock(return_value=_mock_response("ok"))
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client)

    llm.complete(model="claude-haiku-4-5", system="", user="hi")

    body = json.loads(route.calls.last.request.content)
    assert body["max_tokens"] == 4096


@respx.mock
def test_llm_max_tokens_is_configurable_per_instance() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=_mock_response("ok"))
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client, max_tokens=8192)

    llm.complete(model="claude-haiku-4-5", system="", user="hi")

    body = json.loads(route.calls.last.request.content)
    assert body["max_tokens"] == 8192


@respx.mock
def test_llm_warns_when_response_is_truncated(caplog) -> None:
    truncated = httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "cut off mid-"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
    respx.post(MESSAGES_URL).mock(return_value=truncated)
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client)

    with caplog.at_level("WARNING"):
        result = llm.complete(model="claude-haiku-4-5", system="", user="hi")

    assert result == "cut off mid-"
    assert any("truncated" in record.message for record in caplog.records)


@respx.mock
def test_llm_translates_auth_error_to_fatal() -> None:
    # Deterministic API errors must abort the whole optimization run instead
    # of being scored 0.0 per example by the optimizer's transient-error net.
    import pytest

    from prompt_forge.llm import FatalLLMError

    respx.post(MESSAGES_URL).mock(
        return_value=httpx.Response(
            401,
            json={"type": "error", "error": {"type": "authentication_error", "message": "bad key"}},
        )
    )
    client = anthropic.Anthropic(api_key="bad-key")
    llm = LLM(client=client)

    with pytest.raises(FatalLLMError):
        llm.complete(model="claude-haiku-4-5", system="", user="hi")
