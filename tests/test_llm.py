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
def test_llm_complete_folds_system_into_user_turn_not_native_system_param() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=_mock_response("ok"))
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client)

    llm.complete(model="claude-haiku-4-5", system="Be terse.", user="Say hi.")

    request_body = json.loads(route.calls.last.request.content)
    assert "system" not in request_body
    assert request_body["messages"] == [{"role": "user", "content": "Be terse.\n\nSay hi."}]


@respx.mock
def test_llm_complete_with_empty_system_sends_user_text_unchanged() -> None:
    route = respx.post(MESSAGES_URL).mock(return_value=_mock_response("ok"))
    client = anthropic.Anthropic(api_key="test-key")
    llm = LLM(client=client)

    llm.complete(model="claude-haiku-4-5", system="", user="Just the user turn.")

    request_body = json.loads(route.calls.last.request.content)
    assert request_body["messages"] == [{"role": "user", "content": "Just the user turn."}]
