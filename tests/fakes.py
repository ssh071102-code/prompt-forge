"""A scripted LLM test double used across the test suite.

FakeLLM implements the same ``complete(model, system, user)`` surface as
``prompt_forge.llm.LLM`` (see ``LLMProtocol``), so it can be passed anywhere
a real LLM is expected -- the optimizer and scorers never know the
difference. This keeps every test in the suite fully offline.
"""

from __future__ import annotations

from collections.abc import Callable

TASK_MODEL = "fake-task-model"
PROPOSER_MODEL = "fake-proposer-model"


class FakeLLM:
    """Dispatches to a task responder or a scripted list of proposer responses.

    - ``task_fn(user_text) -> str`` decides the task model's output. It is
      given the full user turn (prompt + input), so tests can key behavior
      off substrings of either.
    - ``proposer_responses`` is a fixed list of responses returned in order,
      one per call to the proposer model -- this is what lets a test script
      "round 2's proposed prompt scores higher than round 1's".

    Every call is recorded in ``calls`` for tests that want to assert on
    call counts or exact arguments.
    """

    def __init__(
        self,
        task_fn: Callable[[str], str],
        proposer_responses: list[str] | None = None,
    ) -> None:
        self.task_fn = task_fn
        self.proposer_responses = list(proposer_responses or [])
        self._proposer_call_count = 0
        self.calls: list[tuple[str, str, str]] = []

    def complete(self, model: str, system: str, user: str) -> str:
        self.calls.append((model, system, user))
        if model == PROPOSER_MODEL:
            if self._proposer_call_count >= len(self.proposer_responses):
                raise AssertionError("FakeLLM proposer called more times than scripted.")
            response = self.proposer_responses[self._proposer_call_count]
            self._proposer_call_count += 1
            return response
        return self.task_fn(user)
