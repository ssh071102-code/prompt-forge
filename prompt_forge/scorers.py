"""Scorers rate how well a task output satisfies an example's expected answer.

A ``Scorer`` is any object with a ``score(output, example) -> float`` method
returning a value in ``[0, 1]``. The optimizer only depends on that
interface, so custom scorers (semantic similarity, unit tests, regex
checks, whatever the task calls for) can be dropped in without touching the
optimization loop.
"""

from __future__ import annotations

import re
from typing import Protocol

from prompt_forge.llm import LLMProtocol
from prompt_forge.models import Example


class Scorer(Protocol):
    """Rates a task output against an example's expected answer, in [0, 1]."""

    def score(self, output: str, example: Example) -> float:
        ...


class ExactMatchScorer:
    """Normalized exact/substring scorer.

    Both ``output`` and ``example.expected`` are lowercased and have
    whitespace collapsed before comparison:

    - 1.0 if they match exactly after normalization.
    - 0.7 if the expected answer appears as a substring of the output
      (e.g. the model wrapped the right answer in a sentence).
    - 0.0 otherwise.
    """

    PARTIAL_CREDIT = 0.7

    def score(self, output: str, example: Example) -> float:
        normalized_output = _normalize(output)
        normalized_expected = _normalize(example.expected)
        if not normalized_expected:
            return 0.0
        if normalized_output == normalized_expected:
            return 1.0
        if normalized_expected in normalized_output:
            return self.PARTIAL_CREDIT
        return 0.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class LLMJudgeScorer:
    """Asks an LLM to rate how well ``output`` satisfies ``example.expected``.

    The judge is instructed to answer with a single number, but real
    responses are messy ("I'd rate this a 7/10." or "0.85"), so the rating
    is extracted with a regex rather than a strict parse. Any value found
    above 1 is treated as a 0-10 score and rescaled; the result is always
    clamped to ``[0, 1]``.
    """

    JUDGE_INSTRUCTIONS = (
        "You are grading a model's output against an expected reference answer. "
        "Rate how well the output satisfies the expected answer on a scale of "
        "0 to 10, where 0 means completely wrong and 10 means it fully "
        "matches the expected answer in meaning. Respond with ONLY the "
        "number, nothing else."
    )

    def __init__(self, llm: LLMProtocol, *, model: str) -> None:
        self._llm = llm
        self._model = model

    def score(self, output: str, example: Example) -> float:
        user = (
            f"Expected answer:\n{example.expected}\n\n"
            f"Model output:\n{output}\n\n"
            "Rating (0-10):"
        )
        response = self._llm.complete(model=self._model, system=self.JUDGE_INSTRUCTIONS, user=user)
        return _parse_rating(response)


def _parse_rating(text: str) -> float:
    """Extract a numeric rating from free-form judge text, clamped to [0, 1]."""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    value = float(match.group())
    if value > 1.0:
        value = value / 10.0
    return max(0.0, min(1.0, value))
