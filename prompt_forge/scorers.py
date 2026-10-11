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
    - 0.7 if the expected answer appears as a whole word/phrase in the output
      (e.g. the model wrapped the right answer in a sentence). The match is
      word-bounded, so "cat" gets no credit for "concatenate", and a directly
      negated answer ("not positive" for expected "positive") gets no credit.
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
        if _contains_answer(normalized_output, normalized_expected):
            return self.PARTIAL_CREDIT
        return 0.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# A negator directly (or one word) ahead of the expected answer flips its
# meaning: "not positive", "never spam", "isn't a cat". Partial credit for
# those would hide systematically inverted answers from the optimizer.
_NEGATORS = (
    r"(?:not|no|never|neither|nor|isn't|aren't|wasn't|don't|doesn't|didn't|cannot|can't|won't)"
)


def _contains_answer(output: str, expected: str) -> bool:
    """Whole-word containment that refuses credit for a negated match."""
    escaped = re.escape(expected)
    bounded = re.compile(rf"(?<!\w){escaped}(?!\w)")
    if not bounded.search(output):
        return False
    negated = re.compile(rf"\b{_NEGATORS}\s+(?:\w+\s+)?{escaped}(?!\w)")
    if negated.search(output) and len(bounded.findall(output)) == len(negated.findall(output)):
        return False
    return True


class LLMJudgeScorer:
    """Asks an LLM to rate how well ``output`` satisfies ``example.expected``.

    The judge is instructed to answer with a single number on a 0-10 scale,
    but real responses are messy ("I'd rate this a 7/10."), so the rating is
    extracted with a regex rather than a strict parse. Explicit rating forms
    (``X/10``, ``X of 10``, ``X out of 10``, ``Score: X``, ``Rating: X``) are
    preferred; otherwise the last bare number is read on the instructed 0-10
    scale and divided by 10. The result is always clamped to ``[0, 1]``.
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


# "on a scale of 0 to 10", "between 0 and 10", "0-10", "out of 10": scale
# preamble a chatty judge echoes before its verdict. Its numbers are not the
# rating and must not be picked up by the fallback number search.
_SCALE_MENTION = re.compile(
    r"(?:scale\s+(?:of|from)\s+)?-?\d+(?:\.\d+)?\s*(?:to|and|-|–)\s*10\b|\bout\s+of\s+10\b",
    re.IGNORECASE,
)


# "7/10", "7 of 10", "7 out of 10": the rating stated as a fraction of 10.
_FRACTION = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:/|\bout\s+of\b|\bof\b)\s*10\b", re.IGNORECASE)

# "Score: 8", "Rating = 6", "score is 7": a labelled rating. Searched after the
# scale preamble is stripped, so "Score of 0 to 10: 7" does not read as 0.
_LABELLED = re.compile(r"\b(?:score|rating)\s*(?::|=|\bis\b)?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_rating(text: str) -> float:
    """Extract a 0-10 judge rating from free-form text, normalized to [0, 1].

    Explicit rating forms are read first, in this order: a fraction of 10
    (``X/10``, ``X of 10``, ``X out of 10``), then a labelled rating
    (``Score: X``, ``Rating: X``). Numbers after the rating ("Score: 8.
    There are 2 minor errors.") are therefore not mistaken for it. Without
    an explicit form, scale-preamble phrases ("on a scale of 0 to 10, ...")
    are stripped and the LAST remaining number is read as the verdict --
    chatty judges state the scale before the rating, so the first number in
    the raw text is often the scale's lower bound, not the answer. A bare
    number is read on the instructed 0-10 scale and divided by 10 -- a bare
    "1" means 1/10 (near-total failure), never a perfect normalized score.
    The result is clamped to ``[0, 1]``.
    """
    fraction = _FRACTION.search(text)
    if fraction:
        value = float(fraction.group(1))
    else:
        stripped = _SCALE_MENTION.sub(" ", text)
        labelled = _LABELLED.search(stripped)
        if labelled:
            value = float(labelled.group(1))
        else:
            numbers = _NUMBER.findall(stripped)
            if not numbers:
                return 0.0
            value = float(numbers[-1])
    return max(0.0, min(1.0, value / 10.0))
