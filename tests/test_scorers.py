from __future__ import annotations

import pytest

from prompt_forge.models import Example
from prompt_forge.scorers import ExactMatchScorer, LLMJudgeScorer
from tests.fakes import FakeLLM


def test_exact_match_scorer_full_score_on_exact_match() -> None:
    scorer = ExactMatchScorer()
    example = Example(input="ignored", expected="positive")

    assert scorer.score("positive", example) == 1.0


def test_exact_match_scorer_is_case_and_whitespace_insensitive() -> None:
    scorer = ExactMatchScorer()
    example = Example(input="ignored", expected="positive")

    assert scorer.score("  Positive \n", example) == 1.0


def test_exact_match_scorer_partial_credit_on_substring() -> None:
    scorer = ExactMatchScorer()
    example = Example(input="ignored", expected="positive")

    assert scorer.score("I'd call this positive overall.", example) == ExactMatchScorer.PARTIAL_CREDIT


def test_exact_match_scorer_zero_on_miss() -> None:
    scorer = ExactMatchScorer()
    example = Example(input="ignored", expected="positive")

    assert scorer.score("negative", example) == 0.0


@pytest.mark.parametrize(
    ("judge_text", "expected_score"),
    [
        ("8", 0.8),
        ("1", 0.1),  # bare "1" means 1/10 -- near-total failure, never a perfect score
        ("10", 1.0),
        ("7/10", 0.7),
        ("I'd rate this a 7/10.", 0.7),
        ("I'd rate this a 7 out of 10.", 0.7),
        ("0.85", 0.085),  # bare numbers are read on the instructed 0-10 scale
        ("-3", 0.0),  # clamped to the bottom of the scale
        ("12", 1.0),  # clamped to the top of the scale
        ("I refuse to give a number.", 0.0),
    ],
)
def test_llm_judge_scorer_normalizes_rating_to_unit_interval(
    judge_text: str, expected_score: float
) -> None:
    fake_llm = FakeLLM(task_fn=lambda user: judge_text)
    scorer = LLMJudgeScorer(fake_llm, model="fake-task-model")
    example = Example(input="ignored", expected="expected answer")

    assert scorer.score("some output", example) == pytest.approx(expected_score)
