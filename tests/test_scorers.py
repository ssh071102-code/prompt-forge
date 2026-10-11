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

    score = scorer.score("I'd call this positive overall.", example)
    assert score == ExactMatchScorer.PARTIAL_CREDIT


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


def test_exact_match_scorer_no_credit_for_embedded_word() -> None:
    scorer = ExactMatchScorer()

    assert scorer.score("concatenate", Example(input="i", expected="cat")) == 0.0
    assert scorer.score("yesterday it rained", Example(input="i", expected="yes")) == 0.0


def test_exact_match_scorer_no_credit_for_negated_answer() -> None:
    scorer = ExactMatchScorer()
    example = Example(input="ignored", expected="positive")

    assert scorer.score("not positive", example) == 0.0
    assert scorer.score("This is definitely not positive.", example) == 0.0
    # A plain wrapped answer still earns partial credit.
    assert scorer.score("The tone is positive.", example) == ExactMatchScorer.PARTIAL_CREDIT


@pytest.mark.parametrize(
    ("judge_text", "expected_score"),
    [
        # The scale preamble's numbers must not be mistaken for the verdict.
        ("On a scale of 0 to 10, I'd say 7", 0.7),
        ("Between 0 and 10 this deserves a 9", 0.9),
        ("I would rate it 7 out of 10", 0.7),
        # An explicit fraction still wins outright.
        ("On a scale of 0 to 10: 6/10", 0.6),
        # Explicit rating forms win over numbers that come after them.
        ("7 of 10", 0.7),
        ("Score: 8. There are 2 minor errors.", 0.8),
        ("Rating: 6 -- 3 issues found", 0.6),
        ("8 out of 10, with 2 small mistakes", 0.8),
    ],
)
def test_judge_scorer_ignores_scale_preamble(judge_text: str, expected_score: float) -> None:
    llm = FakeLLM(task_fn=lambda user: judge_text)
    scorer = LLMJudgeScorer(llm, model="fake-task-model")

    score = scorer.score("some output", Example(input="i", expected="e"))
    assert score == pytest.approx(expected_score)
