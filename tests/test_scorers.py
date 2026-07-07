from __future__ import annotations

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


def test_llm_judge_scorer_parses_plain_integer() -> None:
    fake_llm = FakeLLM(task_fn=lambda user: "8")
    scorer = LLMJudgeScorer(fake_llm, model="fake-task-model")
    example = Example(input="ignored", expected="a great review")

    assert scorer.score("a pretty great review", example) == 0.8


def test_llm_judge_scorer_parses_number_embedded_in_prose() -> None:
    fake_llm = FakeLLM(task_fn=lambda user: "I'd rate this a 7 out of 10.")
    scorer = LLMJudgeScorer(fake_llm, model="fake-task-model")
    example = Example(input="ignored", expected="expected answer")

    assert scorer.score("some output", example) == 0.7


def test_llm_judge_scorer_accepts_already_normalized_0_to_1_score() -> None:
    fake_llm = FakeLLM(task_fn=lambda user: "0.9")
    scorer = LLMJudgeScorer(fake_llm, model="fake-task-model")
    example = Example(input="ignored", expected="expected answer")

    assert scorer.score("some output", example) == 0.9


def test_llm_judge_scorer_defaults_to_zero_when_unparseable() -> None:
    fake_llm = FakeLLM(task_fn=lambda user: "I refuse to give a number.")
    scorer = LLMJudgeScorer(fake_llm, model="fake-task-model")
    example = Example(input="ignored", expected="expected answer")

    assert scorer.score("some output", example) == 0.0
