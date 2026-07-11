from __future__ import annotations

from prompt_forge.models import Example
from prompt_forge.optimizer import PROMPT_BLOCK_END, PROMPT_BLOCK_START, optimize
from prompt_forge.scorers import ExactMatchScorer
from tests.fakes import PROPOSER_MODEL, TASK_MODEL, FakeLLM

EXAMPLES = [
    Example(input="I love this movie", expected="positive"),
    Example(input="This movie was terrible", expected="negative"),
]


def _wrap(prompt_text: str) -> str:
    return f"{PROMPT_BLOCK_START}\n{prompt_text}\n{PROMPT_BLOCK_END}"


def test_optimizer_improves_score_across_rounds() -> None:
    # v0 (initial prompt) gets everything wrong. The proposer's round-1
    # candidate (v1) gets one example right; its round-2 candidate (v2)
    # gets both right. Each round should be kept because each one improves
    # on the last.
    def task_fn(user: str) -> str:
        if "PROMPT_V2" in user:
            return "positive" if "love" in user else "negative"
        if "PROMPT_V1" in user:
            return "positive" if "love" in user else "neutral"
        return "unknown"

    fake_llm = FakeLLM(
        task_fn=task_fn,
        proposer_responses=[
            _wrap("PROMPT_V1: Classify sentiment as positive or negative."),
            _wrap("PROMPT_V2: Classify sentiment as positive or negative, paying close attention to negative tone."),
        ],
    )

    result = optimize(
        "PROMPT_V0: Classify sentiment.",
        EXAMPLES,
        ExactMatchScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=5,
    )

    assert result.rounds == 2
    assert result.best_score == 1.0
    assert result.best_prompt == "PROMPT_V2: Classify sentiment as positive or negative, paying close attention to negative tone."
    # trace has round 0 (initial) plus one entry per round actually run
    assert [t.round for t in result.trace] == [0, 1, 2]
    assert [t.score for t in result.trace] == [0.0, 0.5, 1.0]
    # best_score strictly increased from the initial evaluation
    assert result.trace[0].score < result.trace[-1].score


def test_optimizer_stops_after_two_consecutive_no_improvement_rounds() -> None:
    # The task model gets everything wrong no matter which prompt version is
    # tried, so every round is a stall. The search should give up after
    # STALL_LIMIT (2) consecutive stalled rounds rather than burning through
    # every configured round.
    def task_fn(user: str) -> str:
        return "wrong answer"

    fake_llm = FakeLLM(
        task_fn=task_fn,
        proposer_responses=[
            _wrap("PROMPT_V1: try harder."),
            _wrap("PROMPT_V2: try even harder."),
            _wrap("PROMPT_V3: this should never be requested."),
        ],
    )

    result = optimize(
        "PROMPT_V0: Classify sentiment.",
        EXAMPLES,
        ExactMatchScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=5,
    )

    assert result.best_score == 0.0
    assert result.rounds == 2
    assert result.rounds < 5
    # only 2 proposer calls should have happened -- the 3rd scripted
    # response must never be requested once the stall limit is hit
    proposer_calls = [call for call in fake_llm.calls if call[0] == PROPOSER_MODEL]
    assert len(proposer_calls) == 2


def test_optimizer_survives_transient_proposer_error_and_keeps_climbing() -> None:
    # The proposer call itself blows up in round 1 (e.g. a transient API
    # error). That round counts as stalled, but the run must survive: round
    # 2's proposer call succeeds and its candidate is kept.
    def task_fn(user: str) -> str:
        if "PROMPT_V1" in user:
            return "positive" if "love" in user else "negative"
        return "unknown"

    fake_llm = FakeLLM(
        task_fn=task_fn,
        proposer_responses=[
            RuntimeError("simulated transient proposer failure"),
            _wrap("PROMPT_V1: Classify sentiment as positive or negative."),
        ],
    )

    result = optimize(
        "PROMPT_V0: Classify sentiment.",
        EXAMPLES,
        ExactMatchScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=5,
    )

    assert result.best_score == 1.0
    assert result.best_prompt == "PROMPT_V1: Classify sentiment as positive or negative."


def test_optimizer_returns_best_so_far_when_proposer_keeps_erroring() -> None:
    # Every proposer call fails. After STALL_LIMIT consecutive stalled
    # rounds the search stops and still returns the initial evaluation
    # instead of raising.
    fake_llm = FakeLLM(
        task_fn=lambda user: "positive" if "love" in user else "wrong",
        proposer_responses=[
            RuntimeError("proposer down"),
            RuntimeError("proposer still down"),
        ],
    )

    result = optimize(
        "PROMPT_V0: Classify sentiment.",
        EXAMPLES,
        ExactMatchScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=5,
    )

    assert result.best_prompt == "PROMPT_V0: Classify sentiment."
    assert result.best_score == 0.5
    # both scripted errors were consumed, then the stall limit stopped the run
    proposer_calls = [call for call in fake_llm.calls if call[0] == PROPOSER_MODEL]
    assert len(proposer_calls) == 2


def test_optimizer_reuses_previous_prompt_when_proposer_omits_delimiters() -> None:
    # A proposer response without the <<<PROMPT>>> markers falls back to the
    # previous prompt, so the round re-evaluates the same prompt, cannot
    # improve, and counts as stalled.
    fake_llm = FakeLLM(
        task_fn=lambda user: "positive" if "love" in user else "neutral",
        proposer_responses=[
            "Here is a much better prompt, trust me.",  # no delimiters
            "Still forgetting the delimiters.",
        ],
    )

    result = optimize(
        "PROMPT_V0: Classify sentiment.",
        EXAMPLES,
        ExactMatchScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=5,
    )

    assert result.best_prompt == "PROMPT_V0: Classify sentiment."
    assert all(t.prompt == "PROMPT_V0: Classify sentiment." for t in result.trace)
    # two consecutive fallback rounds stall out the search
    assert result.rounds == 2


def test_optimizer_scores_erroring_example_as_zero_without_crashing() -> None:
    def task_fn(user: str) -> str:
        if "TRIGGER_ERROR" in user:
            raise RuntimeError("simulated transient failure")
        return "positive"

    fake_llm = FakeLLM(task_fn=task_fn)

    examples = [
        Example(input="I love this movie", expected="positive"),
        Example(input="TRIGGER_ERROR this input blows up", expected="positive"),
    ]

    # max_rounds=0 isolates the initial evaluation pass -- no proposer calls
    # are scripted, so this only exercises _evaluate's error handling.
    result = optimize(
        "Classify sentiment.",
        examples,
        ExactMatchScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=0,
    )

    assert result.rounds == 0
    # one example scored 1.0, the erroring one scored 0.0 -> average 0.5
    assert result.best_score == 0.5
    assert len(result.trace[0].failures) == 1
    assert "TRIGGER_ERROR" in result.trace[0].failures[0]
