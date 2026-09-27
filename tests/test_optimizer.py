from __future__ import annotations

import pytest

from prompt_forge.models import Example
from prompt_forge.optimizer import (
    MAX_FAILURE_SAMPLES,
    PROMPT_BLOCK_END,
    PROMPT_BLOCK_START,
    EvaluationFailedError,
    optimize,
    split_examples,
)
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

    improved_prompt = (
        "PROMPT_V2: Classify sentiment as positive or negative, "
        "paying close attention to negative tone."
    )
    fake_llm = FakeLLM(
        task_fn=task_fn,
        proposer_responses=[
            _wrap("PROMPT_V1: Classify sentiment as positive or negative."),
            _wrap(improved_prompt),
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
    assert result.best_prompt == improved_prompt
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
    assert result.trace[0].total_failures == 1
    assert "TRIGGER_ERROR" in result.trace[0].failures[0]


def test_optimizer_caps_trace_failures_but_records_true_count() -> None:
    # With more failing examples than MAX_FAILURE_SAMPLES, the trace stores
    # only a readable sample while total_failures keeps the real count.
    examples = [
        Example(input=f"example number {i}", expected="positive")
        for i in range(MAX_FAILURE_SAMPLES + 2)
    ]
    fake_llm = FakeLLM(task_fn=lambda user: "wrong answer")

    result = optimize(
        "Classify sentiment.",
        examples,
        ExactMatchScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=0,
    )

    assert len(result.trace[0].failures) == MAX_FAILURE_SAMPLES
    assert result.trace[0].total_failures == MAX_FAILURE_SAMPLES + 2


def test_noop_proposal_stalls_without_reevaluation() -> None:
    # Malformed proposer responses (no delimiters) fall back to the current
    # best prompt. Re-evaluating that prompt would spend one task call per
    # example re-measuring a known score -- and could accept the unchanged
    # prompt on sampling noise -- so the round must count as a stall with no
    # evaluation at all.
    fake_llm = FakeLLM(
        task_fn=lambda user: "wrong answer",
        proposer_responses=["no delimiters here", "still no delimiters"],
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

    # Only round 0 evaluated: one task call per example, nothing more.
    task_calls = [c for c in fake_llm.calls if c[0] == TASK_MODEL]
    assert len(task_calls) == len(EXAMPLES)
    # Both stalled rounds are recorded in the trace with the prompt unchanged.
    assert result.rounds == 2
    assert [t.round for t in result.trace] == [0, 1, 2]
    assert result.trace[1].prompt == result.trace[0].prompt
    assert result.trace[1].score == result.trace[0].score


def test_uniform_midrange_scores_keep_optimizing() -> None:
    # A scorer that never drops below the failure threshold used to end the
    # run at round 0 ("no failures to learn from") even at a mediocre 0.7
    # average. The optimizer must keep going, showing the proposer its
    # weakest examples instead.
    class MidrangeScorer:
        def score(self, output: str, example: Example) -> float:
            return 1.0 if output == "great answer" else 0.7

    def task_fn(user: str) -> str:
        return "great answer" if "PROMPT_V1" in user else "mediocre answer"

    fake_llm = FakeLLM(
        task_fn=task_fn,
        proposer_responses=[_wrap("PROMPT_V1: be better.")],
    )

    result = optimize(
        "PROMPT_V0: Classify sentiment.",
        EXAMPLES,
        MidrangeScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=5,
    )

    assert result.best_score == 1.0
    assert result.rounds == 1
    # The proposer saw the weakest examples annotated with their scores.
    proposer_calls = [c for c in fake_llm.calls if c[0] == PROPOSER_MODEL]
    assert len(proposer_calls) == 1
    assert "(scored 0.70)" in proposer_calls[0][2]


def test_fatal_llm_error_aborts_the_run() -> None:
    # A deterministic API failure (bad key, unknown model) fails every call;
    # scoring it 0.0 per example would report an all-zero "success".
    import pytest

    from prompt_forge.llm import FatalLLMError

    fake_llm = FakeLLM(task_fn=lambda user: (_ for _ in ()).throw(FatalLLMError("bad key")))

    with pytest.raises(FatalLLMError):
        optimize(
            "PROMPT_V0: Classify sentiment.",
            EXAMPLES,
            ExactMatchScorer(),
            fake_llm,
            task_model=TASK_MODEL,
            proposer_model=PROPOSER_MODEL,
        )


def test_fatal_proposer_error_aborts_the_run() -> None:
    import pytest

    from prompt_forge.llm import FatalLLMError

    fake_llm = FakeLLM(
        task_fn=lambda user: "wrong answer",
        proposer_responses=[FatalLLMError("model not found")],
    )

    with pytest.raises(FatalLLMError):
        optimize(
            "PROMPT_V0: Classify sentiment.",
            EXAMPLES,
            ExactMatchScorer(),
            fake_llm,
            task_model=TASK_MODEL,
            proposer_model=PROPOSER_MODEL,
        )


def test_proposer_error_round_is_recorded_in_trace() -> None:
    # A transient proposer failure costs a round; the trace must record it
    # (prompt and score unchanged) so round numbers stay contiguous and
    # Result.rounds counts every round attempted.
    def task_fn(user: str) -> str:
        if "PROMPT_V1" in user:
            return "positive" if "love" in user else "negative"
        return "unknown"

    fake_llm = FakeLLM(
        task_fn=task_fn,
        proposer_responses=[RuntimeError("transient boom"), _wrap("PROMPT_V1: improved.")],
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
    assert result.rounds == 2
    assert [t.round for t in result.trace] == [0, 1, 2]
    # Round 1 is the stalled round: unchanged prompt, unchanged score.
    assert result.trace[1].prompt == result.trace[0].prompt
    assert result.trace[1].score == result.trace[0].score
    assert result.trace[2].score == 1.0


def test_every_example_erroring_on_the_initial_prompt_aborts() -> None:
    # A misconfiguration that is not recognised as fatal (for example a
    # custom provider raising a plain exception for a missing key) makes
    # every example fail. That must stop the run, not report 0.00.
    def task_fn(user: str) -> str:
        raise RuntimeError("no credentials")

    with pytest.raises(EvaluationFailedError, match="all 2 examples errored"):
        optimize(
            "PROMPT_V0",
            EXAMPLES,
            ExactMatchScorer(),
            FakeLLM(task_fn=task_fn),
            task_model=TASK_MODEL,
            proposer_model=PROPOSER_MODEL,
        )


def test_split_examples_is_deterministic_and_disjoint() -> None:
    examples = [Example(input=f"in {i}", expected=str(i)) for i in range(10)]

    train, val = split_examples(examples, 0.3, seed=7)
    train_again, val_again = split_examples(examples, 0.3, seed=7)

    assert (train, val) == (train_again, val_again)
    assert len(train) == 7 and len(val) == 3
    assert {e.input for e in train}.isdisjoint({e.input for e in val})
    assert sorted(e.input for e in train + val) == sorted(e.input for e in examples)


def test_split_examples_keeps_at_least_one_on_each_side() -> None:
    examples = [Example(input=f"in {i}", expected=str(i)) for i in range(3)]

    train, val = split_examples(examples, 0.05)
    assert (len(train), len(val)) == (2, 1)
    train, val = split_examples(examples, 0.9)
    assert (len(train), len(val)) == (1, 2)


def test_split_examples_zero_fraction_returns_everything_as_train() -> None:
    assert split_examples(EXAMPLES, 0.0) == (EXAMPLES, [])


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_split_examples_rejects_out_of_range_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        split_examples(EXAMPLES, fraction)


def test_validation_set_is_scored_but_never_shown_to_the_proposer() -> None:
    # The candidate is chosen on the training example only. The validation
    # example is scored for the initial and the final prompt after the
    # search, and never appears in a proposer request.
    def task_fn(user: str) -> str:
        if "PROMPT_V1" in user:
            return "positive" if "love" in user else "negative"
        return "unknown"

    train = [Example(input="I love this movie", expected="positive")]
    validation = [Example(input="VALIDATION_ONLY terrible film", expected="negative")]
    fake_llm = FakeLLM(task_fn=task_fn, proposer_responses=[_wrap("PROMPT_V1: better.")])

    result = optimize(
        "PROMPT_V0",
        train,
        ExactMatchScorer(),
        fake_llm,
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        validation=validation,
    )

    assert result.best_score == 1.0
    assert result.initial_val_score == 0.0
    assert result.val_score == 1.0
    proposer_text = " ".join(c[2] for c in fake_llm.calls if c[0] == PROPOSER_MODEL)
    assert "VALIDATION_ONLY" not in proposer_text
    # Round 0 and round 1 on train, then initial and best prompt on validation.
    assert len([c for c in fake_llm.calls if c[0] == TASK_MODEL]) == 4


def test_no_validation_leaves_val_scores_empty() -> None:
    result = optimize(
        "PROMPT_V0",
        EXAMPLES,
        ExactMatchScorer(),
        FakeLLM(task_fn=lambda user: "positive"),
        task_model=TASK_MODEL,
        proposer_model=PROPOSER_MODEL,
        max_rounds=0,
    )

    assert result.val_score is None
    assert result.initial_val_score is None
