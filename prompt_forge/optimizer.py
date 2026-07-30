"""The hill-climbing optimization loop.

``optimize`` runs the current best prompt over a set of labeled examples,
scores the outputs, and hands the failures to an LLM "proposer" that tries
to write a better prompt. The candidate is kept only if it scores
meaningfully higher than the current best; the search stops once it runs
out of rounds or stalls for two rounds in a row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from prompt_forge.llm import FatalLLMError, LLMProtocol
from prompt_forge.models import Example, Result, RoundTrace
from prompt_forge.scorers import Scorer

logger = logging.getLogger(__name__)

# An example scoring below this is considered a failure worth showing to the proposer.
FAILURE_THRESHOLD = 0.5

# An average score at or above this means there is nothing meaningful left to
# learn from -- the search stops. Below it, even a run with zero sub-threshold
# failures keeps optimizing, learning from its weakest examples instead
# (an LLM judge that scores 6-8/10 everywhere never produces a "failure",
# but a 0.7 average is nowhere near done).
PERFECT_SCORE = 0.99

# How many failures to show the proposer per round -- enough to find a pattern
# without blowing the prompt budget or overfitting to one bad example.
MAX_FAILURE_SAMPLES = 5

# Number of consecutive no-improvement rounds before the search gives up.
STALL_LIMIT = 2

# Delimiters the proposer is asked to wrap its candidate prompt in, so it can
# be extracted reliably even if the model adds commentary around it.
PROMPT_BLOCK_START = "<<<PROMPT>>>"
PROMPT_BLOCK_END = "<<<END_PROMPT>>>"


@dataclass
class _Evaluation:
    score: float
    failures: list[str]
    weakest: list[str]

    @property
    def learn_samples(self) -> list[str]:
        """What to show the proposer: failures, else the weakest examples."""
        return self.failures if self.failures else self.weakest


def _run_example(llm: LLMProtocol, task_model: str, prompt: str, example: Example) -> str:
    """Run ``prompt`` on a single example's input via the task model."""
    user = f"{prompt}\n\nInput:\n{example.input}"
    return llm.complete(model=task_model, system="", user=user)


def _evaluate(
    llm: LLMProtocol,
    task_model: str,
    prompt: str,
    examples: list[Example],
    scorer: Scorer,
) -> _Evaluation:
    """Run ``prompt`` over every example and score each output.

    An example that raises a *transient* error during generation or scoring is
    scored 0.0 and recorded as a failure rather than crashing the run -- a
    single flaky example (a timeout, a malformed response) should never take
    down an entire optimization pass. A ``FatalLLMError`` (bad API key,
    unknown model) is deterministic and re-raised immediately: scoring it 0.0
    would silently turn every run into an all-zero "success".
    """
    scores: list[float] = []
    failures: list[str] = []
    scored_lines: list[tuple[float, str]] = []
    for example in examples:
        try:
            output = _run_example(llm, task_model, prompt, example)
            example_score = scorer.score(output, example)
        except FatalLLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - isolate per-example failures
            logger.warning("Example errored during evaluation, scoring 0.0: %s", exc)
            output = f"<error: {exc}>"
            example_score = 0.0
        scores.append(example_score)
        line = f"input: {example.input!r} | expected: {example.expected!r} | got: {output!r}"
        if example_score < FAILURE_THRESHOLD:
            failures.append(line)
        elif example_score < PERFECT_SCORE:
            scored_lines.append((example_score, f"{line} (scored {example_score:.2f})"))
    scored_lines.sort(key=lambda pair: pair[0])
    weakest = [line for _, line in scored_lines[:MAX_FAILURE_SAMPLES]]
    average = sum(scores) / len(scores) if scores else 0.0
    return _Evaluation(score=average, failures=failures, weakest=weakest)


def _build_proposer_prompt(current_prompt: str, failures: list[str]) -> str:
    """Build the request asking the proposer model for an improved prompt."""
    sample = failures[:MAX_FAILURE_SAMPLES]
    failure_block = "\n".join(f"- {failure}" for failure in sample) or "- (no recorded failures)"
    return (
        "You are improving a prompt that instructs another language model to "
        "perform a task.\n\n"
        "Current prompt:\n"
        f"{current_prompt}\n\n"
        "The current prompt produced these failures on held-out examples:\n"
        f"{failure_block}\n\n"
        "Write an improved version of the prompt that addresses these "
        "failures while remaining general -- it must not hardcode answers "
        "to the specific examples above, only fix the underlying "
        "instruction. Return ONLY the improved prompt, wrapped between "
        f"{PROMPT_BLOCK_START} and {PROMPT_BLOCK_END} markers, with nothing "
        "else outside the markers."
    )


def _parse_proposed_prompt(response: str, fallback: str) -> str:
    """Extract the improved prompt from the proposer's delimited block.

    Falls back to the previous prompt (a no-op proposal) if the markers are
    missing or malformed, so a badly formatted proposer response never
    crashes the run -- it just fails to improve that round.
    """
    start = response.find(PROMPT_BLOCK_START)
    end = response.find(PROMPT_BLOCK_END)
    if start == -1 or end == -1 or end <= start:
        logger.warning("Proposer response missing delimiters; reusing previous prompt.")
        return fallback
    extracted = response[start + len(PROMPT_BLOCK_START) : end].strip()
    return extracted or fallback


def _trace_round(round_number: int, prompt: str, evaluation: _Evaluation) -> RoundTrace:
    """Build a trace entry, capping stored failures at ``MAX_FAILURE_SAMPLES``.

    The full failure list can be as long as the example set; the trace keeps
    a readable sample plus the true count in ``total_failures``.
    """
    return RoundTrace(
        round=round_number,
        prompt=prompt,
        score=evaluation.score,
        failures=evaluation.failures[:MAX_FAILURE_SAMPLES],
        total_failures=len(evaluation.failures),
    )


def optimize(
    initial_prompt: str,
    examples: list[Example],
    scorer: Scorer,
    llm: LLMProtocol,
    *,
    task_model: str,
    proposer_model: str,
    max_rounds: int = 5,
    min_gain: float = 0.01,
) -> Result:
    """Hill-climb ``initial_prompt`` toward a higher average score.

    Round 0 evaluates ``initial_prompt`` as-is. Each following round:

    1. Shows the proposer model the current best prompt and a sample of its
       failures on ``examples``.
    2. Asks it to propose an improved prompt.
    3. Evaluates the candidate on the same examples.
    4. Keeps the candidate only if it beats the best score by at least
       ``min_gain``; otherwise counts a stalled round.

    The search stops after ``max_rounds`` rounds, after ``STALL_LIMIT``
    consecutive rounds without improvement, or as soon as the average score
    reaches ``PERFECT_SCORE`` -- whichever comes first. A run with no
    sub-threshold failures but a mediocre average keeps going, showing the
    proposer its weakest examples instead.

    Rounds that produce no candidate to evaluate -- a failed proposer call, or
    a malformed response with no prompt delimiters -- are recorded in the
    trace as stalled rounds (prompt and score unchanged) rather than silently
    skipped, so trace round numbers stay contiguous and ``Result.rounds``
    counts every round attempted.
    """
    evaluation = _evaluate(llm, task_model, initial_prompt, examples, scorer)
    best_prompt = initial_prompt
    best_score = evaluation.score
    trace = [_trace_round(0, initial_prompt, evaluation)]

    stalled_rounds = 0

    def _stall(round_number: int) -> bool:
        """Record a stalled round in the trace; True means stop the search."""
        nonlocal stalled_rounds
        trace.append(_trace_round(round_number, best_prompt, evaluation))
        stalled_rounds += 1
        if stalled_rounds >= STALL_LIMIT:
            logger.info("No improvement for %d consecutive rounds; stopping.", stalled_rounds)
            return True
        return False

    for round_number in range(1, max_rounds + 1):
        samples = evaluation.learn_samples
        if not samples:
            logger.info("Score %.2f leaves nothing to learn from; stopping early.", best_score)
            break

        proposer_prompt = _build_proposer_prompt(best_prompt, samples)
        try:
            proposer_response = llm.complete(model=proposer_model, system="", user=proposer_prompt)
        except FatalLLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - a flaky proposer must not kill the run
            # Mirror the per-example isolation in _evaluate: one transient
            # proposer error costs a round, not the whole run.
            logger.warning("Proposer call failed in round %d: %s", round_number, exc)
            if _stall(round_number):
                break
            continue
        candidate_prompt = _parse_proposed_prompt(proposer_response, fallback=best_prompt)

        if candidate_prompt == best_prompt:
            # A no-op proposal (missing delimiters, or the proposer returned
            # the prompt unchanged). Re-evaluating it would spend one task
            # call per example re-measuring a known score -- and, since
            # sampling is stochastic, could "beat" best_score on noise alone
            # and accept the unchanged prompt as an improvement.
            logger.warning(
                "Round %d produced no new prompt; counting a stalled round.", round_number
            )
            if _stall(round_number):
                break
            continue

        candidate_evaluation = _evaluate(llm, task_model, candidate_prompt, examples, scorer)
        trace.append(_trace_round(round_number, candidate_prompt, candidate_evaluation))

        if candidate_evaluation.score - best_score >= min_gain:
            best_prompt = candidate_prompt
            best_score = candidate_evaluation.score
            evaluation = candidate_evaluation
            stalled_rounds = 0
        else:
            stalled_rounds += 1
            if stalled_rounds >= STALL_LIMIT:
                logger.info("No improvement for %d consecutive rounds; stopping.", stalled_rounds)
                break

    return Result(
        best_prompt=best_prompt, best_score=best_score, rounds=len(trace) - 1, trace=trace
    )
