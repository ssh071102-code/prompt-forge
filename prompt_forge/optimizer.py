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

from prompt_forge.llm import LLMProtocol
from prompt_forge.models import Example, Result, RoundTrace
from prompt_forge.scorers import Scorer

logger = logging.getLogger(__name__)

# An example scoring below this is considered a failure worth showing to the proposer.
FAILURE_THRESHOLD = 0.5

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

    An example that raises during generation or scoring is scored 0.0 and
    recorded as a failure rather than crashing the run -- a single flaky
    example (a timeout, a malformed response) should never take down an
    entire optimization pass.
    """
    scores: list[float] = []
    failures: list[str] = []
    for example in examples:
        try:
            output = _run_example(llm, task_model, prompt, example)
            example_score = scorer.score(output, example)
        except Exception as exc:  # noqa: BLE001 - isolate per-example failures
            logger.warning("Example errored during evaluation, scoring 0.0: %s", exc)
            output = f"<error: {exc}>"
            example_score = 0.0
        scores.append(example_score)
        if example_score < FAILURE_THRESHOLD:
            failures.append(
                f"input: {example.input!r} | expected: {example.expected!r} | got: {output!r}"
            )
    average = sum(scores) / len(scores) if scores else 0.0
    return _Evaluation(score=average, failures=failures)


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
    consecutive rounds without improvement, or as soon as the best prompt
    has no more failures to learn from -- whichever comes first.
    """
    evaluation = _evaluate(llm, task_model, initial_prompt, examples, scorer)
    best_prompt = initial_prompt
    best_score = evaluation.score
    trace = [
        RoundTrace(round=0, prompt=initial_prompt, score=best_score, failures=evaluation.failures)
    ]

    stalled_rounds = 0

    for round_number in range(1, max_rounds + 1):
        if not evaluation.failures:
            logger.info("No failures left to learn from; stopping early.")
            break

        proposer_prompt = _build_proposer_prompt(best_prompt, evaluation.failures)
        try:
            proposer_response = llm.complete(model=proposer_model, system="", user=proposer_prompt)
        except Exception as exc:  # noqa: BLE001 - a flaky proposer must not kill the run
            # Mirror the per-example isolation in _evaluate: one transient
            # proposer error costs a round, not the whole run. The round
            # counts as stalled and the best-so-far result is preserved.
            logger.warning("Proposer call failed in round %d: %s", round_number, exc)
            stalled_rounds += 1
            if stalled_rounds >= STALL_LIMIT:
                logger.info("No improvement for %d consecutive rounds; stopping.", stalled_rounds)
                break
            continue
        candidate_prompt = _parse_proposed_prompt(proposer_response, fallback=best_prompt)

        candidate_evaluation = _evaluate(llm, task_model, candidate_prompt, examples, scorer)
        trace.append(
            RoundTrace(
                round=round_number,
                prompt=candidate_prompt,
                score=candidate_evaluation.score,
                failures=candidate_evaluation.failures,
            )
        )

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

    return Result(best_prompt=best_prompt, best_score=best_score, rounds=len(trace) - 1, trace=trace)
