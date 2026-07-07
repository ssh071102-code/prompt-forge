# prompt-forge

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**prompt-forge** automatically improves a prompt by hill-climbing: run it on labeled examples, score the outputs, ask an LLM to fix the failures, and keep the result only if it's actually better.

Give it a starting prompt, a handful of labeled examples, and a scorer. It returns the best prompt it found, plus a full trace of every round so you can see exactly how it got there.

## What it does

1. **Evaluate.** Run the current prompt over every labeled example through the task model and score each output.
2. **Diagnose.** Collect the examples that scored below a failure threshold, with their input and the wrong output.
3. **Propose.** Show a proposer model the current prompt and a sample of its failures; ask it to write an improved prompt that generalizes, not one that just memorizes the examples.
4. **Compare.** Evaluate the candidate on the same examples. Keep it only if it beats the current best by at least `min_gain`; otherwise it's a stalled round.
5. **Repeat** until the prompt has no failures left to learn from, the round budget runs out, or two consecutive rounds fail to improve anything.

The result is the best prompt found, its score, and a round-by-round trace -- not just a black box that hands back a string.

## Architecture

| Module | Responsibility |
|---|---|
| `prompt_forge/models.py` | Pydantic models for examples, candidates, round traces, and the final result. |
| `prompt_forge/llm.py` | Thin wrapper around the Anthropic SDK. Everything else in the library depends on its `complete(model, system, user)` interface, not on `anthropic` directly. |
| `prompt_forge/scorers.py` | Pluggable scoring: `ExactMatchScorer` for deterministic tasks, `LLMJudgeScorer` for anything that needs semantic judgment. |
| `prompt_forge/optimizer.py` | The hill-climbing loop itself: evaluate, diagnose, propose, compare, repeat. |
| `prompt_forge/cli.py` | A `typer` CLI that loads examples from YAML and runs the loop end to end. |

```
                    ┌─────────────────────────────┐
                    │   evaluate(prompt, examples) │
                    │   -> average score, failures │
                    └───────────────┬──────────────┘
                                    │
                    failures empty? │ yes -> done
                                    │ no
                                    v
                    ┌─────────────────────────────┐
                    │ propose(prompt, failures)    │
                    │ -> candidate prompt          │
                    └───────────────┬──────────────┘
                                    │
                                    v
                    ┌─────────────────────────────┐
                    │ evaluate(candidate, examples)│
                    └───────────────┬──────────────┘
                                    │
                score improved by   │
                >= min_gain?        │
                 yes: keep, reset   │  no: count a stalled round
                 stall counter      │  2 stalls in a row -> stop
                                    │
                                    v
                        loop until max_rounds
```

## Design notes

- **Pluggable scoring.** `Scorer` is a `Protocol`, not a base class. Exact-match tasks (classification, extraction) get `ExactMatchScorer`; open-ended tasks get `LLMJudgeScorer`, which asks a model to rate the output 0-10 and parses the number back out of whatever free-form text comes back. Bring your own scorer for anything else -- the optimizer doesn't care.
- **Fully testable offline.** The optimizer and scorers only ever depend on an object with a `complete(model, system, user)` method (`LLMProtocol`). Tests inject a scripted `FakeLLM` instead of a real Anthropic client, so the entire hill-climbing loop -- including multi-round improvement and early stopping -- is verified without a network call or an API key.
- **Instructions folded into the user turn.** `LLM.complete` never uses the API's native `system` parameter; it concatenates system instructions into the user message instead. That keeps every request a single, self-contained turn that's easy to log, replay, and pass through proxies that don't forward `system` correctly.
- **Budget and early-stop control.** `max_rounds` caps total cost; `min_gain` sets the bar for what counts as real improvement (so noise doesn't get mistaken for progress); the search also gives up after two consecutive stalled rounds rather than grinding through a budget that isn't paying off.
- **Defensive per-example evaluation.** A single example that throws (a malformed response, a transient API error) is scored 0.0 and logged -- it never crashes the run. Optimization results should degrade gracefully, not stop dead on one bad example.

## Quickstart

```bash
pip install -e ".[dev]"
cp secrets/.env.example secrets/.env   # fill in ANTHROPIC_API_KEY
```

```bash
prompt-forge optimize examples/sentiment.yaml --prompt "Classify the sentiment of the input."
```

```
Best score: 0.86 (after 3 round(s))

Best prompt:
Classify the sentiment of the input as exactly one of: positive, negative,
or neutral. Base the label only on the emotional tone expressed, not on
whether the described product or event was itself good or bad...

Trace:
round  score  failures
    0   0.57  3
    1   0.71  2
    2   0.86  1
```

Options: `--task-model`, `--proposer-model`, `--max-rounds`, `--min-gain`, and `--judge` to score with an LLM judge instead of exact match.

## Tests

The whole suite runs offline against a scripted `FakeLLM` -- no API key, no network access.

```bash
pip install -e ".[dev]"
pytest
```

Coverage includes: scorer correctness (exact match and LLM-judge rating parsing), multi-round score improvement, early stopping on stalled rounds, and defensive handling of an example that errors mid-evaluation. The Anthropic wrapper itself is tested against a mocked HTTP layer (via `respx`) to confirm the real SDK request/response path works as expected.

## Tech

Python · [pydantic](https://docs.pydantic.dev/) · [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) · [typer](https://typer.tiangolo.com/) · [pytest](https://docs.pytest.org/)
