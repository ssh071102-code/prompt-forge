# prompt-forge

[![CI](https://github.com/ssh071102-code/prompt-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/ssh071102-code/prompt-forge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
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

## Why prompt-forge instead of DSPy or promptfoo?

DSPy (and OPRO-style optimizers generally) also rewrite prompts from feedback, but they live inside a heavyweight framework you adopt wholesale -- signatures, modules, compilers, its own program model. promptfoo is excellent at *evaluating* prompts against test cases, but it doesn't close the loop and rewrite them for you. prompt-forge is deliberately the smallest thing that does close the loop: under 600 lines across five modules, a handful of dependencies, one `optimize()` function, and a full round-by-round trace of what changed and why. If you need multi-stage pipelines or program synthesis, use DSPy; if you need a prompt that stops failing your examples by this afternoon, this is the tool-shaped version of that.

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

- **Pluggable scoring.** `Scorer` is a `Protocol`, not a base class. Exact-match tasks (classification, extraction) get `ExactMatchScorer`; open-ended tasks get `LLMJudgeScorer`, which asks a model to rate the output on a 0-10 scale and parses the number back out of whatever free-form text comes back (an explicit `X/10` wins; a bare number is read on the instructed 0-10 scale). Bring your own scorer for anything else -- the optimizer doesn't care.
- **Fully testable offline.** The optimizer and scorers only ever depend on an object with a `complete(model, system, user)` method (`LLMProtocol`). Tests inject a scripted `FakeLLM` instead of a real Anthropic client, so the entire hill-climbing loop -- including multi-round improvement, early stopping, and the CLI itself -- is verified without a network call or an API key.
- **Native system parameter, with an escape hatch.** `LLM.complete` sends system text through the API's native `system` parameter by default, which is how the models are trained to receive instructions. `LLM(fold_system_into_user=True)` is a compatibility escape hatch that prepends the system text to the user message instead, for API gateways that strip or drop the `system` field.
- **Budget and early-stop control.** `max_rounds` caps total cost; `min_gain` sets the bar for what counts as real improvement (so noise doesn't get mistaken for progress); the search also gives up after two consecutive stalled rounds rather than grinding through a budget that isn't paying off.
- **Defensive evaluation, everywhere.** A single example that throws (a malformed response, a transient API error) is scored 0.0 and logged -- it never crashes the run. A failed proposer call gets the same treatment: it costs one stalled round, and the run still returns the best prompt found so far.

## Quickstart

```bash
pip install -e ".[dev]"
```

Provide your API key either through a `.env` file (loaded automatically at CLI startup):

```bash
cp secrets/.env.example secrets/.env   # then fill in ANTHROPIC_API_KEY
```

or directly in the environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # bash/zsh
```

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."  # PowerShell
```

Variables already set in the process environment take precedence over `secrets/.env`. Then:

```bash
prompt-forge optimize examples/sentiment.yaml --prompt "Classify the sentiment of the input."
```

Illustrative output (what a run looks like -- real scores depend on the models; prompt text wrapped for readability):

```
Best score: 0.96 (after 2 round(s))

Best prompt:
Classify the sentiment of the input as exactly one of: positive, negative,
or neutral. Respond with only the single lowercase label and nothing else.
Judge only the emotional tone the author expresses, not whether the
described product or event was itself good or bad. If praise and criticism
are mixed, pick the sentiment that dominates the author's overall verdict,
and reserve neutral for genuinely indifferent statements.

Trace:
round   score  failures
    0    0.57  3
    1    0.81  1
    2    0.96  0
```

(That block is a captured run of the real CLI against a scripted stand-in model: with the 7 bundled sentiment examples, 4/7 exact matches round to 0.57, then 0.81, then 0.96 as failures drop from 3 to 0.)

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--task-model` | `claude-haiku-4-5` | Model that runs the prompt on each example. |
| `--proposer-model` | `claude-opus-4-5` | Model that writes improved prompt candidates. |
| `--judge-model` | `claude-opus-4-5` | Model used by the LLM judge (only with `--judge`). Defaults to a stronger model than the task model, so the weakest model never grades its own output. |
| `--max-rounds` | `5` | Maximum optimization rounds. |
| `--min-gain` | `0.01` | Minimum score improvement required to keep a candidate. |
| `--judge` | off | Score outputs with an LLM judge instead of exact match. |

Three more knobs live as constants in `prompt_forge/optimizer.py`: `FAILURE_THRESHOLD` (0.5 -- scores below this count as failures), `MAX_FAILURE_SAMPLES` (5 -- failures shown to the proposer and kept per trace round), and `STALL_LIMIT` (2 -- consecutive no-improvement rounds before giving up).

## Use as a library

Everything is importable from the package root, and any object with a 3-line `score` method is a valid scorer:

```python
from prompt_forge import LLM, Example, optimize

class KeywordScorer:
    def score(self, output: str, example: Example) -> float:
        return 1.0 if example.expected.lower() in output.lower() else 0.0

examples = [
    Example(input="The refund never arrived.", expected="billing"),
    Example(input="The app crashes on launch.", expected="bug"),
]

result = optimize(
    "Tag the support ticket with one topic label.",
    examples, KeywordScorer(), LLM(),  # LLM() reads ANTHROPIC_API_KEY from the env
    task_model="claude-haiku-4-5", proposer_model="claude-opus-4-5",
)
print(f"{result.best_score:.2f}", result.best_prompt)
```

`result` is a pydantic model carrying `best_prompt`, `best_score`, `rounds`, and the full `trace` (per-round prompt, score, failure samples, and total failure count).

## Limitations & cost

- **Greedy search.** Hill-climbing keeps only strict improvements, so it can park in a local optimum: a rewrite that fixes three failures but temporarily breaks one scoring pattern gets rejected. If the trace shows early stalling, try a different starting prompt rather than more rounds.
- **Small example sets overfit.** With 5-10 examples, a "better" prompt may just fit those examples' quirks. For decisions that matter, use 20+ examples and hold some out for a manual check.
- **LLM-judge scores are noisy.** The same output can get a 6 or an 8 on different calls. Keep `min_gain` meaningfully above the judge's noise floor so the search doesn't chase noise, and prefer `ExactMatchScorer` whenever the task allows it.
- **Cost shape.** Round 0 costs one task-model call per example. Every later round costs one proposer call plus one task-model call per example -- and with `--judge`, one judge call per example on top. Total calls grow linearly in `max_rounds * len(examples)`; `max_rounds` and the stall limit are the cost caps.

## Tests

The whole suite (28 tests) runs offline against a scripted `FakeLLM` -- no API key, no network access.

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Coverage includes: scorer correctness (exact match and LLM-judge rating normalization), multi-round score improvement, early stopping on stalled rounds, survival of both per-example errors and a failed proposer call, the undelimited-proposer-response fallback, trace failure capping, and the CLI end to end (via a monkeypatched LLM factory), including example-file validation errors. The Anthropic wrapper itself is tested against a mocked HTTP layer (via `respx`) to confirm the real SDK request/response path, the native `system` parameter, and the fold-in escape hatch.

## Tech

Python · [pydantic](https://docs.pydantic.dev/) · [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) · [typer](https://typer.tiangolo.com/) · [python-dotenv](https://github.com/theskumar/python-dotenv) · [pytest](https://docs.pytest.org/) · [ruff](https://docs.astral.sh/ruff/)
