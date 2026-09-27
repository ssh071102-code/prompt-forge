# prompt-forge

[![CI](https://github.com/ssh071102-code/prompt-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/ssh071102-code/prompt-forge/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

prompt-forge improves a prompt by hill-climbing on labeled training examples and then reports how the winning prompt scores on validation examples the search never saw.

```
$ python examples/offline_demo.py

Train score: 1.00 on 5 example(s) (after 1 round(s))
Validation score: 0.50 on 2 held-out example(s) (initial prompt: 0.00)

Best prompt:
Classify the sentiment of the input as exactly one of: positive, negative, or neutral. Respond with only the single lowercase label.

Trace:
round   train  failures
    0    0.70  0
    1    1.00  0
```

That output comes from the real CLI running on the 7 bundled sentiment examples, with the model swapped for a scripted keyword heuristic (see [`examples/offline_demo.py`](examples/offline_demo.py)), so it needs no API key. The numbers only describe that heuristic. They do show the reason for the validation set: the rewrite reaches 1.00 on the 5 training examples and 0.50 on the 2 held-out ones.

## Quickstart

```bash
git clone https://github.com/ssh071102-code/prompt-forge.git
cd prompt-forge
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python examples/offline_demo.py    # offline, prints the block above
pytest                             # offline, 61 tests
```

To run against Anthropic models, provide an API key, either in `secrets/.env` (loaded at CLI startup) or in the environment. Variables already set in the environment take precedence.

```bash
cp secrets/.env.example secrets/.env   # then fill in ANTHROPIC_API_KEY
prompt-forge optimize examples/sentiment.yaml --prompt "Classify the sentiment of the input."
```

A missing or rejected key, or an unknown model name, stops the run with `Error: ...` and exit code 1.

## How it works

1. **Split.** Shuffle the examples with a fixed seed and hold out 30% (`--val-split`) as a validation set, or read validation examples from a separate file (`--val`).
2. **Evaluate.** Run the current prompt over every training example through the task model and score each output.
3. **Diagnose.** Collect the training examples that scored below a failure threshold. If none did but the average is below 0.99, use the lowest-scoring ones instead.
4. **Propose.** Show a proposer model the current prompt and up to 5 of those examples, and ask for an improved prompt that fixes the instruction without hardcoding answers.
5. **Compare.** Score the candidate on the training examples. Keep it if it beats the current best by at least `min_gain`, otherwise count a stalled round. A proposal identical to the current prompt is counted as stalled without being re-evaluated.
6. **Repeat** until every training example scores 0.99 or more, `max_rounds` runs out, or two rounds in a row stall.
7. **Validate.** Score the initial prompt and the best prompt once each on the validation set.

The proposer never sees validation examples, and no search decision uses them. The training score is therefore optimistic; the validation score is the one to compare prompts by.

```
                ┌──────────────────────────────┐
                │ split(examples) -> train, val │
                └───────────────┬──────────────┘
                                v
                ┌──────────────────────────────┐
                │ evaluate(prompt, train)       │
                └───────────────┬──────────────┘
                                v
                ┌──────────────────────────────┐
                │ propose(prompt, failures)     │<──┐
                └───────────────┬──────────────┘   │
                                v                  │
                ┌──────────────────────────────┐   │
                │ evaluate(candidate, train)    │   │ until max_rounds,
                │ keep if gain >= min_gain      │───┘ 2 stalls, or 0.99
                └───────────────┬──────────────┘
                                v
                ┌──────────────────────────────┐
                │ evaluate(initial, val)        │
                │ evaluate(best, val)           │
                └──────────────────────────────┘
```

## Error handling

- A single example that raises a transient error (a timeout, a rate limit, a malformed response) is scored 0.0, logged as a warning, and the run continues.
- Errors that would fail on every call abort the run: authentication, permission, not-found and bad-request API errors, and missing credentials. `LLM` raises these as `FatalLLMError`.
- If every training example errors on the initial prompt, the run stops with `EvaluationFailedError`. This catches configuration problems that come from a custom provider or scorer and are not recognised as fatal.
- A failed or malformed proposer response costs one stalled round and is recorded in the trace.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--val-split` | `0.3` | Fraction of the examples held out for validation. `0` disables validation. |
| `--val` | none | Separate YAML file of validation examples. All of `EXAMPLES_PATH` is then used for training. |
| `--seed` | `0` | Seed for the train/validation shuffle. |
| `--task-model` | `claude-haiku-4-5` | Model that runs the prompt on each example. |
| `--proposer-model` | `claude-opus-5` | Model that writes improved prompt candidates. |
| `--judge-model` | `claude-opus-5` | Model used by the LLM judge (only with `--judge`). It defaults to a stronger model than the task model so a model does not grade its own output. |
| `--max-rounds` | `5` | Maximum optimization rounds. |
| `--min-gain` | `0.01` | Minimum training-score improvement required to keep a candidate. |
| `--judge` | off | Score outputs with an LLM judge instead of exact match. |

Three more knobs are constants in `prompt_forge/optimizer.py`: `FAILURE_THRESHOLD` (0.5, scores below this count as failures), `MAX_FAILURE_SAMPLES` (5, examples shown to the proposer and kept per trace round) and `STALL_LIMIT` (2, consecutive stalled rounds before giving up).

## Use as a library

Everything is importable from the package root. Any object with a `score(output, example) -> float` method is a valid scorer.

```python
from prompt_forge import LLM, Example, optimize, split_examples

class KeywordScorer:
    def score(self, output: str, example: Example) -> float:
        return 1.0 if example.expected.lower() in output.lower() else 0.0

examples = [
    Example(input="The refund never arrived.", expected="billing"),
    Example(input="I was charged twice this month.", expected="billing"),
    Example(input="The app crashes on launch.", expected="bug"),
    Example(input="Search returns a blank page.", expected="bug"),
]
train, val = split_examples(examples, 0.5, seed=0)

result = optimize(
    "Tag the support ticket with one topic label.",
    train, KeywordScorer(), LLM(),  # LLM() reads ANTHROPIC_API_KEY from the env
    task_model="claude-haiku-4-5", proposer_model="claude-opus-5",
    validation=val,
)
print(f"train {result.best_score:.2f}, validation {result.val_score:.2f}")
print(result.best_prompt)
```

`result` is a pydantic model with `best_prompt`, `best_score` (training), `val_score` and `initial_val_score` (`None` without a validation set), `rounds`, and the per-round `trace`.

## Modules

| Module | Responsibility |
|---|---|
| `prompt_forge/models.py` | Pydantic models for examples, round traces and the result. |
| `prompt_forge/llm.py` | Thin wrapper around the Anthropic SDK. The rest of the package depends only on its `complete(model, system, user)` interface (`LLMProtocol`). |
| `prompt_forge/scorers.py` | `ExactMatchScorer` (1.0 exact, 0.7 when the answer appears as a whole word and is not negated, else 0.0) and `LLMJudgeScorer` (a 0-10 rating normalised to 0-1). |
| `prompt_forge/optimizer.py` | The train/validation split and the hill-climbing loop. |
| `prompt_forge/cli.py` | A `typer` CLI that loads YAML examples and prints the report. |

`LLM.complete` sends system text through the API's `system` parameter. `LLM(fold_system_into_user=True)` prepends it to the user message instead, for API gateways that drop the `system` field.

## Limitations and cost

- **Small validation sets are coarse.** With the 7 bundled examples the validation set has 2, so its score can only move in steps of about 0.5. Use 20 or more examples, or pass a larger `--val` file, before trusting a comparison.
- **Greedy search.** Only candidates that beat the best training score by `min_gain` are kept, so the search can stop at a local optimum. If the trace stalls early, try a different starting prompt.
- **One validation pass.** The validation score is measured once at the end. It is not used for early stopping or model selection between rounds.
- **LLM-judge scores are noisy.** The same output can get a 6 or an 8 on different calls. Keep `min_gain` above the judge's noise and prefer `ExactMatchScorer` when the task allows it.
- **Anthropic only out of the box.** Other providers need an adapter with a `complete(model, system, user)` method.
- **Cost.** Round 0 costs one task call per training example. Each later round costs one proposer call plus one task call per training example, and with `--judge` one judge call per example on top. Validation adds up to two task calls per validation example.

## Development

```bash
pip install -e ".[dev]"
pytest          # 61 tests, offline
ruff check .
mypy            # strict, configured in pyproject.toml
```

Tests use a scripted `FakeLLM` for the optimizer, scorers and CLI, and a fake Anthropic client object for `LLM`, so the suite makes no network calls and needs no API key. CI runs all three commands on Python 3.10 and 3.12.

## AI assistance
This project was built with AI assistance (Claude Code). Commits it co-authored carry a `Co-Authored-By` trailer.

## License

MIT
