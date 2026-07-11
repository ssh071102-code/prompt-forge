"""Command-line interface for prompt-forge.

    prompt-forge optimize examples/sentiment.yaml --prompt "Classify sentiment."
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from prompt_forge.llm import LLM
from prompt_forge.models import Example, Result
from prompt_forge.optimizer import optimize
from prompt_forge.scorers import ExactMatchScorer, LLMJudgeScorer

app = typer.Typer(help="Iteratively optimize a prompt against labeled examples.")

DEFAULT_TASK_MODEL = "claude-haiku-4-5"
DEFAULT_PROPOSER_MODEL = "claude-opus-4-5"
# The judge grades the task model's outputs, so it must not default to the
# task model itself -- a weak model grading its own answers inflates scores
# through self-preference. Default to a stronger model, same as the proposer.
DEFAULT_JUDGE_MODEL = "claude-opus-4-5"

ENV_FILE = Path("secrets") / ".env"


@app.callback()
def _callback() -> None:
    """prompt-forge: an iterative LLM prompt optimizer."""
    # Registering a callback keeps `optimize` a required subcommand name
    # (e.g. `prompt-forge optimize examples.yaml --prompt ...`) instead of
    # typer collapsing a single-command app into an implicit default command.
    #
    # Load ANTHROPIC_API_KEY (and friends) from secrets/.env if it exists.
    # A missing file is fine: variables already in the process environment
    # always take precedence, and the SDK reads them directly.
    load_dotenv(ENV_FILE)


def _make_llm() -> LLM:
    """Build the Anthropic-backed LLM used by the CLI.

    A module-level seam so tests can monkeypatch it with a FakeLLM and run
    the whole command end to end without a network call or an API key.
    """
    return LLM()


def _load_examples(path: Path) -> list[Example]:
    """Load a YAML file of ``{input, expected}`` records into ``Example`` models."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise typer.BadParameter(f"{path} must contain a YAML list of examples.")
    try:
        return [Example(**item) for item in data]
    except (TypeError, ValidationError) as exc:
        raise typer.BadParameter(
            f"{path}: every example needs string 'input' and 'expected' fields.\n{exc}"
        ) from exc


def _print_report(result: Result) -> None:
    typer.echo(f"\nBest score: {result.best_score:.2f} (after {result.rounds} round(s))\n")
    typer.echo("Best prompt:")
    typer.echo(result.best_prompt)
    typer.echo("\nTrace:")
    typer.echo(f"{'round':>5}  {'score':>6}  failures")
    for entry in result.trace:
        typer.echo(f"{entry.round:>5}  {entry.score:>6.2f}  {entry.total_failures}")


@app.command("optimize")
def optimize_command(
    examples_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="YAML file of {input, expected} examples."),
    ],
    prompt: Annotated[str, typer.Option("--prompt", help="The initial prompt to optimize.")],
    task_model: Annotated[
        str,
        typer.Option("--task-model", help="Model used to run the prompt on each example."),
    ] = DEFAULT_TASK_MODEL,
    proposer_model: Annotated[
        str,
        typer.Option("--proposer-model", help="Model used to propose improved prompts."),
    ] = DEFAULT_PROPOSER_MODEL,
    judge_model: Annotated[
        str,
        typer.Option("--judge-model", help="Model used by the LLM judge (only with --judge)."),
    ] = DEFAULT_JUDGE_MODEL,
    max_rounds: Annotated[
        int, typer.Option("--max-rounds", help="Maximum optimization rounds.")
    ] = 5,
    min_gain: Annotated[
        float,
        typer.Option("--min-gain", help="Minimum score improvement required to keep a candidate."),
    ] = 0.01,
    judge: Annotated[
        bool,
        typer.Option("--judge", help="Score outputs with an LLM judge instead of exact match."),
    ] = False,
) -> None:
    """Optimize --prompt against the labeled examples in EXAMPLES_PATH."""
    examples = _load_examples(examples_path)
    llm = _make_llm()
    scorer = LLMJudgeScorer(llm, model=judge_model) if judge else ExactMatchScorer()

    result = optimize(
        prompt,
        examples,
        scorer,
        llm,
        task_model=task_model,
        proposer_model=proposer_model,
        max_rounds=max_rounds,
        min_gain=min_gain,
    )

    _print_report(result)


if __name__ == "__main__":
    app()
