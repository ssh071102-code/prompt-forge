"""Command-line interface for prompt-forge.

    prompt-forge optimize examples/sentiment.yaml --prompt "Classify sentiment."
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from prompt_forge.llm import LLM
from prompt_forge.models import Example, Result
from prompt_forge.optimizer import optimize
from prompt_forge.scorers import ExactMatchScorer, LLMJudgeScorer

app = typer.Typer(help="Iteratively optimize a prompt against labeled examples.")

DEFAULT_TASK_MODEL = "claude-haiku-4-5"
DEFAULT_PROPOSER_MODEL = "claude-opus-4-5"


@app.callback()
def _callback() -> None:
    """prompt-forge: an iterative LLM prompt optimizer."""
    # Registering a callback keeps `optimize` a required subcommand name
    # (e.g. `prompt-forge optimize examples.yaml --prompt ...`) instead of
    # typer collapsing a single-command app into an implicit default command.


def _load_examples(path: Path) -> list[Example]:
    """Load a YAML file of ``{input, expected}`` records into ``Example`` models."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise typer.BadParameter(f"{path} must contain a YAML list of examples.")
    return [Example(**item) for item in data]


def _print_report(result: Result) -> None:
    typer.echo(f"\nBest score: {result.best_score:.2f} (after {result.rounds} round(s))\n")
    typer.echo("Best prompt:")
    typer.echo(result.best_prompt)
    typer.echo("\nTrace:")
    typer.echo(f"{'round':>5}  {'score':>6}  failures")
    for round_trace in result.trace:
        typer.echo(f"{round_trace.round:>5}  {round_trace.score:>6.2f}  {len(round_trace.failures)}")


@app.command("optimize")
def optimize_command(
    examples_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="YAML file of {input, expected} examples."
    ),
    prompt: str = typer.Option(..., "--prompt", help="The initial prompt to optimize."),
    task_model: str = typer.Option(
        DEFAULT_TASK_MODEL, "--task-model", help="Model used to run the prompt on each example."
    ),
    proposer_model: str = typer.Option(
        DEFAULT_PROPOSER_MODEL, "--proposer-model", help="Model used to propose improved prompts."
    ),
    max_rounds: int = typer.Option(5, "--max-rounds", help="Maximum optimization rounds."),
    min_gain: float = typer.Option(
        0.01, "--min-gain", help="Minimum score improvement required to keep a candidate."
    ),
    judge: bool = typer.Option(
        False, "--judge", help="Score outputs with an LLM judge instead of exact match."
    ),
) -> None:
    """Optimize --prompt against the labeled examples in EXAMPLES_PATH."""
    examples = _load_examples(examples_path)
    llm = LLM()
    scorer = LLMJudgeScorer(llm, model=task_model) if judge else ExactMatchScorer()

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
