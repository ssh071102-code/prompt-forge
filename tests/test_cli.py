"""End-to-end CLI tests.

These drive the real typer app through CliRunner, with the Anthropic-backed
LLM swapped for a FakeLLM by monkeypatching the ``_make_llm`` seam -- the
full command path (YAML loading, optimization, report printing) runs without
a network call or an API key.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from prompt_forge import cli
from prompt_forge.optimizer import PROMPT_BLOCK_END, PROMPT_BLOCK_START
from tests.fakes import PROPOSER_MODEL, TASK_MODEL, FakeLLM

runner = CliRunner()

EXAMPLES_YAML = """\
- input: "I love this movie"
  expected: "positive"
- input: "This movie was terrible"
  expected: "negative"
"""


def _all_output(result) -> str:  # noqa: ANN001 - click's Result type varies across versions
    """Combine stdout and stderr, flattened for substring assertions.

    Error output goes to stderr on newer click versions (older ones mix it
    into stdout), and typer renders it in a rich panel that wraps long lines
    with box-drawing characters. Rich also force-enables ANSI color on CI
    terminals (e.g. GitHub Actions), so escape sequences must be stripped
    whole -- dropping just the ESC byte would leave "[33m" fragments inside
    the message text. Strip the decoration and collapse the whitespace so
    tests can assert on the plain message text.
    """
    combined = result.output
    try:
        combined += result.stderr
    except (ValueError, AttributeError):
        pass  # older click mixes stderr into output already
    no_ansi = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", combined)
    flattened = re.sub(r"[^\x20-\x7e]", " ", no_ansi)  # drop rich box-drawing characters
    return re.sub(r"\s+", " ", flattened)


def _write_examples(tmp_path: Path, content: str = EXAMPLES_YAML) -> Path:
    path = tmp_path / "examples.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_cli_optimize_runs_end_to_end_with_fake_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def task_fn(user: str) -> str:
        if "PROMPT_V1" in user:
            return "positive" if "love" in user else "negative"
        return "unknown"

    fake_llm = FakeLLM(
        task_fn=task_fn,
        proposer_responses=[
            f"{PROMPT_BLOCK_START}\n"
            "PROMPT_V1: Classify sentiment as positive or negative.\n"
            f"{PROMPT_BLOCK_END}"
        ],
    )
    monkeypatch.setattr(cli, "_make_llm", lambda: fake_llm)
    examples_path = _write_examples(tmp_path)

    result = runner.invoke(
        cli.app,
        [
            "optimize",
            str(examples_path),
            "--prompt",
            "PROMPT_V0: Classify sentiment.",
            "--task-model",
            TASK_MODEL,
            "--proposer-model",
            PROPOSER_MODEL,
        ],
    )

    assert result.exit_code == 0, _all_output(result)
    assert "Best score: 1.00 (after 1 round(s))" in result.output
    assert "PROMPT_V1: Classify sentiment as positive or negative." in result.output


def test_cli_rejects_examples_file_that_is_not_a_list(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("just a string, not a list of examples", encoding="utf-8")

    result = runner.invoke(cli.app, ["optimize", str(bad_path), "--prompt", "p"])

    assert result.exit_code != 0
    assert "must contain a YAML list" in _all_output(result)


def test_cli_rejects_example_missing_required_fields(tmp_path: Path) -> None:
    bad_path = _write_examples(tmp_path, content="- input: 'has input but no expected'\n")

    result = runner.invoke(cli.app, ["optimize", str(bad_path), "--prompt", "p"])

    assert result.exit_code != 0
    assert "every example needs string 'input' and 'expected' fields" in _all_output(result)
