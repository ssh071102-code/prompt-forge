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
from typer.testing import CliRunner, Result

from prompt_forge import cli
from prompt_forge.llm import FatalLLMError
from prompt_forge.optimizer import PROMPT_BLOCK_END, PROMPT_BLOCK_START
from tests.fakes import PROPOSER_MODEL, TASK_MODEL, FakeLLM

runner = CliRunner()

EXAMPLES_YAML = """\
- input: "I love this movie"
  expected: "positive"
- input: "This movie was terrible"
  expected: "negative"
"""


def _all_output(result: Result) -> str:
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
    # default --val-split 0.3 on 2 examples: 1 for training, 1 held out
    assert "Train score: 1.00 on 1 example(s) (after 1 round(s))" in result.output
    assert "Validation score: 1.00 on 1 held-out example(s) (initial prompt: 0.00)" in (
        result.output
    )
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


def _sentiment_task_fn(user: str) -> str:
    if "PROMPT_V1" in user:
        return "positive" if "love" in user else "negative"
    return "unknown"


def _run(args: list[str], fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch) -> Result:
    monkeypatch.setattr(cli, "_make_llm", lambda: fake_llm)
    return runner.invoke(
        cli.app,
        ["optimize", *args, "--task-model", TASK_MODEL, "--proposer-model", PROPOSER_MODEL],
    )


def test_cli_val_split_zero_reports_no_validation_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_llm = FakeLLM(task_fn=_sentiment_task_fn, proposer_responses=["no delimiters"] * 2)
    examples_path = _write_examples(tmp_path)

    result = _run([str(examples_path), "--prompt", "p", "--val-split", "0"], fake_llm, monkeypatch)

    assert result.exit_code == 0, _all_output(result)
    assert "Train score: 0.00 on 2 example(s)" in result.output
    assert "Validation score: n/a" in result.output


def test_cli_separate_val_file_trains_on_every_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    val_path = tmp_path / "val.yaml"
    val_path.write_text(
        "- input: 'I love it'\n  expected: positive\n"
        "- input: 'I hate it'\n  expected: negative\n"
        "- input: 'love love love'\n  expected: positive\n",
        encoding="utf-8",
    )
    fake_llm = FakeLLM(
        task_fn=_sentiment_task_fn,
        proposer_responses=[f"{PROMPT_BLOCK_START}PROMPT_V1: better.{PROMPT_BLOCK_END}"],
    )
    examples_path = _write_examples(tmp_path)

    result = _run(
        [str(examples_path), "--prompt", "PROMPT_V0", "--val", str(val_path)],
        fake_llm,
        monkeypatch,
    )

    assert result.exit_code == 0, _all_output(result)
    assert "Train score: 1.00 on 2 example(s)" in result.output
    assert "Validation score: 1.00 on 3 held-out example(s) (initial prompt: 0.00)" in (
        result.output
    )
    # The proposer only ever saw training inputs.
    proposer_text = " ".join(c[2] for c in fake_llm.calls if c[0] == PROPOSER_MODEL)
    assert "I hate it" not in proposer_text


def test_cli_rejects_val_split_with_a_single_example(tmp_path: Path) -> None:
    one = _write_examples(tmp_path, content="- input: 'a'\n  expected: 'b'\n")

    result = runner.invoke(cli.app, ["optimize", str(one), "--prompt", "p"])

    assert result.exit_code != 0
    assert "need at least 2 examples" in _all_output(result)


def test_cli_fatal_llm_error_exits_nonzero_with_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def task_fn(user: str) -> str:
        raise FatalLLMError("invalid x-api-key")

    examples_path = _write_examples(tmp_path)

    result = _run([str(examples_path), "--prompt", "p"], FakeLLM(task_fn=task_fn), monkeypatch)

    assert result.exit_code == 1
    assert "Error: invalid x-api-key" in _all_output(result)
