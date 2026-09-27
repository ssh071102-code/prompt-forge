"""Pin the output of examples/offline_demo.py, which the README quotes verbatim."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from prompt_forge import cli

DEMO = Path(__file__).resolve().parent.parent / "examples" / "offline_demo.py"


def test_offline_demo_output_matches_readme(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The demo swaps cli._make_llm; registering it here restores it afterwards.
    monkeypatch.setattr(cli, "_make_llm", cli._make_llm)
    monkeypatch.setattr(sys, "argv", [str(DEMO)])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(DEMO), run_name="__main__")

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "Train score: 1.00 on 5 example(s) (after 1 round(s))" in out
    assert "Validation score: 0.50 on 2 held-out example(s) (initial prompt: 0.00)" in out
    assert "    0    0.70  0\n    1    1.00  0" in out
