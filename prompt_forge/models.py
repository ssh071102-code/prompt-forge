"""Data models shared across prompt-forge.

These are plain pydantic models: no behavior, just validated shapes that
flow between the optimizer, the scorers, and the CLI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Example(BaseModel):
    """A single labeled input/expected-output pair used to evaluate a prompt."""

    input: str
    expected: str


class Candidate(BaseModel):
    """A prompt variant produced during optimization, with its measured score."""

    prompt: str
    score: float
    round: int


class RoundTrace(BaseModel):
    """A record of what happened in a single optimization round.

    ``round`` 0 always records the evaluation of the initial prompt, before
    any proposer round has run. ``failures`` holds human-readable
    descriptions of the examples that scored below the failure threshold,
    capped at a small sample so the trace stays readable.
    """

    round: int
    prompt: str
    score: float
    failures: list[str] = Field(default_factory=list)


class Result(BaseModel):
    """The final output of an optimization run."""

    best_prompt: str
    best_score: float
    rounds: int
    trace: list[RoundTrace] = Field(default_factory=list)
