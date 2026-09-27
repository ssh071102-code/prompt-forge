"""Run the real CLI on examples/sentiment.yaml with a scripted stand-in model.

    python examples/offline_demo.py

No API key and no network: ``cli._make_llm`` is swapped for ``ScriptedLLM``,
whose "task model" is a keyword heuristic that gets better as the prompt
mentions more of the rules, and whose "proposer" returns three canned
rewrites in order. Everything else (YAML loading, the train/validation
split, the optimization loop, scoring and the report) is the code a real run
uses. The scores it prints come from that heuristic and say nothing about
how a real model would do.
"""

from __future__ import annotations

import sys
from pathlib import Path

from prompt_forge import cli
from prompt_forge.optimizer import PROMPT_BLOCK_END, PROMPT_BLOCK_START

TASK_MODEL = "scripted-task-model"
PROPOSER_MODEL = "scripted-proposer-model"

PROPOSALS = [
    "Classify the sentiment of the input as exactly one of: positive, negative, "
    "or neutral. Respond with only the single lowercase label.",
    "Classify the sentiment of the input as exactly one of: positive, negative, "
    "or neutral. Respond with only the single lowercase label. If praise and "
    "criticism are mixed, pick the sentiment that dominates the author's overall "
    "verdict.",
    "Classify the sentiment of the input as exactly one of: positive, negative, "
    "or neutral. Respond with only the single lowercase label. If praise and "
    "criticism are mixed, pick the sentiment that dominates the author's overall "
    "verdict, and reserve neutral for genuinely indifferent statements.",
]

NEGATIVE_WORDS = ("waste", "terrible", "rude", "cold", "broke", "never", "predictable", "bad")
INDIFFERENT_PHRASES = ("nothing special", "it was fine")


def _label(prompt: str, text: str) -> str:
    """A keyword heuristic that follows whichever rules the prompt spells out."""
    lowered = text.lower()
    if "neutral" in prompt and any(p in lowered for p in INDIFFERENT_PHRASES):
        return "neutral"
    if "dominates" in prompt and " but " in lowered:
        lowered = lowered.split(" but ", 1)[1]  # the verdict after "but" wins
    return "negative" if any(w in lowered for w in NEGATIVE_WORDS) else "positive"


class ScriptedLLM:
    """Implements ``LLMProtocol`` without calling any model."""

    def __init__(self) -> None:
        self._proposals = iter(PROPOSALS)

    def complete(self, model: str, system: str, user: str) -> str:
        if model == PROPOSER_MODEL:
            return f"{PROMPT_BLOCK_START}\n{next(self._proposals)}\n{PROMPT_BLOCK_END}"
        prompt, _, text = user.partition("\n\nInput:\n")
        label = _label(prompt, text)
        if "only the single lowercase label" in prompt:
            return label
        return f"The sentiment of this text is {label}."


def main() -> None:
    examples = Path(__file__).with_name("sentiment.yaml")
    cli._make_llm = ScriptedLLM  # type: ignore[assignment]  # the CLI's test seam
    sys.argv = [
        "prompt-forge",
        "optimize",
        str(examples),
        "--prompt",
        "Classify the sentiment of the input.",
        "--task-model",
        TASK_MODEL,
        "--proposer-model",
        PROPOSER_MODEL,
        *sys.argv[1:],
    ]
    cli.app()


if __name__ == "__main__":
    main()
