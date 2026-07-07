"""prompt-forge: an iterative LLM prompt optimizer.

Given a task, a handful of labeled examples, and a scorer, prompt-forge
hill-climbs an initial prompt toward higher quality by repeatedly:

1. Running the current prompt over the examples and scoring the outputs.
2. Collecting the failures.
3. Asking an LLM to propose an improved prompt from those failures.
4. Keeping the candidate only if it scores higher than the current best.

See ``prompt_forge.optimizer.optimize`` for the entry point.
"""

from prompt_forge.models import Candidate, Example, Result, RoundTrace

__version__ = "0.1.0"

__all__ = [
    "Candidate",
    "Example",
    "Result",
    "RoundTrace",
    "__version__",
]
