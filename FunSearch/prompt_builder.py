"""
prompt_builder.py
-----------------
Constructs LLM prompts for the FunSearch evolution loop.

One prompt per generation asks the LLM to generate both model() and
estimate_params() together, following EDGAR's linked prompt design.

Two modes:
  explore : high creativity, invent something new
  exploit : refine and simplify the best existing program

Function names in the output are always the canonical 'model' and 'estimate_params'.
Versioning is handled entirely by the output folder structure in run_loop.py:
    island_1/gen_1/program_1/program.py

The problem description is loaded from:
    problems/<name>/code/prompt_context.txt
"""

from pathlib import Path
from typing import Literal

from FunSearch.program_parser import ProgramStrings

Mode = Literal["explore", "exploit"]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_prompt(
    parents: list[tuple[ProgramStrings, float]],
    mode: Mode,
    context: str,
) -> str:
    """
    Create a prompt to generate a new model and estimate_params based on
    2 existing programs, following EDGAR's linked prompt design.

    Parameters
    ----------
    parents : list of exactly 2 (ProgramStrings, score) tuples,
              ordered worst-first so the LLM sees a clear improvement direction
    mode    : "explore" or "exploit"
    context : problem description string (from prompt_context.txt)

    Returns
    -------
    Prompt string ready to send to the LLM.
    """
    assert len(parents) == 2, "build_prompt expects exactly 2 parent programs"
    assert mode in ("explore", "exploit"), f"Unknown mode: {mode!r}"

    prompt = f"""\
{context}

You are an AI scientist. The programs below are models and their associated \
parameter estimators. The models are sorted from highest loss to lowest loss.

Your task is to create a new model() and the associated estimate_params() \
that has a lower loss than the models below.

*Analyze* the progression of the models, *generalize* the improvements, \
and *create* a new model that is better than the *all* previous models.

"""

    if mode == "explore":
        prompt += """\
Use the models and parameter estimators below as inspiration, but be *creative* \
and *invent* something new. Which features in the models below correlate with \
lower loss? Find those and *extrapolate* them. You should also *combine* features \
from several models, and *experiment* with new ideas.

"""
    else:
        prompt += """\
Use the models and parameter estimators below as a *template* to create an \
improved, simpler model. Focus on *exploiting* the strengths of the existing \
models and *eliminating* their weaknesses or *redundancies*. You will be \
*penalized* for complexity, so make the new model and parameter estimator as \
*simple* as possible while still being better than the previous models.

"""

    prompt += """\
**Code Generation Guidelines:**
* Import any packages you use.
* Do not include any text other than the code.
* Ensure all free parameters are numeric, not strings.
* The functions must be named exactly `model` and `estimate_params`.
* `model(x, params)` takes an array of inputs and a parameter vector, \
and returns predicted outputs.
* `estimate_params(x, y)` takes inputs and observed outputs, and returns \
an initial parameter guess. Do not use curve_fit, minimize, or any iterative \
fitting — this is a starting point for gradient-based optimisation.

"""

    for i, (prog, score) in enumerate(parents):
        prompt += f"loss of model {i + 1}: {score:.4f}\n"
        prompt += f"```python {prog.model_src}```\n\n"
        prompt += f"```python {prog.estimator_src}```\n"

    return prompt.strip()


# ---------------------------------------------------------------------------
# Load problem context
# ---------------------------------------------------------------------------

def load_context(problem_code_dir: Path) -> str:
    """
    Load the plain-text problem description from prompt_context.txt.
    Falls back to a generic description if the file does not exist.
    """
    context_path = Path(problem_code_dir) / "prompt_context.txt"
    if context_path.exists():
        return context_path.read_text().strip()
    print(f"[prompt_builder] Warning: no prompt_context.txt found at {context_path}")
    return ("")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from FunSearch.program_parser import script_to_strings

    seed_path = Path("problems/sinewave/code/seed_programs.py")
    context_path = Path("problems/sinewave/code")

    programs = script_to_strings(seed_path)
    context = load_context(context_path)

    parents = [(programs[0], -0.45), (programs[1], 0.82)]

    print("=" * 60)
    print("PROMPT — explore mode")
    print("=" * 60)
    print(build_prompt(parents, mode="explore", context=context))

    print("\n" + "=" * 60)
    print("PROMPT — exploit mode")
    print("=" * 60)
    print(build_prompt(parents, mode="exploit", context=context))