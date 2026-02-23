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
    image_paths: list[Path | None] | None = None,
    use_image: bool = True,
) -> list[tuple[str, list[Path] | None]]:
    """
    Build an interleaved prompt for the LLM as a list of (text, images) segments.

    Segment order:
      1. Task instructions + output format (+ image analysis note if use_image)
      2. Parent program 1 text [+ figure 1 if use_image and available]
      3. Parent program 2 text [+ figure 2 if use_image and available]

    Parameters
    ----------
    parents     : list of exactly 2 (ProgramStrings, score) tuples,
                  ordered worst-first so the LLM sees a clear improvement direction
    mode        : "explore" or "exploit"
    context     : problem description string (from prompt_context.txt)
    image_paths : optional list of 2 Path-or-None values, one per parent
    use_image   : whether to include evaluation figures in the prompt

    Returns
    -------
    List of (text, image_paths) segments for llm_caller.call_interleaved().
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
and *create* a new model that is better than *all* previous models.

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

    prompt += """**Output format — strictly follow this:**
Output a SINGLE fenced Python code block containing ONLY the two new functions.

```python
import numpy as np

def model(x, params):
    # your implementation
    ...

def estimate_params(x, y):
    # your implementation
    ...
```

**Code guidelines:**
* Import packages inside the code block.
* Name the functions exactly `model` and `estimate_params` — no version suffixes.
* `model(x, params)`: x and params are np.ndarrays, return np.ndarray of predictions.
* `estimate_params(x, y)`: return an initial parameter guess as np.ndarray. \
Do not use curve_fit, minimize, or any iterative fitting.
* All free parameters must be numeric, not strings.

"""

    # Resolve image paths — None if not provided or use_image is False
    n = len(parents)
    imgs: list[Path | None] = []
    if use_image and image_paths:
        for i in range(n):
            p = image_paths[i] if i < len(image_paths) else None
            imgs.append(Path(p) if p is not None else None)
    else:
        imgs = [None] * n

    # Append image analysis instructions to the task prompt if using images
    if use_image:
        prompt += """
**Image Analysis Instructions:**
Attached are scatter plots where the data is plotted and each model curve is drawn **using the functions shown below**.
Pay close attention to the data and what kind of model() function may fit it.
Then focus on how the parameter values affect model fit, and how estimate_params() can improve the model() fit.

"""

    # Segment 1: task instructions (no image — sent first so the LLM reads
    # the objective before seeing the programs)
    segments: list[tuple[str, list[Path] | None]] = [(prompt, None)]

    # One segment per parent: program text [+ figure if available]
    for i, (prog, score) in enumerate(parents):
        parent_text = f"loss of model {i + 1}: {score:.4f}\n"
        parent_text += f"{prog.model_src}\n\n"
        parent_text += f"{prog.estimator_src}\n"
        parent_text += "\n----------------------------\n"
        img = [imgs[i]] if imgs[i] is not None else None
        segments.append((parent_text, img))

    return segments


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
    return (
        "You are an AI scientist performing symbolic regression. "
        "Your goal is to discover a compact mathematical function that fits the data well."
    )


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