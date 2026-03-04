"""
prompt_builder.py
-----------------
Constructs LLM prompts for the FunSearch evolution loop.

Prompt blocks are defined as module-level constants, making it easy to
see, edit, and recombine them. Each build_* function assembles blocks
into an interleaved list of (text, image_paths) segments for
llm_caller.call_interleaved().

Two public builders:

  build_prompt(parents, mode, context, ...)
      Standard evolution prompt — shown parent programs + task instructions.

  build_seed_prompt(data_plot_path, context)
      No-seed initialisation prompt — shown only the data plot, asked to
      write model() and estimate_params() from scratch.

Segment format returned by both builders:
  list[tuple[str, list[Path] | None]]
  Each tuple is (text_block, image_paths_or_None).
  Consumed by llm_caller.call_interleaved().
"""

from pathlib import Path
from typing import Literal

from FunSearch.program_parser import ProgramStrings

Mode = Literal["explore", "exploit"]

Segments = list[tuple[str, list[Path] | None]]


# ---------------------------------------------------------------------------
# Prompt blocks — edit these to change prompt behaviour globally
# ---------------------------------------------------------------------------

TASK = """\
You are an AI scientist. The programs below are models and their associated \
parameter estimators. The models are sorted from highest loss to lowest loss.

Your task is to create a new model() and the associated estimate_params() \
that has a lower loss than the models below.

*Analyze* the progression of the models, *generalize* the improvements, \
and *create* a new model that is better than *all* previous models.

"""

EXPLORE = """\
Use the models and parameter estimators below as inspiration, but be *creative* \
and *invent* something new. Which features in the models below correlate with \
lower loss? Find those and *extrapolate* them. You should also *combine* features \
from several models, and *experiment* with new ideas.

"""

EXPLOIT = """\
Use the models and parameter estimators below as a *template* to create an \
improved, simpler model. Focus on *exploiting* the strengths of the existing \
models and *eliminating* their weaknesses or *redundancies*. You will be \
*penalized* for complexity, so make the new model and parameter estimator as \
*simple* as possible while still being better than the previous models.

"""

CODE_GUIDE = """\
**Code Generation Guidelines:**
* Import any packages you use.
* Do not include any text other than the code.
* Ensure all free parameters are numeric, not strings.
* Output only a single improved program, which must have two functions named \
exactly `model` and `estimate_params`.
* `model(x, params)` takes an array of inputs and a parameter vector, and \
returns predicted outputs.
* `estimate_params(x, y)` takes inputs and observed outputs, and returns an \
initial parameter guess. Do not use curve_fit, minimize, or any iterative \
fitting — this is a starting point for gradient-based optimisation.
"""

IMAGE_GUIDE = """\

**Image Analysis Instructions:**
Attached are scatter plots where the data is plotted and each model curve is \
drawn **using the functions shown below**.
Pay attention to the data and what kind of model() function may fit it.
Then focus on how the parameter values affect model fit, and how \
estimate_params() can improve the model() fit.

"""

SEED_TASK = """\
You are an AI scientist doing symbolic regression. \
Pay close attention to the attached image. It shows some data (blue scatter \
points) that we need to model the relationship for. \
Please write two python functions which will be used to fit the data. \
Make sure to import all the packages you will be using, such as \
`import numpy as np`.\

`def model(x, params):`

`def estimate_params(x, y):`
"""


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_prompt(
    parents: list[tuple[ProgramStrings, float]],
    mode: Mode,
    context: str,
    image_paths: list[Path | None] | None = None,
    use_image: bool = True,
) -> Segments:
    """
    Build the standard evolution prompt as interleaved (text, images) segments.

    Segment order:
      1. context + TASK + EXPLORE/EXPLOIT + CODE_GUIDE [+ IMAGE_GUIDE]  (no image)
      2. parent program 1 text  [+ figure 1 if use_image and available]
      3. parent program 2 text  [+ figure 2 if use_image and available]

    Parameters
    ----------
    parents     : exactly 2 (ProgramStrings, score) tuples, worst-first
    mode        : "explore" or "exploit"
    context     : problem description (from prompt_context.txt)
    image_paths : one Path-or-None per parent; ignored if use_image=False
    use_image   : whether to attach evaluation figures and IMAGE_GUIDE

    Returns
    -------
    Segments for llm_caller.call_interleaved().
    """
    assert len(parents) == 2, "build_prompt expects exactly 2 parent programs"
    assert mode in ("explore", "exploit"), f"Unknown mode: {mode!r}"

    mode_block = EXPLORE if mode == "explore" else EXPLOIT
    image_block = IMAGE_GUIDE if use_image else ""

    instruction = context + "\n\n" + TASK + mode_block + CODE_GUIDE + image_block

    imgs = _resolve_image_paths(parents, image_paths, use_image)

    # Segment 1: instructions (no image — LLM reads objective before programs)
    segments: Segments = [(instruction, None)]

    # One segment per parent: code text [+ figure]
    for i, (prog, score) in enumerate(parents):
        text = (
            f"loss of model {i + 1}: {score:.4f}\n"
            f"{prog.model_src}\n\n"
            f"{prog.estimator_src}\n"
            "\n----------------------------\n"
        )
        img = [imgs[i]] if imgs[i] is not None else None
        segments.append((text, img))

    return segments


def build_seed_prompt(
    data_plot_path: Path,
    context: str,
) -> Segments:
    """
    Build a no-seed initialisation prompt: the LLM sees only the data plot
    and writes model() + estimate_params() from scratch.

    Used when use_seed_programs=False in the search loop.

    Parameters
    ----------
    data_plot_path : path to a PNG scatter plot of the raw data
    context        : problem description (from prompt_context.txt)

    Returns
    -------
    A single-segment Segments list with the data plot attached.
    """
    text = SEED_TASK + CODE_GUIDE
    image = [Path(data_plot_path)] if data_plot_path is not None else None
    return [(text, image)]


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
    print(f"[prompt_builder] Warning: no prompt_context.txt at {context_path}")
    return ("")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_image_paths(
    parents: list,
    image_paths: list[Path | None] | None,
    use_image: bool,
) -> list[Path | None]:
    """Return one resolved Path-or-None per parent."""
    if not use_image or not image_paths:
        return [None] * len(parents)
    result = []
    for i in range(len(parents)):
        p = image_paths[i] if i < len(image_paths) else None
        result.append(Path(p) if p is not None else None)
    return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from FunSearch.program_parser import script_to_strings

    context = load_context(Path("problems/sinewave/code"))

    print("=" * 60)
    print("SEED PROMPT (no seed programs)")
    print("=" * 60)
    seed_segs = build_seed_prompt(
        data_plot_path=Path("problems/sinewave/data/data_plot.png"),
        context=context,
    )
    for i, (text, imgs) in enumerate(seed_segs):
        print(f"Segment {i+1}: {len(text)} chars | images={imgs}")
        print(text[:300])

    seed_path = Path("problems/sinewave/code/seed_programs.py")
    if seed_path.exists():
        programs = script_to_strings(seed_path)
        parents = [(programs[0], -0.45), (programs[1], 0.82)]

        print("\n" + "=" * 60)
        print("EVOLUTION PROMPT — explore, no images")
        print("=" * 60)
        segs = build_prompt(parents, mode="explore", context=context, use_image=True)
        for i, (text, imgs) in enumerate(segs):
            print(f"Segment {i+1}: {len(text)} chars | images={imgs}")

        print("\n" + "=" * 60)
        print("EVOLUTION PROMPT — exploit, with fake image paths")
        print("=" * 60)
        segs = build_prompt(
            parents, mode="exploit", context=context,
            image_paths=[Path("fig1.png"), Path("fig2.png")],
            use_image=True,
        )
        for i, (text, imgs) in enumerate(segs):
            print(f"Segment {i+1}: {len(text)} chars | images={imgs}")