"""
seed_generator.py
-----------------
Generates seed programs from a scatter plot of the raw data using an LLM.

Used when Config.use_seed_programs=False in search_loop.py. Instead of
loading hand-written seeds from seed_programs.py, this module generates
n_seeds programs by showing the LLM only the data plot and asking it to
write model() and estimate_params() from scratch.

Usage
-----
    from seed_generator import generate_seeds

    progs = generate_seeds(
        problem_dir=Path("problems/sinewave"),
        n_seeds=2,
        provider="google",
        model="gemini-2.5-flash",
    )

The data plot is created automatically if it does not already exist, by
calling plot() from load_data.py (or falling back to a simple matplotlib
scatter if no plot() is defined).

API keys are read from .env (GOOGLE_API_KEY / ANTHROPIC_API_KEY).
"""

import asyncio
import importlib.util
import matplotlib.pyplot as plt
from pathlib import Path

from FunSearch.llm_caller import LLMCaller
from FunSearch.program_parser import ProgramStrings, llm_output_to_strings
from prompt_builder import SEED_TASK, CODE_GUIDE

# Default number of seed programs to generate
DEFAULT_N_SEEDS = 2

# Default model per provider — choose something capable for seed generation
DEFAULT_SEED_MODELS = {
    "google": "gemma-3-27b-it",
    "anthropic": "claude-sonnet-4-6",
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_seeds(
    problem_dir: Path,
    n_seeds: int = DEFAULT_N_SEEDS,
    provider: str = "google",
    model: str | None = None,
    temperature: float = 1.0,
) -> list[ProgramStrings]:
    """
    Generate seed programs from the raw data plot.

    Creates the data plot if it does not already exist, then fires
    n_seeds LLM calls simultaneously and returns successfully parsed
    programs.

    Parameters
    ----------
    problem_dir : root directory of the problem (contains code/ and data/)
    n_seeds     : number of seed programs to request (default: 2)
    provider    : "google" or "anthropic"
    model       : model name; defaults to a capable model per provider
    temperature : sampling temperature

    Returns
    -------
    List of ProgramStrings. May be shorter than n_seeds if some LLM
    calls fail to produce parseable code.
    """
    return asyncio.run(
        _generate_seeds_async(problem_dir, n_seeds, provider, model, temperature)
    )


# ---------------------------------------------------------------------------
# Async core
# ---------------------------------------------------------------------------

async def _generate_seeds_async(
    problem_dir: Path,
    n_seeds: int,
    provider: str,
    model: str | None,
    temperature: float,
) -> list[ProgramStrings]:
    resolved_model = model or DEFAULT_SEED_MODELS.get(provider)
    data_plot_path = _make_data_plot(problem_dir)

    caller = LLMCaller(provider=provider, model=resolved_model, temperature=temperature)
    segments = _build_seed_segments(data_plot_path)

    print(f"  [seed_generator] Firing {n_seeds} seed LLM calls simultaneously "
          f"(model={resolved_model}) ...")

    # Fire all calls simultaneously
    tasks = [caller.call_interleaved(segments) for _ in range(n_seeds)]
    responses = await asyncio.gather(*tasks)

    programs: list[ProgramStrings] = []
    for i, response in enumerate(responses):
        prog, fail_reason = _parse_response(response)
        if prog is None:
            print(f"  [seed_generator] seed {i + 1}: FAILED ({fail_reason})")
        else:
            print(f"  [seed_generator] seed {i + 1}: OK")
            programs.append(prog)

    print(f"  [seed_generator] {len(programs)}/{n_seeds} seeds generated successfully.")
    return programs


# ---------------------------------------------------------------------------
# Data plot
# ---------------------------------------------------------------------------

def _make_data_plot(problem_dir: Path) -> Path:
    """
    Generate a scatter plot of the raw data for use in seed prompts.

    Calls load_data.py's plot() function if it exists, saving the figure
    to <problem_dir>/data/data_plot.png. Returns the path to the plot.

    If load_data.py has no plot() function, falls back to a simple
    matplotlib scatter using the data returned by load_data().
    """
    plot_path = problem_dir / "data" / "data_plot.png"
    if plot_path.exists():
        print(f"  [seed_generator] Data plot already exists: {plot_path}")
        return plot_path

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    load_data_path = problem_dir / "code" / "load_data.py"
    spec = importlib.util.spec_from_file_location("load_data", load_data_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "plot"):
        module.plot(save_path=plot_path)
    else:
        data = module.load_data()
        x, y = data[0], data[1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x, y, s=12, alpha=0.7, color="steelblue", label="data")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Data")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

    print(f"  [seed_generator] Data plot saved: {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_seed_segments(data_plot_path: Path) -> list[tuple[str, list[Path] | None]]:
    """
    Build prompt segments for seed generation: task description + data plot.
    Reuses SEED_TASK and CODE_GUIDE from prompt_builder to stay consistent.
    """
    text = SEED_TASK + CODE_GUIDE
    image = [Path(data_plot_path)] if data_plot_path is not None else None
    return [(text, image)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_response(response: str | None) -> tuple[ProgramStrings | None, str]:
    if response is None:
        return None, "empty response from LLM"
    prog = llm_output_to_strings(response)
    if prog is None:
        return None, "could not extract model() and estimate_params() from response"
    return prog, ""


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    problem_dir = Path("problems/sinewave")
    print(f"Generating seeds for: {problem_dir}\n")

    programs = generate_seeds(
        problem_dir=problem_dir,
        n_seeds=DEFAULT_N_SEEDS,
        provider="google",
    )

    for i, prog in enumerate(programs):
        print(f"\n{'=' * 50}")
        print(f"Seed {i + 1}")
        print(f"{'=' * 50}")
        print(prog.combined())