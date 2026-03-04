"""
search_loop.py
--------------
Main orchestration loop for FunSearch.

Workflow:
  1. Load seed programs, evaluate them, write to funsearch/seed/
  2. Initialise n_islands by generating n_programs_per_generation programs
     from seeds via LLM, writing each to funsearch/island_k/
  3. Run the main evolution loop:
       a. Build/update the island DataFrame
       b. For each island: select parents, call LLM, evaluate, write to disk
       c. Prune, deduplicate, and occasionally migrate
       d. Repeat for n_generations

Failed LLM responses (parse errors, empty outputs) are saved to
funsearch/failed/ for manual inspection.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from FunSearch import island_manager
from FunSearch.llm_caller import LLMCaller
from FunSearch.program_parser import (
    ProgramStrings,
    script_to_strings,
    llm_output_to_strings,
)
from FunSearch.prompt_builder import build_prompt, build_seed_prompt, load_context

Mode = Literal["explore", "exploit"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    problem_dir: Path

    n_islands: int = 3
    n_programs_per_generation: int = 4
    n_generations: int = 20
    max_population: int = 10
    migrate_every: int = 5
    n_migrate: int = 1

    provider: str = "google"
    model: str | None = None

    # Annealing schedule: temperature declines linearly from explore -> exploit
    # over all generations. Mode switches from explore to exploit once the
    # temperature crosses mode_threshold.
    temperature_explore: float = 1.2
    temperature_exploit: float = 0.6
    mode_threshold: float = 1.0   # temperature below this -> exploit

    use_seed_programs: bool = True   # False -> LLM generates seeds from data plot
    use_image: bool = True
    random_seed: int = 42

    @property
    def seed_dir(self) -> Path:
        return self.problem_dir / "funsearch" / "seed"

    @property
    def funsearch_dir(self) -> Path:
        return self.problem_dir / "funsearch"

    @property
    def code_dir(self) -> Path:
        return self.problem_dir / "code"

    def island_dir(self, k: int) -> Path:
        return self.funsearch_dir / f"island_{k}"

    def temperature_for(self, generation: int) -> float:
        """Linearly anneal from temperature_explore to temperature_exploit."""
        if self.n_generations <= 1:
            return self.temperature_explore
        t = generation / (self.n_generations - 1)   # 0.0 -> 1.0
        return self.temperature_explore + t * (self.temperature_exploit - self.temperature_explore)

    def mode_for(self, generation: int) -> Mode:
        """Explore while temperature is above threshold, exploit below."""
        return "explore" if self.temperature_for(generation) >= self.mode_threshold else "exploit"


# ---------------------------------------------------------------------------
# Seed stage
# ---------------------------------------------------------------------------

def run_seed_stage(config: Config) -> None:
    """
    Parse seed_programs.py, evaluate each seed, and write to funsearch/seed/.
    Skips seeds already evaluated (score.json exists).
    """
    print("\n=== Seed stage ===")
    config.seed_dir.mkdir(parents=True, exist_ok=True)

    seed_path = config.code_dir / "seed_programs.py"
    seeds = script_to_strings(seed_path)
    if not seeds:
        raise RuntimeError(f"No seed programs found in {seed_path}")
    print(f"Found {len(seeds)} seed program(s).")

    score_fn = _load_score_fn(config)

    for i, prog in enumerate(seeds):
        program_name = f"program_{i + 1}"
        program_dir = config.seed_dir / program_name

        if (program_dir / "score.json").exists():
            print(f"  {program_name}: already evaluated, skipping.")
            continue

        program_dir.mkdir(parents=True, exist_ok=True)
        _write_program(prog, program_dir)
        score, metrics = score_fn(prog, output_dir=program_dir)
        _write_score(metrics, program_dir)
        print(f"  {program_name}: score={score:.4f}  status={metrics['status']}")


# ---------------------------------------------------------------------------
# Island initialisation
# ---------------------------------------------------------------------------

def initialise_islands(config: Config) -> None:
    """
    For each island, generate n_programs_per_generation programs from seeds
    via LLM, evaluate, and write to disk. Islands with existing programs
    are skipped (resumable).
    """
    print("\n=== Island initialisation ===")

    seed_parents = _load_seed_parents(config)
    if len(seed_parents) < 2:
        raise RuntimeError("Need at least 2 evaluated seed programs to initialise islands.")

    context = load_context(config.code_dir)
    score_fn = _load_score_fn(config)
    rng = np.random.default_rng(config.random_seed)

    caller = LLMCaller(
        provider=config.provider,
        model=config.model,
        temperature=config.temperature_explore,
    )

    for k in range(1, config.n_islands + 1):
        island_dir = config.island_dir(k)
        island_name = f"island_{k}"

        existing = list(island_dir.glob("program_*/score.json")) if island_dir.exists() else []
        if existing:
            print(f"  {island_name}: {len(existing)} program(s) already exist, skipping.")
            continue

        print(f"\n  Initialising {island_name} ...")
        island_dir.mkdir(parents=True, exist_ok=True)

        for j in range(config.n_programs_per_generation):
            parents = _sample_two(seed_parents, rng)
            segments = build_prompt(parents, mode="explore", context=context, use_image=config.use_image)
            print(f"    [{island_name}] program {j + 1}/{config.n_programs_per_generation} ...", end=" ", flush=True)

            response = caller.call_interleaved_sync(segments)
            prog, fail_reason = _parse_response(response)

            if prog is None:
                island_manager.save_failed(config.funsearch_dir, response, fail_reason, prompt=prompt)
                print(f"FAILED ({fail_reason})")
                continue

            program_name = island_manager.next_program_name(island_dir)
            program_dir = island_dir / program_name
            program_dir.mkdir(parents=True, exist_ok=True)

            _write_program(prog, program_dir)
            score, metrics = score_fn(prog, output_dir=program_dir)
            _write_score(metrics, program_dir)
            print(f"score={score:.4f}  status={metrics['status']}")


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def run_evolution(config: Config) -> None:
    """
    Main FunSearch loop. Assumes seed stage and initialisation are complete.
    """
    print("\n=== Evolution loop ===")

    context = load_context(config.code_dir)
    score_fn = _load_score_fn(config)
    rng = np.random.default_rng(config.random_seed + 1)

    for gen in range(config.n_generations):
        temperature = config.temperature_for(gen)
        mode = config.mode_for(gen)
        print(f"\n--- Generation {gen + 1}/{config.n_generations}  mode={mode}  temp={temperature:.3f} ---")

        df = island_manager.build_dataframe(config.funsearch_dir)
        island_manager.print_summary(df)

        caller = LLMCaller(
            provider=config.provider,
            model=config.model,
            temperature=temperature,
        )

        for k in range(1, config.n_islands + 1):
            island_name = f"island_{k}"
            island_dir = config.island_dir(k)

            try:
                parent_rows = island_manager.select_parents(df, island_name, n_parents=2, rng=rng)
            except ValueError as e:
                print(f"  [{island_name}] Skipping: {e}")
                continue

            parents = _rows_to_parent_tuples(parent_rows)
            if len(parents) < 2:
                print(f"  [{island_name}] Could not load parent programs from disk, skipping.")
                continue

            # Collect evaluation figures for each parent (worst-first order)
            image_paths = [row.get("image_dir") for row in parent_rows]

            segments = build_prompt(parents, mode=mode, context=context, image_paths=image_paths, use_image=config.use_image)
            print(f"  [{island_name}] Generating ...", end=" ", flush=True)

            response = caller.call_interleaved_sync(segments)
            prog, fail_reason = _parse_response(response)

            if prog is None:
                island_manager.save_failed(config.funsearch_dir, response, fail_reason, prompt=prompt)
                print(f"FAILED ({fail_reason})")
                continue

            program_name = island_manager.next_program_name(island_dir)
            program_dir = island_dir / program_name
            program_dir.mkdir(parents=True, exist_ok=True)

            _write_program(prog, program_dir)
            score, metrics = score_fn(prog, output_dir=program_dir)
            _write_score(metrics, program_dir)
            print(f"score={score:.4f}  status={metrics['status']}")

        # Prune and deduplicate
        df = island_manager.build_dataframe(config.funsearch_dir)
        for k in range(1, config.n_islands + 1):
            island_name = f"island_{k}"
            df = island_manager.prune_island(df, island_name, config.max_population)
            df = island_manager.deduplicate_island(df, island_name)

        # Periodic migration
        if (gen + 1) % config.migrate_every == 0:
            print("  [migration] Migrating programs across islands ...")
            df = island_manager.migrate_programs(
                df, config.funsearch_dir, n_migrate=config.n_migrate, rng=rng
            )

    print("\n=== Evolution complete ===")
    df = island_manager.build_dataframe(config.funsearch_dir)
    island_manager.print_summary(df)
    successful = df[df["status"] == "success"]
    if not successful.empty:
        best = successful.iloc[0]
        print(f"Best program : {best['island']}/{best['program']}  score={best['score']:.4f}")
        print(f"Location     : {best['program_dir']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: Config) -> None:
    """Full FunSearch run: seed -> initialise -> evolve."""
    if config.use_seed_programs:
        run_seed_stage(config)
        initialise_islands(config)
    else:
        initialise_islands_from_data(config)
    run_evolution(config)



def initialise_islands_from_data(config: Config) -> None:
    """
    Initialise islands without seed programs.

    Generates n_islands * n_programs_per_generation programs by prompting
    the LLM with only a scatter plot of the raw data. No human-written
    seed functions are used — the LLM must infer a suitable model form
    from the data plot alone.

    The data plot is created by calling load_data.py's plot() function,
    or generated fresh if it does not already exist.
    """
    print("\n=== Seedless initialisation (LLM from data plot) ===")

    data_plot_path = _make_data_plot(config)
    context = load_context(config.code_dir)
    score_fn = _load_score_fn(config)
    rng = np.random.default_rng(config.random_seed)

    caller = LLMCaller(
        provider=config.provider,
        model=config.model,
        temperature=config.temperature_explore,
    )

    for k in range(1, config.n_islands + 1):
        island_dir = config.island_dir(k)
        island_name = f"island_{k}"

        existing = list(island_dir.glob("program_*/score.json")) if island_dir.exists() else []
        if existing:
            print(f"  {island_name}: {len(existing)} program(s) already exist, skipping.")
            continue

        print(f"\n  Initialising {island_name} ...")
        island_dir.mkdir(parents=True, exist_ok=True)

        for j in range(config.n_programs_per_generation):
            segments = build_seed_prompt(data_plot_path=data_plot_path, context=context)
            print(f"    [{island_name}] program {j + 1}/{config.n_programs_per_generation} ...", end=" ", flush=True)

            response = caller.call_interleaved_sync(segments)
            prog, fail_reason = _parse_response(response)

            if prog is None:
                # Reconstruct prompt text for saving alongside failure
                prompt_text = segments[0][0] if segments else ""
                island_manager.save_failed(config.funsearch_dir, response, fail_reason, prompt=prompt_text)
                print(f"FAILED ({fail_reason})")
                continue

            program_name = island_manager.next_program_name(island_dir)
            program_dir = island_dir / program_name
            program_dir.mkdir(parents=True, exist_ok=True)

            _write_program(prog, program_dir)
            score, metrics = score_fn(prog, output_dir=program_dir)
            _write_score(metrics, program_dir)
            print(f"score={score:.4f}  status={metrics['status']}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_score_fn(config: Config):
    """Dynamically import evaluate from the problem's evaluate_programs.py."""
    score_path = config.code_dir / "evaluate_programs.py"
    spec = importlib.util.spec_from_file_location("evaluate_programs", score_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


def _make_data_plot(config: Config) -> Path:
    """
    Generate a scatter plot of the raw data for use in seedless prompts.

    Calls load_data.py's plot() function if it exists, saving the figure
    to problems/<n>/data/data_plot.png. Returns the path to the plot.

    If load_data.py has no plot() function, falls back to a simple
    matplotlib scatter using the data returned by load_data().
    """
    import importlib.util
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = config.problem_dir / "data" / "data_plot.png"
    if plot_path.exists():
        print(f"  Data plot already exists: {plot_path}")
        return plot_path

    load_data_path = config.code_dir / "load_data.py"
    spec = importlib.util.spec_from_file_location("load", load_data_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Use plot() if the problem defines one, otherwise build a simple scatter
    if hasattr(module, "plot"):
        module.plot(save_path=plot_path)
    else:
        data = module.load()
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

    print(f"  Data plot saved: {plot_path}")
    return plot_path


def _load_seed_parents(config: Config) -> list[tuple[ProgramStrings, float]]:
    """Read evaluated seed programs and return as (ProgramStrings, score) tuples."""
    df = island_manager.build_dataframe(config.funsearch_dir)
    seed_df = df[(df["island"] == "seed") & (df["status"] == "success")]
    seed_df = seed_df.sort_values("score", ascending=True)  # worst-first

    parents = []
    for _, row in seed_df.iterrows():
        prog_path = Path(row["program_dir"]) / "program.py"
        if not prog_path.exists():
            continue
        prog = llm_output_to_strings(prog_path.read_text())
        if prog is not None:
            parents.append((prog, float(row["score"])))
    return parents


def _sample_two(
    parents: list[tuple[ProgramStrings, float]],
    rng: np.random.Generator,
) -> list[tuple[ProgramStrings, float]]:
    """Score-weighted sample of 2 parents. Returns worst-first."""
    if len(parents) == 2:
        return sorted(parents, key=lambda x: x[1])

    scores = np.array([s for _, s in parents], dtype=float)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    idx = rng.choice(len(parents), size=2, replace=False, p=weights)
    return sorted([parents[i] for i in idx], key=lambda x: x[1])


def _rows_to_parent_tuples(rows: list[dict]) -> list[tuple[ProgramStrings, float]]:
    """Convert island_manager row dicts to (ProgramStrings, score) tuples."""
    parents = []
    for row in rows:
        prog_path = Path(row["program_dir"]) / "program.py"
        if not prog_path.exists():
            continue
        prog = llm_output_to_strings(prog_path.read_text())
        if prog is not None:
            parents.append((prog, float(row["score"])))
    return parents


def _parse_response(response: str | None) -> tuple[ProgramStrings | None, str]:
    """
    Try to parse an LLM response into ProgramStrings.
    Returns (ProgramStrings, '') on success, (None, reason) on failure.
    """
    if response is None:
        return None, "empty response from LLM"
    prog = llm_output_to_strings(response)
    if prog is None:
        return None, "could not extract model() and estimate_params() from response"
    return prog, ""


def _write_program(prog: ProgramStrings, program_dir: Path) -> None:
    (program_dir / "program.py").write_text(prog.combined())


def _write_score(metrics: dict, program_dir: Path) -> None:
    with open(program_dir / "score.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config(
        problem_dir=Path("problems/sinewave"),
        n_islands=6,
        n_programs_per_generation=8,
        n_generations=10,
        max_population=10,
        migrate_every=2,
        use_seed_programs=False,   # try seedless initialisation
        use_image=True,
        provider="google",
        random_seed=42,
    )

    print("FunSearch configuration:")
    print(f"  Problem       : {config.problem_dir}")
    print(f"  Islands       : {config.n_islands}")
    print(f"  Programs/gen  : {config.n_programs_per_generation}")
    print(f"  Generations   : {config.n_generations}")
    print(f"  Use seeds     : {config.use_seed_programs}")
    print(f"  Use images    : {config.use_image}")
    print(f"  Provider      : {config.provider}")

    run(config)