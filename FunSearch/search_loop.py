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
from FunSearch.prompt_builder import build_prompt, load_context

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

    # Mode schedule cycles over generations
    mode_schedule: list[Mode] = field(default_factory=lambda: ["explore", "exploit"])

    provider: str = "google"
    model: str | None = None
    temperature_explore: float = 1.2
    temperature_exploit: float = 0.6

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

    def mode_for(self, generation: int) -> Mode:
        return self.mode_schedule[generation % len(self.mode_schedule)]

    def temperature_for(self, mode: Mode) -> float:
        return self.temperature_explore if mode == "explore" else self.temperature_exploit


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
            prompt = build_prompt(parents, mode="explore", context=context)
            print(f"    [{island_name}] program {j + 1}/{config.n_programs_per_generation} ...", end=" ", flush=True)

            response = caller.call_sync(prompt)
            prog, fail_reason = _parse_response(response)

            if prog is None:
                island_manager.save_failed(config.funsearch_dir, response, fail_reason)
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
        mode = config.mode_for(gen)
        temperature = config.temperature_for(mode)
        print(f"\n--- Generation {gen + 1}/{config.n_generations}  mode={mode}  temp={temperature} ---")

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

            # Optionally include the best parent's evaluation figure
            best_row = max(parent_rows, key=lambda r: r["score"])
            image_path = best_row.get("image_dir")

            prompt = build_prompt(parents, mode=mode, context=context)
            print(f"  [{island_name}] Generating ...", end=" ", flush=True)

            response = caller.call_sync(prompt, image_path=image_path)
            prog, fail_reason = _parse_response(response)

            if prog is None:
                island_manager.save_failed(config.funsearch_dir, response, fail_reason)
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
        best = successful.iloc[0] #the dataframe is sorted in ascending, so ranking best first.
        print(f"Best program : {best['island']}/{best['program']}  score={best['score']:.4f}")
        print(f"Location     : {best['program_dir']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: Config) -> None:
    """Full FunSearch run: seed -> initialise -> evolve."""
    run_seed_stage(config)
    initialise_islands(config)
    run_evolution(config)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_score_fn(config: Config):
    """Dynamically import score.evaluate from the problem's code directory."""
    score_path = config.code_dir / "evaluate_programs.py"
    spec = importlib.util.spec_from_file_location("score", score_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate


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
        n_islands=4,
        n_programs_per_generation=6,
        n_generations=10,
        max_population=10,
        migrate_every=2,
        provider="google",
        random_seed=42,
    )

    print("FunSearch configuration:")
    print(f"  Problem     : {config.problem_dir}")
    print(f"  Islands     : {config.n_islands}")
    print(f"  Programs/gen: {config.n_programs_per_generation}")
    print(f"  Generations : {config.n_generations}")
    print(f"  Provider    : {config.provider}")

    run(config)