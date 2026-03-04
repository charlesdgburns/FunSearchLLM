"""
search_loop.py
--------------
Main orchestration loop for FunSearch.

Workflow
--------
  1. run_seed_stage
       If config.use_seed_programs=True  → parse seed_programs.py, evaluate, write to funsearch/seed/
       If config.use_seed_programs=False → call seed_generator.generate_seeds(), evaluate, write to funsearch/seed/
  2. initialise_islands
       Generate n_programs_per_generation programs per island from seeds via LLM.
  3. run_evolution
       a. Build/update the island DataFrame
       b. For ALL islands simultaneously: gather LLM calls with asyncio.gather
       c. Evaluate ALL resulting programs in parallel via ProcessPoolExecutor
       d. Prune, deduplicate, and occasionally migrate
       e. Repeat for n_generations

Failed LLM responses (parse errors, empty outputs) are saved to
funsearch/failed/ for manual inspection.

Parallelism
-----------
  LLM calls  : asyncio.gather fires all n_islands × n_programs_per_generation
               requests simultaneously.
  evaluate() : ProcessPoolExecutor runs one worker per program. Each worker
               re-imports evaluate_programs.py, avoiding pickling issues.
               Safe because evaluate() is pure numpy/scipy.
"""

import asyncio
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
from seed_generator import generate_seeds

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

    # Annealing schedule
    temperature_explore: float = 1.5
    temperature_exploit: float = 0.5
    mode_threshold: float = 0.5

    # If True  → load seeds from seed_programs.py
    # If False → generate seeds via seed_generator.generate_seeds()
    use_seed_programs: bool = True
    use_image: bool = True
    random_seed: int = 42

    # Seed generation options (only used when use_seed_programs=False)
    seed_model: str | None = None      # overrides `model` for seed generation only
    seed_provider: str | None = None   # overrides `provider` for seed generation only
    n_seeds: int = 2                   # number of seeds to generate

    # Max parallel evaluate() workers. None → os.cpu_count()
    max_eval_workers: int | None = None

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
        if self.n_generations <= 1:
            return self.temperature_explore
        t = generation / (self.n_generations - 1)
        return self.temperature_explore + t * (self.temperature_exploit - self.temperature_explore)

    def mode_for(self, generation: int) -> Mode:
        return "explore" if self.temperature_for(generation) >= self.mode_threshold else "exploit"


# ---------------------------------------------------------------------------
# Worker function for ProcessPoolExecutor (must be module-level to be picklable)
# ---------------------------------------------------------------------------

def _evaluate_worker(args: tuple) -> tuple[float, dict]:
    """
    Re-imports evaluate_programs.py inside the worker process and calls
    evaluate(). Only plain serialisable arguments cross the process boundary.
    """
    score_path_str, prog, output_dir_str = args
    output_dir = Path(output_dir_str) if output_dir_str is not None else None

    spec = importlib.util.spec_from_file_location("evaluate_programs", score_path_str)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.evaluate(prog, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate_batch(
    jobs: list[tuple],
    max_workers: int | None,
) -> list[tuple[float, dict]]:
    """Run evaluate() for every job in parallel. Returns results in job order."""
    if not jobs:
        return []
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_evaluate_worker, job): i
                         for i, job in enumerate(jobs)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  [evaluate_batch] worker {idx} raised: {e}")
                results[idx] = (999.0, {"status": "failed", "reason": str(e), "score": 999.0})
    return results


async def _llm_call_one(
    caller: LLMCaller,
    segments: list[tuple[str, list[Path] | None]],
    label: str,
) -> tuple[str | None, str]:
    response = await caller.call_interleaved(segments)
    return response, label


async def _gather_llm_calls(
    tasks: list[tuple[LLMCaller, list, str]],
) -> list[tuple[str | None, str]]:
    """Fire all LLM calls simultaneously and return results in submission order."""
    coroutines = [_llm_call_one(caller, segments, label)
                  for caller, segments, label in tasks]
    return await asyncio.gather(*coroutines)


def _load_seed_parents(config: Config) -> list[tuple[ProgramStrings, float]]:
    df = island_manager.build_dataframe(config.funsearch_dir)
    seed_df = df[(df["island"] == "seed") & (df["status"] == "success")]
    seed_df = seed_df.sort_values("score", ascending=True)
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
    if len(parents) == 2:
        return sorted(parents, key=lambda x: x[1])
    scores = np.array([s for _, s in parents], dtype=float)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    idx = rng.choice(len(parents), size=2, replace=False, p=weights)
    return sorted([parents[i] for i in idx], key=lambda x: x[1])


def _rows_to_parent_tuples(rows: list[dict]) -> list[tuple[ProgramStrings, float]]:
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


def _write_prompt(segments: list[tuple[str, list | None]], program_dir: Path) -> None:
    """Save the text portions of the prompt to prompt.txt for inspection."""
    text = "\n\n--- [image] ---\n\n".join(
        block for block, _ in segments
    )
    (program_dir / "prompt.txt").write_text(text)


# ---------------------------------------------------------------------------
# Seed stage — handles both seeded and seedless runs
# ---------------------------------------------------------------------------

def run_seed_stage(config: Config) -> None:
    """
    Obtain seed programs, evaluate them, and write to funsearch/seed/.

    If config.use_seed_programs=True:
        Parses seed_programs.py and evaluates each seed found there.

    If config.use_seed_programs=False:
        Calls seed_generator.generate_seeds() to produce config.n_seeds
        programs from the data plot using the LLM, then evaluates them.

    In both cases, already-evaluated seeds (score.json exists) are skipped
    so the stage is safely resumable.
    """
    print("\n=== Seed stage ===")
    config.seed_dir.mkdir(parents=True, exist_ok=True)
    score_path = str(config.code_dir / "evaluate_programs.py")

    # ------------------------------------------------------------------
    # Obtain the list of ProgramStrings to evaluate
    # ------------------------------------------------------------------
    if config.use_seed_programs:
        seed_path = config.code_dir / "seed_programs.py"
        seeds = script_to_strings(seed_path)
        if not seeds:
            raise RuntimeError(f"No seed programs found in {seed_path}")
        print(f"  Loaded {len(seeds)} seed program(s) from {seed_path}.")
    else:
        print(f"  Generating {config.n_seeds} seed program(s) via LLM ...")
        seeds = generate_seeds(
            problem_dir=config.problem_dir,
            n_seeds=config.n_seeds,
            provider=config.seed_provider or config.provider,
            model=config.seed_model or config.model,
            temperature=config.temperature_explore,
        )
        if not seeds:
            raise RuntimeError("seed_generator produced no valid programs.")
        print(f"  Generated {len(seeds)} seed program(s).")

    # ------------------------------------------------------------------
    # Evaluate seeds not yet scored (in parallel)
    # ------------------------------------------------------------------
    jobs = []
    job_meta = []
    for i, prog in enumerate(seeds):
        program_name = f"program_{i + 1}"
        program_dir = config.seed_dir / program_name
        if (program_dir / "score.json").exists():
            print(f"  {program_name}: already evaluated, skipping.")
            continue
        program_dir.mkdir(parents=True, exist_ok=True)
        _write_program(prog, program_dir)
        jobs.append((score_path, prog, str(program_dir)))
        job_meta.append((program_name, program_dir))

    if jobs:
        print(f"  Evaluating {len(jobs)} seed(s) in parallel ...")
        results = _evaluate_batch(jobs, max_workers=config.max_eval_workers)
        for (program_name, program_dir), (score, metrics) in zip(job_meta, results):
            _write_score(metrics, program_dir)
            print(f"  {program_name}: score={score:.4f}  status={metrics['status']}")


# ---------------------------------------------------------------------------
# Island initialisation
# ---------------------------------------------------------------------------

async def _initialise_islands_async(config: Config) -> None:
    seed_parents = _load_seed_parents(config)
    if len(seed_parents) < 2:
        raise RuntimeError("Need at least 2 evaluated seed programs to initialise islands.")

    context = load_context(config.code_dir)
    score_path = str(config.code_dir / "evaluate_programs.py")
    rng = np.random.default_rng(config.random_seed)

    caller = LLMCaller(
        provider=config.provider,
        model=config.model,
        temperature=config.temperature_explore,
    )

    pending_islands = []
    for k in range(1, config.n_islands + 1):
        island_dir = config.island_dir(k)
        existing = list(island_dir.glob("program_*/score.json")) if island_dir.exists() else []
        if existing:
            print(f"  island_{k}: {len(existing)} program(s) already exist, skipping.")
            continue
        island_dir.mkdir(parents=True, exist_ok=True)
        pending_islands.append(k)

    if not pending_islands:
        return

    llm_tasks = []
    task_meta = []
    for k in pending_islands:
        for j in range(config.n_programs_per_generation):
            parents = _sample_two(seed_parents, rng)
            segments = build_prompt(parents, mode="explore", context=context,
                                    use_image=config.use_image)
            label = f"island_{k}_prog_{j+1}"
            llm_tasks.append((caller, segments, label))
            task_meta.append((k, j, segments))   # carry segments for prompt saving

    print(f"\n  Firing {len(llm_tasks)} LLM calls simultaneously across "
          f"{len(pending_islands)} islands ...")
    llm_results = await _gather_llm_calls(llm_tasks)

    eval_jobs = []
    eval_meta = []
    for (response, label), (k, j, segments) in zip(llm_results, task_meta):
        island_dir = config.island_dir(k)
        prog, fail_reason = _parse_response(response)
        if prog is None:
            island_manager.save_failed(config.funsearch_dir, response, fail_reason,
                                       prompt=label)
            print(f"  [{label}] FAILED ({fail_reason})")
            continue
        program_name = island_manager.next_program_name(island_dir)
        program_dir = island_dir / program_name
        program_dir.mkdir(parents=True, exist_ok=True)
        _write_program(prog, program_dir)
        _write_prompt(segments, program_dir)
        eval_jobs.append((str(score_path), prog, str(program_dir)))
        eval_meta.append((label, program_dir))

    print(f"  Evaluating {len(eval_jobs)} programs in parallel ...")
    eval_results = _evaluate_batch(eval_jobs, max_workers=config.max_eval_workers)
    for (label, program_dir), (score, metrics) in zip(eval_meta, eval_results):
        _write_score(metrics, program_dir)
        print(f"  [{label}] score={score:.4f}  status={metrics['status']}")


def initialise_islands(config: Config) -> None:
    print("\n=== Island initialisation ===")
    asyncio.run(_initialise_islands_async(config))


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

async def _run_evolution_async(config: Config) -> None:
    print("\n=== Evolution loop ===")

    context = load_context(config.code_dir)
    score_path = str(config.code_dir / "evaluate_programs.py")
    rng = np.random.default_rng(config.random_seed + 1)

    for gen in range(config.n_generations):
        temperature = config.temperature_for(gen)
        mode = config.mode_for(gen)
        print(f"\n--- Generation {gen + 1}/{config.n_generations}  "
              f"mode={mode}  temp={temperature:.3f} ---")

        df = island_manager.build_dataframe(config.funsearch_dir)
        island_manager.print_summary(df)

        caller = LLMCaller(
            provider=config.provider,
            model=config.model,
            temperature=temperature,
        )

        # Build all prompts for all islands × programs_per_generation
        llm_tasks = []
        task_meta = []

        for k in range(1, config.n_islands + 1):
            island_name = f"island_{k}"
            island_df = df[(df["island"] == island_name) & (df["status"] == "success")]
            if len(island_df) < 2:
                print(f"  [{island_name}] Skipping: fewer than 2 successful programs.")
                continue

            for j in range(config.n_programs_per_generation):
                try:
                    parent_rows = island_manager.select_parents(
                        df, island_name, n_parents=2, rng=rng)
                except ValueError as e:
                    print(f"  [{island_name}] {j + 1}: skipping ({e})")
                    continue

                parents = _rows_to_parent_tuples(parent_rows)
                if len(parents) < 2:
                    print(f"  [{island_name}] {j + 1}: could not load parents, skipping.")
                    continue

                image_paths = [row.get("image_dir") for row in parent_rows]
                segments = build_prompt(parents, mode=mode, context=context,
                                        image_paths=image_paths,
                                        use_image=config.use_image)
                prompt_text = " ".join(s[0] for s in segments)
                label = f"island_{k}_gen{gen+1}_prog{j+1}"
                llm_tasks.append((caller, segments, label))
                task_meta.append((k, j, prompt_text, segments))  # carry segments

        if not llm_tasks:
            print("  No valid tasks this generation, skipping.")
            continue

        # Fire ALL LLM calls simultaneously
        print(f"  Firing {len(llm_tasks)} LLM calls simultaneously ...")
        llm_results = await _gather_llm_calls(llm_tasks)

        # Parse responses, prepare evaluate() jobs
        eval_jobs = []
        eval_meta = []

        for (response, label), (k, j, prompt_text, segments) in zip(llm_results, task_meta):
            island_dir = config.island_dir(k)
            prog, fail_reason = _parse_response(response)
            if prog is None:
                island_manager.save_failed(config.funsearch_dir, response,
                                           fail_reason, prompt=prompt_text)
                print(f"  [{label}] FAILED ({fail_reason})")
                continue
            program_name = island_manager.next_program_name(island_dir)
            program_dir = island_dir / program_name
            program_dir.mkdir(parents=True, exist_ok=True)
            _write_program(prog, program_dir)
            _write_prompt(segments, program_dir)
            eval_jobs.append((score_path, prog, str(program_dir)))
            eval_meta.append((label, program_dir))

        # Evaluate all in parallel
        print(f"  Evaluating {len(eval_jobs)} programs in parallel ...")
        eval_results = _evaluate_batch(eval_jobs, max_workers=config.max_eval_workers)
        for (label, program_dir), (score, metrics) in zip(eval_meta, eval_results):
            _write_score(metrics, program_dir)
            print(f"  [{label}] score={score:.4f}  status={metrics['status']}")

        # Prune, deduplicate, migrate
        df = island_manager.build_dataframe(config.funsearch_dir)
        for k in range(1, config.n_islands + 1):
            island_name = f"island_{k}"
            df = island_manager.prune_island(df, island_name, config.max_population)
            df = island_manager.deduplicate_island(df, island_name)

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


def run_evolution(config: Config) -> None:
    asyncio.run(_run_evolution_async(config))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config: Config) -> None:
    """Full FunSearch run: seed stage → initialise islands → evolve."""
    run_seed_stage(config)
    initialise_islands(config)
    run_evolution(config)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config(
        problem_dir=Path("problems/sinewave"),
        n_islands=4,
        n_programs_per_generation=4,
        n_generations=10,
        max_population=10,
        migrate_every=2,
        use_seed_programs=False,   # ← set False to generate seeds via LLM
        use_image=True,
        provider="google",
        random_seed=42,
        # seed_provider="google",         # optional: different provider for seeds
        # seed_model="gemini-2.5-flash",  # optional: different model for seeds
        # n_seeds=2,                      # only used when use_seed_programs=False
        max_eval_workers=None,
    )

    print("FunSearch configuration:")
    print(f"  Problem       : {config.problem_dir}")
    print(f"  Islands       : {config.n_islands}")
    print(f"  Programs/gen  : {config.n_programs_per_generation}")
    print(f"  Generations   : {config.n_generations}")
    print(f"  Use seeds     : {config.use_seed_programs}")
    print(f"  Use images    : {config.use_image}")
    print(f"  Provider      : {config.provider}")
    print(f"  Eval workers  : {config.max_eval_workers or 'os.cpu_count()'}")

    run(config)