"""
island_manager.py
-----------------
Manages the population of programs across islands.

Responsibilities:
  - Build and maintain a pandas DataFrame index of all evaluated programs
  - Select parents for the next generation (score-weighted sampling)
  - Prune islands to max_population, removing worst-performing program folders
  - Deduplicate similar programs within an island (by score similarity)
  - Migrate top programs across islands periodically

The DataFrame is rebuilt from disk at startup, making runs resumable.

Columns:
    island      : str  — 'seed', 'island_1', 'island_2', ...
    program     : str  — 'program_1', 'program_2', ...
    program_dir : Path — path to the program folder
    image_dir   : Path — path to evaluation_figure.png (or None if absent)
    score       : float
    status      : str  — 'success' or 'failed'
    + any additional metrics from score.json

Failed programs and raw LLM responses are saved under:
    funsearch/failed/program_k/
        raw_response.txt   — the raw LLM text that failed to parse
        reason.txt         — brief description of what went wrong
"""

import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path


FIGURE_NAME = "evaluation_figure.png"
FAILED_DIR_NAME = "failed"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def build_dataframe(funsearch_dir: Path) -> pd.DataFrame:
    """
    Scan the funsearch output directory and build a DataFrame of all
    evaluated programs found on disk.

    Reads score.json from each program folder. Programs without a score.json
    are skipped with a warning. Failed programs (in funsearch/failed/) are
    included with status='failed'.

    Parameters
    ----------
    funsearch_dir : path to problems/<n>/funsearch/

    Returns
    -------
    DataFrame with one row per program, sorted by score descending.
    """
    rows = []
    funsearch_dir = Path(funsearch_dir)

    for island_dir in sorted(funsearch_dir.iterdir()):
        if not island_dir.is_dir():
            continue
        island_name = island_dir.name  # 'seed', 'island_1', 'failed', etc.

        for program_dir in sorted(island_dir.iterdir()):
            if not program_dir.is_dir():
                continue

            score_path = program_dir / "score.json"
            if not score_path.exists():
                print(f"[island_manager] No score.json in {program_dir}, skipping.")
                continue

            with open(score_path) as f:
                metrics = json.load(f)

            figure_path = program_dir / FIGURE_NAME
            image_dir = figure_path if figure_path.exists() else None

            row = {
                "island": island_name,
                "program": program_dir.name,
                "program_dir": program_dir,
                "image_dir": image_dir,
                "score": float(metrics.get("score", 999.0)),
                "status": metrics.get("status", "unknown"),
            }
            for k, v in metrics.items():
                if k not in row:
                    row[k] = v
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["island", "program", "program_dir", "image_dir", "score", "status"]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=True).reset_index(drop=True)
    df.to_csv(funsearch_dir/'model_rankings.htsv', sep='\t')
    return df


def save_failed(
    funsearch_dir: Path,
    raw_response: str | None,
    reason: str,
    prompt: str | None,
) -> None:
    """
    Save a failed LLM response to funsearch/failed/ for manual inspection.

    Parameters
    ----------
    funsearch_dir : root funsearch output directory
    raw_response  : the raw LLM text that failed to parse (may be None)
    reason        : brief description of the failure
    """
    failed_root = Path(funsearch_dir) / FAILED_DIR_NAME
    failed_root.mkdir(parents=True, exist_ok=True)

    # Find next available program folder name
    program_name = next_program_name(failed_root)
    program_dir = failed_root / program_name
    program_dir.mkdir(parents=True, exist_ok=True)

    # Write reason
    (program_dir / "reason.txt").write_text(reason)

    #Write prompt if available
    if prompt is not None:
        (program_dir/"raw_response.txt").write_text(prompt)
    # Write raw LLM response if available
    if raw_response is not None:
        (program_dir / "raw_response.txt").write_text(raw_response)

    # Write a minimal score.json so build_dataframe can read it
    metrics = {"status": "failed", "reason": reason, "score": 999.0}
    with open(program_dir / "score.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[island_manager] Saved failed response to {program_dir}")


def select_parents(
    df: pd.DataFrame,
    island: str,
    n_parents: int = 2,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """
    Select parent programs from an island using score-weighted sampling
    from the top half of the island population.

    Parameters
    ----------
    df        : full program DataFrame (from build_dataframe)
    island    : island name to sample from
    n_parents : number of parents to select (default 2)
    rng       : numpy random Generator for reproducibility

    Returns
    -------
    List of n_parents row dicts, ordered worst-first so the LLM sees
    a clear improvement direction.
    """
    if rng is None:
        rng = np.random.default_rng()

    island_df = df[(df["island"] == island) & (df["status"] == "success")].copy()

    if len(island_df) < n_parents:
        raise ValueError(
            f"Island '{island}' has {len(island_df)} successful programs, "
            f"need at least {n_parents}."
        )

    # Restrict to top half
    top_half = island_df.head(max(n_parents, len(island_df) // 2))

    # Softmax-weighted sampling without replacement
    scores = top_half["score"].values.astype(float)
    weights = np.exp(1/scores)
    weights /= weights.sum()

    chosen_idx = rng.choice(len(top_half), size=n_parents, replace=False, p=weights)
    chosen = top_half.iloc[chosen_idx]

    # Worst-first for the prompt
    return chosen.sort_values("score", ascending=False).to_dict(orient="records")


def prune_island(
    df: pd.DataFrame,
    island: str,
    max_population: int,
) -> pd.DataFrame:
    """
    Remove the worst-scoring program folders from an island until it has
    at most max_population successful programs.
    Failed programs are always removed first.
    """
    island_df = df[df["island"] == island].copy()

    # Remove failed programs first
    failed = island_df[island_df["status"] != "success"]
    for _, row in failed.iterrows():
        _remove_program_dir(row["program_dir"])
    df = df.drop(failed.index)
    island_df = df[df["island"] == island].copy()

    # Remove worst-scoring successes if over limit
    if len(island_df) > max_population:
        n_remove = len(island_df) - max_population
        to_remove = island_df.sort_values("score", ascending=False).head(n_remove)
        for _, row in to_remove.iterrows():
            _remove_program_dir(row["program_dir"])
        df = df.drop(to_remove.index)

    return df.reset_index(drop=True)


def deduplicate_island(
    df: pd.DataFrame,
    island: str,
    score_tolerance: float = 1e-4,
) -> pd.DataFrame:
    """
    Remove near-duplicate programs within an island.
    Two programs are duplicates if their scores differ by less than
    score_tolerance. The lower-scoring duplicate is removed.
    """
    island_df = df[df["island"] == island].sort_values("score", ascending=False)
    seen_scores: list[float] = []
    to_remove = []

    for idx, row in island_df.iterrows():
        score = row["score"]
        if any(abs(score - s) < score_tolerance for s in seen_scores):
            to_remove.append(idx)
            _remove_program_dir(row["program_dir"])
        else:
            seen_scores.append(score)

    if to_remove:
        print(f"[island_manager] Removed {len(to_remove)} duplicate(s) from {island}.")

    return df.drop(to_remove).reset_index(drop=True)


def migrate_programs(
    df: pd.DataFrame,
    funsearch_dir: Path,
    n_migrate: int = 1,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Move top-scoring programs from each island to a randomly chosen
    different island. The DataFrame is updated to reflect new locations.
    """
    if rng is None:
        rng = np.random.default_rng()

    funsearch_dir = Path(funsearch_dir)
    islands = [i for i in df["island"].unique() if i not in ("seed", FAILED_DIR_NAME)]

    if len(islands) < 2:
        return df

    updates = []
    for island in islands:
        island_df = df[(df["island"] == island) & (df["status"] == "success")]
        if island_df.empty:
            continue

        migrants = island_df.sort_values("score", ascending=False).head(n_migrate)
        other_islands = [i for i in islands if i != island]
        target = rng.choice(other_islands)
        target_dir = funsearch_dir / target

        for _, row in migrants.iterrows():
            src = Path(row["program_dir"])
            new_name = next_program_name(target_dir)
            dst = target_dir / new_name
            shutil.move(str(src), str(dst))
            print(f"[island_manager] Migrated {src.name}: {island} -> {target}/{new_name}")

            # Update image_dir path if figure exists in new location
            new_figure = dst / FIGURE_NAME
            new_image_dir = new_figure if new_figure.exists() else None

            updates.append((row.name, target, new_name, dst, new_image_dir))

    for idx, new_island, new_program, new_dir, new_image_dir in updates:
        if idx in df.index:
            df.at[idx, "island"] = new_island
            df.at[idx, "program"] = new_program
            df.at[idx, "program_dir"] = new_dir
            df.at[idx, "image_dir"] = new_image_dir

    return df.reset_index(drop=True)


def next_program_name(island_dir: Path) -> str:
    """
    Return the next available program folder name within an island directory.
    e.g. if program_1 and program_2 exist, returns 'program_3'.
    """
    existing = [
        d.name for d in Path(island_dir).iterdir()
        if d.is_dir() and d.name.startswith("program_")
    ] if Path(island_dir).exists() else []

    indices = []
    for name in existing:
        try:
            indices.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass

    return f"program_{max(indices) + 1 if indices else 1}"


def print_summary(df: pd.DataFrame) -> None:
    """Print a brief per-island summary of the current population."""
    print("\n[island_manager] Population summary:")
    skip = {FAILED_DIR_NAME}
    for island, group in df.groupby("island"):
        if island in skip:
            continue
        successful = group[group["status"] == "success"]
        best = f"{successful['score'].min():.4f}" if not successful.empty else "—"
        print(f"  {island:15s}: {len(group):3d} programs | best score: {best}")

    failed = df[df["island"] == FAILED_DIR_NAME]
    if not failed.empty:
        print(f"  {'failed':15s}: {len(failed):3d} responses saved for inspection")
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_program_dir(program_dir: Path) -> None:
    program_dir = Path(program_dir)
    if program_dir.exists():
        shutil.rmtree(program_dir)
        print(f"[island_manager] Removed {program_dir}")