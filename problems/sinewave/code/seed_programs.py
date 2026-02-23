"""
seed_programs.py
----------------
Starting programs for the sine wave FunSearch run.

Write seed programs as normal Python functions — no string embedding needed.
The program_parser will extract them automatically.

Naming convention:
  - Model functions    : model(x, params) or model_v2(x, params), etc.
  - Estimator functions: estimate_params(x, y) or estimate_params_v2(x, y), etc.

Functions are matched in definition order: first model_* with first estimate_params_*,
second with second, and so on.

Two seeds are provided:
  1. A poor baseline (constant prediction) — establishes a floor score
  2. A reasonable guess (single sinusoid) — correct structure, parameters need fitting
"""

import numpy as np


# ---------------------------------------------------------------------------
# Seed 1: Deliberately poor — constant prediction
# Rationale: establishes a floor; LLM should easily improve on this.
# ---------------------------------------------------------------------------

def model(x, params):
    """Predict a constant value regardless of x."""
    return np.full_like(x, params[0])


def estimate_params(x, y):
    """Initial guess: mean of y."""
    return np.array([np.mean(y)])


# ---------------------------------------------------------------------------
# Seed 2: Another bad guess — just a linear model based on the mean.
# ---------------------------------------------------------------------------

def model_v2(x, slope=1, intercept=1):
    """Basic linear model:"""
    return slope* x + intercept


def estimate_params_v2(x, y):
    """
    Naive initial guess:
      slope ~ half the peak-to-peak range
      intercept ~ mean of all values
    """
    slope = (np.max(y) - np.min(y)) / 2.0
    intercept = np.mean(y)
    return np.array([intercept])


# ---------------------------------------------------------------------------
# Standalone check — parse and report what program_parser finds
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "FunSearch"))
    from FunSearch.program_parser import script_to_strings

    programs = script_to_strings(__file__)
    print(f"{len(programs)} seed program(s) found:")
    for i, p in enumerate(programs):
        print(f"\n--- Seed {i + 1} ---")
        print("model_src     :", p.model_src[:80], "...")
        print("estimator_src :", p.estimator_src[:80], "...")