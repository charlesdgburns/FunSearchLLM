"""
seed_programs.py
----------------
Starting programs for the sine wave FunSearch run.

Each entry is a dict with:
  - "model"           : source of a function  model(x, params) -> np.ndarray
  - "estimate_params" : source of a function  estimate_params(x, y) -> np.ndarray
                        returning an initial parameter guess for the optimizer
  - "description"     : plain-English note shown in the first LLM prompt

Two seeds are provided:
  1. A poor baseline (constant prediction) — gives the LLM something to clearly beat
  2. A reasonable guess (single sinusoid, wrong parameters) — gives structure to build on

The engine will evaluate both before the first LLM call and use their scores
as the initial population.
"""


SEEDS = [
    # ------------------------------------------------------------------
    # Seed 1: Deliberately poor — constant prediction
    # Rationale: establishes a floor score; LLM should easily improve on this.
    # ------------------------------------------------------------------
    {
        "description": "Constant prediction at the mean of y. A trivial baseline.",
        "model": """
def model(x, params):
    \"\"\"Predict a constant value regardless of x.\"\"\"
    return np.full_like(x, params[0])
""",
        "estimate_params": """
def estimate_params(x, y):
    \"\"\"Initial guess: mean of y.\"\"\"
    return np.array([np.mean(y)])
""",
    },

    # ------------------------------------------------------------------
    # Seed 2: Reasonable — sinusoid with naive parameter guess
    # Rationale: correct functional form but wrong parameters;
    # tests whether the optimizer + LLM can jointly recover A, B, C, D.
    # ------------------------------------------------------------------
    {
        "description": (
            "Single sinusoid A*sin(B*x + C) + D with a naive initial parameter guess. "
            "Correct structure but parameters need fitting."
        ),
        "model": """
def model(x, params):
    \"\"\"Single sinusoid: A*sin(B*x + C) + D.\"\"\"
    A, B, C, D = params
    return A * np.sin(B * x + C) + D
""",
        "estimate_params": """
def estimate_params(x, y):
    \"\"\"
    Naive initial guess:
      A ~ half the range of y
      B ~ 1.0 (assume ~1 cycle over the domain)
      C ~ 0.0 (no phase shift)
      D ~ mean of y
    \"\"\"
    A_guess = (np.max(y) - np.min(y)) / 2.0
    B_guess = 1.0
    C_guess = 0.0
    D_guess = np.mean(y)
    return np.array([A_guess, B_guess, C_guess, D_guess])
""",
    },
]


if __name__ == "__main__":
    print(f"{len(SEEDS)} seed programs defined:")
    for i, s in enumerate(SEEDS):
        print(f"\n--- Seed {i+1}: {s['description']}")
        print("  model():", s["model"].strip().splitlines()[0])
        print("  estimate_params():", s["estimate_params"].strip().splitlines()[0])
