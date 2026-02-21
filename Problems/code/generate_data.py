"""
generate_data.py
----------------
Generates synthetic sine wave data of the form:
    y = A * sin(B * x + C) + D + noise

Saves a single human-readable TSV file: sine_data.tsv
Columns: x, y

Run once before starting a FunSearch run:
    python generate_data.py

Ground truth parameters are saved to ground_truth.json for reference.
"""

import numpy as np
import json
from pathlib import Path

# --- Ground truth parameters ---
A = 2.5       # amplitude
B = 1.3       # frequency
C = 0.8       # phase shift
D = -1.0      # vertical offset
NOISE_STD = 0.1

# --- Sampling settings ---
N_POINTS = 280
X_MIN = -4 * np.pi
X_MAX = 4 * np.pi
RANDOM_SEED = 42

if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    out_dir = Path(__file__).parent/'data'

    x = rng.uniform(X_MIN, X_MAX, size=N_POINTS)
    noise = rng.normal(0, NOISE_STD, size=N_POINTS)
    y = A * np.sin(B * x + C) + D + noise

    # Write TSV
    tsv_path = out_dir / "sine_data.tsv"
    with open(tsv_path, "w") as f:
        f.write("x\ty\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.8f}\t{yi:.8f}\n")

    # Save ground truth
    ground_truth = {"A": A, "B": B, "C": C, "D": D, "noise_std": NOISE_STD,
                    "n_points": N_POINTS, "random_seed": RANDOM_SEED}
    with open(out_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Saved {N_POINTS} points to {tsv_path}")
    print(f"Ground truth: y = {A}*sin({B}*x + {C}) + {D}  [noise std={NOISE_STD}]")
