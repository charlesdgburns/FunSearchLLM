"""
generate_data.py
----------------
Generates synthetic sine wave data of the form:
    y = A * sin(B * x + C) + D + noise

Saves a single human-readable TSV file: sine_data.htsv
Columns: x, y

Run once before starting a FunSearch run:
    python generate_data.py

"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# --- Ground truth parameters ---
A = 2.5       # amplitude
B = 1.3       # frequency
C = 0.8       # phase shift
D = -1.0      # vertical offset


# --- Sampling settings ---
N_POINTS = 280
X_MIN = -4 * np.pi
X_MAX = 4 * np.pi
RANDOM_SEED = 42

if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    out_dir = Path(__file__).parent.parent
    x = rng.uniform(X_MIN, X_MAX, size=N_POINTS)
    y = A * np.sin(B * x + C) + D

    # Write htsv
    filepath = out_dir / "data/data.htsv"
    df = pd.DataFrame({'x':x,'y':y})
    df.to_csv(filepath, sep = '\t', index=False)
    
