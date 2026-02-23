"""
load_data.py
------------
Loads sine_data.tsv and splits it into train and validation sets.

The split is deterministic given TRAIN_FRACTION and RANDOM_SEED, so every
call returns the same split. Change these constants here if needed.

Returns four numpy arrays: x_train, y_train, x_val, y_val.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TSV_FILE = DATA_DIR / "data.htsv"

# --- Split settings ---
TRAIN_FRACTION = 0.75
RANDOM_SEED = 0        # separate from data generation seed


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads sine_data.tsv, shuffles with a fixed seed, and splits into
    train and validation sets.

    Returns
    -------
    x_train, y_train, x_val, y_val  — all float64 numpy arrays
    """
    data = pd.read_csv(TSV_FILE, sep="\t")
    x, y = data.iloc[:, 0].to_numpy(), data.iloc[:, 1].to_numpy()

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(x))
    split = int(len(x) * TRAIN_FRACTION)

    train_idx, val_idx = indices[:split], indices[split:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


if __name__ == "__main__":
    x_tr, y_tr, x_v, y_v = load()
    print(f"Train: {x_tr.shape}  Val: {x_v.shape}")
    print(f"y_train range: [{y_tr.min():.3f}, {y_tr.max():.3f}]")
    print(f"y_val   range: [{y_v.min():.3f}, {y_v.max():.3f}]")
