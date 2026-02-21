"""
score.py
--------
Top-level evaluation entry point for the sine wave problem.

The engine calls evaluate(program_str) with a candidate program string and
receives a scalar score and a metrics dict. Everything else — loading data,
fitting parameters, computing metrics — is handled internally here.

Two-phase evaluation:
  1. FIT   : run estimate_params(x_train, y_train) then scipy optimize on train set
  2. EVAL  : score model(x_val, optimized_params) against y_val

Visual feedback (evaluation_figure.png) is generated from the TRAIN split only,
so the validation set is never exposed to the LLM via figures.

Users can adjust the penalty weights and optimizer settings below.
"""

import ast
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

# --- Import problem data loader ---
import sys
sys.path.insert(0, str(Path(__file__).parent))
from load_data import load

# --- Penalty weights (set to 0.0 to disable) ---
PARAM_COUNT_WEIGHT = 0.02
AST_DEPTH_WEIGHT = 0.005

# --- Optimizer settings ---
MAX_OPTIMIZER_ITER = 2000
OPTIMIZER_METHOD = "L-BFGS-B"

# --- Sentinel score returned on failure ---
FAILURE_SCORE = -999.0


# ---------------------------------------------------------------------------
# Public interface (called by the engine)
# ---------------------------------------------------------------------------

def evaluate(program_str: str, output_dir: Path | None = None) -> tuple[float, dict]:
    """
    Evaluate a candidate program string end-to-end.

    Parameters
    ----------
    program_str : source code string containing model() and estimate_params()
    output_dir  : if provided, saves evaluation_figure.png here (train split only)

    Returns
    -------
    score   : scalar, higher is better. FAILURE_SCORE on any error.
    metrics : dict suitable for writing to score.json
    """
    try:
        model_fn, param_fn = _exec_program(program_str)
    except Exception as e:
        return FAILURE_SCORE, _failure_metrics(f"exec error: {e}")

    x_train, y_train, x_val, y_val = load()

    # --- Phase 1: fit on training data ---
    try:
        p0 = param_fn(x_train, y_train)
        p0 = np.atleast_1d(np.array(p0, dtype=float))
    except Exception as e:
        return FAILURE_SCORE, _failure_metrics(f"estimate_params error: {e}")

    try:
        def loss(params):
            y_pred = model_fn(x_train, params)
            return float(np.mean((y_pred - y_train) ** 2))

        result = minimize(loss, p0, method=OPTIMIZER_METHOD,
                          options={"maxiter": MAX_OPTIMIZER_ITER})
        params_opt = result.x
    except Exception as e:
        return FAILURE_SCORE, _failure_metrics(f"optimizer error: {e}")

    # --- Phase 2: evaluate on validation data ---
    try:
        y_val_pred = model_fn(x_val, params_opt)
        if not np.all(np.isfinite(y_val_pred)):
            return FAILURE_SCORE, _failure_metrics("non-finite predictions on val set")
    except Exception as e:
        return FAILURE_SCORE, _failure_metrics(f"model eval error: {e}")

    mse = float(np.mean((y_val_pred - y_val) ** 2))
    r2 = _r2(y_val_pred, y_val)
    n_params = len(params_opt)
    depth = _ast_depth(program_str)

    param_penalty = PARAM_COUNT_WEIGHT * n_params
    complexity_penalty = AST_DEPTH_WEIGHT * depth
    score = r2 - param_penalty - complexity_penalty

    metrics = {
        "status": "success",
        "score": float(score),
        "mse_val": mse,
        "r2_val": float(r2),
        "n_params": int(n_params),
        "ast_depth": int(depth),
        "param_penalty": float(param_penalty),
        "complexity_penalty": float(complexity_penalty),
        "optimizer_converged": bool(result.success),
        "params_opt": params_opt.tolist(),
    }

    # --- Optional: save evaluation figure from TRAIN split only ---
    if output_dir is not None:
        try:
            y_train_pred = model_fn(x_train, params_opt)
            _save_figure(x_train, y_train, y_train_pred, score, output_dir)
            metrics["figure"] = "evaluation_figure.png"
        except Exception:
            pass  # figure failure should never block a score result

    return float(score), metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _exec_program(program_str: str):
    """
    Execute program_str in a restricted namespace.
    Returns (model_fn, estimate_params_fn).
    Raises on syntax error or missing required functions.
    """
    namespace = {"np": np}
    exec(compile(program_str, "<candidate>", "exec"), namespace)

    if "model" not in namespace:
        raise ValueError("program must define model(x, params)")
    if "estimate_params" not in namespace:
        raise ValueError("program must define estimate_params(x, y)")

    return namespace["model"], namespace["estimate_params"]


def _r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)


def _ast_depth(source: str) -> int:
    try:
        return _tree_depth(ast.parse(source))
    except SyntaxError:
        return 999


def _tree_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_tree_depth(c) for c in children)


def _failure_metrics(reason: str) -> dict:
    return {"status": "failed", "reason": reason, "score": FAILURE_SCORE}


def _save_figure(x_train, y_train, y_train_pred, score, output_dir: Path):
    sort_idx = np.argsort(x_train)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_train, y_train, s=10, alpha=0.5, label="train data")
    ax.plot(x_train[sort_idx], y_train_pred[sort_idx],
            color="crimson", linewidth=2, label=f"model (score={score:.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Model fit — training data only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_figure.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_program = """
def model(x, params):
    A, B, C, D = params
    return A * np.sin(B * x + C) + D

def estimate_params(x, y):
    A_guess = (np.max(y) - np.min(y)) / 2.0
    return np.array([A_guess, 1.0, 0.0, np.mean(y)])
"""
    score, metrics = evaluate(test_program, output_dir=Path("."))
    print(f"Score : {score:.4f}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
