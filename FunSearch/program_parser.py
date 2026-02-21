"""
program_parser.py
-----------------
Utilities for transforming candidate programs between three representations:

  1. SCRIPT  : a .py file with real function definitions (e.g. seed_programs.py)
  2. STRING  : source code as a plain string (stored in score.json, passed to LLM)
  3. CALLABLE: live Python functions ready to call (used by score.py / optimizer)

Three main public functions:

  script_to_strings(path)          -> list of ProgramStrings
  llm_output_to_strings(text)      -> ProgramStrings | None
  strings_to_callables(prog)       -> ProgramCallables | None

Plus one convenience dataclass for each representation.

Expected function names in any program:
  model(x, params)              -> np.ndarray
  estimate_params(x, y)         -> np.ndarray

These names are enforced at parse time. The LLM prompt should instruct
the model to use exactly these names.
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Expected function names — change here if the interface ever changes
# ---------------------------------------------------------------------------

MODEL_NAME = "model"
ESTIMATOR_NAME = "estimate_params"

# Namespace always injected when executing program strings.
# Programs that omit 'import numpy as np' will still work.
_BASE_NAMESPACE: dict = {"np": np}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProgramStrings:
    """
    A candidate program represented as source-code strings.
    Both strings are self-contained: they include any necessary imports.
    This is the form stored to disk and passed to the LLM.
    """
    model_src: str          # source of model(x, params)
    estimator_src: str      # source of estimate_params(x, y)

    def combined(self) -> str:
        """Single string with both functions — used when writing program.py."""
        return self.model_src + "\n\n" + self.estimator_src

    def __repr__(self) -> str:
        return f"ProgramStrings(model_src={self.model_src[:40]!r}...)"


@dataclass
class ProgramCallables:
    """
    A candidate program represented as live callable functions.
    This is the form used by the optimizer and score.py.
    """
    model: Callable           # model(x, params) -> np.ndarray
    estimate_params: Callable # estimate_params(x, y) -> np.ndarray
    source: ProgramStrings    # keep the strings around for scoring / logging


# ---------------------------------------------------------------------------
# 1. SCRIPT -> STRINGS
#    Parse a real .py file and extract all (model, estimate_params) pairs.
#    seed_programs.py contains multiple pairs; each becomes one ProgramStrings.
# ---------------------------------------------------------------------------

def script_to_strings(path: str | Path) -> list[ProgramStrings]:
    """
    Parse a Python script file and extract all pairs of
    (model_*, estimate_params_*) functions.

    Functions are matched in order of definition:
      - first model_* pairs with first estimate_params_*
      - second model_* pairs with second estimate_params_*, etc.

    Imports from the script are deduplicated and prepended to each function string.

    Parameters
    ----------
    path : path to a .py script file

    Returns
    -------
    list of ProgramStrings, one per matched pair found in the file.
    Empty list if no valid pairs are found.

    Example
    -------
    >>> programs = script_to_strings("problems/sine/code/seed_programs.py")
    >>> print(programs[0].model_src)
    """
    source = Path(path).read_text()
    try:
        module = ast.parse(source)
    except SyntaxError as e:
        print(f"[program_parser] SyntaxError in {path}: {e}")
        return []

    imports = _extract_unique_imports(module)
    funcs = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}

    models = sorted(
        [n for name, n in funcs.items() if name.startswith(MODEL_NAME)],
        key=lambda n: n.lineno,
    )
    estimators = sorted(
        [n for name, n in funcs.items() if name.startswith(ESTIMATOR_NAME)],
        key=lambda n: n.lineno,
    )

    if not models or not estimators:
        print(f"[program_parser] No valid function pairs found in {path}")
        return []

    pairs = []
    for model_node, est_node in zip(models, estimators):
        model_src = _node_to_src(imports, model_node, canonical_name=MODEL_NAME)
        est_src = _node_to_src(imports, est_node, canonical_name=ESTIMATOR_NAME)
        pairs.append(ProgramStrings(model_src=model_src, estimator_src=est_src))

    return pairs


# ---------------------------------------------------------------------------
# 2. LLM OUTPUT -> STRINGS
#    Extract the two required functions from raw LLM text.
#    Handles markdown fences, extra prose, and partial outputs gracefully.
# ---------------------------------------------------------------------------

def llm_output_to_strings(text: str | None) -> ProgramStrings | None:
    """
    Extract model() and estimate_params() from raw LLM output.

    Handles:
      - Markdown code fences (```python ... ```)
      - Extra prose before/after the code block
      - Missing or malformed functions (returns None)
      - Duplicate imports

    Parameters
    ----------
    text : raw string returned by the LLM API

    Returns
    -------
    ProgramStrings if both functions are found, None otherwise.
    """
    if text is None:
        return None

    code = _strip_markdown(text)

    try:
        module = ast.parse(code)
    except SyntaxError as e:
        print(f"[program_parser] SyntaxError in LLM output: {e}")
        return None

    imports = _extract_unique_imports(module)
    funcs = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}

    model_node = _find_function(funcs, MODEL_NAME)
    est_node = _find_function(funcs, ESTIMATOR_NAME)

    if model_node is None:
        print(f"[program_parser] LLM output missing '{MODEL_NAME}' function")
        return None
    if est_node is None:
        print(f"[program_parser] LLM output missing '{ESTIMATOR_NAME}' function")
        return None

    model_src = _node_to_src(imports, model_node, canonical_name=MODEL_NAME)
    est_src = _node_to_src(imports, est_node, canonical_name=ESTIMATOR_NAME)
    return ProgramStrings(model_src=model_src, estimator_src=est_src)


# ---------------------------------------------------------------------------
# 3. STRINGS -> CALLABLES
#    Execute program strings and return live callable functions.
# ---------------------------------------------------------------------------

def strings_to_callables(prog: ProgramStrings) -> ProgramCallables | None:
    """
    Execute a ProgramStrings and return live callables.

    Both functions are executed in a shared namespace that always includes
    numpy as 'np', so programs that omit the import still work.

    Parameters
    ----------
    prog : ProgramStrings to execute

    Returns
    -------
    ProgramCallables if execution succeeds, None otherwise.
    """
    namespace = dict(_BASE_NAMESPACE)  # fresh copy each time
    combined = prog.combined()

    try:
        exec(compile(combined, "<candidate>", "exec"), namespace)
    except Exception as e:
        print(f"[program_parser] exec error: {e}")
        return None

    model_fn = namespace.get(MODEL_NAME)
    est_fn = namespace.get(ESTIMATOR_NAME)

    if model_fn is None or not callable(model_fn):
        print(f"[program_parser] '{MODEL_NAME}' not found after exec")
        return None
    if est_fn is None or not callable(est_fn):
        print(f"[program_parser] '{ESTIMATOR_NAME}' not found after exec")
        return None

    return ProgramCallables(
        model=model_fn,
        estimate_params=est_fn,
        source=prog,
    )


# ---------------------------------------------------------------------------
# Convenience: script path -> callables (combines steps 1 and 3)
# ---------------------------------------------------------------------------

def script_to_callables(path: str | Path) -> list[ProgramCallables]:
    """
    Parse a script file and return all programs as live callables.
    Convenience wrapper around script_to_strings + strings_to_callables.
    """
    programs = script_to_strings(path)
    callables = []
    for prog in programs:
        c = strings_to_callables(prog)
        if c is not None:
            callables.append(c)
    return callables


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_unique_imports(module: ast.Module) -> list[ast.stmt]:
    """Return deduplicated import nodes from a parsed module."""
    seen = set()
    unique = []
    for node in module.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            src = ast.unparse(node)
            if src not in seen:
                seen.add(src)
                unique.append(node)
    return unique


def _find_function(
    funcs: dict[str, ast.FunctionDef],
    name: str,
) -> ast.FunctionDef | None:
    """
    Find a function node by exact name, then by prefix match.
    Renames the node to the canonical name before returning.
    """
    node = funcs.get(name)
    if node is None:
        # Try prefix match (e.g. 'model_v2' -> 'model')
        candidates = [n for k, n in funcs.items() if k.startswith(name)]
        node = candidates[0] if candidates else None
    if node is not None:
        node.name = name  # normalise to canonical name
    return node


def _node_to_src(
    imports: list[ast.stmt],
    func_node: ast.FunctionDef,
    canonical_name: str,
) -> str:
    """Build a self-contained source string: imports + renamed function."""
    func_node.name = canonical_name
    mini_module = ast.Module(body=imports + [func_node], type_ignores=[])
    ast.fix_missing_locations(mini_module)
    return ast.unparse(mini_module)


def _strip_markdown(text: str) -> str:
    """
    Remove markdown code fences and leading/trailing prose.
    Extracts the content of the first ```python ... ``` block if present,
    otherwise returns the full text stripped of fence markers.
    """
    lines = text.splitlines()

    # Find first ```python or ``` fence
    in_block = False
    block_lines = []
    for line in lines:
        stripped = line.strip()
        if not in_block and stripped.startswith("```"):
            in_block = True
            continue  # skip the opening fence line
        if in_block and stripped.startswith("```"):
            break  # end of block
        if in_block:
            block_lines.append(line)

    if block_lines:
        return "\n".join(block_lines)

    # No fences found — return full text, stripping any stray backticks
    return text.replace("```", "").strip()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Test LLM output parsing ---
    fake_llm_output = """
Here is my improved function:

```python
import numpy as np

def model(x, params):
    A, B, C, D = params
    return A * np.sin(B * x + C) + D

def estimate_params(x, y):
    return np.array([(np.max(y) - np.min(y)) / 2, 1.0, 0.0, np.mean(y)])
```

This should fit better because of the phase term.
"""

    print("=== Test: llm_output_to_strings ===")
    prog = llm_output_to_strings(fake_llm_output)
    if prog:
        print("model_src:\n", prog.model_src)
        print("\nestimator_src:\n", prog.estimator_src)

    print("\n=== Test: strings_to_callables ===")
    callables = strings_to_callables(prog)
    if callables:
        x = np.linspace(-3, 3, 10)
        params = np.array([2.5, 1.3, 0.8, -1.0])
        y_pred = callables.model(x, params)
        p0 = callables.estimate_params(x, y_pred)
        print(f"model output shape : {y_pred.shape}")
        print(f"estimate_params p0 : {p0}")

    print("\n=== Test: missing function ===")
    bad_output = "def model(x, params):\n    return params[0]"
    result = llm_output_to_strings(bad_output)
    print(f"Result (should be None): {result}")