"""
program_parser.py
-----------------
Utilities for transforming candidate programs between three representations:

  1. SCRIPT  : a .py file with real function definitions (e.g. seed_programs.py)
  2. STRING  : source code as a plain string (stored in program.py, passed to LLM)
  3. CALLABLE: live Python functions ready to call (used by score.py / optimizer)

Three main public functions:

  script_to_strings(path)       -> list[ProgramStrings]
  llm_output_to_strings(text)   -> ProgramStrings | None
  strings_to_callables(prog)    -> ProgramCallables | None

Expected function names in any program:
  model(x, params)          -> np.ndarray
  estimate_params(x, y)     -> np.ndarray
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

MODEL_NAME = "model"
ESTIMATOR_NAME = "estimate_params"

_BASE_NAMESPACE: dict = {"np": np}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProgramStrings:
    """
    A candidate program as source-code strings.
    Both strings are self-contained (include necessary imports).
    """
    model_src: str
    estimator_src: str

    def combined(self) -> str:
        """Single string with both functions — written to program.py."""
        return self.model_src + "\n\n" + self.estimator_src

    def __repr__(self) -> str:
        return f"ProgramStrings(model_src={self.model_src[:40]!r}...)"


@dataclass
class ProgramCallables:
    """A candidate program as live callable functions."""
    model: Callable
    estimate_params: Callable
    source: ProgramStrings


# ---------------------------------------------------------------------------
# 1. SCRIPT -> STRINGS
# ---------------------------------------------------------------------------

def script_to_strings(path: str | Path) -> list[ProgramStrings]:
    """
    Parse a Python script and extract all (model, estimate_params) pairs.
    Functions are matched in definition order.
    Imports are deduplicated and prepended to each function string.
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
# ---------------------------------------------------------------------------

def llm_output_to_strings(text: str | None) -> ProgramStrings | None:
    """
    Extract model() and estimate_params() from raw LLM output.

    Strategy:
      1. If markdown fences are present, extract all fenced blocks and
         concatenate them (handles functions split across multiple blocks).
      2. If no fences, isolate the last occurrence of 'def model' and take
         everything from there — this handles LLMs that echo the full prompt
         (parent programs + separators) before writing the new program.
      3. Parse with AST and extract the two canonical functions.

    Returns ProgramStrings if both functions found, None otherwise.
    """
    if text is None:
        return None

    code = _isolate_new_program(text)

    try:
        module = ast.parse(code)
    except SyntaxError as e:
        print(f"[program_parser] SyntaxError after extraction: {e}")
        print(f"[program_parser] First 200 chars: {code[:200]!r}")
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
# ---------------------------------------------------------------------------

def strings_to_callables(prog: ProgramStrings) -> ProgramCallables | None:
    """
    Execute a ProgramStrings and return live callables.
    numpy is always injected as 'np' regardless of imports.
    """
    namespace = dict(_BASE_NAMESPACE)
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

    return ProgramCallables(model=model_fn, estimate_params=est_fn, source=prog)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def script_to_callables(path: str | Path) -> list[ProgramCallables]:
    """Parse a script and return all programs as live callables."""
    return [c for p in script_to_strings(path)
            if (c := strings_to_callables(p)) is not None]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _isolate_new_program(text: str) -> str:
    """
    Extract clean Python code from raw LLM output.

    If markdown fences are present, extract all fenced blocks and concatenate
    them — this handles functions split across multiple blocks.

    If no fences, find the LAST 'def model' in the text and take everything
    from the nearest preceding import onward. This handles the common LLM
    pattern of echoing all parent programs before writing the new one.
    """
    # --- Path 1: markdown fences present ---
    fence_pattern = re.compile(
        r'^\s*```(?:python|py)?\s*\n(.*?)^\s*```',
        re.MULTILINE | re.DOTALL,
    )
    blocks = fence_pattern.findall(text)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks)

    # --- Path 2: no fences — isolate from last def model onward ---
    lines = text.splitlines()

    # Find the last zero-indented 'def model'
    last_model_line = None
    for i, line in enumerate(lines):
        if re.match(r'^def model', line):
            last_model_line = i

    if last_model_line is None:
        # No def model found at all — return as-is for error reporting
        return re.sub(r'`+', '', text).strip()

    # Walk backward from def model to pick up any preceding imports
    start = last_model_line
    for i in range(last_model_line - 1, -1, -1):
        line = lines[i].strip()
        if re.match(r'^(import |from )', lines[i]):
            start = i
        elif line == '':
            continue
        else:
            break

    return "\n".join(lines[start:])


def _extract_unique_imports(module: ast.Module) -> list[ast.stmt]:
    seen, unique = set(), []
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
    """Find by exact name, then by prefix. Renames to canonical."""
    node = funcs.get(name)
    if node is None:
        candidates = [n for k, n in funcs.items() if k.startswith(name)]
        node = candidates[0] if candidates else None
    if node is not None:
        node.name = name
    return node


def _node_to_src(
    imports: list[ast.stmt],
    func_node: ast.FunctionDef,
    canonical_name: str,
) -> str:
    func_node.name = canonical_name
    mini_module = ast.Module(body=imports + [func_node], type_ignores=[])
    ast.fix_missing_locations(mini_module)
    return ast.unparse(mini_module)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Test 1: clean single block ===")
    prog = llm_output_to_strings("""
```python
import numpy as np

def model(x, params):
    A, B, C, D = params
    return A * np.sin(B * x + C) + D

def estimate_params(x, y):
    return np.array([(np.max(y) - np.min(y)) / 2, 1.0, 0.0, np.mean(y)])
```
""")
    print("Parsed:", prog is not None)

    print("\n=== Test 2: split across two fenced blocks ===")
    prog = llm_output_to_strings("""
Here is the model:
```python
import numpy as np
def model(x, params):
    A, B, C, D = params
    return A * np.sin(B * x + C) + D
```
And the estimator:
```python
def estimate_params(x, y):
    return np.array([(np.max(y) - np.min(y)) / 2, 1.0, 0.0, np.mean(y)])
```
""")
    print("Parsed:", prog is not None)

    print("\n=== Test 3: full prompt echo (no fences) ===")
    prog = llm_output_to_strings("""loss of model 3: 0.0412
import numpy as np
def model(x, params):
    \"\"\"Linear model.\"\"\"
    A, B = params
    return A * x + B
import numpy as np
def estimate_params(x, y):
    A_guess = (y[1] - y[0]) / (x[1] - x[0]) if (x[1] - x[0]) != 0 else 0
    return np.array([A_guess, 0.0])
----------------------------
loss of model 4: 0.0321
import numpy as np
def model(x, params):
    \"\"\"Quadratic model.\"\"\"
    A, B, C = params
    return A * x**2 + B * x + C
import numpy as np
def estimate_params(x, y):
    return np.array([1.0, 0.0, 0.0])""")
    print("Parsed:", prog is not None)
    if prog:
        print("Is quadratic (last model taken):", "A, B, C" in prog.model_src)

    print("\n=== Test 4: missing estimate_params ===")
    prog = llm_output_to_strings("def model(x, params):\n    return params[0]")
    print(f"Result (should be None): {prog}")