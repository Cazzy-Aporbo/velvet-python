#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cazzy Learning Lab — Multi-Modal Task Explorer
Author: Cazandra "Cazzy" Aporbo

I designed this as a lab to transform any of my Python tasks
(yes, even the messy draft ones) into four complementary ways of learning and doing:

  1) Bottom-Up Walkthrough  — "trace every step, demystify everything"
  2) Top-Down Pipeline      — "start with the architecture, then drill down"
  3) Expert Optimization    — "measure, benchmark, and sketch faster variants"
  4) Playful Challenge      — "turn it into a tiny puzzle/game"

The lab auto-discovers my uploaded *.py files, separates tasks by module,
and offers an interactive CLI to choose a task and a mode. If a file defines
functions, we leverage them; otherwise we sandbox-run the script and capture output.

No external deps needed. If advanced packages are installed (numpy/pandas), the
lab will opportunistically use them for small demos, but it works fine without them.

How to run:
  python Cazzy_LearningLab.py             # interactive menu
  python Cazzy_LearningLab.py --quick     # quick run: first task in all 4 modes

Design notes:
- Written from my perspective (Cazzy), speaking like a top educator + playful puzzle dev.
- Clean, heavily commented code so learners see not only *what* I do, but *why*.
- Safe: we execute my own files in a controlled subprocess to avoid polluting state.
"""

from __future__ import annotations

import os, sys, ast, io, time, json, textwrap, types, inspect, subprocess, importlib.util, contextlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/data"

# --------- utility: safe, quiet import of a module by path ---------
def load_module_safely(path: str) -> Optional[types.ModuleType]:
    """Attempt to import a module from a given file path without executing top-level code twice.
    If import fails, return None. We don't execute in __main__; we import under a unique name."""
    try:
        name = f"cazzy_mod_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(name, path)
        if not spec or not spec.loader:
            return None
        mod = importlib.util.module_from_spec(spec)
        # Exec the module code in a controlled module namespace.
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    except Exception as e:
        return None

# --------- utility: run a script in subprocess and capture output ---------
def run_script_subprocess(path: str, timeout: int = 10) -> Tuple[int, str, str]:
    """Run 'python path' in a subprocess, capture stdout/stderr, return (code, out, err)."""
    try:
        proc = subprocess.Popen([sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()  # type: ignore
        return 124, "", f"Timed out after {timeout}s"
    except Exception as e:
        return 1, "", f"Failed: {e}"

# --------- AST-based discovery of top-level functions ---------
def discover_functions(path: str) -> List[str]:
    """Return a list of top-level function names defined in the file (no dunders)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        fnames = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    fnames.append(node.name)
        return fnames
    except Exception:
        return []

# --------- capture prints utility ---------
@contextlib.contextmanager
def capture_io():
    buf_out, buf_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = buf_out, buf_err
        yield buf_out, buf_err
    finally:
        sys.stdout, sys.stderr = old_out, old_err

# --------- data classes to model tasks & modes ---------
@dataclass
class Task:
    name: str
    path: str
    functions: List[str] = field(default_factory=list)

    def has_functions(self) -> bool:
        return len(self.functions) > 0

@dataclass
class ModeResult:

# --------- Variant Registry: multiple unique implementations per task ---------
def _have_numpy():
    try:
        import numpy as _np  # type: ignore
        return True
    except Exception:
        return False

def _have_sympy():
    try:
        import sympy as _sp  # type: ignore
        return True
    except Exception:
        return False

def get_variants(task: Task):
    """
    Return a list of (variant_name, callable) for this task.
    Each callable takes no arguments and prints something meaningful.
    Variants are intentionally different in technique/architecture.
    """
    name = task.name.lower()
    variants = []

    # -------- DICE: simulation techniques --------
    if "dice" in name:
        def v1():
            """Pure Python RNG with histogram for 2d6 sum."""
            import random, collections
            rolls = 5000
            cnt = collections.Counter(sum((random.randint(1,6), random.randint(1,6))) for _ in range(rolls))
            print("V1 Pure Python — 2d6 histogram:")
            for s in range(2,13):
                print(f"{s:2d}: {'#' * (cnt[s]//50)} ({cnt[s]})")

        def v2():
            """Vectorized NumPy simulation for Nd6, convolution for exact pmf if numpy present."""
            print("V2 NumPy vectorized — 5e5×2d6 Monte Carlo + exact pmf via convolution (if available)")
            if _have_numpy():
                import numpy as np
                n = 500000
                s = np.random.randint(1,7,size=(n,2)).sum(axis=1)
                hist = np.bincount(s, minlength=13)[2:]
                print("MC mean:", s.mean(), "MC var:", s.var())
                # exact pmf via discrete conv of two uniform(1..6)
                pmf = np.convolve(np.ones(6)/6, np.ones(6)/6)
                # pmf indices 0..10 correspond to sums 2..12
                for k, p in enumerate(pmf, start=2):
                    print(f"sum={k:2d}  pmf≈{p:.4f}  freq={hist[k-2]}")
            else:
                print("NumPy not available — skipping vectorized demo.")

        def v3():
            """Cryptographically secure sampling for a few fair rolls."""
            import secrets
            print("V3 SystemRandom — 10 secure 1d6 rolls:")
            print([secrets.choice(range(1,7)) for _ in range(10)])

        def v4():
            """Functional composition + mapping to game events."""
            import random
            def roll(n=2, sides=6):
                return sum(random.randint(1,sides) for _ in range(n))
            outcome = {7:"Lucky Seven!", 2:"Snake Eyes!", 12:"Boxcars!"}
            draws = [roll() for _ in range(20)]
            annotated = [(x, outcome.get(x,"")) for x in draws]
            print("V4 Functional + annotation:")
            print(annotated)

        variants += [("Dice/V1 Pure Python Histogram", v1),
                     ("Dice/V2 NumPy Vectorized + PMF", v2),
                     ("Dice/V3 Secure RNG", v3),
                     ("Dice/V4 Functional Game Mapping", v4)]

    # -------- TEMPERATURE CONVERTER: multiple scales / APIs --------
    if "temp" in name:
        def t1():
            """Simple C↔F CLI-like function with validation."""
            def cf(val, to="F"):
                return val*9/5+32 if to.upper()=="F" else (val-32)*5/9
            for c in (-40, 0, 37, 100):
                print(f"{c}C -> {cf(c,'F'):.2f}F")
            for f in (-40, 32, 98.6, 212):
                print(f"{f}F -> {cf(f,'C'):.2f}C")

        def t2():
            """Generalized scale graph C/F/K/Rankine using dictionary of lambdas."""
            conv = {
                ("C","F"): lambda x: x*9/5+32,
                ("F","C"): lambda x: (x-32)*5/9,
                ("C","K"): lambda x: x+273.15,
                ("K","C"): lambda x: x-273.15,
                ("F","K"): lambda x: (x+459.67)*5/9,
                ("K","F"): lambda x: x*9/5-459.67,
                ("K","R"): lambda x: x*1.8,
                ("R","K"): lambda x: x/1.8,
                ("F","R"): lambda x: x+459.67,
                ("R","F"): lambda x: x-459.67,
                ("C","R"): lambda x: (x+273.15)*1.8,
                ("R","C"): lambda x: x/1.8-273.15,
            }
            samples = [("C","F",25), ("F","C",451), ("C","K",0), ("K","F",300), ("R","C",672)]
            for a,b,v in samples:
                print(f"{v}{a} -> {conv[(a,b)](v):.2f}{b}")

        def t3():
            """Batch conversion with NumPy arrays (if available)."""
            print("Batch convert array of Celsius → Fahrenheit")
            if _have_numpy():
                import numpy as np
                arr = np.array([-40, 0, 20, 37, 100], dtype=float)
                out = arr*9/5+32
                print("in C:", arr)
                print("out F:", out)
            else:
                print("NumPy not available.")

        def t4():
            """Object-oriented converter with dataclass and guards."""
            from dataclasses import dataclass
            @dataclass
            class Converter:
                scale: str
                def to_f(self, c: float) -> float:
                    if self.scale.upper()!="C": raise ValueError("Use Celsius input")
                    return c*9/5+32
                def to_c(self, f: float) -> float:
                    if self.scale.upper()!="F": raise ValueError("Use Fahrenheit input")
                    return (f-32)*5/9
            print(Converter("C").to_f(25.0))
            print(Converter("F").to_c(77.0))

        variants += [("Temp/V1 CF basics", t1),
                     ("Temp/V2 Multi-scale graph", t2),
                     ("Temp/V3 Numpy batch", t3),
                     ("Temp/V4 OO dataclass", t4)]

    # -------- QUADRATIC: multiple solvers --------
    if "quadratic" in name or "quardatic" in name:
        def q1():
            """Closed-form quadratic formula with real/complex handling."""
            import cmath
            a,b,c = 1, -3, 2  # roots 1 and 2
            d = b*b-4*a*c
            r1 = (-b+cmath.sqrt(d))/(2*a)
            r2 = (-b-cmath.sqrt(d))/(2*a)
            print("V1 formula:", r1, r2)
        def q2():
            """NumPy roots if available."""
            if _have_numpy():
                import numpy as np
                print("V2 numpy.roots:", np.roots([1,-3,2]))
            else:
                print("NumPy not available.")
        def q3():
            """Sympy symbolic solve if available."""
            if _have_sympy():
                import sympy as sp
                x = sp.symbols('x')
                print("V3 sympy:", sp.solve(sp.Eq(x**2-3*x+2,0), x))
            else:
                print("Sympy not available.")
        def q4():
            """Newton-Raphson numeric root-finding starting from two seeds."""
            def newton(a,b,c,x0,steps=10):
                f  = lambda x: a*x*x + b*x + c
                df = lambda x: 2*a*x + b
                x = x0
                for _ in range(steps):
                    x = x - f(x)/df(x)
                return x
            print("V4 Newton near 0.5:", newton(1,-3,2,0.5))
            print("V4 Newton near 2.5:", newton(1,-3,2,2.5))
        variants += [("Quad/V1 Formula", q1), ("Quad/V2 NumPy", q2),
                     ("Quad/V3 Sympy", q3), ("Quad/V4 Newton", q4)]

    # -------- CRAMER'S RULE / LINEAR SYSTEMS --------
    if "cramer" in name or "cramers_rule" in name:
        def c1():
            """Cramer's rule for 2x2 system."""
            # ax + by = e; cx + dy = f
            a,b,c,d,e,f = 2,1,1,3,  8,13
            D = a*d - b*c
            x = (e*d - b*f)/D
            y = (a*f - e*c)/D
            print("V1 Cramers 2x2:", x, y)
        def c2():
            """Manual Gauss-Jordan elimination 2x2."""
            import copy
            A = [[2,1,8],[1,3,13]]
            # forward eliminate
            f = A[1][0]/A[0][0]
            A[1][0] -= f*A[0][0]; A[1][1] -= f*A[0][1]; A[1][2] -= f*A[0][2]
            y = A[1][2]/A[1][1]
            x = (A[0][2] - A[0][1]*y)/A[0][0]
            print("V2 Gauss-Jordan:", x, y)
        def c3():
            """NumPy linear solve."""
            if _have_numpy():
                import numpy as np
                A = np.array([[2,1],[1,3.]], float)
                b = np.array([8,13.], float)
                sol = np.linalg.solve(A,b)
                print("V3 numpy.linalg.solve:", sol)
            else:
                print("NumPy not available.")
        def c4():
            """Sympy Matrix solve if available."""
            if _have_sympy():
                import sympy as sp
                A = sp.Matrix([[2,1],[1,3]])
                b = sp.Matrix([8,13])
                print("V4 sympy solve:", A.LUsolve(b))
            else:
                print("Sympy not available.")
        variants += [("Cramer/V1", c1), ("Cramer/V2 Gauss-Jordan", c2),
                     ("Cramer/V3 NumPy", c3), ("Cramer/V4 Sympy", c4)]

    # -------- CELEBRITIES: search, fuzzy, trivia skeleton --------
    if "celebr" in name:
        def s1():
            """Dictionary lookup demo."""
            celebs = {"Adele":{"field":"Music","origin":"UK"},
                      "Serena Williams":{"field":"Tennis","origin":"USA"},
                      "Zendaya":{"field":"Acting","origin":"USA"}}
            print("Lookup Adele:", celebs["Adele"])
        def s2():
            """Fuzzy search with difflib."""
            import difflib
            names = ["Adele","Serena Williams","Zendaya","Meryl Streep"]
            q = "Serina Willams"
            print("Closest to", q, "→", difflib.get_close_matches(q, names, n=1))
        def s3():
            """Trivia quiz skeleton (no external I/O)."""
            import random
            questions = [("Who is known for the album '21'?", "Adele"),
                         ("Which star has 23 Grand Slams?", "Serena Williams")]
            q,a = random.choice(questions)
            print("Trivia:", q, "| Answer:", a)
        def s4():
            """DataClass + sorting demo."""
            from dataclasses import dataclass
            @dataclass
            class Person: name:str; field:str; awards:int
            roster = [Person("Adele","Music",15), Person("Zendaya","Acting",10), Person("Serena Williams","Tennis",23)]
            print(sorted(roster, key=lambda p:(-p.awards,p.name)))
        variants += [("Celeb/V1 Dict Lookup", s1), ("Celeb/V2 Fuzzy", s2),
                     ("Celeb/V3 Trivia", s3), ("Celeb/V4 Dataclass Sort", s4)]

    # -------- RANDOM UTILITIES --------
    if "my_random" in name or name.endswith("random.py"):
        def r1():
            """LCG pseudo-RNG demonstration."""
            m,a,c,seed = 2**31-1, 1103515245, 12345, 42
            x = seed
            seq = []
            for _ in range(10):
                x = (a*x + c) % m
                seq.append(x/m)
            print("V1 LCG:", seq)
        def r2():
            """SystemRandom / secrets."""
            import secrets
            print("V2 secrets.randbelow samples:", [secrets.randbelow(100) for _ in range(5)])
        def r3():
            """NumPy Generator PCG64 (if available)."""
            if _have_numpy():
                import numpy as np
                rng = np.random.default_rng()
                print("V3 PCG64 normals:", rng.normal(size=5))
            else:
                print("NumPy not available.")
        def r4():
            """Sampling without replacement functional style."""
            import random
            print("V4 sample:", random.sample(range(100), k=10))
        variants += [("Rand/V1 LCG", r1), ("Rand/V2 Secrets", r2),
                     ("Rand/V3 NumPy PCG64", r3), ("Rand/V4 Sample", r4)]

    # Generic fallback: run script + echo functions
    if not variants:
        def g1():
            code, out, err = run_script_subprocess(task.path)
            print(f"Fallback run exit={code}")
            if out.strip(): print(out[:1000])
            if err.strip(): print("stderr:", err[:500])
        def g2():
            mod = load_module_safely(task.path)
            if not mod:
                print("Import failed — fallback only.")
                return
            fns = [getattr(mod, fn) for fn in task.functions if hasattr(mod, fn)]
            fns = [f for f in fns if callable(f)]
            print("Discovered functions:", [f.__name__ for f in fns])
            for f in fns[:3]:
                try:
                    print(f.__name__, "->", repr(f())[:200])
                except TypeError:
                    print(f.__name__, "(needs params)")
        variants += [("Generic/V1 Run", g1), ("Generic/V2 Import+Introspect", g2)]

    return variants
    success: bool
    summary: str
    stdout: str = ""
    stderr: str = ""

# --------- Mode 1: Bottom-Up Walkthrough ---------
def mode_bottom_up(task: Task) -> ModeResult:
    """
    Goal: demystify the task step-by-step.
    If the file defines functions, we import and call them with no args (best effort).
    If not, we run it and provide a cleaned transcript.
    """
    lines = []
    lines.append(f"👣 Bottom-Up Walkthrough for '{task.name}'")
    lines.append("I trace the execution and narrate what happens, like a lab partner explaining aloud.\n")

    if task.has_functions():
        mod = load_module_safely(task.path)
        if not mod:
            return ModeResult(False, "Could not import module safely.")
        lines.append("Discovered functions: " + ", ".join(task.functions))
        with capture_io() as (o, e):
            for fn in task.functions:
                func = getattr(mod, fn, None)
                if callable(func):
                    lines.append(f"\n— Calling {fn}() with no arguments (best-effort demo)...")
                    try:
                        res = func()
                        print(f"[return] {fn} -> {repr(res)[:200]}")
                    except TypeError:
                        print(f"[skip] {fn} requires parameters — leaving for Top-Down mode.")
                    except Exception as ex:
                        print(f"[error] {fn} raised: {ex}")
        out = o.getvalue()
        err = e.getvalue()
        return ModeResult(True, "\n".join(lines), out, err)
    else:
        code, out, err = run_script_subprocess(task.path)
        lines.append("Script executed in a sandbox. Below is the clean output transcript.")
        return ModeResult(code == 0, "\n".join(lines), out, err)

# --------- Mode 2: Top-Down Pipeline ---------
def mode_top_down(task: Task) -> ModeResult:
    """
    Goal: architect first, then execute. We create an ordered plan:
      - Identify key functions (if any)
      - Orchestrate them as a pipeline (setup -> core -> finalize)
    If no functions exist, we still run the script but we wrap the output with a 'plan'.
    """
    plan = []
    plan.append(f"🏗️ Top-Down Pipeline for '{task.name}'")
    plan.append("I start with architecture: inputs → transforms → outputs.\n")

    mod = load_module_safely(task.path)
    if mod and task.has_functions():
        plan.append("Functions detected — sketching a pipeline:")
        # naive heuristic: first = setup-like, middle = core, last = finalize
        fns = task.functions
        segments = {
            "setup": fns[:1],
            "core": fns[1:-1] if len(fns) > 2 else fns[1:],
            "finalize": fns[-1:] if len(fns) > 1 else [],
        }
        for stage, names in segments.items():
            plan.append(f"  • {stage}: " + (", ".join(names) if names else "—"))
        with capture_io() as (o, e):
            for stage in ("setup", "core", "finalize"):
                for fn in segments[stage]:
                    func = getattr(mod, fn, None)
                    if callable(func):
                        try:
                            print(f"[{stage}] {fn}() →", end=" ")
                            res = func()
                            print(repr(res)[:200])
                        except TypeError:
                            print(f"{fn} needs args — leaving as an exercise with docstring guidance.")
                            doc = inspect.getdoc(func) or "No docstring available."
                            print("Doc:", doc[:300])
                        except Exception as ex:
                            print(f"{fn} error: {ex}")
        return ModeResult(True, "\n".join(plan), o.getvalue(), e.getvalue())
    else:
        plan.append("No functions found — running as a monolithic script inside a 'pipeline harness'.")
        code, out, err = run_script_subprocess(task.path)
        return ModeResult(code == 0, "\n".join(plan), out, err)

# --------- Mode 3: Expert Optimization ---------
def mode_expert(task: Task) -> ModeResult:
    """
    Goal: think like a performance-minded expert.
    We benchmark the baseline run and sketch faster-friendly hooks.
    If numpy/pandas are available, we offer vectorization hints and micro-demos.
    """
    summary = [f"⚡ Expert Optimization for '{task.name}'"]
    summary.append("Benchmarking baseline, then suggesting refactors & vectorization paths.\n")

    # Baseline timing via subprocess
    t0 = time.perf_counter()
    code, out, err = run_script_subprocess(task.path)
    t1 = time.perf_counter()
    dt = (t1 - t0) * 1000.0
    summary.append(f"Baseline wall time (subprocess): ~{dt:.1f} ms, exit={code}")

    # Heuristics: if functions exist, show how to time them (no-arg only demo)
    if task.has_functions():
        mod = load_module_safely(task.path)
        if mod:
            with capture_io() as (o, e):
                print("Per-function micro-benchmarks (no-arg calls only):")
                for fn in task.functions:
                    f = getattr(mod, fn, None)
                    if callable(f):
                        try:
                            t0 = time.perf_counter()
                            f()
                            t1 = time.perf_counter()
                            print(f"  {fn:<24} → {1000*(t1-t0):6.2f} ms")
                        except TypeError:
                            print(f"  {fn:<24} → needs params (skipped)")
                        except Exception as ex:
                            print(f"  {fn:<24} → error: {ex}")
            out += "\n" + o.getvalue()
            err += "\n" + e.getvalue()

    # Vectorization sketch
    vectorization = textwrap.dedent("""
        Vectorization / Parallelism ideas:
        - Replace Python loops with list comprehensions or built-ins (sum/map/any/all).
        - If numeric-heavy, port to NumPy arrays (vectorized ops) or use pandas for tabular transforms.
        - For embarrassingly parallel tasks, try concurrent.futures ThreadPool/ProcessPool.
        - Cache repeated pure computations with functools.lru_cache.
        - Push I/O to the edges; compute in-memory where possible.
    """).strip()
    summary.append(vectorization)
    return ModeResult(code == 0, "\n".join(summary), out, err)

# --------- Mode 4: Playful Challenge ---------
def mode_playful(task: Task) -> ModeResult:
    """
    Goal: turn the task into a tiny game.
    If the script prints output, we turn lines into a 'predict-the-next-line' quiz.
    If functions exist, we ask the learner to guess return types before we reveal.
    """
    header = [f"🎮 Playful Challenge for '{task.name}'"]
    code, out, err = run_script_subprocess(task.path)
    if out.strip():
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        quiz = ["I ran the task secretly. Can you predict the next line? Type 'quit' to stop."]
        score = 0
        import sys as _sys
        inp = input
        for i, line in enumerate(lines[:5]):  # cap to 5 rounds to keep it snappy
            guess = inp(f"Round {i+1} — your guess: ").strip()
            if guess.lower() == "quit":
                break
            correct = (guess.strip() == line.strip())
            score += 1 if correct else 0
            print("✓ Correct!" if correct else f"✗ Not quite. Actual: {line}")
        header.append(f"Your score: {score}/{min(5, len(lines))}")
        return ModeResult(True, "\n".join(header), "", err)
    else:
        # No stdout to gamify; try function return-type guessing
        mod = load_module_safely(task.path)
        if mod:
            funs = [getattr(mod, fn) for fn in task.functions if hasattr(mod, fn)]
            funs = [f for f in funs if callable(f)]
            score = 0
            total = 0
            for f in funs[:5]:
                total += 1
                ans = input(f"Predict the return TYPE of {f.__name__}() (e.g., int/str/list/None): ").strip().lower()
                try:
                    res = f()
                except TypeError:
                    print("Needs args — counting as a freebie!")
                    total -= 1
                    continue
                actual = type(res).__name__.lower()
                if ans == actual:
                    print("✓ Nailed it.")
                    score += 1
                else:
                    print(f"✗ Close. Actual type: {actual}")
            header.append(f"Your score: {score}/{max(1,total)}")
            return ModeResult(True, "\n".join(header), "", "")
        return ModeResult(False, "Nothing to gamify (no output, no functions).")


# --------- Mode 5: Variants Showcase ---------
def mode_variants(task: Task) -> ModeResult:
    """
    Goal: actually *re-implement* the idea multiple unique ways.
    We detect canonical problems by filename (dice, quadratic, temp, linear systems ...)
    and run 3–4 different techniques for each.
    """
    header = [f"🧪 Variants Showcase for '{task.name}'"]
    vars = get_variants(task)
    if not vars:
        return ModeResult(False, "No variants available.")
    from time import perf_counter
    with capture_io() as (o, e):
        for label, fn in vars:
            t0 = perf_counter()
            print("\n---", label, "---")
            try:
                fn()
            except Exception as ex:
                print(f"[variant error] {ex}")
            dt = (perf_counter() - t0) * 1000.0
            print(f"[time] {dt:.2f} ms")
    return ModeResult(True, "\n".join(header), o.getvalue(), e.getvalue())

# --------- Discover tasks from /mnt/data ---------
def discover_tasks() -> List[Task]:
    preferred = [
        "Cazandra_Aporbo_ProgrammingAssignment1.py",
        "bottles.py",
        "bottles_2.py",
        "game_day.py",
        "week9_3005.py",
        "cazandra_Aporbo_Programming_Assignment_3.py",
        "cazandra.aporbo.programmingAssignment4.py",
        "cazandra.aporboProgrammingAssignment7.py",
        "cazandra.aporbo.programmingassignment5redo.py",
    
        "Dice.py",
        "Celebrities.py",
        "test_temp.py",
        "Temperature_converter.py",
        "quadratic.py",
        "base_class.py",
        "quardatic_2.py",
        "my_random.py",
        "week_nine.py",
        "Cramers_rule.py",
    ]
seen = set()
    tasks: List[Task] = []
    # Include preferred first (preserving order) if present
    for name in preferred:
        path = os.path.join(DATA, name)
        if os.path.isfile(path):
            fns = discover_functions(path)
            tasks.append(Task(name=name, path=path, functions=fns))
            seen.add(path)
    # Add any other .py files in /mnt/data as extras
    for entry in os.listdir(DATA):
        if entry.endswith(".py"):
            path = os.path.join(DATA, entry)
            if path not in seen:
                fns = discover_functions(path)
                tasks.append(Task(name=entry, path=path, functions=fns))
    return tasks

# --------- Pretty print helpers ---------
def banner(txt: str) -> None:
    print("=" * 88)
    print(txt)
    print("=" * 88)

def describe_task(task: Task) -> None:
    print(f"Task: {task.name}")
    print(f"Path: {task.path}")
    if task.functions:
        print(f"Functions: {', '.join(task.functions)}")
    else:
        print("Functions: (none detected — will run as a script)")

# --------- Interactive menu ---------
def interactive() -> None:
    tasks = discover_tasks()
    if not tasks:
        print("No tasks found in /mnt/data. Please add *.py files and re-run.")
        return
    banner("Cazzy Learning Lab — choose a task and a mode")
    for idx, t in enumerate(tasks, 1):
        print(f"[{idx}] {t.name} {'(functions)' if t.has_functions() else ''}")
    try:
        choice = int(input("\nSelect task # (or 0 to exit): ").strip())
    except Exception:
        print("Exiting.")
        return
    if choice <= 0 or choice > len(tasks):
        print("Bye!")
        return
    task = tasks[choice-1]
    banner("Selected task")
    describe_task(task)

    print("\nModes: 1) Bottom-Up  2) Top-Down  3) Expert  4) Playful  5) Variants  9) Run All  A) All Tasks × All Modes")
    mode = input("Choose mode: ").strip()
    try:
        repeat = int(input("Repeat each run how many times? (default 1): ") or 1)
    except Exception:
        repeat = 1
    results: List[Tuple[str, ModeResult]] = []

    def run_and_show(label: str, fn: Callable[[Task], ModeResult]):
        banner(label)
        for _ in range(repeat):
            res = fn(task)
            print(res.summary)
        if res.stdout.strip():
            print("\n--- OUTPUT ---\n" + res.stdout)
        if res.stderr.strip():
            print("\n--- ERRORS ---\n" + res.stderr)
        results.append((label, res))

    if mode == "1":
        for _ in range(repeat):
        run_and_show("Bottom-Up Walkthrough", mode_bottom_up)
    elif mode == "2":
        run_and_show("Top-Down Pipeline", mode_top_down)
    elif mode == "3":
        run_and_show("Expert Optimization", mode_expert)
    elif mode == "4":
        run_and_show("Playful Challenge", mode_playful)
        run_and_show("Variants Showcase", mode_variants)
    elif mode == "5":
        run_and_show("Variants Showcase", mode_variants)
    elif mode == "9":
        run_and_show("Bottom-Up Walkthrough", mode_bottom_up)
        run_and_show("Top-Down Pipeline", mode_top_down)
        run_and_show("Expert Optimization", mode_expert)
        run_and_show("Playful Challenge", mode_playful)
        run_and_show("Variants Showcase", mode_variants)
    elif mode.upper() == "A":
        # All tasks × all modes
        tasks = discover_tasks()
        for t in tasks:
            banner(f"[All] {t.name}")
            for label, fn in [
                ("Bottom-Up Walkthrough", mode_bottom_up),
                ("Top-Down Pipeline", mode_top_down),
                ("Expert Optimization", mode_expert),
                ("Playful Challenge", mode_playful),
        ("Variants Showcase", mode_variants),
            ]:
                banner(label + f" ×{repeat}")
                for _ in range(repeat):
                    res = fn(t)
                    print(res.summary)
                    if res.stdout.strip():
                        print("\n--- OUTPUT ---\n" + res.stdout)
                    if res.stderr.strip():
                        print("\n--- ERRORS ---\n" + res.stderr)
        run_and_show("Bottom-Up Walkthrough", mode_bottom_up)
        run_and_show("Top-Down Pipeline", mode_top_down)
        run_and_show("Expert Optimization", mode_expert)
        run_and_show("Playful Challenge", mode_playful)
        run_and_show("Variants Showcase", mode_variants)
    else:
        print("Unknown mode. Exiting.")

# --------- Quick mode (first task, all four) ---------

def parse_repeat(argv) -> int:
    try:
        if "--repeat" in argv:
            idx = argv.index("--repeat")
            return max(1, int(argv[idx+1]))
    except Exception:
        pass
    return 1


def quick_mode() -> None:
    tasks = discover_tasks()
    repeat = parse_repeat(sys.argv)
    if not tasks:
        print("No tasks found.")
        return
    task = tasks[0]
    banner(f"Quick on: {task.name}")
    for label, fn in [
        ("Bottom-Up Walkthrough", mode_bottom_up),
        ("Top-Down Pipeline", mode_top_down),
        ("Expert Optimization", mode_expert),
        ("Playful Challenge", mode_playful),
        ("Variants Showcase", mode_variants),
    ]:
        banner(label)
        for _ in range(repeat):
            res = fn(task)
            print(res.summary)
        if res.stdout.strip():
            print("\n--- OUTPUT ---\n" + res.stdout)
        if res.stderr.strip():
            print("\n--- ERRORS ---\n" + res.stderr)


def batch_all() -> None:
    """Run all discovered tasks across all modes. Supports --repeat N."""
    tasks = discover_tasks()
    if not tasks:
        print("No tasks found.")
        return
    repeat = parse_repeat(sys.argv)
    for task in tasks:
        banner(f"[Batch] {task.name}")
        for label, fn in [
            ("Bottom-Up Walkthrough", mode_bottom_up),
            ("Top-Down Pipeline", mode_top_down),
            ("Expert Optimization", mode_expert),
            ("Playful Challenge", mode_playful),
        ("Variants Showcase", mode_variants),
        ]:
            banner(label + f" ×{repeat}")
            for _ in range(repeat):
                res = fn(task)
                print(res.summary)
                if res.stdout.strip():
                    print("\n--- OUTPUT ---\n" + res.stdout)
                if res.stderr.strip():
                    print("\n--- ERRORS ---\n" + res.stderr)


if __name__ == "__main__":
    if "--batch-all" in sys.argv:
        batch_all()
    elif "--quick" in sys.argv:
        quick_mode()
    else:
        interactive()
