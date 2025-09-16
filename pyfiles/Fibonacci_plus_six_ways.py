"""
Fibonacci — Six (plus) Ways: a guided climb from approachable to abstract.

I wrote this to show learners how the *same* idea can be expressed with
different mental models and data structures — from loops to DP to algebra.
We’ll compare tradeoffs, note pitfalls, and even play a quick console mini-game.

Author: Cazzy Aporbo  •  December 3, 2024
Updated: August 13, 2025

What you’ll see (and why it matters)
------------------------------------
  1) Iterative list build (bottom-up)          → state updates, O(n), tiny RAM
  2) Streaming generator (lazy)                → backpressure-friendly pipelines
  3) Top-down recursion + memoization          → optimal substructure, cache hits
  4) Bottom-up DP with dict *and* list forms   → same recurrence, two containers
  5) Fast doubling (divide & conquer)          → T(n) = O(log n), elegant algebra
  6) 2×2 matrix power (exponentiation by squaring) → linear recurrences as linear algebra
  7) (Bonus) itertools “pair-walk” stream      → functional, compact, beginner-friendly

“Game-style” bits
-----------------
 • Quick timing table for each method (same n).
 • A tiny console quiz: guess F(k); I reveal hints and teach estimation.
 • Clear notes on what to look out for: base cases, off-by-one, rounding traps.

Run
---
$ python fibonacci_six_plus_ways.py
"""

# ---- Imports (top only) ------------------------------------------------------
import sys
import math
import time
from functools import lru_cache
from collections import deque
from itertools import islice

# Optional niceties: pretty table if available; otherwise plain prints.
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None

# Optional linear algebra speedup (not required). Falls back to pure ints.
try:
    import numpy as np  # noqa: F401
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


# ---- Utility: defensively parse n -------------------------------------------
def ask_n(prompt="How many Fibonacci numbers do you want? (default 20): ", default=20) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        n = int(raw)
        if n < 0:
            raise ValueError
        return n
    except ValueError:
        print("Please provide a non-negative integer.")
        return ask_n(prompt, default)


# ---- Method 1: Iterative list build (bottom-up) ------------------------------
def fib_list(n: int):
    """
    Build the first n Fibonacci numbers bottom-up using a list.
    Skills: state updates; off-by-one discipline; O(n) time, O(1) extra space (aside from output).
    """
    seq = []
    a, b = 0, 1
    for _ in range(n):
        seq.append(a)   # emit current
        a, b = b, a + b # advance state
    return seq


# ---- Method 2: Streaming generator (lazy) ------------------------------------
def fib_gen(n: int):
    """
    Yield the first n Fibonacci numbers lazily (stream style).
    Skills: generators; backpressure; pipeline-friendly iteration.
    """
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


# ---- Method 3: Top-down recursion + memoization (lru_cache) ------------------
@lru_cache(maxsize=None)
def _fib_td_single(k: int) -> int:
    # Base cases (decide them early; the rest is mechanical)
    if k < 2:
        return k
    return _fib_td_single(k - 1) + _fib_td_single(k - 2)

def fib_topdown(n: int):
    """
    First n numbers via top-down recursion with memoization.
    Skills: optimal substructure; overlapping subproblems; cache hits vs naive blow-up.
    """
    # Warm the cache and collect in order
    return [_fib_td_single(k) for k in range(n)]


# ---- Method 4: Bottom-up DP (dict and list flavors) --------------------------
def fib_bottomup_list(n: int):
    """
    Bottom-up DP with a list. Identical values to fib_list; more “DP” in spirit.
    Skills: tabulation; explicit base seeds; index discipline.
    """
    if n == 0:
        return []
    if n == 1:
        return [0]
    dp = [0, 1] + [0] * (n - 2)
    for i in range(2, n):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp

def fib_bottomup_dict(n: int):
    """
    Bottom-up DP with a dict. Overkill for a line recurrence, but useful
    when indices aren’t contiguous (sparse DP) or keys are tuples.
    Skills: mapping state explicitly; clarity over compactness.
    """
    d = {0: 0, 1: 1}
    for i in range(2, n):
        d[i] = d[i - 1] + d[i - 2]
    # materialize a list for consistent interface
    return [d[i] for i in range(n)]


# ---- Method 5: Fast doubling (divide & conquer) ------------------------------
def _fib_fast_doubling(k: int):
    """
    Return (F(k), F(k+1)) using identities:
      F(2m)   = F(m) * [2*F(m+1) − F(m)]
      F(2m+1) = F(m+1)^2 + F(m)^2
    T(k) = O(log k). Pure integer arithmetic (exact).
    """
    if k == 0:
        return (0, 1)
    a, b = _fib_fast_doubling(k // 2)  # (F(m), F(m+1))
    c = a * ((b << 1) - a)             # F(2m)
    d = a * a + b * b                  # F(2m+1)
    if k % 2 == 0:
        return (c, d)
    else:
        return (d, c + d)

def fib_fast_doubling_list(n: int):
    """
    First n numbers using fast doubling per index (still quick for n up to ~1e5).
    For huge n, you’d stream with a generator or keep last two only.
    Skills: divide-and-conquer algebra; bit-level intuition (<<1 is ×2).
    """
    return [_fib_fast_doubling(k)[0] for k in range(n)]


# ---- Method 6: 2×2 matrix power (exponentiation by squaring) -----------------
def _matmul2(A, B):
    # Minimal 2×2 integer multiply (kept explicit to avoid external deps)
    return (
        A[0]*B[0] + A[1]*B[2],
        A[0]*B[1] + A[1]*B[3],
        A[2]*B[0] + A[3]*B[2],
        A[2]*B[1] + A[3]*B[3],
    )

def _matpow2(M, e: int):
    # Fast power with squaring; identity is (1,0,0,1)
    R = (1, 0, 0, 1)
    while e > 0:
        if e & 1:
            R = _matmul2(R, M)
        M = _matmul2(M, M)
        e >>= 1
    return R

def fib_matrix_power(n: int):
    """
    First n numbers from powers of the Fibonacci Q-matrix:
      Q = [[1,1],[1,0]]  and  Q^k = [[F(k+1), F(k)], [F(k), F(k-1)]]
    Skills: recurrences ↔ linear algebra; exponentiation by squaring.
    """
    out = []
    Q = (1, 1, 1, 0)
    for k in range(n):
        if k == 0:
            out.append(0)
        else:
            R = _matpow2(Q, k)
            out.append(R[1]) # top-right = F(k)
    return out


# ---- Method 7 (Bonus): “pair-walk” stream (functional flavor) ----------------
def fib_pairwalk(n: int):
    """
    Generator that walks (a,b)->(b,a+b); feels like itertools but stays explicit.
    Skills: state as a tuple; readable streaming; zero extra containers.
    """
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


# ---- Game-style: tiny console quiz ------------------------------------------
def quiz_mode(seq):
    """
    Pick a random index and quiz the user. Estimation hint:
      F(n) ~ phi^n / sqrt(5), where phi ≈ 1.618…
    """
    if not seq:
        return
    import random
    n = len(seq)
    k = random.randint(5, max(6, n - 1))
    target = seq[k]
    print(f"\nQuick quiz: What is F({k})?")
    hint = int(round((1.618033988749895 ** k) / 2.23606797749979))  # phi^k / sqrt(5)
    print(f"(Hint: ≈ {hint}, exact integer nearby)")

    guess_raw = input("Your guess: ").strip()
    try:
        guess = int(guess_raw)
    except ValueError:
        print("Numbers only! No worry—we’re here to learn.")
        return
    if guess == target:
        print("Nice! Nailed it.")
    else:
        delta = abs(guess - target)
        print(f"Close! True F({k}) = {target}  (you were off by {delta})")


# ---- Pretty print / timing harness ------------------------------------------
def time_method(fn, n: int, label: str):
    t0 = time.perf_counter()
    res = fn(n)
    dt = time.perf_counter() - t0
    # Normalize to a list for comparison (generators)
    if not isinstance(res, list):
        res = list(res)
    return label, res, dt

def show_table(rows):
    if RICH:
        table = Table(title="Fibonacci — Methods Comparison", box=box.SIMPLE_HEAVY)
        table.add_column("Method", style="bold")
        table.add_column("Time (ms)", justify="right")
        table.add_column("First 10 terms", overflow="fold")
        for name, seq, dt in rows:
            preview = ", ".join(map(str, seq[:10]))
            table.add_row(name, f"{dt*1e3:.2f}", preview)
        console.print(table)
    else:
        print("\nFibonacci — Methods Comparison")
        print("-" * 72)
        for name, seq, dt in rows:
            preview = ", ".join(map(str, seq[:10]))
            print(f"{name:28s} | {dt*1e3:7.2f} ms | {preview}")

def sanity_check_equal(rows, n: int):
    # All sequences should match for the first n terms.
    base = rows[0][1]
    ok = all(r[1] == base for r in rows[1:])
    if ok:
        print("Sanity: ✅ all sequences match.")
    else:
        print("Sanity: ❌ mismatch detected. Investigate rounding or base cases.")


# ---- Things to watch out for -------------------------------------------------
NOTES = """
What to look out for:
 • Base cases: define F(0)=0, F(1)=1 once; the rest follows.
 • Off-by-one: “first n numbers” means indices [0..n-1]; be consistent.
 • Recursion without memoization explodes (≈ φ^n). Memoization fixes it.
 • Fast methods (doubling/matrix) compute F(k) in O(log k); to build a *list*,
   either loop k=0..n-1 (still cheap) or stream with a generator.
 • Binet’s formula (closed form) is beautiful but floats round off—avoid for big n
   unless you use Decimal with enough precision (int still wins for exactness).
"""


# ---- Main menu ---------------------------------------------------------------
def choose_methods():
    print("\nWhich methods do you want to run?")
    print("  1) Iterative list (bottom-up)")
    print("  2) Streaming generator (lazy)")
    print("  3) Top-down recursion + memoization")
    print("  4) Bottom-up DP (list) + (dict)")
    print("  5) Fast doubling (divide & conquer)")
    print("  6) 2×2 matrix power (exp by squaring)")
    print("  7) Bonus: pair-walk generator")
    print("  A) All of the above")
    choice = input("Enter numbers/letters (e.g., '1 4 5' or 'A'): ").strip().lower()
    if not choice or choice == "a":
        return ["1","2","3","4","5","6","7"]
    return choice.split()

def main():
    print("\nFibonacci — Six (plus) Ways  •  by Cazzy Aporbo")
    n = ask_n()
    picks = choose_methods()

    # Build the requested rows
    rows = []

    if "1" in picks:
        rows.append(time_method(fib_list, n, "1) iterative list"))

    if "2" in picks:
        rows.append(time_method(lambda k: fib_gen(k), n, "2) generator (lazy)"))

    if "3" in picks:
        # clear the cache so timing is fair across runs
        _fib_td_single.cache_clear()
        rows.append(time_method(fib_topdown, n, "3) top-down + lru_cache"))

    if "4" in picks:
        rows.append(time_method(fib_bottomup_list, n, "4a) bottom-up DP (list)"))
        rows.append(time_method(fib_bottomup_dict, n, "4b) bottom-up DP (dict)"))

    if "5" in picks:
        rows.append(time_method(fib_fast_doubling_list, n, "5) fast doubling"))

    if "6" in picks:
        rows.append(time_method(fib_matrix_power, n, "6) matrix power"))

    if "7" in picks:
        rows.append(time_method(lambda k: fib_pairwalk(k), n, "7) pair-walk stream"))

    # Show timing table + sanity check
    show_table(rows)
    sanity_check_equal(rows, n)

    # Print “what to watch for”
    print(NOTES)

    # Tiny quiz to make it feel like a game
    wanna_quiz = input("Play a quick F(n) quiz? [y/N]: ").strip().lower()
    if wanna_quiz == "y":
        # Use a consistent reference sequence (fast, exact)
        seq = rows[0][1] if rows else fib_bottomup_list(n)
        # If the list is short, extend a bit so quiz index is meaningful
        if len(seq) < 15:
            seq = fib_bottomup_list(25)
        quiz_mode(seq)

    print("\nThanks for coding along — same math, many paths. ✨")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
