# recursion_masterclass.py
# -----------------------------------------------------------------------------
# Title: Recursion Masterclass — From First Principles to Expert Techniques
# Author: Cazzy Aporbo, 
# Started: April 14, 2024
# Last Updated: April 19, 2025
# Intent: Teach recursion in Python with maximum educational value:
#         multiple approaches for each problem, careful base cases, and
#         practical alternatives when recursion is not the right tool.
# -----------------------------------------------------------------------------

from __future__ import annotations  # forward reference convenience (Python 3.7+)

import sys                      # used to talk about recursion limits
from functools import lru_cache # memoization for performance
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

# Utility: console divider so this reads like a guided lesson when executed

def block(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)

# -----------------------------------------------------------------------------
# SECTION 0 — What is Recursion?
# -----------------------------------------------------------------------------

def lesson_intro() -> None:
    block("0) What is Recursion?")
    # Recursion is when a definition or process refers to itself.
    # The mirror analogy: two mirrors face each other and reflect recursively.
    print("Recursion is defining a solution in terms of smaller versions of itself.")
    print("The two-mirrors image is a good metaphor: reflection inside reflection.")
    # In code, a function can call itself to solve a smaller subproblem.
    # The two keys for safe recursion are:
    print("Key #1: A BASE CASE that returns a direct answer and stops recursing.")
    print("Key #2: A RECURSIVE STEP that moves the input closer to the base case.")

# -----------------------------------------------------------------------------
# SECTION 1 — Factorial: multiple ways (recursive, iterative, with validation)
# -----------------------------------------------------------------------------

def factorial_recursive(n: int) -> int:
    """Classic recursive factorial with explicit base case and input checks."""
    if n < 0:                      # guard against invalid input
        raise ValueError("factorial is undefined for negative integers")
    if n in (0, 1):                # base case: 0! == 1! == 1
        return 1                   # stop recursing here
    return n * factorial_recursive(n - 1)  # recursive step: shrink toward base


def factorial_iterative(n: int) -> int:
    """Iterative version: same math, no recursion; useful to avoid depth limits."""
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    result = 1                      # neutral element of multiplication
    for k in range(2, n + 1):       # multiply 2*3*...*n in a loop
        result *= k
    return result


def lesson_factorial() -> None:
    block("1) Factorial — recursive vs iterative, and why base cases matter")
    print("factorial_recursive(5) ->", factorial_recursive(5))
    print("factorial_iterative(5) ->", factorial_iterative(5))
    # Quick correctness checks across small values
    for i in range(6):
        assert factorial_recursive(i) == factorial_iterative(i)
    print("Validated 0..5 for both implementations.")

# -----------------------------------------------------------------------------
# SECTION 2 — Fibonacci: naive recursion, memoized recursion, bottom-up DP
# -----------------------------------------------------------------------------

def fib_recursive_naive(n: int) -> int:
    """Naive recursive Fibonacci: clear but exponentially slow."""
    if n < 0:
        raise ValueError("Fibonacci is undefined for negative integers")
    if n in (0, 1):      # base cases
        return n
    return fib_recursive_naive(n - 1) + fib_recursive_naive(n - 2)


@lru_cache(maxsize=None)
def fib_recursive_memo(n: int) -> int:
    """Memoized recursion: same recurrence, cached results (linear time)."""
    if n < 0:
        raise ValueError("Fibonacci is undefined for negative integers")
    if n in (0, 1):
        return n
    return fib_recursive_memo(n - 1) + fib_recursive_memo(n - 2)


def fib_bottom_up(n: int) -> int:
    """Dynamic programming (iterative) — efficient and avoids recursion depth."""
    if n < 0:
        raise ValueError("Fibonacci is undefined for negative integers")
    if n in (0, 1):
        return n
    a, b = 0, 1                    # F(0), F(1)
    for _ in range(2, n + 1):      # build upward
        a, b = b, a + b
    return b


def lesson_fibonacci() -> None:
    block("2) Fibonacci — naive recursion, memoization, dynamic programming")
    # Demonstrate small values so naive version doesn't explode in time
    for n in range(8):
        r1 = fib_recursive_naive(n)
        r2 = fib_recursive_memo(n)
        r3 = fib_bottom_up(n)
        assert r1 == r2 == r3, "All approaches must agree"
    print("Fibonacci 0..7 agree across naive, memoized, and bottom-up.")

# -----------------------------------------------------------------------------
# SECTION 3 — Recursion Depth and Errors (safe demo)
# -----------------------------------------------------------------------------

def lesson_depth() -> None:
    block("3) Recursion depth in CPython and safe error handling")
    limit = sys.getrecursionlimit()      # current safety cap (often 1000)
    print("Current recursion limit:", limit)

    # We'll trigger a RecursionError in a controlled way and catch it.
    def recursor(k: int) -> None:
        return recursor(k + 1)   # no base case → infinite recursion in theory

    try:
        recursor(0)
    except RecursionError as e:
        print("Caught RecursionError as expected:", e.__class__.__name__)

    # You *can* raise the limit, but be careful: you only buy risk of crashing.
    # Example (commented for safety):
    # sys.setrecursionlimit(5000)

# -----------------------------------------------------------------------------
# SECTION 4 — Backtracking: permutations and N-Queens (n=4 for speed)
# -----------------------------------------------------------------------------

def permutations_backtracking(items: List[Any]) -> List[List[Any]]:
    """Generate permutations via backtracking (classic recursive template)."""
    result: List[List[Any]] = []        # where we'll store full permutations
    used = [False] * len(items)         # track which positions we have taken
    path: List[Any] = []                # current partial permutation being built

    def dfs() -> None:                  # inner recursive depth-first search
        if len(path) == len(items):     # base: full-length permutation constructed
            result.append(path.copy())  # record a snapshot of the path
            return                      # stop going deeper here
        for i, val in enumerate(items): # try each choice
            if used[i]:
                continue                # skip values already in the path
            used[i] = True              # choose
            path.append(val)
            dfs()                       # explore deeper
            path.pop()                  # un-choose (backtrack)
            used[i] = False

    dfs()                               # kick off the recursion
    return result


def n_queens(n: int = 4) -> List[List[int]]:
    """Solve n-queens by backtracking; returns list of column positions per row."""
    cols: List[int] = []                # cols[r] = c means queen at row r, col c
    solutions: List[List[int]] = []

    def safe(c: int) -> bool:
        r = len(cols)
        for pr, pc in enumerate(cols):  # compare against all previously placed
            if pc == c or abs(pc - c) == abs(pr - r):
                return False            # same column or same diagonal
        return True

    def place() -> None:
        if len(cols) == n:              # base: placed n queens successfully
            solutions.append(cols.copy())
            return
        for c in range(n):              # try each column in this row
            if safe(c):
                cols.append(c)          # choose
                place()                 # explore
                cols.pop()              # backtrack

    place()
    return solutions


def lesson_backtracking() -> None:
    block("4) Backtracking — exploring choices and undoing them cleanly")
    print("permutations_backtracking([1,2,3]) ->", permutations_backtracking([1,2,3]))
    sols = n_queens(4)
    print("n_queens(4) solutions (as column indices per row) ->", sols)

# -----------------------------------------------------------------------------
# SECTION 5 — Trees and Graphs: recursive DFS traversal
# -----------------------------------------------------------------------------

def dfs_tree(tree: Dict[str, Any]) -> List[str]:
    """Depth-first traversal of a small tree represented as nested dicts."""
    order: List[str] = []

    def visit(node: Dict[str, Any]) -> None:
        order.append(node["name"])                      # pre-order: record node
        for child in node.get("children", []):          # then visit children
            visit(child)

    visit(tree)
    return order


def lesson_tree_traversal() -> None:
    block("5) Trees — recursive traversals in real data structures")
    tree = {
        "name": "A",  # root
        "children": [
            {"name": "B", "children": [{"name": "D"}, {"name": "E"}]},
            {"name": "C", "children": [{"name": "F"}]},
        ],
    }
    print("DFS pre-order ->", dfs_tree(tree))

# -----------------------------------------------------------------------------
# SECTION 6 — Mutual Recursion and Simulating Recursion with an Explicit Stack
# -----------------------------------------------------------------------------

def is_even(n: int) -> bool:
    """Mutual recursion demo: is_even uses is_odd and vice versa."""
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n: int) -> bool:
    if n == 0:
        return False
    return is_even(n - 1)


def factorial_with_explicit_stack(n: int) -> int:
    """Replace recursion with our own stack to avoid depth limits."""
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    stack: List[int] = []          # this will mimic the call stack of recursion
    while n > 1:                   # push frames (defer multiplications)
        stack.append(n)
        n -= 1
    result = 1
    while stack:                   # pop frames and apply the deferred work
        result *= stack.pop()
    return result


def lesson_mutual_and_stack() -> None:
    block("6) Mutual recursion + replacing recursion with an explicit stack")
    print("is_even(10) ->", is_even(10))
    print("is_odd(11)  ->", is_odd(11))
    print("factorial_with_explicit_stack(6) ->", factorial_with_explicit_stack(6))

# -----------------------------------------------------------------------------
# SECTION 7 — Tail Recursion Note (and why Python doesn't optimize it)
# -----------------------------------------------------------------------------

def factorial_tail_like(n: int, acc: int = 1) -> int:
    """Tail-style factorial; Python will NOT optimize this (no TCO in CPython)."""
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    if n in (0, 1):
        return acc                # base case returns the accumulator
    return factorial_tail_like(n - 1, acc * n)  # final action is the recursive call


def lesson_tail_recursion() -> None:
    block("7) Tail recursion in theory vs. Python's reality")
    print("Tail-style factorial works for small n but still uses one frame per call.")
    print("factorial_tail_like(5) ->", factorial_tail_like(5))

# -----------------------------------------------------------------------------
# SECTION 8 — A Recursion Design Checklist (thinking recursively)
# -----------------------------------------------------------------------------

def lesson_design_checklist() -> None:
    block("8) Recursion design checklist — what I go through every time")
    print("1. Identify a crisp base case that requires no further work.")
    print("2. Define the recursive step so each call shrinks the problem.")
    print("3. Ensure progress toward the base case (no infinite recursion).")
    print("4. Combine results from subproblems carefully (order matters?).")
    print("5. Consider stack depth and input size; switch to iteration if needed.")
    print("6. Add tests, especially edge cases (empty, 0, negative, singletons).")
    print("7. Profile and memoize if overlapping subproblems exist.")

# -----------------------------------------------------------------------------
# SECTION 9 — Tests (never ship recursion without them)
# -----------------------------------------------------------------------------

def lesson_tests() -> None:
    block("9) Tests — quick asserts to keep us honest")
    # Factorial
    for i, v in [(0, 1), (1, 1), (5, 120), (6, 720)]:
        assert factorial_recursive(i) == v
        assert factorial_iterative(i) == v
        assert factorial_with_explicit_stack(i) == v
    # Fibonacci
    fib_expected = [0, 1, 1, 2, 3, 5, 8, 13]
    for n, v in enumerate(fib_expected):
        assert fib_recursive_memo(n) == v
        assert fib_bottom_up(n) == v
    # Permutations
    perms_123 = permutations_backtracking([1, 2, 3])
    assert sorted(perms_123) == sorted([
        [1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]
    ])
    # N-Queens (4 has 2 solutions)
    assert len(n_queens(4)) == 2
    # Mutual recursion sanity
    assert is_even(0) and not is_odd(0)
    assert is_even(10) and is_odd(11)
    # DFS order
    tree = {"name": "A", "children": [{"name": "B"}, {"name": "C"}]}
    assert dfs_tree(tree) == ["A", "B", "C"]
    print("All tests passed.")

# -----------------------------------------------------------------------------
# MASTER RUNNER — present as a narrative when executed
# -----------------------------------------------------------------------------

def run_all_recursion_lessons() -> None:
    lesson_intro()
    lesson_factorial()
    lesson_fibonacci()
    lesson_depth()
    lesson_backtracking()
    lesson_tree_traversal()
    lesson_mutual_and_stack()
    lesson_tail_recursion()
    lesson_design_checklist()
    lesson_tests()


if __name__ == "__main__":
    run_all_recursion_lessons()
