"""
NCoins — Six Ways (Greedy traps, Top/Bottom DP, Graph Search, Bitsets, and a Game)

Perspective
-----------
I wrote this to teach *how pros think*: multiple mental models for the same problem,
and how to stress-test them with adversarial coin systems. You'll see when greedy
is perfect, when it quietly fails, and how DP/graph/bitset methods recover optimality.

What’s inside (six distinct solution strategies)
------------------------------------------------
  1) Greedy baseline                         → fast, intuitive, often wrong in generalized systems
  2) Divide & Conquer (plain recursion)      → clarity over performance; shows exponential blow-up
  3) Top-down DP (memoization)               → same recurrence, optimal substructure, no repeats
  4) Bottom-up DP (tabulation + traceback)   → classic O(V·m) with reconstruction (coins used)
  5) BFS on the state graph                  → edges have unit cost; shortest path = min #coins
  6) Bitset layer-BFS (integer bitset hack)  → O(minCoins·maxAmt/wordsize), elegant for “reachability by k”

Bonus (analysis hooks)
----------------------
  • Minimal counterexample finder (where Greedy ≠ Optimal for a coin system)
  • Timing harness (same target across methods)
  • Tiny console **game**: predict whether Greedy equals Optimal; guess coin counts

Datasets
--------
A few coin systems (in cents) that matter pedagogically:
  - Canonical (greedy is optimal): USD classic, EU cents, Binary
  - Non-canonical (greedy fails): add a 12c piece, or {1,3,4}, {1,7,10}, etc.

Author: Cazzy Aporbo  •  December 2024
Updated: August 13, 2025
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports (top-only, as requested)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
import sys
import math
import time
import random
from dataclasses import dataclass
from collections import deque, defaultdict
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

# Optional: pretty tables if rich is present; else, fall back to prints.
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None


# ──────────────────────────────────────────────────────────────────────────────
# Built-in “dataset” of coin systems (deliberately mixed)
# ──────────────────────────────────────────────────────────────────────────────
COIN_SYSTEMS: Dict[str, List[int]] = {
    # Canonical: Greedy == Optimal for all targets (with 1 present)
    "USD_Classic":      [1, 5, 10, 25, 50],
    "EU_Cents":         [1, 2, 5, 10, 20, 50],
    "Binary_Coins":     [1, 2, 4, 8, 16, 32, 64],

    # Non-canonical (greedy fails for some targets)
    "USD_With_12c":     [1, 5, 10, 12, 25],         # famous greedy trap
    "One_Three_Four":   [1, 3, 4],                  # greedy fails at 6 (3+3 vs 4+1+1)
    "One_Seven_Ten":    [1, 7, 10],                 # different trap profile
    "Random_Primey":    [1, 2, 3, 5, 7, 11, 13],    # behaves oddly at certain n
    "Dozenal_Flavor":   [1, 3, 12, 24],             # good lesson for structure thinkers
}

DEFAULT_TARGETS = [29, 63, 99, 117, 188]


# ──────────────────────────────────────────────────────────────────────────────
# Shared structures
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ChangeResult:
    method: str
    coinset: List[int]
    target: int
    count: int
    combo: List[int]           # multiset of coins used (values, not counts)
    elapsed_ms: float
    note: str = ""             # optional: warnings, optimality notes, etc.


# ──────────────────────────────────────────────────────────────────────────────
# Guardrails & helpers
# ──────────────────────────────────────────────────────────────────────────────
def validate_coinset(coins: List[int]) -> None:
    if not coins or any(c <= 0 for c in coins):
        raise ValueError("Coins must be a non-empty list of positive integers (cents).")
    if len(set(coins)) != len(coins):
        raise ValueError("Coin list has duplicates; remove them.")
    if math.gcd(*coins) != 1:
        # Without gcd=1, some amounts are unreachable; I warn but allow.
        print("⚠️  gcd(coins) > 1; some targets may be unreachable.", file=sys.stderr)

def normalize_combo(combo: List[int]) -> List[int]:
    return sorted(combo, reverse=True)

def pretty_combo(combo: List[int]) -> str:
    if not combo:
        return "∅"
    # collapse runs (e.g., 25×2 + 10×1 + 1×4)
    counts = defaultdict(int)
    for c in combo:
        counts[c] += 1
    parts = [f"{v}×{k}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[0], kv[1]))]
    return " + ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 1) Greedy baseline (descending coin choice)
# ──────────────────────────────────────────────────────────────────────────────
def change_greedy(coins: List[int], target: int) -> Optional[List[int]]:
    """
    Descending coin pick. Great intuition, wrong in general.
    Returns a combo or None if unreachable.
    """
    coins = sorted(coins, reverse=True)
    remaining = target
    out: List[int] = []
    for c in coins:
        k, remaining = divmod(remaining, c)
        out.extend([c] * k)
    if remaining != 0:
        return None
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2) Divide & Conquer (plain recursion): exponential, pedagogical
# ──────────────────────────────────────────────────────────────────────────────
def change_recursion(coins: List[int], target: int) -> Optional[List[int]]:
    """
    Try all first moves (subtract a coin) and recurse; pick minimal length.
    Exponential without memoization; perfect for illustrating repeated work.
    """
    if target == 0:
        return []
    if target < 0:
        return None

    best: Optional[List[int]] = None
    for c in coins:
        sub = change_recursion(coins, target - c)
        if sub is not None:
            candidate = sub + [c]
            if best is None or len(candidate) < len(best):
                best = candidate
    return best


# ──────────────────────────────────────────────────────────────────────────────
# 3) Top-down DP (memoization): same recurrence, no repeats
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=None)
def _memo_inner(target: int, coin_tuple: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    if target == 0:
        return tuple()
    if target < 0:
        return None
    best: Optional[Tuple[int, ...]] = None
    for c in coin_tuple:
        sub = _memo_inner(target - c, coin_tuple)
        if sub is not None:
            cand = sub + (c,)
            if best is None or len(cand) < len(best):
                best = cand
    return best

def change_memo(coins: List[int], target: int) -> Optional[List[int]]:
    _memo_inner.cache_clear()
    tup = tuple(sorted(coins))
    res = _memo_inner(target, tup)
    return list(res) if res is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# 4) Bottom-up DP (tabulation + traceback)
# ──────────────────────────────────────────────────────────────────────────────
def change_bottomup(coins: List[int], target: int) -> Optional[List[int]]:
    """
    dp[v] = min #coins to reach value v; parent[v] = coin used last.
    """
    maxV = target
    dp = [math.inf] * (maxV + 1)
    parent = [-1] * (maxV + 1)
    dp[0] = 0
    for v in range(1, maxV + 1):
        for c in coins:
            if v - c >= 0 and dp[v - c] + 1 < dp[v]:
                dp[v] = dp[v - c] + 1
                parent[v] = c
    if math.isinf(dp[target]):
        return None
    # reconstruct
    out: List[int] = []
    v = target
    while v > 0:
        c = parent[v]
        out.append(c)
        v -= c
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5) BFS on the state graph (0 → target), each edge adds a coin (unit cost)
# ──────────────────────────────────────────────────────────────────────────────
def change_bfs(coins: List[int], target: int) -> Optional[List[int]]:
    """
    Each node is a value (sum); start at 0; edges: +c for c in coins (if <= target).
    Uniform costs → BFS finds min edges = min #coins.
    """
    if target == 0:
        return []
    q = deque([0])
    parent: Dict[int, Tuple[int, int]] = {0: (-1, -1)}  # value -> (prev, coin)
    while q:
        v = q.popleft()
        for c in coins:
            u = v + c
            if u > target or u in parent:
                continue
            parent[u] = (v, c)
            if u == target:
                # reconstruct
                out: List[int] = []
                cur = u
                while cur != 0:
                    prev, coin = parent[cur]
                    out.append(coin)
                    cur = prev
                return out
            q.append(u)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 6) Bitset layer-BFS: grow reachability by #coins using integer bit ops
# ──────────────────────────────────────────────────────────────────────────────
def change_bitset(coins: List[int], target: int, max_layers: Optional[int] = None) -> Optional[List[int]]:
    """
    Idea: Let R_k be a bitset where bit v is 1 if value v is reachable using ≤ k coins.
    Then R_{k+1} = OR over c in coins of (R_k << c). Find the smallest k with bit[target]=1.
    After k is known, reconstruct by walking backwards (check membership in previous layer).
    This runs fast in CPython because Python's big ints are optimized C loops.
    """
    if target == 0:
        return []

    # bound layers: worst-case ≤ target if coin 1 exists; else we can cap somewhat
    if max_layers is None:
        max_layers = target

    # Reachability layers
    layers: List[int] = []  # store bitsets for reconstruction
    R = 1  # bit 0 set
    mask = (1 << (target + 1)) - 1  # keep within [0..target]
    goal_layer = -1

    for k in range(1, max_layers + 1):
        shifted_or = 0
        for c in coins:
            shifted_or |= (R << c)
        R = (R | shifted_or) & mask
        layers.append(R)
        if (R >> target) & 1:
            goal_layer = k
            break

    if goal_layer == -1:
        return None

    # Reconstruct by going backwards: at layer k, check which coin c yields membership in layer k-1
    combo: List[int] = []
    v = target
    prev_R = 1  # R_0
    for k in range(goal_layer, 0, -1):
        Rk = layers[k - 1]
        # ensure strictly increasing with each layer for reconstruction logic
        # find a coin c such that v - c is reachable in R_{k-1}
        chosen = None
        for c in sorted(coins, reverse=True):
            if v - c >= 0 and ((prev_R >> (v - c)) & 1):
                chosen = c
                break
        if chosen is None:
            # fallback: membership test against R_{k-1} by recomputing prev_R
            prev_prev = 1
            for kk in range(1, k):
                tmp = 0
                for cc in coins:
                    tmp |= (prev_prev << cc)
                prev_prev = (prev_prev | tmp) & mask
            for c in sorted(coins, reverse=True):
                if v - c >= 0 and ((prev_prev >> (v - c)) & 1):
                    chosen = c
                    prev_R = prev_prev
                    break
        else:
            # advance prev_R one layer to align with next iteration
            tmp = 0
            for cc in coins:
                tmp |= (prev_R << cc)
            prev_R = (prev_R | tmp) & mask

        if chosen is None:
            # If reconstruction failed due to a degenerate set, bail out (should be rare)
            return None
        combo.append(chosen)
        v -= chosen

    return combo


# ──────────────────────────────────────────────────────────────────────────────
# Minimal counterexample finder: where Greedy != Optimal
# ──────────────────────────────────────────────────────────────────────────────
def first_greedy_failure(coins: List[int], limit: int = 500) -> Optional[Tuple[int, List[int], List[int]]]:
    """
    Return the smallest amount ≤ limit where greedy differs from bottom-up optimal.
    """
    for t in range(1, limit + 1):
        g = change_greedy(coins, t)
        o = change_bottomup(coins, t)
        if (g is None) != (o is None):
            return (t, g or [], o or [])
        if g is not None and len(g) != len(o or []):
            return (t, g, o or [])
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Uniform runner + timing
# ──────────────────────────────────────────────────────────────────────────────
def run_method(label: str, fn, coins: List[int], target: int) -> ChangeResult:
    t0 = time.perf_counter()
    combo = fn(coins, target)
    dt = (time.perf_counter() - t0) * 1e3
    if combo is None:
        return ChangeResult(label, coins, target, math.inf, [], dt, note="unreachable")
    return ChangeResult(label, coins, target, len(combo), normalize_combo(combo), dt)

def print_table(rows: List[ChangeResult]) -> None:
    if RICH:
        table = Table(title="NCoins — Methods Comparison", box=box.SIMPLE_HEAVY)
        table.add_column("Method", style="bold")
        table.add_column("#", justify="right")
        table.add_column("Combo (value×coin)", overflow="fold")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Note", overflow="fold")
        for r in rows:
            table.add_row(r.method, "∞" if math.isinf(r.count) else str(r.count),
                          pretty_combo(r.combo), f"{r.elapsed_ms:.2f}", r.note)
        console.print(table)
    else:
        print("\nNCoins — Methods Comparison")
        print("-" * 84)
        for r in rows:
            cc = "∞" if math.isinf(r.count) else str(r.count)
            print(f"{r.method:30s} | {cc:>3s} | {pretty_combo(r.combo):40s} | {r.elapsed_ms:8.2f} ms | {r.note}")


# ──────────────────────────────────────────────────────────────────────────────
# Game: Predict greedy correctness, guess minimal coin count
# ──────────────────────────────────────────────────────────────────────────────
def game(coins: List[int]) -> None:
    """
    Short console game:
      • I draw a random target.
      • You predict whether Greedy is optimal (Y/N).
      • Then guess the minimal #coins. I reveal both and show the combo.
    """
    print("\n— Mini Game —")
    print(f"Coin system: {coins}")
    target = random.choice(range(18, 131))
    print(f"Target is: {target}")
    ans = input("Will Greedy be optimal? (y/n): ").strip().lower()
    g_combo = change_greedy(coins, target)
    o_combo = change_bottomup(coins, target)
    greedy_ok = (g_combo is not None) and (o_combo is not None) and (len(g_combo) == len(o_combo))
    guess = input("Your guess for minimal #coins: ").strip()
    try:
        guess_n = int(guess)
    except Exception:
        guess_n = None

    print("\nResults:")
    print(f"  Greedy optimal? {'YES' if greedy_ok else 'NO'}")
    print(f"  Greedy combo : {pretty_combo(g_combo or [])}")
    print(f"  Optimal combo: {pretty_combo(o_combo or [])}")
    if guess_n is not None and o_combo is not None:
        delta = abs(guess_n - len(o_combo))
        print(f"  Your guess was {guess_n} (off by {delta}).")

    fail = first_greedy_failure(coins, limit=200)
    if fail:
        amt, g, o = fail
        print(f"\nSmallest counterexample ≤200: amount={amt}")
        print(f"  Greedy : {pretty_combo(g)}  (#{len(g)})")
        print(f"  Optimal: {pretty_combo(o)}  (#{len(o)})")


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────
def choose_coin_system() -> Tuple[str, List[int]]:
    print("\nAvailable coin systems (cents):")
    keys = list(COIN_SYSTEMS.keys())
    for i, k in enumerate(keys, 1):
        print(f"  {i:2d}) {k:16s}  -> {COIN_SYSTEMS[k]}")
    raw = input("Pick system by number (default 1): ").strip()
    idx = 1 if not raw else max(1, min(len(keys), int(raw)))
    name = keys[idx - 1]
    coins = COIN_SYSTEMS[name]
    validate_coinset(coins)
    return name, coins

def ask_target() -> int:
    raw = input(f"Target amount in cents (Enter for random from {DEFAULT_TARGETS}): ").strip()
    if not raw:
        return random.choice(DEFAULT_TARGETS)
    try:
        v = int(raw)
        if v < 0:
            raise ValueError
        return v
    except ValueError:
        print("Please enter a non-negative integer.")
        return ask_target()

def main() -> int:
    print("\nNCoins — Six Ways  •  by Cazzy Aporbo")
    sys.setrecursionlimit(10000)

    sys_name, coins = choose_coin_system()
    target = ask_target()
    print(f"\nUsing {sys_name}: coins={coins}, target={target}")

    methods = [
        ("Greedy (descending pick)", change_greedy),
        ("Divide & Conquer (plain rec)", change_recursion),
        ("Top-down + memoization", change_memo),
        ("Bottom-up DP (tabulation)", change_bottomup),
        ("BFS on value graph", change_bfs),
        ("Bitset layer-BFS", change_bitset),
    ]

    rows: List[ChangeResult] = []
    for label, fn in methods:
        try:
            rows.append(run_method(label, fn, coins, target))
        except RecursionError:
            rows.append(ChangeResult(label, coins, target, math.inf, [],
                                     0.0, note="recursion depth"))
        except Exception as e:
            rows.append(ChangeResult(label, coins, target, math.inf, [],
                                     0.0, note=f"error: {type(e).__name__}"))

    # Flag greedy suboptimality explicitly
    greedy_row = next(r for r in rows if r.method.startswith("Greedy"))
    optimal_row = min((r for r in rows if math.isfinite(r.count)), key=lambda r: r.count)
    for r in rows:
        if r.method.startswith("Greedy") and math.isfinite(optimal_row.count) and r.count != optimal_row.count:
            r.note = "⚠️ suboptimal vs DP"
        if r.method == "Divide & Conquer (plain rec)":
            r.note = (r.note + " | " if r.note else "") + "exponential demo"

    print_table(rows)

    # If greedy fails, show smallest counterexample
    if greedy_row.count != optimal_row.count and math.isfinite(optimal_row.count):
        fail = first_greedy_failure(coins, limit=max(200, target))
        if fail:
            amt, g, o = fail
            print(f"\nSmallest amount where Greedy ≠ Optimal: {amt}")
            print(f"  Greedy : {pretty_combo(g)}  (#{len(g)})")
            print(f"  Optimal: {pretty_combo(o)}  (#{len(o)})")

    print("\nNotes:")
    print(" • Greedy is optimal for canonical systems (USD_Classic, EU_Cents, Binary_Coins).")
    print(" • It fails for systems like USD_With_12c or One_Three_Four. Bottom-up DP (or BFS/bitset) fixes it.")
    print(" • Bitset trick is faster than it looks: CPython bigint bit ops run in C, not Python loops.")

    # Offer the mini game
    play = input("\nPlay the mini game? [y/N]: ").strip().lower()
    if play == "y":
        game(coins)

    print("\nDone. Keep the systems adversarial — that’s where intuition stays honest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
