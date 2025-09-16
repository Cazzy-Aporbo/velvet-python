"""
Matrix Chain Multiplication — Four Ways, with Progressive Visualization

What this program teaches
-------------------------
1) Why parenthesization matters for chained matrix multiplication (A1·A2·...·An)
   — multiplication is associative (order of grouping matters) but not commutative.
2) Four approaches:
   (A) Naive brute force (enumerate all parenthesizations) — for small n, educational.
   (B) Divide & Conquer with memoization (top–down DP).
   (C) Bottom–up Dynamic Programming (tabulation) with parenthesis reconstruction.
   (D) A "looks reasonable but wrong" greedy heuristic — to show pitfalls.
3) Progressive visuals with palette control (Primary / Black & White / Pastel),
   rendered with subtle ombré gradients:
   F1: Cost comparison for every parenthesization (small n).
   F2: DP cost table heatmap with split (k) annotations.
   F3: Recursion call counts — naive vs memoized (why DP matters).
   F4: (Optional) Timing sanity check by actually multiplying random matrices
       in the optimal vs greedy orders to show correlation with cost model.

Data model
----------
For n matrices A1..An with dimensions:
  A1 is (p0 × p1), A2 is (p1 × p2), ..., An is (p_{n-1} × p_n)
The "dims" vector is [p0, p1, ..., p_n] (length n+1).
The scalar multiply cost for (Ai..Ak)·(A{k+1}..Aj) at split k is:
  dp[i][k] + dp[k+1][j] + p_i * p_{k+1} * p_{j+1}

Run
---
$ python matrix_chain_four_ways.py
(choose a palette when prompted; outputs saved to Desktop/MatrixChain_Outputs)

Author’s note
-------------
Approachable for beginners, still precise for experts. Clean top–down orchestration,
bottom–up utilities, and guardrails for real-world usage.
"""

from __future__ import annotations

# ----- Imports (top only) -----------------------------------------------------
import os
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ----- Palette utilities (ombré gradients) -----------------------------------
def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def _lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))

def make_ombre(start_hex: str, end_hex: str, n: int) -> List[str]:
    r1, g1, b1 = _hex_to_rgb(start_hex)
    r2, g2, b2 = _hex_to_rgb(end_hex)
    steps = []
    for i in range(max(1, n)):
        t = i / max(1, n - 1)
        steps.append(_rgb_to_hex((_lerp(r1, r2, t), _lerp(g1, g2, t), _lerp(b1, b2, t))))
    return steps

def choose_palette() -> Dict[str, List[str]]:
    """
    Ask the user for a theme. Return ombré lists we’ll reuse consistently.
    """
    print("\nChoose a visualization palette:")
    print(" [1] Primary (bold)")
    print(" [2] Black & White (publication style)")
    print(" [3] Pastel (soft)")
    choice = (input("Enter 1 / 2 / 3 (default: 1): ").strip() or "1")
    theme = {"1": "primary", "2": "bw", "3": "pastel"}.get(choice, "primary")

    if theme == "primary":
        scatter = make_ombre("#1F77B4", "#FF7F0E", 200)  # blue → orange
        line    = make_ombre("#2CA02C", "#D62728", 60)   # green → red
        heat    = make_ombre("#F7FBFF", "#084594", 200)  # light → deep blue
    elif theme == "bw":
        scatter = make_ombre("#111111", "#CCCCCC", 200)  # black → gray
        line    = make_ombre("#333333", "#AAAAAA", 60)
        heat    = make_ombre("#FFFFFF", "#000000", 200)
    else:  # pastel
        scatter = make_ombre("#FFB5CC", "#D4FFE4", 200)  # pink → mint
        line    = make_ombre("#D8B5D8", "#C8A8C8", 60)   # lavender ombré
        heat    = make_ombre("#FFF8FD", "#C8A8C8", 200)

    return {"scatter": scatter, "line": line, "heat": heat}


# ----- Output location: Desktop/MatrixChain_Outputs ---------------------------
def ensure_output_dir(path: Optional[str] = None) -> str:
    """
    Default: <Desktop>/MatrixChain_Outputs (cross-platform, OneDrive aware).
    """
    if path:
        out = Path(path).expanduser().resolve()
    else:
        home = Path.home()
        candidates = [
            home / "Desktop",                                   # macOS/Linux
            Path(os.environ.get("USERPROFILE", "")) / "Desktop",# Windows
            Path(os.environ.get("OneDrive", "")) / "Desktop",   # OneDrive
        ]
        desktop = next((p for p in candidates if p and p.exists()), home / "Desktop")
        out = desktop / "MatrixChain_Outputs"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


# ----- Core math helpers ------------------------------------------------------
def validate_dims(dims: List[int]) -> None:
    """
    dims must be length n+1 and all positive. For A1..An, A_i has shape (dims[i], dims[i+1]).
    """
    if not isinstance(dims, (list, tuple)) or len(dims) < 2:
        raise ValueError("dims must be a list like [p0, p1, ..., p_n] with length >= 2")
    if any(int(d) <= 0 for d in dims):
        raise ValueError("All dimensions must be positive integers")
    # Nothing else to enforce; chain validity is encoded by shared dims.

def split_cost(dims: List[int], i: int, k: int, j: int) -> int:
    """
    Cost of multiplying (Ai..Ak) by (A{k+1}..Aj) after the subproblems are solved:
    p_i * p_{k+1} * p_{j+1}
    """
    return dims[i] * dims[k + 1] * dims[j + 1]


# ----- Four approaches --------------------------------------------------------
# A) Brute force (enumerate all parenthesizations) — educational; n should be small.
def brute_force_min_cost(dims: List[int]) -> Tuple[int, str, int]:
    """
    Return (min_cost, parenthesization_string, call_count).
    Exponential; safe for n <= ~10 (Catalan growth). We enforce a small guard.
    """
    n = len(dims) - 1
    if n > 10:
        raise ValueError("Brute force is restricted to n<=10 for pedagogy. Use DP for larger n.")

    call_counter = 0
    from functools import lru_cache

    def name(i: int) -> str:
        return f"A{i+1}"

    @lru_cache(maxsize=None)
    def best(i: int, j: int) -> Tuple[int, str]:
        nonlocal call_counter
        call_counter += 1

        if i == j:
            return 0, name(i)

        best_cost = math.inf
        best_expr = ""
        for k in range(i, j):
            c_left, e_left = best(i, k)
            c_right, e_right = best(k + 1, j)
            c_split = split_cost(dims, i, k, j)
            cost = c_left + c_right + c_split
            if cost < best_cost:
                best_cost = cost
                best_expr = f"({e_left} × {e_right})"
        return best_cost, best_expr

    min_cost, expr = best(0, n - 1)
    return int(min_cost), expr, call_counter

# B) Divide & Conquer with memoization (explicit cache + reconstruction)
def topdown_memo(dims: List[int]) -> Tuple[int, List[List[int]]]:
    """
    Return (min_cost, split_table).
    split_table[i][j] stores the argmin k that achieves dp[i][j].
    """
    n = len(dims) - 1
    dp = [[None for _ in range(n)] for _ in range(n)]
    split = [[-1 for _ in range(n)] for _ in range(n)]
    call_counter = 0

    def solve(i: int, j: int) -> int:
        nonlocal call_counter
        call_counter += 1

        if i == j:
            dp[i][j] = 0
            return 0
        if dp[i][j] is not None:
            return dp[i][j]  # memoized

        best_cost = math.inf
        best_k = -1
        for k in range(i, j):
            left = solve(i, k)
            right = solve(k + 1, j)
            sc = split_cost(dims, i, k, j)
            cost = left + right + sc
            if cost < best_cost:
                best_cost = cost
                best_k = k
        dp[i][j] = int(best_cost)
        split[i][j] = best_k
        return dp[i][j]

    total = solve(0, n - 1)
    return int(total), split  # dp table not returned to keep API compact

# C) Bottom–up Dynamic Programming (tabulation) + reconstruction
def bottomup_dp(dims: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build dp (min scalar multiplies) and split (argmin k) tables.
    dp[i][i] = 0; dp[i][j] = min_k dp[i][k] + dp[k+1][j] + p_i*p_{k+1}*p_{j+1}
    """
    n = len(dims) - 1
    dp = [[0 if i == j else math.inf for j in range(n)] for i in range(n)]
    split = [[-1 for _ in range(n)] for _ in range(n)]

    for gap in range(1, n):             # chain length - 1
        for i in range(0, n - gap):
            j = i + gap
            best_c, best_k = math.inf, -1
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + split_cost(dims, i, k, j)
                if cost < best_c:
                    best_c, best_k = cost, k
            dp[i][j] = int(best_c)
            split[i][j] = best_k
    return dp, split

def reconstruct(split: List[List[int]], i: int, j: int) -> str:
    """Return parenthesization string using the split table."""
    if i == j:
        return f"A{i+1}"
    k = split[i][j]
    return f"({reconstruct(split, i, k)} × {reconstruct(split, k + 1, j)})"

# D) A simple greedy pitfall: always split where the immediate multiplication p_i * p_{k+1} * p_{j+1} is smallest
def greedy_bad(dims: List[int]) -> Tuple[int, str]:
    """
    Not optimal in general. Demonstrates why "local best" can be globally bad.
    """
    n = len(dims) - 1
    ranges = [(i, i) for i in range(n)]  # keep contiguous blocks
    exprs = [f"A{i+1}" for i in range(n)]
    total_cost = 0

    while len(ranges) > 1:
        # try all adjacent merges; pick the cheapest immediate cost
        best_idx, best_c = -1, math.inf
        for t in range(len(ranges) - 1):
            i = ranges[t][0]
            j = ranges[t + 1][1]
            k = ranges[t][1]  # split between these two blocks
            c = split_cost(dims, i, k, j)
            if c < best_c:
                best_c = c
                best_idx = t
        total_cost += best_c
        # merge at best_idx
        i0, _ = ranges[best_idx]
        _, j1 = ranges[best_idx + 1]
        new_range = (i0, j1)
        new_expr = f"({exprs[best_idx]} × {exprs[best_idx+1]})"
        ranges = ranges[:best_idx] + [new_range] + ranges[best_idx + 2:]
        exprs = exprs[:best_idx] + [new_expr] + exprs[best_idx + 2:]

    return int(total_cost), exprs[0]


# ----- Visualization helpers --------------------------------------------------
def save_bar_all_parenthesizations(labels: List[str], costs: List[int], best_idx: int,
                                   palette: Dict[str, List[str]], outdir: str, tag: str) -> None:
    plt.figure(figsize=(10, 5.2))
    xs = np.arange(len(labels))
    # color bars with ombré
    base = np.array(palette["scatter"])
    col_idx = (np.linspace(0, len(base) - 1, len(labels))).astype(int)
    colors = base[col_idx]
    plt.bar(xs, costs, color=colors, edgecolor="white", linewidth=0.6)
    # highlight best
    plt.bar(xs[best_idx], costs[best_idx], color=palette["line"][-1], edgecolor="black", linewidth=1.2)

    plt.xticks(xs, labels, rotation=30, ha="right")
    plt.ylabel("Scalar multiplications (cost)")
    plt.title("All Parenthesizations — Cost Comparison")
    plt.tight_layout()
    path = os.path.join(outdir, f"F1_all_parenthesizations_{tag}.png")
    plt.savefig(path, dpi=180); plt.close()
    print(f"Saved {path} — (learn: grouping changes cost dramatically)")

def save_heatmap_dp(dp: List[List[int]], split: List[List[int]],
                    palette: Dict[str, List[str]], outdir: str, tag: str) -> None:
    arr = np.array(dp, dtype=float)
    n = arr.shape[0]
    # log-scale to make contrast visible if values vary widely
    arr_log = np.log(arr + 1.0)

    plt.figure(figsize=(6.8, 6))
    # build a custom colormap from ombré
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(palette["heat"])
    im = plt.imshow(arr_log, cmap=cmap, origin="upper")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="log(cost+1)")

    # annotate dp values and split choices
    for i in range(n):
        for j in range(n):
            if j < i:  # lower triangle unused
                continue
            txt = "0" if i == j else f"{dp[i][j]}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8, color="#111111")
            if i < j:
                k = split[i][j]
                plt.text(j, i + 0.3, f"k={k+1}", ha="center", va="center", fontsize=7, color="#444444")

    plt.title("DP Cost Table (numbers) + Split Choices (k indices)")
    plt.xlabel("j (end index)"); plt.ylabel("i (start index)")
    plt.tight_layout()
    path = os.path.join(outdir, f"F2_dp_heatmap_{tag}.png")
    plt.savefig(path, dpi=180); plt.close()
    print(f"Saved {path} — (learn: how DP fills and where each split occurs)")

def save_callcount_plot(naive_calls: int, memo_calls: int,
                        palette: Dict[str, List[str]], outdir: str, tag: str) -> None:
    plt.figure(figsize=(6, 4.2))
    cats = ["Naive recursion", "Memoized (top–down)"]
    vals = [naive_calls, memo_calls]
    col = [palette["scatter"][10], palette["scatter"][-30]]
    plt.bar(cats, vals, color=col, edgecolor="white", linewidth=0.8)
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Recursive calls")
    plt.title("Why Memoization Matters")
    plt.tight_layout()
    path = os.path.join(outdir, f"F3_call_counts_{tag}.png")
    plt.savefig(path, dpi=180); plt.close()
    print(f"Saved {path} — (learn: DP collapses repeated work)")

def time_actual_multiply(dims: List[int], expr: str, trials: int = 3) -> float:
    """
    Build random matrices matching dims and multiply according to the parenthesization string.
    This is for sanity (cost ↔ runtime). For clarity we parse the expression recursively.
    """
    # Build matrices
    mats = [np.random.randn(dims[i], dims[i+1]).astype(np.float64) for i in range(len(dims) - 1)]
    # Parser: returns a function that when called multiplies NumPy arrays in the specified order.
    def parse(expr_str: str):
        expr_str = expr_str.strip()
        if expr_str.startswith("A") and " " not in expr_str:
            idx = int(expr_str[1:]) - 1
            return lambda arrs: arrs[idx]
        assert expr_str[0] == "(" and expr_str[-1] == ")", "Malformed expression"
        # split outer "( left × right )"
        depth, mid = 0, -1
        for i, ch in enumerate(expr_str):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "×" and depth == 1:
                mid = i
                break
        left = expr_str[1:mid].strip()
        right = expr_str[mid+1:-1].strip()
        left_fn = parse(left)
        right_fn = parse(right)
        return lambda arrs: left_fn(arrs) @ right_fn(arrs)

    fn = parse(expr)
    # Warmup + timing
    best = math.inf
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = fn(mats)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best

def save_timing_compare(dims: List[int], best_expr: str, greedy_expr: str,
                        palette: Dict[str, List[str]], outdir: str, tag: str) -> None:
    t_best = time_actual_multiply(dims, best_expr)
    t_greedy = time_actual_multiply(dims, greedy_expr)
    plt.figure(figsize=(6.2, 4.2))
    xs = np.arange(2)
    vals = [t_best, t_greedy]
    cols = [palette["line"][-10], palette["scatter"][80]]
    plt.bar(xs, vals, color=cols, edgecolor="white", linewidth=0.8)
    plt.xticks(xs, ["Optimal (DP)", "Greedy (pitfall)"])
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v*1000:.1f} ms", ha="center", va="bottom")
    plt.ylabel("Runtime (seconds, min of 3)")
    plt.title("Runtime sanity check (random matrices with given dims)")
    plt.tight_layout()
    path = os.path.join(outdir, f"F4_timing_compare_{tag}.png")
    plt.savefig(path, dpi=180); plt.close()
    print(f"Saved {path} — (learn: cost model correlates with actual runtime)")


# ----- Orchestration ----------------------------------------------------------
def main() -> int:
    palette = choose_palette()
    outdir = ensure_output_dir()

    # Example chains to mirror your outline (you can edit or extend):
    examples: Dict[str, List[int]] = {
        # A(20×2), B(2×30), C(30×12), D(12×8)
        "ABCD_prompt": [20, 2, 30, 12, 8],
        # A1(10×4), A2(4×5), A3(5×20), A4(20×2)
        "A1..A4_alt":  [10, 4, 5, 20, 2],
        # Slightly longer (6 matrices) for a richer DP table; still small enough to view
        "six_mats":    [5, 10, 3, 12, 5, 50, 6],
    }

    print("\nAvailable dimension sets (dims = [p0, p1, ..., p_n]):")
    for k, v in examples.items():
        print(f"  - {k}: {v}")
    key = input("Pick one (default: ABCD_prompt): ").strip() or "ABCD_prompt"
    dims = examples.get(key, examples["ABCD_prompt"])
    validate_dims(dims)

    n = len(dims) - 1
    tag = f"{key}_n{n}"
    print(f"\nUsing dims={dims} (n={n} matrices).")

    # A) Brute force (for small n) — enumerate *all* parenthesizations and costs
    bf_cost, bf_expr, bf_calls = brute_force_min_cost(dims)
    # Build labels & costs for figure (only when n is small enough to enumerate)
    labels, costs = [], []
    # Re-enumerate to collect all options for plotting
    def enumerate_all(i: int, j: int) -> List[Tuple[str, int]]:
        if i == j:
            return [(f"A{i+1}", 0)]
        res = []
        for k in range(i, j):
            left = enumerate_all(i, k)
            right = enumerate_all(k + 1, j)
            for eL, cL in left:
                for eR, cR in right:
                    cost = cL + cR + split_cost(dims, i, k, j)
                    res.append((f"({eL} × {eR})", cost))
        return res
    all_expr_cost = enumerate_all(0, n - 1)
    # stable sort by cost then lexicographically to have reproducible bars
    all_expr_cost.sort(key=lambda x: (x[1], x[0]))
    labels = [e for e, _ in all_expr_cost]
    costs = [c for _, c in all_expr_cost]
    best_idx = labels.index(bf_expr)
    save_bar_all_parenthesizations(labels, costs, best_idx, palette, outdir, tag)

    # B) Top–down memoization
    td_cost, td_split = topdown_memo(dims)

    # C) Bottom–up DP
    dp, split = bottomup_dp(dims)
    bu_cost = dp[0][n - 1]
    bu_expr = reconstruct(split, 0, n - 1)
    save_heatmap_dp(dp, split, palette, outdir, tag)

    # D) Greedy pitfall
    gr_cost, gr_expr = greedy_bad(dims)

    # Educational comparison in the console
    print("\n=== Results ===")
    print(f"Brute force (enumerate)   : cost={bf_cost:>8d}, expr={bf_expr}")
    print(f"Top–down memoization      : cost={td_cost:>8d}, expr={reconstruct(td_split,0,n-1)}")
    print(f"Bottom–up dynamic program : cost={bu_cost:>8d}, expr={bu_expr}")
    print(f"Greedy (pitfall)          : cost={gr_cost:>8d}, expr={gr_expr}")

    # F3: Call-count comparison (naive vs memo). Recompute naive without cache to count "true" calls.
    naive_calls_true = count_calls_naive(dims)
    memo_calls = count_calls_memoized(dims)
    save_callcount_plot(naive_calls_true, memo_calls, palette, outdir, tag)

    # F4: Optional timing sanity (random mats; small dims recommended)
    if n <= 6:
        save_timing_compare(dims, bu_expr, gr_expr, palette, outdir, tag)
    else:
        print("Skipping timing sanity for long chains (set n<=6 to enable).")

    print(f"\nAll figures saved to: {outdir}\n")
    return 0


# ----- Instrumentation helpers for call counting ------------------------------
def count_calls_naive(dims: List[int]) -> int:
    n = len(dims) - 1
    calls = 0
    def solve(i: int, j: int) -> int:
        nonlocal calls
        calls += 1
        if i == j:
            return 0
        best = math.inf
        for k in range(i, j):
            best = min(best, solve(i, k) + solve(k + 1, j) + split_cost(dims, i, k, j))
        return best
    _ = solve(0, n - 1)
    return calls

def count_calls_memoized(dims: List[int]) -> int:
    n = len(dims) - 1
    calls = 0
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> int:
        nonlocal calls
        calls += 1
        if i == j:
            return 0
        best = math.inf
        for k in range(i, j):
            best = min(best, solve(i, k) + solve(k + 1, j) + split_cost(dims, i, k, j))
        return best
    _ = solve(0, n - 1)
    return calls


# ----- Entrypoint -------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())
