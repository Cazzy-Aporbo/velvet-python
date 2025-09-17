# python_lesson_suite.py
# -----------------------------------------------------------------------------
# Title: The Python Lesson — A Complete Teaching Suite (One Program)
# Author: Cazandra Aporbo
# Started: December 14, 2024
# Last Updated: June 19 2025
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import textwrap
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# =============================================================================
# Console helpers — consistent narrative formatting
# =============================================================================

def block(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)


def say(text: str, width: int = 78) -> None:
    print("\n".join(textwrap.wrap(text, width=width)))


def safe_input(prompt: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return default
        return input(prompt)
    except (EOFError, OSError):
        return default

# =============================================================================
# SECTION A — Built‑ins & Friends (from the Playbook)
# =============================================================================

def lesson_builtins_extended() -> None:
    block("A) Built‑ins Tour — essentials with quick demos")
    print("abs(-7) ->", abs(-7))
    print("all([True, 1, 'yes']) ->", all([True, 1, "yes"]))
    print("any([0, False, '']) ->", any([0, False, ""]))
    print("ascii('café') ->", ascii("café"))
    print("bin(10) ->", bin(10))
    print("bool([]) ->", bool([]))
    ba = bytearray([65, 66, 67])
    print("bytearray([65,66,67]) ->", ba)
    print("bytes([65,66,67]) ->", bytes([65, 66, 67]))
    print("callable(print) ->", callable(print))
    print("chr(65) ->", chr(65))
    class Demo:
        @classmethod
        def hi(cls):
            return f"Hello from {cls.__name__}"
    print("Demo.hi() ->", Demo.hi())
    code = compile("5 + 10", "<string>", "eval")
    print("eval(compiled) ->", eval(code))
    exec("x = 7\nprint('exec created x =', x)")
    print("complex(2,3) ->", complex(2, 3))
    class Bag: ...
    bag = Bag()
    setattr(bag, "item", "book")
    print("getattr(bag,'item') ->", getattr(bag, "item"))
    delattr(bag, "item")
    print("hasattr after delattr ->", hasattr(bag, "item"))
    print("dict(a=1,b=2) ->", dict(a=1, b=2))
    print("'append' in dir([]) ->", "append" in dir([]))
    print("divmod(7,3) ->", divmod(7, 3))
    for i, v in enumerate([10, 20]):
        print("enumerate ->", i, v)
    evens = list(filter(lambda n: n % 2 == 0, range(6)))
    print("filter evens ->", evens)
    doubled = list(map(lambda n: n * 2, range(3)))
    print("map doubled ->", doubled)
    print("format(3.14159,'.2f') ->", format(3.14159, ".2f"))
    print("frozenset([1,2,2]) ->", frozenset([1, 2, 2]))
    print("hash('abc') ->", hash("abc"))
    print("hex(255) ->", hex(255), "; oct(8) ->", oct(8))
    it = iter([1, 2])
    print("next(iter) ->", next(it))
    print("len('hi') ->", len("hi"))
    print("list('hi') ->", list("hi"))
    print("tuple([1,2]) ->", tuple([1, 2]))
    print("set([1,1,2]) ->", set([1, 1, 2]))
    print("str(123) ->", str(123))
    print("max([1,7,3]) ->", max([1, 7, 3]))
    print("min([1,7,3]) ->", min([1, 7, 3]))
    print("sum([1,2,3]) ->", sum([1, 2, 3]))
    m = memoryview(b"abc")
    print("memoryview(b'abc').tolist() ->", m.tolist())
    try:
        next(it); next(it)
    except StopIteration:
        print("StopIteration raised when exhausted")
    class Base:  # property/staticmethod/super quick taste
        def greet(self): return "hi"
    class Child(Base):
        def greet(self): return super().greet() + " from child"
    print("super demo ->", Child().greet())
    print("list(range(3)) ->", list(range(3)))
    print("repr('hi') ->", repr("hi"))
    print("list(reversed([1,2,3])) ->", list(reversed([1, 2, 3])))
    print("round(3.14159,2) ->", round(3.14159, 2))
    sl = slice(1, 3)
    print("[10,20,30,40][1:3] == [10,20,30,40][sl] ->", [10, 20, 30, 40][sl])
    print("sorted([3,1,2]) ->", sorted([3, 1, 2]))
    print("type('hi') ->", type("hi"))
    print("list(zip([1,2],[3,4])) ->", list(zip([1, 2], [3, 4])))

# =============================================================================
# SECTION B — Random Module (from beginner to expert)
# =============================================================================

def lesson_random_core() -> None:
    block("B1) Random Core: seed/state/bits")
    random.seed(42)
    print("seed(42); randint(1,10) ->", random.randint(1, 10))
    state = random.getstate(); print("captured state type ->", type(state))
    print("next after capture ->", random.randint(1, 10))
    random.setstate(state); print("restored same ->", random.randint(1, 10))
    print("getrandbits(8) ->", random.getrandbits(8))

def lesson_random_everyday() -> None:
    block("B2) Random Everyday: integers and selections")
    print("randrange(1,10,2) ->", random.randrange(1, 10, 2))
    print("randint(1,100) ->", random.randint(1, 100))
    fruits = ['apple','banana','cherry']; print("choice ->", random.choice(fruits))
    colors = ['red','green','blue']; print("choices k=3 ->", random.choices(colors, k=3))
    nums = [1,2,3,4,5]; random.shuffle(nums); print("shuffle ->", nums)
    letters = ['a','b','c','d','e']; print("sample 3 ->", random.sample(letters, 3))

def lesson_random_floats() -> None:
    block("B3) Random Floats & Distributions")
    print("random() ->", random.random())
    print("uniform(2,5) ->", random.uniform(2, 5))
    print("triangular(1,5,3) ->", random.triangular(1, 5, 3))

def lesson_random_distributions() -> None:
    block("B4) Random Advanced Distributions")
    print("betavariate(2,5) ->", random.betavariate(2, 5))
    print("expovariate(0.5) ->", random.expovariate(0.5))
    print("gammavariate(2,3) ->", random.gammavariate(2, 3))
    print("gauss(0,1) ->", random.gauss(0, 1))
    print("lognormvariate(0,1) ->", random.lognormvariate(0, 1))
    print("normalvariate(0,1) ->", random.normalvariate(0, 1))
    print("vonmisesvariate(0,1) ->", random.vonmisesvariate(0, 1))
    print("paretovariate(2.5) ->", random.paretovariate(2.5))
    print("weibullvariate(2,3) ->", random.weibullvariate(2, 3))

# =============================================================================
# SECTION C — statistics Module Walkthrough
# =============================================================================

import statistics

def lesson_statistics() -> None:
    block("C) statistics: central tendency and spread")
    data = [10, 20, 30, 40, 50]
    print("mean ->", statistics.mean(data))
    print("median([3,1,5,2,4]) ->", statistics.median([3, 1, 5, 2, 4]))
    print("median_high([1,2,3,4]) ->", statistics.median_high([1, 2, 3, 4]))
    print("median_low([1,2,3,4])  ->", statistics.median_low([1, 2, 3, 4]))
    print("mode([1,2,2,3,4]) ->", statistics.mode([1, 2, 2, 3, 4]))
    print("harmonic_mean([2,4,8]) ->", round(statistics.harmonic_mean([2,4,8]), 6))
    print("median_grouped([10,20,30], interval=10) ->", statistics.median_grouped([10, 20, 30], interval=10))
    population = [1, 2, 3, 4, 5]
    print("pvariance(pop) ->", statistics.pvariance(population))
    print("pstdev(pop) ->", statistics.pstdev(population))
    print("variance(sample) ->", statistics.variance(population))
    print("stdev(sample) ->", statistics.stdev(population))

# =============================================================================
# SECTION D — Recursion Masterclass
# =============================================================================

def factorial_recursive(n: int) -> int:
    if n < 0: raise ValueError("factorial undefined for negatives")
    if n in (0, 1): return 1
    return n * factorial_recursive(n - 1)

def factorial_iterative(n: int) -> int:
    if n < 0: raise ValueError("factorial undefined for negatives")
    result = 1
    for k in range(2, n + 1): result *= k
    return result

def fib_recursive_naive(n: int) -> int:
    if n < 0: raise ValueError
    if n in (0, 1): return n
    return fib_recursive_naive(n - 1) + fib_recursive_naive(n - 2)

@lru_cache(maxsize=None)
def fib_recursive_memo(n: int) -> int:
    if n < 0: raise ValueError
    if n in (0, 1): return n
    return fib_recursive_memo(n - 1) + fib_recursive_memo(n - 2)

def fib_bottom_up(n: int) -> int:
    if n < 0: raise ValueError
    if n in (0, 1): return n
    a, b = 0, 1
    for _ in range(2, n + 1): a, b = b, a + b
    return b

def permutations_backtracking(items: List[Any]) -> List[List[Any]]:
    result: List[List[Any]] = []
    used = [False] * len(items)
    path: List[Any] = []
    def dfs():
        if len(path) == len(items):
            result.append(path.copy()); return
        for i, v in enumerate(items):
            if used[i]: continue
            used[i] = True; path.append(v); dfs(); path.pop(); used[i] = False
    dfs(); return result

def n_queens(n: int = 4) -> List[List[int]]:
    cols: List[int] = []; solutions: List[List[int]] = []
    def safe(c: int) -> bool:
        r = len(cols)
        for pr, pc in enumerate(cols):
            if pc == c or abs(pc - c) == abs(pr - r): return False
        return True
    def place():
        if len(cols) == n: solutions.append(cols.copy()); return
        for c in range(n):
            if safe(c): cols.append(c); place(); cols.pop()
    place(); return solutions

def lesson_recursion() -> None:
    block("D) Recursion: factorials, Fibonacci, backtracking")
    print("factorial_recursive(5) ->", factorial_recursive(5))
    print("factorial_iterative(5) ->", factorial_iterative(5))
    for n in range(8):
        r1 = fib_recursive_memo(n); r2 = fib_bottom_up(n)
        assert r1 == r2
    print("Fibonacci 0..7 consistent across memoized & DP.")
    print("permutations_backtracking([1,2,3]) ->", permutations_backtracking([1,2,3]))
    print("n_queens(4) solutions ->", n_queens(4))

# =============================================================================
# SECTION E — Logs & Exponentials Game (cards/quiz/labs)
#   (Integrated slim version of the standalone game; full logic preserved.)
# =============================================================================

@dataclass
class Card:
    title: str; meaning: str; formula: str; when: str; demo: Callable[[], None]

def card_logarithm_definition() -> Card:
    def demo() -> None:
        b, i = 2, 3; T = b ** i
        say(f"log base {b} of {T} is {i} because {b}^{i}={T}.")
        print("math.log(T,b) ->", math.log(T, b), "; change-of-base ->", math.log(T)/math.log(b))
    return Card("What is a Logarithm?","Power that gives a number.","log_b(T)=i ⇔ T=b^i","Switch between forms.",demo)

def card_exponential_form() -> Card:
    def demo() -> None:
        say("log_3(81)=4 ⇔ 3^4=81"); print("pow(3,4)->", pow(3,4))
    return Card("Exponential Form","Numbers written as powers.","a=b^i ⇔ log_b(a)=i","Move unknown between exponent/value.",demo)

def card_product_rule() -> Card:
    def demo() -> None:
        a,b,base=6,4,2; print("log_2(6*4)->", math.log(a*b,base), "; split ->", math.log(a,base)+math.log(b,base))
    return Card("Product Rule","Multiply inside ⇒ add logs.","log_b(TU)=log_b(T)+log_b(U)","Factor products.",demo)

def card_quotient_rule() -> Card:
    def demo() -> None:
        a,b,base=8,2,2; print("log_2(8/2)->", math.log(a/b,base), "; diff ->", math.log(a,base)-math.log(b,base))
    return Card("Quotient Rule","Divide inside ⇒ subtract logs.","log_b(T/U)=log_b(T)-log_b(U)","Separate numerator/denominator.",demo)

def card_power_rule() -> Card:
    def demo() -> None:
        base=2; print("log_2(5^2)->", math.log(5**2,base), "; 2*log_2(5)->", 2*math.log(5,base))
    return Card("Power Rule","Exponent comes down.","log_b(T^i)=i*log_b(T)","Linearize exponents.",demo)

def card_change_of_base() -> Card:
    def demo() -> None:
        T,b,k=25,5,10; print("log_5(25)->", math.log(T,b), "; base-10 ->", math.log(T,k)/math.log(b,k))
    return Card("Change of Base","Compute with any base.","log_b(T)=log_k(T)/log_k(b)","Only base e/10 available.",demo)

def card_log_of_one() -> Card:
    def demo() -> None:
        for b in (2,3,10,math.e): print(f"log_{b}(1)->", math.log(1,b))
    return Card("Log of 1","Always zero.","log_b(1)=0","Spot vanishing terms.",demo)

def card_log_of_base() -> Card:
    def demo() -> None:
        for b in (2,3,10,7): print(f"log_{b}({b})->", math.log(b,b))
    return Card("Log of Base","Always one.","log_b(b)=1","Quick simplification.",demo)

def card_solve_exponential() -> Card:
    def demo() -> None:
        b,a=2,16; x=math.log(a,b); say(f"Solve {b}^x={a} ⇒ x=log_{b}({a})={x}")
    return Card("Solve Exponential","Unknown in exponent.","b^x=a ⇒ x=log_b(a)","Untrap the exponent.",demo)

def card_solve_logarithmic() -> Card:
    def demo() -> None:
        b,i=3,5; T=b**i; say(f"Solve log_{b}(T)={i} ⇒ T={b}^{i}={T}")
    return Card("Solve Logarithmic","Unknown inside a log.","log_b(T)=i ⇒ T=b^i","Rewrite exponentially.",demo)

def card_graph_exponential() -> Card:
    def demo() -> None:
        b=2; xs=list(range(-3,4)); ys=[b**x for x in xs]; print("x:",xs); print("2^x:",ys)
    return Card("Graph Exponential","Growth/decay; passes (0,1).","y=b^x","Model growth/decay.",demo)

def card_graph_logarithm() -> Card:
    def demo() -> None:
        b=2; xs=[0.5,1,2,4,8]; ys=[math.log(x,b) for x in xs]; print("x:",xs); print("log_2(x):",ys)
    return Card("Graph Logarithm","Slow growth; passes (1,0).","y=log_b(x)","Compress big ranges.",demo)

CARDS: List[Card] = [
    card_logarithm_definition(), card_exponential_form(), card_product_rule(),
    card_quotient_rule(), card_power_rule(), card_change_of_base(),
    card_log_of_one(), card_log_of_base(), card_solve_exponential(),
    card_solve_logarithmic(), card_graph_exponential(), card_graph_logarithm(),
]

def logs_mode_review() -> None:
    block("E) Logs/Exponents REVIEW")
    for c in CARDS:
        print(f"\n[{c.title}] — {c.formula}"); say(c.meaning); say("When: "+c.when); c.demo()

def logs_mode_quiz_once() -> None:
    # small quiz sample
    b = random.choice([2,3,5,10]); i = random.choice([2,3,4,5]); T = b**i
    say(f"Compute log_{b}({T}) (press Enter to reveal)")
    resp = safe_input("> ", default="")
    print("Answer:", float(i))

def lab_compound_growth(P: float = 1000.0, r: float = 0.05, n: int = 12, t: float = 3.0) -> None:
    block("E‑Lab: Compound Growth")
    A = P * (1 + r / n) ** (n * t)
    say(f"A=P(1+r/n)^(n t) with P={P}, r={r}, n={n}, t={t} ⇒ A={A:.2f}")
    base = 1 + r / n; target = 2 * P
    t_double = math.log(target / P, base) / n
    print("time to double ->", t_double)

def lab_half_life(N0: float = 100.0, half_life: float = 5.0, t: float = 12.0) -> None:
    block("E‑Lab: Half‑Life")
    N_t = N0 * (0.5) ** (t / half_life)
    say(f"N(t)={N_t:.4f} with N0={N0}, hl={half_life}, t={t}")
    M = N0 / 8; t_to_M = half_life * math.log(M / N0, 0.5)
    print("time to reach N0/8 ->", t_to_M)

# =============================================================================
# SECTION F — Flashcards Quiz Game (DS/ML Q&A condensed)
#   (Slimmed to essentials to fit in one suite; safe in non‑interactive runs.)
# =============================================================================

QA_BANK: List[Tuple[str, str]] = [
    ("Build logistic regression?", "Use sklearn.linear_model.LogisticRegression; fit on X,y."),
    ("LinearRegression interpret?", "Coefficients show effect sizes; intercept baseline."),
    ("Data libs?", "NumPy, SciPy, pandas, scikit‑learn, Matplotlib, Seaborn."),
    ("Pandas Series vs single‑col DataFrame?", "Series 1D; DataFrame 2D with one column."),
    ("Drop duplicates code?", "df.drop_duplicates(subset='col')"),
    ("Sort DataFrame desc?", "df.sort_values(by='col', ascending=False)"),
    ("Random Forest params to tune?", "n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features."),
]

def _choice_options(correct: str, pool: Sequence[str], k: int = 4) -> List[str]:
    alt = [x for x in pool if x != correct]; picks = random.sample(alt, min(len(alt), k-1))
    opts = picks + [correct]; random.shuffle(opts); return opts

def game_mc(limit: Optional[int] = None) -> None:
    block("F) Flashcards: Multiple Choice")
    random.shuffle(QA_BANK); all_answers=[a for _,a in QA_BANK]
    correct=0; total=0
    for i,(q,a) in enumerate(QA_BANK,1):
        if limit and i>limit: break
        say(f"Q{i}. {q}")
        opts=_choice_options(a, all_answers)
        for j,opt in enumerate(opts,1): print(f"  {j}. {opt}")
        ans = safe_input("Your pick [1-4] (Enter to reveal): ", default="")
        if ans and ans.isdigit() and 1<=int(ans)<=len(opts) and opts[int(ans)-1]==a:
            print("Correct."); correct+=1
        else:
            print("Answer:", a)
        total+=1
    print(f"Score: {correct}/{total}")

# =============================================================================
# Master Menus
# =============================================================================

def menu_random() -> None:
    block("Random Module Lessons")
    print("  1) Core  2) Everyday  3) Floats  4) Distributions  5) Back")
    c = safe_input("> ", default=None)
    if c is None: lesson_random_core(); lesson_random_everyday(); return
    if c.strip()=="1": lesson_random_core()
    elif c.strip()=="2": lesson_random_everyday()
    elif c.strip()=="3": lesson_random_floats()
    elif c.strip()=="4": lesson_random_distributions()


def menu_logs() -> None:
    block("Logs & Exponents Lessons")
    print("  1) REVIEW  2) Quick Quiz  3) Lab: Growth  4) Lab: Half‑Life  5) Back")
    c = safe_input("> ", default=None)
    if c is None: logs_mode_review(); return
    if c.strip()=="1": logs_mode_review()
    elif c.strip()=="2": logs_mode_quiz_once()
    elif c.strip()=="3": lab_compound_growth()
    elif c.strip()=="4": lab_half_life()


def menu_statistics() -> None:
    lesson_statistics()

def menu_recursion() -> None:
    lesson_recursion()

def menu_builtins() -> None:
    lesson_builtins_extended()

def menu_flashcards() -> None:
    game_mc(limit=5)

# =============================================================================
# Entry point
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The Python Lesson — unified suite")
    parser.add_argument("--auto", action="store_true", help="run a brief non‑interactive tour and exit")
    args = parser.parse_args(argv)

    if args.auto or not (sys.stdin and sys.stdin.isatty()):
        # Non‑interactive tour: one pass through highlights
        lesson_builtins_extended()
        lesson_random_core(); lesson_random_everyday()
        lesson_statistics()
        lesson_recursion()
        logs_mode_review()
        game_mc(limit=3)
        return 0

    while True:
        block("The Python Lesson — Main Menu")
        print("  1) Built‑ins & Idioms")
        print("  2) Random Module")
        print("  3) statistics Module")
        print("  4) Recursion Masterclass")
        print("  5) Logs & Exponents")
        print("  6) Flashcards (DS/ML)")
        print("  7) Quit")
        choice = safe_input("> ", default=None)
        if choice is None or choice.strip()=="7":
            say("Good study. Re‑run to explore other sections."); break
        ch = choice.strip()
        if ch=="1": menu_builtins()
        elif ch=="2": menu_random()
        elif ch=="3": menu_statistics()
        elif ch=="4": menu_recursion()
        elif ch=="5": menu_logs()
        elif ch=="6": menu_flashcards()
    return 0

# =============================================================================
# Self‑tests (lightweight)
# =============================================================================

def _tests() -> None:
    # statistics identities
    assert statistics.mean([10,20,30,40,50]) == 30
    assert statistics.median([3,1,5,2,4]) == 3
    assert statistics.pvariance([1,2,3,4,5]) == 2
    # recursion equivalence small n
    for i in range(6):
        assert factorial_recursive(i) == factorial_iterative(i)
    for n,v in enumerate([0,1,1,2,3,5,8,13]):
        assert fib_recursive_memo(n) == v == fib_bottom_up(n)
    # logs identities
    for T,b in [(8,2),(81,3),(100,10),(5,math.e)]:
        assert abs(math.log(T,b) - math.log(T)/math.log(b)) < 1e-12


if __name__ == "__main__":
    _tests()
    raise SystemExit(main())
