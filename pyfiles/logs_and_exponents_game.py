# logs_exponents_game.py
# -----------------------------------------------------------------------------
# Title: Logs & Exponents — Interactive Mastery Lab (Game + Tutor)
# Author: Cazzy Aporbo, MS
# Started: December, 2024
# Intent: Teach logarithms and exponentials to high‑schoolers and refresh
#         undergrads/experts through an interactive terminal game that blends
#         rules, intuition, proofs, and real‑world modeling. Multiple ways to
#         solve, careful line‑by‑line narration, and expert sidebars.
# -----------------------------------------------------------------------------

from __future__ import annotations  # allows forward references in type hints

import math      # mathematical functions (log, exp, pow)
import random    # to generate unique practice questions
import sys       # to detect TTY and handle non‑interactive environments
import textwrap  # to wrap long lines in the console
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Console helpers — readable output that looks like a teaching transcript
# -----------------------------------------------------------------------------

def block(title: str) -> None:
    """Print a visual section divider so each idea stands on its own."""
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)


def say(text: str, width: int = 78) -> None:
    """Wrap and print text so longer explanations read comfortably."""
    print("\n".join(textwrap.wrap(text, width=width)))


def safe_input(prompt: str, default: Optional[str] = None) -> Optional[str]:
    """Ask for input, but never crash in non‑interactive sandboxes.
    Returns default if stdin is not a TTY or if input raises EOFError/OSError.
    """
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return default
        return input(prompt)
    except (EOFError, OSError):
        return default

# -----------------------------------------------------------------------------
# Knowledge cards — everything from the brief.
# Each rule prints: concept, meaning, formula, when to use, and a demo.
# -----------------------------------------------------------------------------

@dataclass
class Card:
    title: str
    meaning: str
    formula: str
    when: str
    demo: Callable[[], None]


def card_logarithm_definition() -> Card:
    def demo() -> None:
        # I show the tight relationship: log_b(T)=i  <=>  T=b**i
        b, i = 2, 3
        T = b ** i
        say(f"Example: log base {b} of {T} is {i} because {b}^{i} = {T}.")
        # Two equivalent computations of log base b: math.log(T, b) and change of base
        v1 = math.log(T, b)               # direct base parameter
        v2 = math.log(T) / math.log(b)    # change-of-base formula
        print("math.log(T, b) ->", v1)
        print("log(T)/log(b) ->", v2)
    return Card(
        title="What is a Logarithm?",
        meaning=("A logarithm answers: 'what power of the base gives this number?'"),
        formula="log_b(T) = i  ⇔  T = b^i",
        when="Switching between exponential and logarithmic forms.",
        demo=demo,
    )


def card_exponential_form() -> Card:
    def demo() -> None:
        b, i = 3, 4
        say(f"log_{b}(81) = {i}  ⇔  {b}^{i} = 81. Rewriting is the whole trick.")
        print("pow(3,4) ->", pow(3, 4))
    return Card(
        title="Exponential Form",
        meaning="Exponential equations state numbers as powers of a base.",
        formula="a = b^i  ⇔  log_b(a) = i",
        when="Moving the unknown between exponent and plain value.",
        demo=demo,
    )


def card_product_rule() -> Card:
    def demo() -> None:
        a, b, base = 6, 4, 2
        lhs = math.log(a * b, base)
        rhs = math.log(a, base) + math.log(b, base)
        print("log_2(6*4) ->", lhs, " ; log_2(6)+log_2(4) ->", rhs)
    return Card(
        title="Product Rule",
        meaning="Multiplication inside a log becomes addition of logs.",
        formula="log_b(TU) = log_b(T) + log_b(U)",
        when="Factor a product to separate difficulty into simpler pieces.",
        demo=demo,
    )


def card_quotient_rule() -> Card:
    def demo() -> None:
        a, b, base = 8, 2, 2
        lhs = math.log(a / b, base)
        rhs = math.log(a, base) - math.log(b, base)
        print("log_2(8/2) ->", lhs, " ; log_2(8)-log_2(2) ->", rhs)
    return Card(
        title="Quotient Rule",
        meaning="Division inside a log becomes subtraction of logs.",
        formula="log_b(T/U) = log_b(T) - log_b(U)",
        when="Separate numerator and denominator to simplify.",
        demo=demo,
    )


def card_power_rule() -> Card:
    def demo() -> None:
        base, x = 2, 5
        lhs = math.log(5 ** 2, base)
        rhs = 2 * math.log(5, base)
        print("log_2(5^2) ->", lhs, " ; 2*log_2(5) ->", rhs)
    return Card(
        title="Power Rule",
        meaning="Powers inside the log can be pulled down as multipliers.",
        formula="log_b(T^i) = i * log_b(T)",
        when="Linearize exponents to solve for unknown powers.",
        demo=demo,
    )


def card_change_of_base() -> Card:
    def demo() -> None:
        T, b, k = 25, 5, 10
        lhs = math.log(T, b)
        rhs = math.log(T, k) / math.log(b, k)
        print("log_5(25) ->", lhs, " ; log_10(25)/log_10(5) ->", rhs)
    return Card(
        title="Change of Base",
        meaning="Compute logs in any base using a convenient base k.",
        formula="log_b(T) = log_k(T) / log_k(b)",
        when="Your calculator/library only supports base e or 10.",
        demo=demo,
    )


def card_log_of_one() -> Card:
    def demo() -> None:
        for b in (2, 3, 10, math.e):
            print(f"log_{b}(1) ->", math.log(1, b))
    return Card(
        title="Log of 1",
        meaning="Always zero, because any base to the 0 power is 1.",
        formula="log_b(1) = 0",
        when="Spot log terms that vanish.",
        demo=demo,
    )


def card_log_of_base() -> Card:
    def demo() -> None:
        for b in (2, 3, 10, 7):
            print(f"log_{b}({b}) ->", math.log(b, b))
    return Card(
        title="Log of the Same Base",
        meaning="Always one, because b^1 = b.",
        formula="log_b(b) = 1",
        when="Clean up expressions quickly.",
        demo=demo,
    )


def card_solve_exponential() -> Card:
    def demo() -> None:
        # Solve 2^x = 16. Take log base 2 of both sides to drop x.
        b, a = 2, 16
        x = math.log(a, b)
        say(f"Solve {b}^x={a} ⇒ take log base {b} ⇒ x = log_{b}({a}) = {x}.")
    return Card(
        title="Solving Exponential Equations",
        meaning="When the variable lives in the exponent, logs pull it down.",
        formula="b^x = a  ⇒  x = log_b(a)",
        when="Unknown is trapped upstairs as an exponent.",
        demo=demo,
    )


def card_solve_logarithmic() -> Card:
    def demo() -> None:
        # Solve log_3(T)=5 ⇒ T=3^5
        b, i = 3, 5
        T = b ** i
        say(f"Solve log_{b}(T)={i} ⇒ rewrite exponentially ⇒ T={b}^{i}={T}.")
    return Card(
        title="Solving Logarithmic Equations",
        meaning="Rewrite the log as an exponential to free the variable.",
        formula="log_b(T)=i  ⇒  T=b^i",
        when="Unknown sits inside a log.",
        demo=demo,
    )


def card_graph_exponential() -> Card:
    def demo() -> None:
        # Instead of plotting, we describe growth numerically to keep stdlib only.
        b = 2
        xs = list(range(-3, 4))
        ys = [b ** x for x in xs]
        print("x:", xs)
        print("2^x:", ys)
        say("Notice how doubling per +1 in x creates rapid growth; negatives decay.")
    return Card(
        title="Graphing Exponential Functions",
        meaning="y=b^x grows if b>1 (decays if 0<b<1). Passes through (0,1).",
        formula="y=b^x",
        when="Model growth/decay (interest, populations, half‑life).",
        demo=demo,
    )


def card_graph_logarithm() -> Card:
    def demo() -> None:
        b = 2
        xs = [0.5, 1, 2, 4, 8]
        ys = [math.log(x, b) for x in xs]
        print("x:", xs)
        print("log_2(x):", ys)
        say("Log passes (1,0), grows slowly; doubling x adds 1 to log base 2.")
    return Card(
        title="Graphing Logarithmic Functions",
        meaning="y=log_b(x) grows slowly, defined only for x>0. Passes (1,0).",
        formula="y=log_b(x)",
        when="Compress big ranges; model diminishing returns.",
        demo=demo,
    )

# Expert sidebars — deeper notes for college/grad refreshers
# -----------------------------------------------------------------------------

def sidebar_expert_notes() -> None:
    block("Expert Notes — numerical stability and identities you actually use")
    # log1p & expm1 preserve accuracy near zero by avoiding catastrophic cancellation.
    x = 1e-12
    naive = math.log(1 + x)
    stable = math.log1p(x)
    print("log(1+x) naive ->", naive, "; log1p(x) stable ->", stable)
    # log-sum-exp trick: log(e^a + e^b) = m + log(e^{a-m} + e^{b-m}), m=max(a,b)
    a, b = 1000.0, 1001.0
    m = max(a, b)
    lse = m + math.log(math.exp(a - m) + math.exp(b - m))
    say("Log‑sum‑exp keeps numbers in range when a,b are large — common in ML.")
    print("log‑sum‑exp(1000,1001) ->", lse)

# -----------------------------------------------------------------------------
# Tutor modes — REVIEW (non‑interactive), FLASHCARDS, QUIZ, and MODEL LABS
# -----------------------------------------------------------------------------

CARDS: List[Card] = [
    card_logarithm_definition(),
    card_exponential_form(),
    card_product_rule(),
    card_quotient_rule(),
    card_power_rule(),
    card_change_of_base(),
    card_log_of_one(),
    card_log_of_base(),
    card_solve_exponential(),
    card_solve_logarithmic(),
    card_graph_exponential(),
    card_graph_logarithm(),
]


def mode_review() -> None:
    block("REVIEW — narrated walkthrough of rules with demos")
    for card in CARDS:
        print(f"\n[{card.title}]  —  {card.formula}")
        say(card.meaning)
        say("When to use: " + card.when)
        card.demo()


def _ask(prompt: str) -> Optional[str]:
    return safe_input(prompt + " ", default=None)


def mode_flashcards() -> None:
    block("FLASHCARDS — self‑marking practice")
    ids = list(range(len(CARDS)))
    random.shuffle(ids)
    correct = 0
    for i in ids:
        c = CARDS[i]
        print("\n" + c.title)
        say(c.meaning)
        _ = _ask("Press Enter to reveal formula (or type q to quit)...")
        if _ and _.lower().startswith("q"):
            break
        print("Formula:", c.formula)
        mark = _ask("Mark correct? [y/N]")
        if mark and mark.strip().lower().startswith("y"):
            correct += 1
    print(f"\nYou marked {correct}/{len(ids)} as correct.")


# Question generators for the quiz — multiple approaches accepted
# -----------------------------------------------------------------------------

def q_change_of_base() -> Tuple[str, float]:
    b = random.choice([2, 3, 5, 10])
    i = random.choice([2, 3, 4, 5])
    T = b ** i
    ans = float(i)
    prompt = f"Compute log_{b}({T}). Tip: change of base if needed."
    return prompt, ans


def q_solve_exponent() -> Tuple[str, float]:
    b = random.choice([2, 3, 10])
    i = random.choice([1, 2, 3, 4])
    T = b ** i
    prompt = f"Solve for x: {b}^x = {T}"
    ans = float(i)
    return prompt, ans


def q_product_rule() -> Tuple[str, float]:
    base = random.choice([2, 3, 5])
    a, b = random.choice([2, 4, 8]), random.choice([3, 9])
    prompt = f"Compute log_{base}({a*b}) using the product rule (exact)."
    ans = math.log(a * b, base)
    return prompt, float(ans)


QUESTIONS: List[Callable[[], Tuple[str, float]]] = [
    q_change_of_base,
    q_solve_exponent,
    q_product_rule,
]


def mode_quiz() -> None:
    block("QUIZ — short, exact answers; demonstrates multiple solution paths")
    random.shuffle(QUESTIONS)
    score = 0
    for gen in QUESTIONS:
        prompt, ans = gen()
        say(prompt)
        resp = _ask("Your answer (or press Enter to reveal):")
        if resp is None or resp.strip() == "":
            print("Answer:", ans)
        else:
            try:
                if abs(float(resp) - ans) < 1e-9:
                    print("Correct.")
                    score += 1
                else:
                    print("Close, but not quite. Expected:", ans)
            except ValueError:
                print("Not a number. Expected:", ans)
    print(f"\nScore: {score}/{len(QUESTIONS)}")

# -----------------------------------------------------------------------------
# Modeling labs — compound growth, half‑life decay, and solving for time
# -----------------------------------------------------------------------------

def lab_compound_growth(P: float = 1000.0, r: float = 0.05, n: int = 12, t: float = 3.0) -> None:
    """Compound interest A = P*(1+r/n)^(n*t). Show log method to solve for t."""
    block("LAB: Compound Growth — from formula to log‑solved time")
    A = P * (1 + r / n) ** (n * t)
    say(f"Balance after t years: A = P(1+r/n)^(n t). With P={P}, r={r}, n={n}, t={t} → A={A:.2f}")
    # Solve for time given target A* (invert using logs)
    target = 2 * P  # time to double
    say("Time to double: solve (1+r/n)^(n t) = target/P ⇒ take logs both sides.")
    base = 1 + r / n
    t_double = math.log(target / P, base) / n
    print("t_double ->", t_double, "years")


def lab_half_life(N0: float = 100.0, half_life: float = 5.0, t: float = 12.0) -> None:
    """Radioactive decay: N(t) = N0 * (1/2)^(t/half_life). Invert with logs."""
    block("LAB: Half‑Life Decay — logs linearize exponents for solving time")
    N_t = N0 * (0.5) ** (t / half_life)
    say(f"Amount after t={t}: N(t) = N0*0.5^(t/hl). With N0={N0}, hl={half_life} → N(t)={N_t:.4f}")
    # Solve for time to reach threshold M using logs
    M = N0 / 8
    say("Solve for t when N(t)=M: N0*(1/2)^(t/hl)=M ⇒ (1/2)^(t/hl)=M/N0 ⇒ logs.")
    t_to_M = half_life * math.log(M / N0, 0.5)
    print("t to reach N0/8 ->", t_to_M)

# -----------------------------------------------------------------------------
# Menu — includes non‑interactive fallback to REVIEW for CI/sandboxes
# -----------------------------------------------------------------------------

def main() -> int:
    block("Logs & Exponents — Interactive Mastery Lab")
    say("Pick a mode. If input is blocked, we'll auto‑run REVIEW.")
    print("  1) REVIEW (guided cards)")
    print("  2) FLASHCARDS (self‑mark)")
    print("  3) QUIZ (exact answers)")
    print("  4) LAB: Compound Growth")
    print("  5) LAB: Half‑Life Decay")
    print("  6) Expert Notes")
    print("  7) Quit")

    choice = safe_input("> ", default=None)
    if choice is None:
        mode_review()
        return 0

    choice = choice.strip()
    if choice == "1":
        mode_review()
    elif choice == "2":
        mode_flashcards()
    elif choice == "3":
        mode_quiz()
    elif choice == "4":
        lab_compound_growth()
    elif choice == "5":
        lab_half_life()
    elif choice == "6":
        sidebar_expert_notes()
    else:
        say("Good study. Re‑run to try other modes.")
    return 0

# -----------------------------------------------------------------------------
# Self‑checks — small asserts to guard math identities
# -----------------------------------------------------------------------------

def _tests() -> None:
    # Change of base correctness
    for T, b in [(8, 2), (81, 3), (100, 10), (5, math.e)]:
        assert abs(math.log(T, b) - math.log(T) / math.log(b)) < 1e-12
    # Product/quotient/power rules numerically
    T, U, b = 6.0, 4.0, 2.0
    assert abs(math.log(T * U, b) - (math.log(T, b) + math.log(U, b))) < 1e-12
    assert abs(math.log(T / U, b) - (math.log(T, b) - math.log(U, b))) < 1e-12
    assert abs(math.log(T ** 3, b) - 3 * math.log(T, b)) < 1e-12
    # Domain checks (manual): log defined only for T>0, base>0 and base!=1 — not run here
    # Stability: log1p vs naive near 0
    x = 1e-12
    assert abs(math.log1p(x) - math.log(1 + x)) < 1e-15


if __name__ == "__main__":
    _tests()
    raise SystemExit(main())
