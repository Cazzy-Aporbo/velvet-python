#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Duel of Sages — a two-player Python terminal duel.

- Show breadth: dataclasses, pattern matching, decorators, closures,
  context managers, caching, enums, generators, ANSI color, pure functions.
- 2-player, turn-based: active player answers under a drawn "obstacle".
  Miss it? The opponent may steal.
- Endless by default: play to a target (e.g., 7) or just /quit anytime.

How to play
-----------
1) Run:  python Duel_of_Sages.py
2) Enter two player names.
3) Each round draws a challenge + an obstacle (e.g., "Answer must be prime").
4) Commands: /hint  /pass  /score  /target N  /quit
"""

from __future__ import annotations

import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from typing import Any, Callable, Iterable, Optional, Tuple

# ---------- tiny UI layer ----------

C = dict(
    reset="\033[0m",
    cyan="\033[96m",
    mag="\033[95m",
    green="\033[92m",
    yellow="\033[93m",
    gray="\033[90m",
    bold="\033[1m",
)

def line(msg: str = "", color: str = "gray") -> None:
    print(C[color] + msg + C["reset"])

def banner(title: str) -> None:
    bar = "═" * 72
    print(C["mag"] + bar + C["reset"])
    print(C["cyan"] + title + C["reset"])
    print(C["mag"] + bar + C["reset"])

def prompt(s: str) -> str:
    try:
        return input(C["yellow"] + s + C["reset"])
    except (EOFError, KeyboardInterrupt):
        return "/quit"

# ---------- timing context manager ----------

class tick:
    """with tick("label"): ...  # prints elapsed ms on exit"""
    def __init__(self, label: str): self.label = label; self.t0 = 0.0
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *_): dt = (time.perf_counter() - self.t0) * 1000; line(f"[{self.label}] {dt:.1f} ms", "gray")

# ---------- core datamodel ----------

class Topic(Enum):
    ARITH = auto()
    WORDS = auto()
    SEQ   = auto()
    LOGIC = auto()
    MIXED = auto()

@dataclass
class Player:
    name: str
    score: int = 0
    passes: int = 1
    hints: int = 2

@dataclass
class Challenge:
    prompt: str
    expected: Any                      # ground truth; may be int|str|bool
    explain: str                       # one-liner explanation/hint
    topic: Topic
    validator: Callable[[str, Any], Tuple[bool, str]]  # (answer_str, expected)-> (ok,msg)

@dataclass
class Obstacle:
    name: str
    describe: str
    wrap: Callable[[Callable[[str, Any], Tuple[bool,str]]], Callable[[str, Any], Tuple[bool,str]]]

@dataclass
class Game:
    p1: Player
    p2: Player
    target: Optional[int] = None
    rng: random.Random = field(default_factory=random.Random)
    steal_enabled: bool = True

    @property
    def players(self) -> Tuple[Player, Player]: return (self.p1, self.p2)

# ---------- utilities ----------

@lru_cache(maxsize=10_000)
def is_prime(n: int) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    r = int(math.sqrt(n))
    for k in range(3, r+1, 2):
        if n % k == 0: return False
    return True

def typed_equal(ans_str: str, expected: Any) -> Tuple[bool, str]:
    """
    Lenient compare: numeric strings vs numbers, casefolded strings, booleans.
    Returns (ok, normalized_display_message).
    """
    num = re.fullmatch(r"-?\d+(?:\.\d+)?", ans_str.strip())
    if isinstance(expected, (int, float)) and num:
        return (float(ans_str) == float(expected), "numeric compare")
    if isinstance(expected, bool):
        return (ans_str.strip().lower() in ("true","t","1") and expected is True) or \
               (ans_str.strip().lower() in ("false","f","0") and expected is False), "bool compare"
    if isinstance(expected, str):
        return (ans_str.strip().casefold() == expected.strip().casefold(), "string compare")
    return (ans_str == str(expected), "stringified compare")

def base_validator(ans: str, expected: Any) -> Tuple[bool, str]:
    ok, mode = typed_equal(ans, expected)
    return (ok, f"via {mode}")

def announce(fn: Callable) -> Callable:
    """Decorator for round narration."""
    @wraps(fn)
    def inner(*a, **kw):
        line(f"→ {fn.__name__.replace('_',' ').title()}", "gray")
        return fn(*a, **kw)
    return inner

# ---------- obstacles (composable constraints) ----------

def obstacle(name: str, describe: str) -> Callable[[Callable], Obstacle]:
    def binder(wrap_fn: Callable[[Callable], Callable]) -> Obstacle:
        return Obstacle(name, describe, wrap_fn)
    return binder

@obstacle("Prime Answer", "Your final answer must be a prime number.")
def OB_PRIME(validate):
    def v(ans: str, expected: Any):
        ok, msg = validate(ans, expected)
        if not ok: return (False, msg)
        try:
            return (is_prime(int(float(ans))), "and it is prime")
        except ValueError:
            return (False, "not an integer → not prime")
    return v

@obstacle("No Vowels", "Your answer (string) must contain no vowels.")
def OB_NOVOWELS(validate):
    def v(ans: str, expected: Any):
        ok, msg = validate(ans, expected)
        if not ok: return (False, msg)
        if not isinstance(expected, str): return (False, "obstacle needs a string answer")
        return (not re.search(r"[aeiou]", ans, re.I), "no vowels check")
    return v

@obstacle("Palindrome", "Your string/numeric answer must read the same forwards and backwards.")
def OB_PAL(validate):
    def v(ans: str, expected: Any):
        ok, msg = validate(ans, expected)
        if not ok: return (False, msg)
        s = re.sub(r"\W","", ans).lower()
        return (s == s[::-1], "palindrome check")
    return v

@obstacle("Even Digits Sum", "Sum of digits of your (numeric) answer must be even.")
def OB_EVEN_SUM(validate):
    def v(ans: str, expected: Any):
        ok, msg = validate(ans, expected)
        if not ok: return (False, msg)
        digits = re.findall(r"\d", ans)
        if not digits: return (False, "no digits found")
        tot = sum(int(d) for d in digits)
        return (tot % 2 == 0, f"digits sum={tot} even?")
    return v

OBSTACLES = [OB_PRIME, OB_NOVOWELS, OB_PAL, OB_EVEN_SUM]

def wrap_with_obstacles(validate: Callable, obs: Iterable[Obstacle]) -> Callable:
    for o in obs: validate = o.wrap(validate)
    return validate

# ---------- challenge factories ----------

def ch_arith(rng: random.Random) -> Challenge:
    a, b = rng.randint(10, 99), rng.randint(2, 12)
    op = rng.choice(["+","-","*","//","%"])
    expr = f"{a} {op} {b}"
    expected = eval(expr)
    return Challenge(
        prompt=f"Compute: {expr}",
        expected=expected,
        explain="Remember: // is floor division; % is remainder.",
        topic=Topic.ARITH,
        validator=base_validator,
    )

def ch_words(rng: random.Random) -> Challenge:
    word = rng.choice(["Observation","Knowledge","Cipher","Palindrome","Pythonista"])
    mode = rng.choice(["reverse","vowels","upper"])
    if mode == "reverse":
        return Challenge(f"Write the word '{word}' reversed.", word[::-1],
                         "Slice with [::-1] to reverse.", Topic.WORDS, base_validator)
    if mode == "vowels":
        v = len(re.findall(r"[aeiou]", word, re.I))
        return Challenge(f"How many vowels are in '{word}'?", v,
                         "Count with regex or loop.", Topic.WORDS, base_validator)
    return Challenge(f"Write '{word}' in uppercase.", word.upper(),
                     ".upper() transforms to uppercase.", Topic.WORDS, base_validator)

def ch_seq(rng: random.Random) -> Challenge:
    kind = rng.choice(["powers2","fibo","arithmetic"])
    if kind == "powers2":
        n = rng.randint(3,5)
        seq = [2**k for k in range(1, n+1)]
        return Challenge(f"Next number in {seq} is …", 2**(n+1),
                         "Powers of two.", Topic.SEQ, base_validator)
    if kind == "arithmetic":
        start = rng.randint(1,9); step = rng.randint(2,6)
        seq = [start + i*step for i in range(4)]
        return Challenge(f"Next in arithmetic seq {seq} is …", start + 4*step,
                         "Constant difference.", Topic.SEQ, base_validator)
    # fibo
    a,b = 1,1
    seq = [a,b]
    for _ in range(3): a,b = b,a+b; seq.append(b)
    return Challenge(f"Next in Fibonacci {seq} is …", a+b,
                     "Next term is sum of previous two.", Topic.SEQ, base_validator)

def ch_logic(rng: random.Random) -> Challenge:
    # simple truthiness or divisibility logic
    n = rng.randint(20, 120)
    prop = rng.choice(["even","multiple3","square"])
    if prop == "even":
        return Challenge(f"Is {n} even? (True/False)", n%2==0,
                         "n % 2 == 0", Topic.LOGIC, base_validator)
    if prop == "multiple3":
        return Challenge(f"Is {n} divisible by 3? (True/False)", n%3==0,
                         "n % 3 == 0", Topic.LOGIC, base_validator)
    r = int(math.sqrt(n))
    return Challenge(f"Is {n} a perfect square? (True/False)", r*r==n,
                     "Compare floor(sqrt(n))^2 to n.", Topic.LOGIC, base_validator)

def ch_mixed(rng: random.Random) -> Challenge:
    # tiny set/dict reasoning
    nums = [rng.randint(1,9) for _ in range(6)]
    uniq = sorted(set(nums))
    return Challenge(
        prompt=f"Given nums={nums}, how many unique values?",
        expected=len(uniq),
        explain="set() drops duplicates; len() counts.",
        topic=Topic.MIXED,
        validator=base_validator,
    )

FACTORIES: Tuple[Callable[[random.Random], Challenge], ...] = (
    ch_arith, ch_words, ch_seq, ch_logic, ch_mixed
)

def draw_challenge(rng: random.Random) -> Challenge:
    with tick("draw"):
        return rng.choice(FACTORIES)(rng)

def draw_obstacle(rng: random.Random) -> Obstacle:
    return rng.choice(OBSTACLES)

# ---------- engine ----------

def scoreline(g: Game) -> str:
    return f"{g.p1.name} {g.p1.score} — {g.p2.score} {g.p2.name}"

def maybe_end(g: Game) -> bool:
    if g.target is None: return False
    lead = abs(g.p1.score - g.p2.score)
    if g.p1.score >= g.target or g.p2.score >= g.target:
        if lead >= 2:  # win by 2 like table tennis
            winner = g.p1 if g.p1.score > g.p2.score else g.p2
            banner(f"🏆 {winner.name} wins!  Final: {scoreline(g)}")
            return True
    return False

def apply_obstacle(validate: Callable, obstacle: Obstacle) -> Callable:
    return obstacle.wrap(validate)

def parse_command(s: str) -> Tuple[str, Optional[str]]:
    s = s.strip()
    if not s.startswith("/"): return ("", None)
    parts = s.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) == 2 else None
    return (cmd, arg)

@announce
def turn(g: Game, who: Player, opp: Player) -> bool:
    """
    Returns False to end the match, True to continue.
    Flow:
      - Draw challenge + obstacle
      - Active player answers; if wrong and stealing allowed, opponent may steal
      - Commands: /hint /pass /score /target N /quit
    """
    ch = draw_challenge(g.rng)
    ob = draw_obstacle(g.rng)

    # Compose constraints
    validate = wrap_with_obstacles(ch.validator, [ob])

    banner(f"🔮 {who.name}'s turn ─ Topic: {ch.topic.name} ─ Obstacle: {ob.name}")
    line(ob.describe, "gray")
    line(ch.prompt)

    # Read input
    ans = prompt("> ")

    cmd, arg = parse_command(ans)
    match cmd:
        case "/quit":
            banner("Goodbye, sages.")
            line(f"Final: {scoreline(g)}")
            return False
        case "/score":
            line(scoreline(g)); return True
        case "/hint":
            if who.hints <= 0: line("No hints left."); return True
            who.hints -= 1; line(f"HINT: {ch.explain}  ({who.hints} left)")
            return True
        case "/pass":
            if who.passes <= 0: line("No passes left."); return True
            who.passes -= 1; line(f"{who.name} passes. ({who.passes} left)")
            return True
        case "/target":
            try:
                t = int(arg) if arg else None
                if t and t < 3: raise ValueError
                g.target = t
                line(f"Target set to {t} (win by 2).") if t else line("Endless play.")
            except ValueError:
                line("Usage: /target N   (N >= 3)")
            return True
        case _ if cmd:  # unknown slash command
            line("Unknown command. Try /hint /score /pass /target N /quit")
            return True
        case _:  # not a command; treat as answer
            ok, msg = validate(ans, ch.expected)
            if ok:
                who.score += 1
                line(C["green"] + "✓ Correct!" + C["reset"])
                line(f"({msg})  Score: {scoreline(g)}")
                return not maybe_end(g)
            else:
                line(C["mag"] + "✗ Not quite." + C["reset"] + f"  ({msg})")
                if g.steal_enabled:
                    line(f"{opp.name}, steal? Same obstacle applies. Prompt:")
                    line(ch.prompt)
                    steal = prompt("steal> ")
                    if steal.strip().lower() == "/quit":
                        banner("Goodbye, sages."); line(f"Final: {scoreline(g)}"); return False
                    ok2, msg2 = validate(steal, ch.expected)
                    if ok2:
                        opp.score += 1
                        line(C["green"] + f"★ Stolen by {opp.name}!" + C["reset"])
                        line(f"({msg2})  Score: {scoreline(g)}")
                        return not maybe_end(g)
                    else:
                        line(f"Steal failed. ({msg2})")
                # reveal after attempts
                line(f"Answer was: {ch.expected!r}")
                line(f"Score: {scoreline(g)}")
                return True

def game_loop(g: Game) -> None:
    banner("Duel of Sages — Two minds. One terminal.")
    line("Commands: /hint  /pass  /score  /target N  /quit", "gray")
    active, other = g.p1, g.p2
    while True:
        if not turn(g, active, other):
            break
        active, other = other, active

# ---------- main ----------

def main() -> None:
    banner("Set the board")
    n1 = prompt("Player 1 name: ").strip() or "Sage A"
    n2 = prompt("Player 2 name: ").strip() or "Sage B"
    seed = int(time.time() * 1000) ^ hash((n1, n2))
    g = Game(Player(n1), Player(n2), target=None, rng=random.Random(seed))
    game_loop(g)

if __name__ == "__main__":
    main()
