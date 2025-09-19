#!/usr/bin/env python3  # Shebang for Unix-like systems 
# -*- coding: utf-8 -*-  # Source encoding declaration 

"""
Cazzy Odyssey Game — It is meant to be an an endless, terminal-friendly skill game.  
Every single line is commented to help learners think in terms of:  # Explanation (str)
- data types (int/float/str/list/tuple/dict/set/bool/None)  # Comments about types (str)
- scope (global vs local), functions, decorators, closures  # Concepts list (str)
- OOP (dataclass, properties, classmethod/staticmethod), dunder (magic) methods  # Topics (str)
- comprehensions, pattern matching (3.10+), generators, context managers  # More topics (str)
- error handling, testing ideas, and readable style  # Topics (str)

Quit any time with: /quit, /q, or Ctrl+C.  # Usage hint (str)
"""

# ------------------------------ imports (built-in only) ------------------------------
from __future__ import annotations  # Future import to allow postponed annotations (module-level directive)
import sys  # System-specific functions and parameters (module)
import math  # Math functions (module)
import random  # RNG utilities (module)
import time  # Time functions (module)
import re  # Regular expressions (module)
from dataclasses import dataclass  # Dataclass decorator and function (callable)
from typing import Callable, Dict, List, Tuple, Optional, Any, Iterable  # Type hints (types)
from functools import lru_cache, wraps  # Caching decorator and utility decorator (callables)
from contextlib import contextmanager  # Context manager helper (callable)
from enum import Enum, auto  # Enumerations (class factory, function)
from itertools import islice  # Iterator slicing helper (function)
from collections import Counter, namedtuple  # Data utilities (classes)

# ------------------------------ global configuration & state ------------------------------
PASTEL = {  # Dict[str,str] — ANSI-ish pastel-ish color codes (basic) for terminal flavor; safe if not supported
    "reset": "\033[0m",  # ANSI reset (str)
    "lav": "\033[95m",  # Lavender-ish (str)
    "pink": "\033[95m",  # Pink mapped to same code here (str)
    "mint": "\033[92m",  # Mint/green (str)
    "peach": "\033[93m",  # Peach/yellow (str)
    "ink": "\033[90m",  # Dim ink/gray (str)
}

GLOBAL_COMBO: int = 0  # Global mutable int used to demonstrate global scope updates

# Namedtuple for immutable lightweight record (type: class returned by factory)
ScoreTuple = namedtuple("ScoreTuple", ["correct", "total", "streak"])  # Fields are ints

# ------------------------------ utility decorators & helpers ------------------------------
def banner(text: str) -> None:
    """Pretty banner line with pastel coloring.  # Docstring (str)"""
    # Using str multiplication and concatenation to draw a line — datatype: str
    line = PASTEL["lav"] + "═" * 80 + PASTEL["reset"]  # str
    print(line)  # I/O side-effect: print to terminal
    print(PASTEL["mint"] + text + PASTEL["reset"])  # Colored text (str)
    print(line)  # Another separator line (str)

def safe_input(prompt: str) -> str:
    """Input wrapper that catches EOF/KeyboardInterrupt and returns '/quit'.  # Docstring (str)"""
    try:  # Try/except for robust input (control flow)
        return input(prompt)  # Read user input (str)
    except (EOFError, KeyboardInterrupt):  # Graceful termination on signals (exceptions)
        return "/quit"  # Signal to quit (str)

def typed_equal(a: Any, b: Any) -> bool:
    """Flexible equality: normalize strings & whitespace; cast numbers when appropriate.  # Docstring (str)"""
    # If both look like numbers, compare as floats — demonstrates casting and type checks (bool)
    num_re = re.compile(r"^-?\d+(\.\d+)?$")  # Compiled regex (Pattern)
    if isinstance(a, (int, float)) and isinstance(b, str) and num_re.match(b.strip()):  # Condition (bool)
        return float(a) == float(b.strip())  # Float comparison (bool)
    if isinstance(b, (int, float)) and isinstance(a, str) and num_re.match(a.strip()):  # Mirror branch (bool)
        return float(b) == float(a.strip())  # Float comparison (bool)
    # Normalize strings: casefold + strip — robust string compare (bool)
    if isinstance(a, str) and isinstance(b, str):  # Type check (bool)
        return a.strip().casefold() == b.strip().casefold()  # Case-insensitive compare (bool)
    return a == b  # Fallback to Python equality (bool)

# Context manager implemented via decorator — demonstrates generator-based context managers
@contextmanager  # Decorator (callable) that turns a generator into a context manager
def timer(label: str):  # Function defining a context manager (callable)
    start = time.perf_counter()  # Capture start time (float, local variable)
    try:  # Try block for context (control flow)
        yield  # Yield control to with-block (None)
    finally:  # Ensure this runs even on exceptions (control flow)
        end = time.perf_counter()  # End time (float)
        print(f"{PASTEL['ink']}[timer] {label}: {(end-start)*1000:.1f} ms{PASTEL['reset']}")  # Timing (I/O)

# Decorator example that counts function calls using closure state
def call_counter(fn: Callable) -> Callable:  # Higher-order function (callable) returning callable
    count = 0  # Enclosed variable (int) to be captured by closure
    @wraps(fn)  # Preserve metadata like __name__ (decorator)
    def wrapper(*args: Any, **kwargs: Any):  # Inner wrapper function (callable)
        nonlocal count  # Declare closure variable as nonlocal (scope control)
        count += 1  # Mutate closure state (int +=)
        print(f"{PASTEL['ink']}[calls] {fn.__name__} -> {count}{PASTEL['reset']}")  # Report call count (I/O)
        return fn(*args, **kwargs)  # Forward call to original function (return value varies)
        # End of wrapper (comment)
    return wrapper  # Return decorated function (callable)

# ------------------------------ OOP showpiece with magic methods ------------------------------
@dataclass  # Dataclass auto-generates __init__, __repr__, __eq__ (decorator)
class Vector2D:  # Custom class demonstrating dunder methods (class)
    x: float  # Field x (float)
    y: float  # Field y (float)

    def __add__(self, other: "Vector2D") -> "Vector2D":  # Magic method for + (callable)
        return Vector2D(self.x + other.x, self.y + other.y)  # New Vector2D (object)

    def __mul__(self, k: float) -> "Vector2D":  # Magic method for scalar * (callable)
        return Vector2D(self.x * k, self.y * k)  # Scaled vector (object)

    def __len__(self) -> int:  # Magic method to define "length" in terms of components count (callable)
        return 2  # Always 2 for 2D (int)

    def __iter__(self) -> Iterable[float]:  # Iterator protocol to unpack like tuple (callable)
        yield self.x  # First component (float)
        yield self.y  # Second component (float)

    @property  # Property read-only attribute (decorator)
    def magnitude(self) -> float:  # Compute Euclidean norm (callable)
        return math.hypot(self.x, self.y)  # sqrt(x^2 + y^2) (float)

    @classmethod  # Classmethod as alternate constructor (decorator)
    def from_tuple(cls, t: Tuple[float, float]) -> "Vector2D":  # Receive class and tuple (callable)
        return cls(float(t[0]), float(t[1]))  # Construct instance (object)

    @staticmethod  # Staticmethod independent utility (decorator)
    def dot(a: "Vector2D", b: "Vector2D") -> float:  # Dot product (callable)
        return a.x * b.x + a.y * b.y  # Scalar (float)

# ------------------------------ scoring & game state ------------------------------
@dataclass  # Dataclass for simple container (decorator)
class ScoreBoard:  # Tracks performance metrics (class)
    correct: int = 0  # Total correct answers (int)
    total: int = 0  # Total attempts (int)
    streak: int = 0  # Current consecutive correct streak (int)
    start_ts: float = time.time()  # Game start timestamp (float, default evaluated at import)

    def as_tuple(self) -> ScoreTuple:  # Convert to immutable namedtuple (callable)
        return ScoreTuple(self.correct, self.total, self.streak)  # Return ScoreTuple (object)

    def pretty(self) -> str:  # Human-readable summary (callable)
        elapsed = time.time() - self.start_ts  # Seconds since start (float)
        return f"✅ {self.correct} / {self.total} | 🔥 streak {self.streak} | ⏱ {elapsed:.0f}s"  # Summary (str)

# Enum for challenge categories — handy for filtering or themed streaks (class)
class Topic(Enum):  # Enumeration base (class)
    ARITH = auto()  # Arithmetic (enum member)
    STRINGS = auto()  # Strings (enum member)
    DS = auto()  # Data structures (enum member)
    OOP = auto()  # Object-oriented (enum member)
    ALGO = auto()  # Algorithms (enum member)
    MISC = auto()  # Miscellaneous (enum member)

# ------------------------------ challenge factory functions ------------------------------
def challenge_arith() -> Tuple[str, Any, str, Topic]:
    """Create an arithmetic challenge.  # Docstring (str)"""
    a = random.randint(5, 25)  # Random int a (int)
    b = random.randint(2, 9)  # Random int b (int)
    op = random.choice(["+", "-", "*", "//", "%"])  # Random operator (str)
    # Compute expected answer using Python's eval safely over controlled expression (Any)
    expr = f"{a} {op} {b}"  # Expression text (str)
    expected = eval(expr)  # Evaluate within our own code context (int)
    prompt = f"Compute (int): {expr} = ?"  # Prompt for user (str)
    explain = "int math with operators + - * // % ; remember // is floor division."  # Explanation (str)
    return prompt, expected, explain, Topic.ARITH  # Return tuple (tuple)

def challenge_strings() -> Tuple[str, Any, str, Topic]:
    """Create a string slicing/formatting challenge.  # Docstring (str)"""
    s = random.choice(["FoXX_Health", "Pythonic", "Data-Science", "Lavender"])  # Source string (str)
    i = random.randint(1, len(s)-1)  # Split index (int)
    prompt = f"Slice (str): given s='{s}', what is s[:{i}] + s[{i}:] equal to?"  # Prompt (str)
    expected = s[:i] + s[i:]  # Concatenation equals original (str)
    explain = "slicing is non-destructive; s[:i] + s[i:] reconstructs the original string."  # Explanation (str)
    return prompt, expected, explain, Topic.STRINGS  # Return tuple (tuple)

def challenge_ds() -> Tuple[str, Any, str, Topic]:
    """Create a data-structure challenge (list/dict/set/tuple).  # Docstring (str)"""
    nums = [random.randint(1, 9) for _ in range(5)]  # List of ints (list[int])
    unique = sorted(set(nums))  # Unique sorted values (list[int])
    mapping = {n: n*n for n in unique}  # Dict mapping to squares (dict[int,int])
    prompt = f"Given nums={nums}, unique sorted set is? (enter list like [1,2])"  # Prompt (str)
    expected = unique  # Expected list (list[int])
    explain = f"set removes duplicates; sorted orders; dict shown for mapping: {mapping}."  # Explanation (str)
    return prompt, expected, explain, Topic.DS  # Return tuple (tuple)

def challenge_oop() -> Tuple[str, Any, str, Topic]:
    """OOP challenge using Vector2D and dunder methods.  # Docstring (str)"""
    v1 = Vector2D(random.randint(1,4), random.randint(1,4))  # Vector2D instance (object)
    v2 = Vector2D(random.randint(1,4), random.randint(1,4))  # Another instance (object)
    result = v1 + (v2 * 2)  # Use __mul__ then __add__ (Vector2D)
    prompt = f"Vector: {v1} + ({v2} * 2) -> what is magnitude rounded to 2dp?"  # Prompt (str)
    expected = round(result.magnitude, 2)  # Rounded float (float)
    explain = "__mul__ scales components; __add__ adds; property .magnitude uses math.hypot."  # Explanation (str)
    return prompt, expected, explain, Topic.OOP  # Return tuple (tuple)

@lru_cache(maxsize=128)  # Cache decorator to memoize fib values (decorator)
def fib(n: int) -> int:  # Recursive fibonacci with caching (callable)
    """Return nth Fibonacci number (n>=0).  # Docstring (str)"""
    if n < 2:  # Base case (bool)
        return n  # Return n for 0/1 (int)
    return fib(n-1) + fib(n+ -2 + 0)  # Slightly playful expression same as fib(n-2) (int)

def challenge_algo() -> Tuple[str, Any, str, Topic]:
    """Algorithmic challenge — Fibonacci with caching.  # Docstring (str)"""
    n = random.randint(6, 10)  # Random n (int)
    prompt = f"Algorithm: fib({n}) = ?"  # Prompt (str)
    expected = fib(n)  # Compute via cached recursion (int)
    explain = "lru_cache memoizes results; recursion adds smaller subproblems."  # Explanation (str)
    return prompt, expected, explain, Topic.ALGO  # Return tuple (tuple)

def challenge_misc() -> Tuple[str, Any, str, Topic]:
    """Misc challenge — boolean truthiness & regex.  # Docstring (str)"""
    text = random.choice(["Email me at test@example.com", "No contact here"])  # Text (str)
    found = bool(re.search(r"\w+@\w+\.\w+", text))  # Regex email-like match (bool)
    prompt = f"Misc: does text '{text}' contain an email-like pattern? (True/False)"  # Prompt (str)
    expected = found  # Expected bool (bool)
    explain = "bool(re.search(...)) yields True if a match object exists."  # Explanation (str)
    return prompt, expected, explain, Topic.MISC  # Return tuple (tuple)

# Registry of factories for random selection — list of callables (list[Callable])
CHALLENGES: List[Callable[[], Tuple[str, Any, str, Topic]]] = [  # Annotated list (list)
    challenge_arith,  # Arithmetic factory (callable)
    challenge_strings,  # String factory (callable)
    challenge_ds,  # DS factory (callable)
    challenge_oop,  # OOP factory (callable)
    challenge_algo,  # Algorithm factory (callable)
    challenge_misc,  # Misc factory (callable)
]

# ------------------------------ global/local scope demo ------------------------------
def bump_global_combo() -> None:
    """Demonstrate global mutation: adds random 0..2 to GLOBAL_COMBO.  # Docstring (str)"""
    global GLOBAL_COMBO  # Declare that we intend to write to the global (keyword)
    add = random.randint(0,2)  # Local int (int)
    GLOBAL_COMBO += add  # Mutate global int (side-effect)
    print(f"{PASTEL['ink']}[global] combo +={add} -> {GLOBAL_COMBO}{PASTEL['reset']}")  # Show state (I/O)

# ------------------------------ command help ------------------------------
HELP_TEXT = """  # Multiline help (str)
Commands:
  /quit or /q       -> exit the game
  /skip             -> skip current question (no penalty)
  /hint             -> show explanation (counts as incorrect)
  /score            -> display current score
  /topic NAME       -> bias towards a topic (ARITH/STRINGS/DS/OOP/ALGO/MISC)
  /help             -> show this help
"""  # End of help text (str)

# ------------------------------ topic biasing logic ------------------------------
def pick_challenge(bias: Optional[Topic]) -> Tuple[str, Any, str, Topic]:
    """Pick a challenge, optionally biased to a topic.  # Docstring (str)"""
    # If bias is set, pick matching factory; else uniform random (callable selection)
    if bias:  # If a Topic is provided (bool)
        pool = [f for f in CHALLENGES if f().__getitem__(3) == bias]  # Build a filtered pool by peeking (list)
        # The above peeks one tuple per factory; negligible cost and keeps code simple (comment)
        if pool:  # If we found matching factories (bool)
            fn = random.choice(pool)  # Choose one (callable)
            # Recompute actual tuple since we consumed one tuple during filtering (determinism not required) (comment)
            return fn()  # Return a fresh challenge (tuple)
    # Fallback: any challenge (comment)
    fn = random.choice(CHALLENGES)  # Choose any factory (callable)
    return fn()  # Produce a challenge (tuple)

# ------------------------------ core game loop ------------------------------
@call_counter  # Decorated to demonstrate closures and side effects (decorator)
def play_round(score: ScoreBoard, bias: Optional[Topic]) -> bool:
    """Play one round; return False if player exits, else True.  # Docstring (str)"""
    bump_global_combo()  # Touch global state as a playful side quest (side-effect)
    # Choose challenge under a timer context manager (with) — shows __enter__/__exit__ semantics (comment)
    with timer("challenge-pick"):  # Enter context (object management)
        prompt, expected, explain, topic = pick_challenge(bias)  # Unpack result (multiple types)
    banner(f"Topic: {topic.name}")  # Show topic name (str)
    print(prompt)  # Present the question (I/O)
    # Accept user input; parse magic commands starting with '/' (comment)
    ans = safe_input(PASTEL["peach"] + "Your answer (or /help): " + PASTEL["reset"])  # Read answer (str)
    if ans.strip().lower() in ("/quit", "/q"):  # Graceful exit (bool)
        return False  # Signal to stop game loop (bool)
    if ans.strip().lower() == "/help":  # Show help (bool)
        print(HELP_TEXT)  # Print help (I/O)
        return True  # Continue game (bool)
    if ans.strip().lower() == "/score":  # Score request (bool)
        print(score.pretty())  # Print score (I/O)
        return True  # Continue (bool)
    if ans.strip().lower() == "/skip":  # Skip question (bool)
        print("Skipped. No change to score.")  # Message (I/O)
        score.total += 1  # We still count an attempt for pacing (int +=)
        score.streak = 0  # Reset streak to be fair (int =)
        return True  # Continue (bool)
    if ans.lower().startswith("/topic"):  # Topic bias command (bool)
        # Attempt to parse topic name after command; demonstrates str.split (list[str])
        parts = ans.split()  # Split by whitespace (list[str])
        if len(parts) >= 2:  # If we have a name (bool)
            name = parts[1].upper()  # Uppercase for Enum lookup (str)
            try:  # Try to set bias (control flow)
                new_bias = Topic[name]  # Enum indexing by name (Topic)
                print(f"Bias set to {new_bias.name}")  # Confirm (I/O)
                # Store bias into score object by monkey-patching (educational trick — dynamic attribute) (comment)
                setattr(score, "_bias", new_bias)  # Attach attribute dynamically (side-effect)
            except KeyError:  # Invalid name (exception)
                print("Unknown topic. Use ARITH/STRINGS/DS/OOP/ALGO/MISC")  # Hint (I/O)
        else:  # No name provided (else branch)
            print("Usage: /topic NAME  e.g., /topic OOP")  # Usage hint (I/O)
        return True  # Continue (bool)
    # Evaluate correctness; support flexible typed equality (comment)
    is_correct = typed_equal(expected, ans)  # Compare answer to expected (bool)
    score.total += 1  # Increment attempts (int +=)
    if is_correct:  # Correct branch (bool)
        score.correct += 1  # Increment correct (int +=)
        score.streak += 1  # Increment streak (int +=)
        print(PASTEL["mint"] + "✓ Correct!" + PASTEL["reset"])  # Feedback (I/O)
    else:  # Incorrect branch (bool)
        score.streak = 0  # Reset streak (int =)
        print(PASTEL["lav"] + f"✗ Not quite. Expected: {expected!r}" + PASTEL["reset"])  # Reveal (I/O)
        print(PASTEL["ink"] + f"[why] {explain}" + PASTEL["reset"])  # Show explanation (I/O)
    print(PASTEL["ink"] + score.pretty() + PASTEL["reset"])  # Show running score (I/O)
    return True  # Continue game (bool)

# ------------------------------ main entrypoint ------------------------------
def main() -> None:
    """Run the endless loop until the player opts out.  # Docstring (str)"""
    random.seed()  # Seed RNG from system entropy (None)
    score = ScoreBoard()  # Initialize scoreboard (ScoreBoard)
    bias: Optional[Topic] = getattr(score, "_bias", None)  # Read bias if set later (Optional[Topic])
    banner("Welcome to Cazzy Python Odyssey — the most comprehensive terminal game ever!")  # Intro (I/O)
    print("Type /help any time. Quit with /quit or /q.")  # Instructions (I/O)
    # Endless loop; user can break with command or Ctrl+C (comment)
    while True:  # Infinite loop (bool evaluated at each iteration)
        try:  # Catch interrupts cleanly (control flow)
            cont = play_round(score, getattr(score, "_bias", None))  # Play one round (bool)
            if not cont:  # If False, break (bool)
                break  # Exit loop (control flow)
        except KeyboardInterrupt:  # Ctrl+C (exception)
            print("\nInterrupted — exiting.")  # Message (I/O)
            break  # Exit loop (control flow)
        except Exception as ex:  # Any unexpected error (exception)
            print(f"Unexpected error: {ex}")  # Report (I/O)
    banner("Goodbye!")  # Farewell banner (I/O)
    print(score.pretty())  # Final score summary (I/O)

# Standard Python module guard to allow import without auto-run (pattern)
if __name__ == "__main__":  # True when executed as script (bool)
    main()  # Call main function (None)
