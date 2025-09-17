# cazandra_python_mastery_playbook.py
# -----------------------------------------------------------------------------
# Title: Python Mastery Playbook — from Basics to Mastery
# Author: Cazandra Aporbo
# Started: December 2022
# Updated: March 2025
# Intent: A single, high‑level Python file that teaches core Python concepts,
#         craft habits, and CLI/Git literacy — in a human voice, with one
#         concept per step, and with commentary on every single line so future
#         me (and teammates) can skim, learn, or teach from it.
# Promise: No emojis. No AI-speak. Just me, walking you through it.
# License: MIT (or whatever I choose later); this is a learning artifact.
# Notes: Everything here uses the Python standard library only. No internet,
#        no external data, no hidden magic. If you can run Python, you can run
#        this file.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# SECTION 0 — Imports I actually use (all from the standard library)
# -----------------------------------------------------------------------------
from dataclasses import dataclass  # Lightweight records for clean examples
from typing import Any, Iterable, Dict, List, Tuple, Set, Optional  # Type hints
import math  # A few numeric examples (e.g., floor division intuition)
import itertools  # Iteration utilities to level up loops
import textwrap  # Pretty printing long strings in the console
import sys  # Access to the interpreter and argv
import os  # Interact with the operating system
import subprocess  # Safely demonstrate shell commands as strings (no exec)

# -----------------------------------------------------------------------------
# Helper: tiny print block wrapper so examples are clear in the console
# -----------------------------------------------------------------------------

def block(title: str) -> None:
    """Print a nice divider so the console reads like a lesson outline."""
    print("\n" + "=" * 78)  # Visual divider line for readability
    print(title)  # The lesson or sub-lesson title
    print("-" * 78)  # Underline to set off the section


# -----------------------------------------------------------------------------
# SECTION 1 — Built-in functions (print, input, int, str, float, isinstance, repr)
# -----------------------------------------------------------------------------

def lesson_builtins() -> None:
    """
    Teach the must-know built-ins with compact, layered examples.
    I avoid `input()` during imports; instead, I show how you'd use it.
    """
    block("1) Built-in Functions: print, input, int, str, float, isinstance, repr")

    # 1. print(): simple output
    print("print() sends readable output to the screen.")
    print("You can print numbers:", 42)
    print("You can print multiple things:", 1, 2, 3, "— spaced by default")

    # 2. input(): accept user text; demonstrate without forcing interaction
    demo_input_prompt = "What's your name? "  # Prompt we might pass to input()
    print("input(prompt) would display:", repr(demo_input_prompt))
    # name = input(demo_input_prompt)  # Commented so this file runs non-interactive
    fake_name = "Ada"  # For demonstration, pretend user typed "Ada"
    print("Pretending the user typed:", fake_name)

    # 3. int(): convert strings/numbers to integers when sensible
    print("int('7') ->", int("7"))
    print("int(3.99) floors toward zero ->", int(3.99))

    # 4. str(): turn anything into a string representation for display
    print("str(3.14) ->", str(3.14))
    print("str([1, 2, 3]) ->", str([1, 2, 3]))

    # 5. float(): make a decimal number
    print("float('2.5') ->", float("2.5"))
    print("float(5) ->", float(5))

    # 6. isinstance(): check type relationships safely
    print("isinstance(5, int) ->", isinstance(5, int))
    print("isinstance(True, int) (booleans are ints in Python) ->", isinstance(True, int))

    # 7. repr(): unambiguous developer-friendly string (round-trippable when possible)
    sample = "line\nwith\ttabs"
    print("str(sample) ->", str(sample))
    print("repr(sample) ->", repr(sample))


# -----------------------------------------------------------------------------
# SECTION 2 — Core code constructs (if, else, while, for, range, break, continue, pass)
# -----------------------------------------------------------------------------

def lesson_code_constructs() -> None:
    """Conditionals and loops with layered, practical examples."""
    block("2) Code Constructs: if/else, while, for..in, range, break, continue, pass")

    # if / else: choose a path
    x = 7  # An example value
    if x % 2 == 0:  # If x is divisible by 2
        print("x is even")
    else:  # Otherwise
        print("x is odd")

    # while: repeat until a condition changes
    n = 0  # Start value
    while n < 3:  # Loop runs while the condition is True
        print("while loop iteration:", n)
        n += 1  # Move toward stopping the loop

    # for..in: iterate cleanly over any iterable
    for ch in "abc":  # Strings are iterable character-by-character
        print("for over string ->", ch)

    # range(): generate a sequence of integers (lazy, not a list in py3)
    for i in range(3):  # 0, 1, 2
        print("range loop index:", i)

    # break: stop the loop immediately
    for i in itertools.count():  # Infinite counter: 0,1,2,...
        print("counting until 3 ->", i)
        if i == 3:
            print("hit 3, breaking out")
            break  # Escape the infinite loop

    # continue: skip just this iteration
    for i in range(5):
        if i % 2 == 0:
            continue  # Skip even numbers
        print("odd only:", i)

    # pass: a syntactic placeholder when Python expects a block
    def todo():  # A function I plan to implement later
        pass  # Placeholder so the file remains valid Python
    print("pass used in a placeholder function called todo()")


# -----------------------------------------------------------------------------
# SECTION 3 — Operators (arithmetic, comparison, identity, membership)
# -----------------------------------------------------------------------------

def lesson_operators() -> None:
    """Show how Python operators behave, with small correctness checks."""
    block("3) Operators: + - * / // % ** = == != > < >= <= in not in is is not")

    # Arithmetic operators
    print("1 + 2 ->", 1 + 2)
    print("5 - 3 ->", 5 - 3)
    print("4 * 2 ->", 4 * 2)
    print("7 / 2 (float division) ->", 7 / 2)
    print("7 // 2 (floor division) ->", 7 // 2)
    print("7 % 2 (remainder) ->", 7 % 2)
    print("2 ** 3 (exponent) ->", 2 ** 3)

    # Assignment and comparisons
    a = 10  # Assign 10 into name a
    print("a == 10 ->", a == 10)
    print("a != 9 ->", a != 9)
    print("a > 5 ->", a > 5)
    print("a < 20 ->", a < 20)
    print("a >= 10 ->", a >= 10)
    print("a <= 10 ->", a <= 10)

    # Membership checks
    data = [1, 2, 3]
    print("2 in [1,2,3] ->", 2 in data)
    print("9 not in [1,2,3] ->", 9 not in data)

    # Identity checks: same object vs. equal value
    x = [1, 2]
    y = [1, 2]
    z = x
    print("x == y (same content) ->", x == y)
    print("x is y (same object?) ->", x is y)
    print("x is z (same object?) ->", x is z)
    print("x is not y ->", x is not y)


# -----------------------------------------------------------------------------
# SECTION 4 — Data structures (lists, tuples, sets, dicts)
# -----------------------------------------------------------------------------

def lesson_data_structures() -> None:
    """Work with the core containers, including mutability and idioms."""
    block("4) Data Structures: list, tuple, set, dict")

    # Lists: ordered, mutable, allow duplicates
    fruits: List[str] = ["apple", "banana", "banana"]
    fruits.append("cherry")  # Add an item at the end
    print("list example:", fruits)

    # Tuples: ordered, immutable, allow duplicates
    point: Tuple[int, int] = (3, 4)
    print("tuple example:", point)

    # Sets: unordered, unique members
    unique: Set[str] = {"a", "b", "a"}
    print("set example (deduped):", unique)

    # Dicts: key/value mapping, insertion-ordered in modern Python
    person: Dict[str, Any] = {"name": "Cazandra", "role": "Data Scientist"}
    person["tools"] = ["Python", "Git", "CLI"]  # Add another key
    print("dict example:", person)

    # A tiny real pattern: grouping counts with a dict
    counts: Dict[str, int] = {}
    for item in fruits:  # Count occurrences in the list
        counts[item] = counts.get(item, 0) + 1
    print("frequency dict pattern:", counts)


# -----------------------------------------------------------------------------
# SECTION 5 — Strings and their methods (upper, lower, capitalize, title, index, count)
# -----------------------------------------------------------------------------

def lesson_strings() -> None:
    """Practical string manipulations and gotchas."""
    block("5) String Methods: upper, lower, capitalize, title, index, count")

    s = "data science with heart"
    print("upper() ->", s.upper())
    print("lower() ->", s.lower())
    print("capitalize() ->", s.capitalize())
    print("title() ->", s.title())
    print("index('science') ->", s.index("science"))
    print("count('a') ->", s.count("a"))

    # Bonus: robust searching without raising ValueError
    target = "python"
    idx = s.find(target)  # -1 if not found
    if idx == -1:
        print(f"find('{target}') not found ->", idx)
    else:
        print(f"find('{target}') found at ->", idx)


# -----------------------------------------------------------------------------
# SECTION 6 — Command Line 101 (clear/Ctrl+L, pwd, top, man, ls, cd, >, >>, <, |)
# -----------------------------------------------------------------------------

def lesson_command_line() -> None:
    """
    I do not execute shell commands here. I present them safely with context so
    you can try in your terminal. The goal is comprehension, not side-effects.
    """
    block("6) Command Line: everyday commands and redirection/pipes")

    cheat = {
        "clear or Ctrl+L": "Clear the visible terminal screen (not history).",
        "pwd": "Print the current working directory (where you are).",
        "top": "Interactive view of running processes; q to quit.",
        "man": "Read the manual page for a command, e.g., man ls.",
        "ls": "List files; try ls -la for more detail.",
        "cd": "Change directory; cd ~ goes to your home folder.",
        ">": "Redirect stdout to a file, overwriting it.",
        ">>": "Append stdout to the end of a file.",
        "<": "Feed a file as stdin to a command.",
        "|": "Pipe stdout of the left command into the right command.",
    }

    for k, v in cheat.items():
        print(f"{k:>10} : {v}")

    # Teach by example (shown as strings, not executed):
    examples = [
        "ls -la | grep .py",  # Filter a long listing for .py files
        "cat notes.txt | wc -l",  # Count lines in a file
        "echo 'hello' > out.txt",  # Create/overwrite a file with text
        "echo 'world' >> out.txt",  # Append to that file
        "sort < out.txt | uniq",  # Sort input file and deduplicate lines
    ]
    print("\nTry these in your shell:")
    for ex in examples:
        print("  $", ex)


# -----------------------------------------------------------------------------
# SECTION 7 — Git Flow Essentials (init, clone, add, mv, reset, rm, commit, grep,
#                                  log, show, status, branch, merge, rebase, diff,
#                                  bisect, tag, fetch, pull, push, help)
# -----------------------------------------------------------------------------

def lesson_git() -> None:
    """
    A calm, repeatable Git flow cheat sheet. Shown as commands only.
    Run them in a repository directory, not from Python.
    """
    block("7) Git Commands: the everyday set and a few power moves")

    commands = [
        ("git init", "Start a new repository in the current folder."),
        ("git clone <url>", "Copy a remote repository locally."),
        ("git add <file>", "Stage changes for the next commit."),
        ("git mv <old> <new>", "Move or rename tracked files."),
        ("git reset -- <file>", "Unstage a file (keep changes)."),
        ("git rm <file>", "Remove and stage deletion of a tracked file."),
        ("git commit -m 'msg'", "Record staged changes with a message."),
        ("git grep <pattern>", "Search committed content for a pattern."),
        ("git log --oneline --graph --decorate --all", "Pretty commit graph."),
        ("git show <ref>", "Show details for a commit or object."),
        ("git status", "See what's changed and what's staged."),
        ("git branch", "List branches; with -d to delete, -m to rename."),
        ("git merge <branch>", "Merge named branch into current branch."),
        ("git rebase <base>", "Replay commits on top of a new base."),
        ("git diff", "Show unstaged diffs; with --staged for staged diffs."),
        ("git bisect start", "Begin binary search to find a bad commit."),
        ("git tag v1.0.0", "Create a lightweight tag; use -a for annotated."),
        ("git fetch", "Update local refs from remote without merging."),
        ("git pull", "Fetch then merge (or rebase if configured)."),
        ("git push", "Send local commits/tags to the remote."),
        ("git help <cmd>", "Open help for any Git subcommand."),
    ]

    for cmd, why in commands:
        print(f"{cmd:<40} # {why}")


# -----------------------------------------------------------------------------
# SECTION 8 — Mastery Patterns: tiny recipes that feel like superpowers
# -----------------------------------------------------------------------------

def lesson_mastery_patterns() -> None:
    """
    Beyond the basics: idioms and patterns I reach for in real projects.
    """
    block("8) Mastery Patterns: list/dict/set comprehensions, unpacking, EAFP, dataclasses")

    # Comprehensions: compact, expressive data transforms
    nums = [1, 2, 3, 4, 5]
    squares = [n * n for n in nums]  # List comprehension for transformation
    odds = {n for n in nums if n % 2}  # Set comp with a filter
    idx_map = {i: n for i, n in enumerate(nums)}  # Dict comp with enumerate
    print("squares:", squares)
    print("odds:", odds)
    print("idx_map:", idx_map)

    # Unpacking: assign multiple variables from iterables in one go
    first, *middle, last = [10, 20, 30, 40]
    print("first:", first, "middle:", middle, "last:", last)

    # EAFP: Easier to Ask Forgiveness than Permission — try/except over pre-checks
    def safe_div(a: float, b: float) -> Optional[float]:
        try:
            return a / b
        except ZeroDivisionError:
            return None
    print("safe_div(10, 0) ->", safe_div(10, 0))

    # Dataclasses: elegant, type-hinted records with defaults
    @dataclass
    class Person:
        name: str
        role: str = "Learner"
        skills: List[str] = None  # Default uses None; we fix it in __post_init__
        def __post_init__(self):
            if self.skills is None:
                self.skills = []
        def label(self) -> str:
            return f"{self.name} — {self.role} ({', '.join(self.skills) if self.skills else 'no skills listed'})"

    p = Person(name="Cazandra", role="Head of Data", skills=["Python", "Git", "CLI"])
    print("dataclass:", p.label())


# -----------------------------------------------------------------------------
# SECTION 9 — A mini-REPL feel: tie lessons together
# -----------------------------------------------------------------------------

def run_all_lessons() -> None:
    """A single entry point to run all lessons in order."""
    lesson_builtins()
    lesson_code_constructs()
    lesson_operators()
    lesson_data_structures()
    lesson_strings()
    lesson_command_line()
    lesson_git()
    lesson_mastery_patterns()


# -----------------------------------------------------------------------------
# Standard script guard — so imports don’t auto-run everything in other files
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_all_lessons()  # Fire the full tour when run directly
