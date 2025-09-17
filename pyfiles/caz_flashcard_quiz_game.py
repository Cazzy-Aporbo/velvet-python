# caz_flashcards_quiz_game.py
# -----------------------------------------------------------------------------
# Title: DS/ML Flashcards + Quiz (Interactive + Non‑Interactive Safe Mode)
# Author: Cazandra Aporbo
# Started: Nov 2023
# Updated: April 2025
# Intent: Turn a Q&A bank into an interactive study game that also runs cleanly
#         in sandboxed/non‑TTY environments (no crashes on input/file I/O).
# Notes: Standard library only.Robust against blocked stdin/stdout
#        or filesystem writes. Includes lightweight self‑tests.
# Run:
#   Interactive (terminal):  python caz_flashcards_quiz_game.py --mode mc
#   Non‑interactive (CI/sandbox):  python caz_flashcards_quiz_game.py --mode review --limit 10
#   Self tests:  python caz_flashcards_quiz_game.py --selftest
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import textwrap
import time
from typing import Dict, Iterable, List, Sequence, Tuple

# -----------------------------------------------------------------------------
# Q&A Bank — condensed, faithful to the source content
# -----------------------------------------------------------------------------

QA_BANK: List[Tuple[str, str]] = [
    ("1) How can you build a simple logistic regression model in Python?",
     "Use sklearn.linear_model.LogisticRegression: instantiate and fit on features/labels."),
    ("2) How can you train and interpret a linear regression model in SciKit learn?",
     "Use sklearn.linear_model.LinearRegression: fit on X,y; coefficients/intercept indicate effect sizes."),
    ("3) Name a few libraries in Python used for Data Analysis and Scientific computations.",
     "NumPy, SciPy, pandas, scikit‑learn, Matplotlib, Seaborn."),
    ("4) Which library would you prefer for plotting in Python language: Seaborn or Matplotlib?",
     "Depends: Seaborn for quick, aesthetic stats plots; Matplotlib for low‑level control/customization."),
    ("5) What is the main difference between a Pandas series and a single‑column DataFrame in Python?",
     "Series is 1D labeled array; single‑column DataFrame is 2D table with one column."),
    ("6) Write code to sort a DataFrame in Python in descending order.",
     "df.sort_values(by='column_name', ascending=False)"),
    ("7) How can you handle duplicate values in a dataset for a variable in Python?",
     "df.drop_duplicates(subset='column_name') or df.duplicated('column_name') to flag."),
    ("8) Which Random Forest parameters can be tuned to enhance the predictive power of the model?",
     "n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap."),
    ("9) Which method in pandas.tools.plotting is used to create scatter plot matrix?",
     "pandas.plotting.scatter_matrix (older: pandas.tools.plotting.scatter_matrix)."),
    ("10) How can you check if a data set or time series is Random?",
     "Lag plot: absence of structure suggests randomness."),
    ("11) Can we create a DataFrame with multiple data types in Python? If yes, how can you do it?",
     "Yes; specify per‑column dtypes or assign after creation (dtype per column)."),
    ("12) Is it possible to plot histogram in Pandas without calling Matplotlib? If yes, then write the code to plot the histogram?",
     "df.plot.hist() or series.plot.hist()."),
    ("13) What are the possible ways to load an array from a text data file in Python? How can the efficiency be improved?",
     "numpy.loadtxt or numpy.genfromtxt; specify dtype; memory‑map large files when needed."),
    ("14) Which is the standard data missing marker used in Pandas?",
     "NaN."),
    ("15) Why should you use NumPy arrays instead of nested Python lists?",
     "Vectorized ops, contiguous memory, performance, broadcasting."),
    ("16) What is the preferred method to check for an empty array in NumPy?",
     "Use arr.size == 0 (arr.any()/arr.all() answer different questions)."),
    ("17) List down some evaluation metrics for regression problems.",
     "MAE, MSE, RMSE, R^2, MAPE."),
    ("18) Which Python library would you prefer to use for Data Munging?",
     "pandas."),
    ("19) Write the code to sort an array in NumPy by the nth column?",
     "X[X[:, n-1].argsort()] or with zero‑based n: X[X[:, n].argsort()]."),
    ("20) How are NumPy and SciPy related?",
     "NumPy: arrays and core ops; SciPy: scientific algorithms on top of NumPy."),
    ("21) Which python library is built on top of matplotlib and Pandas to ease data plotting?",
     "Seaborn."),
    ("22) Which plot will you use to assess the uncertainty of a statistic?",
     "Bootstrap plot / resampling distribution."),
    ("23) What are some features of Pandas that you like or dislike?",
     "Flexible wrangling, time series, groupby; downsides: memory use, chained assignment gotchas."),
    ("24) Which scientific libraries in SciPy have you worked with in your project?",
     "scipy.stats, scipy.optimize, scipy.signal, scipy.sparse, etc."),
    ("25) What is pylab?",
     "Convenience namespace mixing NumPy/SciPy/Matplotlib; discouraged in modern code."),
    ("26) Which python library is used for Machine Learning?",
     "scikit‑learn."),
    ("27) How can you copy objects in Python?",
     "copy.copy for shallow; copy.deepcopy for deep; some types have custom methods."),
    ("28) What is the difference between tuples and lists in Python?",
     "Tuples immutable, lists mutable."),
    ("29) What is PEP8?",
     "Python style guide for readable code."),
    ("30) Is all the memory freed when Python exits?",
     "Not guaranteed; OS frees process memory; objects may still be referenced."),
    ("31) What does __init__.py do?",
     "Marks a directory as a package (namespace packages may omit it)."),
    ("32) What is the difference between range() and xrange() functions in Python?",
     "Python 2: xrange was lazy; Python 3: range is lazy; xrange removed."),
    ("33) How can you randomize the items of a list in place in Python?",
     "random.shuffle(lst)."),
    ("34) What is a pass in Python?",
     "No‑op placeholder statement."),
    ("35) If you are given the first and last names of employees, which data type will you use?",
     "List of dicts with 'first'/'last' or a DataFrame with columns."),
    ("36) What happens when you execute the statement mango=banana in Python?",
     "NameError if both names are undefined."),
    ("37) Write a sorting algorithm for a numerical dataset in Python.",
     "Use built‑in sorted (Timsort) or implement quicksort/mergesort for teaching."),
    ("38) Optimize the code: print word.__len__ ()",
     "Use print(len(word)) in Python 3."),
    ("39) What is monkey patching in Python?",
     "Runtime modification/injection of attributes/behavior; useful in tests; risky in prod."),
    ("40) Which tool in Python will you use to find bugs if any?",
     "Linters/analyzers like pylint, flake8; type checker mypy."),
    ("41) How are arguments passed in Python - by reference or by value?",
     "Call‑by‑object reference: values are references to objects."),
    ("42) Single comprehension: even values from even indices.",
     "[x for i, x in enumerate(seq) if i % 2 == 0 and x % 2 == 0]"),
    ("43) Explain the usage of decorators.",
     "Wrap functions/classes to add behavior (logging, auth, caching)."),
    ("44) How can you check whether a pandas data frame is empty or not?",
     "df.empty."),
    ("45) What will be the output of print [m(2) for m in multipliers()]?",
     "[6, 6, 6, 6] due to late binding of the loop variable."),
    ("46) What do you mean by list comprehension?",
     "Concise list construction: [expr for item in iterable if cond]."),
    ("47) What will be the output of word[:3] + word[3:]?",
     "The original string (concatenation of full split)."),
    ("48) list = ['a','e','i','o','u']; print list[8:]",
     "[] — slice beyond end yields empty list."),
    ("49) What will be the output of the code?",
     "[1] and then [1, 1] because a default mutable list persists across calls."),
    ("50) Can the lambda forms in Python contain statements?",
     "No, only expressions."),
    ("51) What will be the data type of x for x = input('Enter a number')?",
     "str."),
    ("52) What do you mean by pickling and unpickling in Python?",
     "Serialize objects to bytes/file vs restore them."),
    ("53) What will be the output of Welcome[1:7:2] for 'Welcome to ProjectPro!'?",
     "'ecm'"),
    ("54) What is wrong with print(\"I love \"ProjectPro\" content.\")?",
     "Unescaped quotes; use single quotes outside or escape inner quotes."),
    ("55) How can you iterate over a few files in Python?",
     "os.listdir + filtering; for recursion use os.walk or glob."),
    ("56) Data type of x for x = input('Enter a number') (dup)",
     "str."),
    ("57) Pickling vs unpickling (dup).",
     "Serialize objects to bytes/file vs restore them."),
    ("58) Welcome[1:7:2] output (dup).",
     "'ecm'"),
    ("59) What is the necessary condition for broadcasting two arrays?",
     "From trailing dimensions: sizes equal or 1."),
    ("60) What is PEP for Python?",
     "Python Enhancement Proposal — design/process document."),
    ("61) What do you mean by overfitting a dataset?",
     "Model fits noise in training; poor generalization."),
    ("62) What do you mean by underfitting a dataset?",
     "Model too simple; poor fit to data."),
    ("63) Difference between a test set and a validation set?",
     "Validation tunes model; test assesses final model."),
    ("64) What is F1‑score for a binary classifier? Which library contains it?",
     "Harmonic mean of precision/recall; sklearn.metrics."),
    ("65) Using sklearn, how to implement ridge regression?",
     "from sklearn.linear_model import Ridge; Ridge(alpha=0.5).fit(X, y)"),
    ("66) Using sklearn, how to implement lasso regression?",
     "from sklearn.linear_model import Lasso; Lasso(alpha=0.4).fit(X, y)"),
    ("67) How is correlation a better metric than covariance?",
     "Correlation is normalized by standard deviations; comparable across scales."),
    ("68) What are confounding factors?",
     "Variables related to both predictors and outcome, distorting relationships."),
    ("69) What is namespace in Python?",
     "Mapping from names to objects; e.g., globals(), locals()."),
    ("70) What is try‑except‑finally in Python?",
     "Error handling: try code, except on errors, finally always runs."),
    ("71) Difference between append() and extend().",
     "append adds one object; extend adds elements from iterable."),
    ("72) What is the use of enumerate()?",
     "Yields (index, value) pairs while iterating."),
    ("73) List immutable and mutable built‑in data types.",
     "Immutable: str, bytes, tuple, frozenset, int/float. Mutable: list, dict, set, bytearray."),
    ("74) What is negative indexing in Python?",
     "Index from the end: a[-1] is last element; slices support negatives."),
]

# -----------------------------------------------------------------------------
# Utilities — formatting, normalization, safe I/O
# -----------------------------------------------------------------------------

RESULTS_FILE = "quiz_results.json"


def wrap(s: str, width: int = 76) -> str:
    return "\n".join(textwrap.wrap(s, width=width)) if s else ""


def normalize(s: str) -> str:
    # Lowercase, trim, strip backticks and newlines for loose comparisons
    return (s or "").strip().lower().replace("`", "").replace("\n", " ")


def safe_input(prompt: str, default: str | None = None) -> str | None:
    """An input() wrapper that never raises in restricted environments.
    Returns default on OSError/EOFError/non‑TTY.
    """
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return default
        return input(prompt)
    except (EOFError, OSError):
        return default


def safe_write_text(path: str, content: str) -> bool:
    """Attempt to write text; return False if the environment blocks I/O."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except (OSError, IOError):
        return False


# -----------------------------------------------------------------------------
# Game mechanics
# -----------------------------------------------------------------------------

def _choice_options(correct_ans: str, all_answers: Sequence[str], k: int = 4) -> List[str]:
    """Return k options including exactly one correct answer."""
    pool = [x for x in all_answers if normalize(x) != normalize(correct_ans)]
    if len(pool) >= k - 1:
        picks = random.sample(list(pool), k - 1)
    else:
        picks = list(pool)
    opts = picks + [correct_ans]
    random.shuffle(opts)
    return opts


def _parse_range(rng: str | None) -> Tuple[int | None, int | None]:
    if not rng:
        return None, None
    try:
        lo_s, hi_s = rng.split("-", 1)
        return int(lo_s), int(hi_s)
    except Exception:
        return None, None


def _subset_by_range(full: List[Tuple[str, str]], rng: str | None) -> List[Tuple[str, str]]:
    lo, hi = _parse_range(rng)
    if lo is None or hi is None:
        return full.copy()
    subset: List[Tuple[str, str]] = []
    for q, a in full:
        # Leading number like "12) ..."
        prefix = q.split(")", 1)[0].strip()
        num = int(prefix) if prefix.isdigit() else None
        if num is None or (lo <= num <= hi):
            subset.append((q, a))
    return subset


# Modes -----------------------------------------------------------------------

def mode_review(pool: List[Tuple[str, str]], limit: int | None = None, delay: float = 0.0) -> Tuple[int, int]:
    """Non‑interactive: show Q then A. Returns (correct, total) where correct==total."""
    total = 0
    for i, (q, a) in enumerate(pool, 1):
        if limit and i > limit:
            break
        print("\n" + "=" * 78)
        print(f"[{i}] REVIEW")
        print("Q:", wrap(q))
        if delay:
            time.sleep(min(delay, 2.0))
        print("A:", wrap(a))
        total += 1
    return total, total


def mode_flashcards(pool: List[Tuple[str, str]], limit: int | None = None) -> Tuple[int, int]:
    random.shuffle(pool)
    correct = 0
    total = 0
    for i, (q, a) in enumerate(pool, 1):
        if limit and i > limit:
            break
        print("\n" + "=" * 78)
        print(f"[{i}] FLASHCARD")
        print("Q:", wrap(q))
        _ = safe_input("\nPress Enter to reveal answer...", default="")
        print("A:", wrap(a))
        mark = safe_input("\nMark correct? [y/N]: ", default="n")
        if (mark or "").strip().lower() == "y":
            correct += 1
        total += 1
    return correct, total


def mode_multiple_choice(pool: List[Tuple[str, str]], limit: int | None = None) -> Tuple[int, int]:
    all_answers = [a for _, a in pool]
    random.shuffle(pool)
    correct = 0
    total = 0
    for i, (q, a) in enumerate(pool, 1):
        if limit and i > limit:
            break
        print("\n" + "=" * 78)
        print(f"[{i}] MULTIPLE CHOICE")
        print("Q:", wrap(q))
        options = _choice_options(a, all_answers)
        for idx, opt in enumerate(options, 1):
            print(f"  {idx}. {wrap(opt)}")
        ans = safe_input("Your choice [1-4]: ", default=None)
        if ans and ans.isdigit():
            pick = int(ans)
        else:
            pick = 0  # non‑interactive: show solution but don't score as correct
        if 1 <= pick <= len(options) and normalize(options[pick - 1]) == normalize(a):
            print("Correct.")
            correct += 1
        else:
            print("Incorrect or skipped.")
            print("Answer:", wrap(a))
        total += 1
    return correct, total


def mode_typing(pool: List[Tuple[str, str]], limit: int | None = None, export_misses: str | None = None, save: bool = True) -> Tuple[int, int]:
    random.shuffle(pool)
    correct = 0
    total = 0
    misses: List[Dict[str, str]] = []
    start = time.time()
    for i, (q, a) in enumerate(pool, 1):
        if limit and i > limit:
            break
        print("\n" + "=" * 78)
        print(f"[{i}] TYPE THE ANSWER")
        print("Q:", wrap(q))
        guess = safe_input("A: ", default=None)
        if guess is None:
            # Non‑interactive: reveal the answer and count as reviewed, not correct
            print("Answer:", wrap(a))
            misses.append({"question": q, "your_answer": "<no input>", "answer": a})
        else:
            if normalize(guess) in normalize(a):
                print("Accepted.")
                correct += 1
            else:
                print("We will count this as a miss.")
                print("Answer:", wrap(a))
                misses.append({"question": q, "your_answer": guess, "answer": a})
        total += 1
    dur = time.time() - start
    print(f"\nCompleted {total} in {dur:.1f}s. Correct: {correct}.")
    if misses and save:
        _save_session({"mode": "typing", "score": correct, "total": total, "misses": misses})
        if export_misses:
            _export_misses_csv(misses, export_misses)
    return correct, total


# Persistence (safe) ----------------------------------------------------------

def _save_session(payload: Dict[str, object]) -> None:
    try:
        hist: List[Dict[str, object]] = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                hist = json.load(f)
        hist.append(payload)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)
    except (OSError, IOError):
        # Filesystem may be read‑only; silently skip persistence.
        pass


def _export_misses_csv(misses: List[Dict[str, str]], path: str) -> None:
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["question", "your_answer", "answer"])
            w.writeheader()
            for row in misses:
                w.writerow(row)
        print(f"Misses exported to {path}")
    except (OSError, IOError):
        print("Could not write misses CSV (environment blocked).")


# CLI / Main ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DS/ML Flashcards + Quiz — interactive and sandbox‑safe")
    p.add_argument("--mode", choices=["auto", "review", "flashcards", "mc", "typing"], default="auto",
                   help="auto picks 'review' if stdin not TTY; otherwise interactive menu")
    p.add_argument("--range", dest="qrange", default=None, help="question range like 1-30")
    p.add_argument("--limit", type=int, default=None, help="limit number of questions")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    p.add_argument("--no-save", action="store_true", help="disable writing quiz_results.json")
    p.add_argument("--export-misses", default=None, help="path to write misses CSV (typing mode)")
    p.add_argument("--delay", type=float, default=0.0, help="seconds between Q and A in review mode")
    p.add_argument("--selftest", action="store_true", help="run built‑in tests and exit")
    return p


def run_menu(selected_pool: List[Tuple[str, str]]) -> None:
    # Interactive menu; use safe_input for resilience
    while True:
        print("\nChoose a mode:")
        print("  1) Flashcards")
        print("  2) Multiple Choice")
        print("  3) Type the Answer")
        print("  4) Quit")
        choice = safe_input("> ", default=None)
        if choice is None:
            print("No interactive input available. Switching to REVIEW mode.")
            mode_review(selected_pool)
            return
        choice = choice.strip()
        if choice == "4":
            print("Good luck out there.")
            return
        if choice == "1":
            mode_flashcards(selected_pool)
        elif choice == "2":
            mode_multiple_choice(selected_pool)
        elif choice == "3":
            mode_typing(selected_pool)
        else:
            print("Unknown choice.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return _run_tests()
    if args.seed is not None:
        random.seed(args.seed)

    pool = _subset_by_range(QA_BANK, args.qrange)
    if args.limit is not None and args.limit > 0:
        pool = pool[: args.limit]

    # Decide operating mode
    stdin_tty = bool(sys.stdin and sys.stdin.isatty())
    mode = args.mode
    if mode == "auto":
        mode = "review" if not stdin_tty else "menu"

    if mode == "menu":
        run_menu(pool)
        return 0
    elif mode == "review":
        mode_review(pool, limit=args.limit, delay=args.delay)
        return 0
    elif mode == "flashcards":
        mode_flashcards(pool, limit=args.limit)
        return 0
    elif mode == "mc":
        mode_multiple_choice(pool, limit=args.limit)
        return 0
    elif mode == "typing":
        mode_typing(pool, limit=args.limit, export_misses=args.export_misses, save=not args.no_save)
        return 0
    else:
        print("Unknown mode.")
        return 2


# -----------------------------------------------------------------------------
# Lightweight test suite (no external deps)
# -----------------------------------------------------------------------------

def _run_tests() -> int:
    failures = 0

    # Test normalize
    a = normalize("  Hello\nWorld  ")
    b = normalize("hello world")
    if a != b:
        print("TEST normalize FAILED")
        failures += 1

    # Test choice options include the correct answer and have <=4 options
    correct = "Answer X"
    options = _choice_options(correct, ["A", "B", correct, "D"], k=4)
    if correct not in options or len(options) != 4:
        print("TEST choice options FAILED")
        failures += 1

    # Test parse range
    if _parse_range("5-10") != (5, 10):
        print("TEST parse range FAILED")
        failures += 1
    if _parse_range("oops") != (None, None):
        print("TEST parse range invalid FAILED")
        failures += 1

    # Test subset by range keeps items without numeric prefix and within bounds
    sample = [("X) no number", "a"), ("2) two", "b"), ("9) nine", "c"]
    sub = _subset_by_range(sample, "3-9")
    labels = [q for q, _ in sub]
    if "X) no number" not in labels or "2) two" in labels or "9) nine" not in labels:
        print("TEST subset by range FAILED")
        failures += 1

    # Test review mode returns (total,total)
    total = 3
    t1, t2 = mode_review([("q1", "a1"), ("q2", "a2"), ("q3", "a3")], limit=total)
    if (t1, t2) != (total, total):
        print("TEST review mode FAILED")
        failures += 1

    if failures == 0:
        print("All tests passed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
