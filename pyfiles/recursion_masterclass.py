from __future__ import annotations

from functools import cache
from typing import Any


def factorial_recursive(n: int) -> int:
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    if n in (0, 1):
        return 1
    return n * factorial_recursive(n - 1)


def factorial_iterative(n: int) -> int:
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def fib_recursive_naive(n: int) -> int:
    if n < 0:
        raise ValueError("Fibonacci is undefined for negative integers")
    if n in (0, 1):
        return n
    return fib_recursive_naive(n - 1) + fib_recursive_naive(n - 2)


@cache
def fib_recursive_memo(n: int) -> int:
    if n < 0:
        raise ValueError("Fibonacci is undefined for negative integers")
    if n in (0, 1):
        return n
    return fib_recursive_memo(n - 1) + fib_recursive_memo(n - 2)


def fib_bottom_up(n: int) -> int:
    if n < 0:
        raise ValueError("Fibonacci is undefined for negative integers")
    if n in (0, 1):
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def permutations_backtracking(items: list[Any]) -> list[list[Any]]:
    result: list[list[Any]] = []
    used = [False] * len(items)
    path: list[Any] = []

    def dfs() -> None:
        if len(path) == len(items):
            result.append(path.copy())
            return
        for i, val in enumerate(items):
            if used[i]:
                continue
            used[i] = True
            path.append(val)
            dfs()
            path.pop()
            used[i] = False

    dfs()
    return result


def n_queens(n: int = 4) -> list[list[int]]:
    cols: list[int] = []
    solutions: list[list[int]] = []

    def safe(c: int) -> bool:
        r = len(cols)
        for pr, pc in enumerate(cols):
            if pc == c or abs(pc - c) == abs(pr - r):
                return False
        return True

    def place() -> None:
        if len(cols) == n:
            solutions.append(cols.copy())
            return
        for c in range(n):
            if safe(c):
                cols.append(c)
                place()
                cols.pop()

    place()
    return solutions


def dfs_tree(tree: dict[str, Any]) -> list[str]:
    order: list[str] = []

    def visit(node: dict[str, Any]) -> None:
        order.append(node["name"])
        for child in node.get("children", []):
            visit(child)

    visit(tree)
    return order


def is_even(n: int) -> bool:
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n: int) -> bool:
    if n == 0:
        return False
    return is_even(n - 1)


def factorial_with_explicit_stack(n: int) -> int:
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    stack: list[int] = []
    while n > 1:
        stack.append(n)
        n -= 1
    result = 1
    while stack:
        result *= stack.pop()
    return result


def factorial_tail_like(n: int, acc: int = 1) -> int:
    if n < 0:
        raise ValueError("factorial is undefined for negative integers")
    if n in (0, 1):
        return acc
    return factorial_tail_like(n - 1, acc * n)


def run_tests() -> None:
    for i, v in [(0, 1), (1, 1), (5, 120), (6, 720)]:
        assert factorial_recursive(i) == v
        assert factorial_iterative(i) == v
        assert factorial_with_explicit_stack(i) == v

    fib_expected = [0, 1, 1, 2, 3, 5, 8, 13]
    for n, v in enumerate(fib_expected):
        assert fib_recursive_memo(n) == v
        assert fib_bottom_up(n) == v

    perms_123 = permutations_backtracking([1, 2, 3])
    assert sorted(perms_123) == sorted([
        [1, 2, 3], [1, 3, 2], [2, 1, 3],
        [2, 3, 1], [3, 1, 2], [3, 2, 1]
    ])

    assert len(n_queens(4)) == 2

    assert is_even(0) and not is_odd(0)
    assert is_even(10) and is_odd(11)

    tree = {"name": "A", "children": [{"name": "B"}, {"name": "C"}]}
    assert dfs_tree(tree) == ["A", "B", "C"]


if __name__ == "__main__":
    run_tests()
