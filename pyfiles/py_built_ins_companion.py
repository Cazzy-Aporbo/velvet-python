# cazandra_builtins_companion.py
# -----------------------------------------------------------------------------
# Title: Python Built-ins — Field Guide from Basics to Mastery
# Author: Cazandra Aporbo
# Started: December 2022
# Updated: March 2025
# Intent: A single, high‑level Python file that walks through a broad set of
#         Python built-ins in my teaching style — line‑by‑line commentary,
#         practical patterns, and safe demos you can run locally.
# Promise: No emojis. No AI-speak. Just me, Cazandra, explaining how I use these.
# Notes: Standard library only; examples are safe and side‑effect aware.
# -----------------------------------------------------------------------------

from typing import Any, Dict  # pared imports down to only what is actually used
import io    # for safe in‑memory file examples with open()

def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)

# --- abs, all, any, ascii, bin, bool -------------------------------------------------

def demo_core_truthiness() -> None:
    section("abs, all, any, ascii, bin, bool")
    print("abs(-8) ->", abs(-8))
    print("abs(-2.5) ->", abs(-2.5))
    print("abs(3-4j) (complex magnitude) ->", abs(3-4j))
    print("all([1, True, 'x']) ->", all([1, True, 'x']))
    print("all([1, 0, 'x']) ->", all([1, 0, 'x']))
    print("any([0, '', None, 5]) ->", any([0, '', None, 5]))
    print("any([]) ->", any([]))
    foreign = "café"
    print("ascii('café') ->", ascii(foreign))
    print("bin(13) ->", bin(13))
    print("bool([]) (empty is False) ->", bool([]))
    print("bool([0]) (non‑empty is True) ->", bool([0]))

# --- bytearray, bytes, callable, chr --------------------------------------------------

def demo_bytes_and_callables() -> None:
    section("bytearray, bytes, callable, chr")
    b = bytes([65, 66, 67])
    print("bytes([65,66,67]) ->", b)
    ba = bytearray(b)
    ba[0] = 97
    print("bytearray mutated ->", ba)
    def f(x):
        return x * 2
    print("callable(f) ->", callable(f))
    print("callable(42) ->", callable(42))
    print("chr(9731) (snowman) ->", chr(9731))

# --- classmethod, staticmethod, property, super --------------------------------------

def demo_method_kinds() -> None:
    section("classmethod, staticmethod, property, super")
    class Counter:
        _count = 0
        def __init__(self) -> None:
            self._n = 0
            Counter._count += 1
        @property
        def n(self) -> int:
            return self._n
        @n.setter
        def n(self, value: int) -> None:
            if value < 0:
                raise ValueError("n cannot be negative")
            self._n = value
        @staticmethod
        def twice(x: int) -> int:
            return x * 2
        @classmethod
        def how_many(cls) -> int:
            return cls._count
    c1, c2 = Counter(), Counter()
    print("Counter.how_many() ->", Counter.how_many())
    print("Counter.twice(5) ->", Counter.twice(5))
    c1.n = 7
    print("c1.n (via property) ->", c1.n)
    class Base:
        def greet(self) -> str:
            return "hello"
    class Child(Base):
        def greet(self) -> str:
            return super().greet() + ", world"
    print("Child().greet() ->", Child().greet())

# --- compile, eval, exec --------------------------------------------------------------

def demo_dynamic_code() -> None:
    section("compile, eval, exec — used carefully")
    expr_src = "3 * (2 + 1)"
    expr_code = compile(expr_src, filename="<expr>", mode="eval")
    print("eval(compile(expr)) ->", eval(expr_code))
    print("eval('1+2+3') ->", eval("1+2+3"))
    ns: Dict[str, Any] = {}
    exec("def add(a,b):\n    return a+b", ns, ns)
    print("exec-defined add(2,3) ->", ns["add"](2, 3))

# --- complex, dict, delattr, dir, divmod, enumerate ----------------------------------

def demo_collections_and_introspection() -> None:
    section("complex, dict, delattr, dir, divmod, enumerate")
    z = complex(2, -3)
    print("complex(2, -3) ->", z)
    d = dict(a=1, b=2)
    print("dict(a=1,b=2) ->", d)
    print("'__len__' in dir(d)? ->", "__len__" in dir(d))
    q, r = divmod(17, 5)
    print("divmod(17,5) ->", (q, r))
    for i, ch in enumerate("abc", start=1):
        print("enumerate ->", i, ch)
    class Tmp:
        def __init__(self):
            self.x = 10  # instance attribute
    t = Tmp()
    print("hasattr(t,'x') before ->", hasattr(t, 'x'))
    # Safely delete the instance attribute
    delattr(t, 'x')
    print("hasattr(t,'x') after  ->", hasattr(t, 'x'))
    # Demonstrate class attribute deletion
    Tmp.z = 99
    print("hasattr(Tmp,'z') before ->", hasattr(Tmp, 'z'))
    delattr(Tmp, 'z')
    print("hasattr(Tmp,'z') after  ->", hasattr(Tmp, 'z'))

# --- filter, map, frozenset, getattr, globals, hasattr, hash, help --------------------

def demo_higher_order_and_meta() -> None:
    section("filter, map, frozenset, getattr, globals, hasattr, hash, help")
    odds = list(filter(lambda n: n % 2, range(6)))
    print("filter odds 0..5 ->", odds)
    doubled = list(map(lambda n: n * 2, odds))
    print("map doubled ->", doubled)
    fs = frozenset({1, 2, 2, 3})
    print("frozenset({1,2,2,3}) ->", fs)
    class Box:
        value = 42
    bx = Box()
    print("getattr(bx,'value') ->", getattr(bx, 'value'))
    print("getattr(bx,'missing', 'fallback') ->", getattr(bx, 'missing', 'fallback'))
    print("'demo_core_truthiness' in globals()? ->", 'demo_core_truthiness' in globals())
    print("hasattr(bx,'value') ->", hasattr(bx, 'value'))
    print("hash('key') ->", hash('key'))
    print("Use help(str) or help('modules') in an interactive session.")

# --- hex, id, input, int, isinstance, issubclass, iter, len, list, locals -------------

def demo_types_and_introspection2() -> None:
    section("hex, id, input, int, isinstance, issubclass, iter, len, list, locals")
    print("hex(255) ->", hex(255))
    x = []
    print("id(x) ->", id(x))
    prompt = "Favorite number? "
    print("input(prompt) would show:", repr(prompt))
    pretend = "7"
    print("pretend user typed ->", pretend)
    print("int('7') ->", int(pretend))
    print("isinstance(3, int) ->", isinstance(3, int))
    class A: pass
    class B(A): pass
    print("issubclass(B, A) ->", issubclass(B, A))
    it = iter([10, 20, 30])
    print("next(iter) ->", next(it))
    print("next(iter) ->", next(it))
    print("len('abc') ->", len('abc'))
    print("list('abc') ->", list('abc'))
    here = locals()
    print("'pretend' in locals()? ->", 'pretend' in here)

# --- max, memoryview, min, next, object, oct, open, ord, pow, print -------------------

def demo_data_access_and_math() -> None:
    section("max, memoryview, min, next, object, oct, open, ord, pow, print")
    data = [("alice", 3), ("bob", 5), ("carol", 4)]
    print("max by score ->", max(data, key=lambda t: t[1]))
    print("min by name  ->", min(data, key=lambda t: t[0]))
    buf = bytearray(b"abcdef")
    mv = memoryview(buf)
    print("memoryview slice ->", bytes(mv[2:5]))
    it = iter([])
    print("next(iter([]), 'empty') ->", next(it, 'empty'))
    sentinel = object()
    print("object() is unique each time? ->", sentinel is not object())
    print("oct(64) ->", oct(64))
    with io.StringIO() as f:
        f.write("hello\nworld\n")
        f.seek(0)
        print("read from StringIO ->", f.read().strip().splitlines())
    print("ord('A') ->", ord('A'))
    print("pow(2, 10) ->", pow(2, 10))
    print("pow(7, 3, 5) ->", pow(7, 3, 5))
    print("x", "y", "z", sep=", ", end=".\n")

# --- range, repr, reversed, round, set, setattr, slice, sorted ------------------------

def demo_sequences_and_ordering() -> None:
    section("range, repr, reversed, round, set, setattr, slice, sorted")
    print("list(range(3)) ->", list(range(3)))
    sample = "line\nwith\ttabs"
    print("repr(sample) ->", repr(sample))
    print("list(reversed([1,2,3])) ->", list(reversed([1, 2, 3])))
    print("round(2.5) ->", round(2.5))
    print("round(3.5) ->", round(3.5))
    print("round(3.14159, 2) ->", round(3.14159, 2))
    print("set('mississippi') ->", set('mississippi'))
    class Bag: pass
    bag = Bag()
    setattr(bag, 'color', 'mint')
    print("bag.color via setattr ->", bag.color)
    s = slice(1, 5, 2)
    print("'abcdef'[s] ->", "abcdef"[s])
    print("sorted([3,1,2]) ->", sorted([3, 1, 2]))
    words = ["Alpha", "beta", "Gamma"]
    print("sorted casefold ->", sorted(words, key=str.casefold))

# --- str, sum, tuple, type, vars, zip -------------------------------------------------

def demo_text_numbers_and_zipping() -> None:
    section("str, sum, tuple, type, vars, zip")
    print("str({'a':1}) ->", str({'a': 1}))
    print("sum([1,2,3]) ->", sum([1, 2, 3]))
    print("sum([[1],[2]], start=[]) ->", sum([[1], [2]], start=[]))
    print("tuple('abc') ->", tuple('abc'))
    print("type(3.14) ->", type(3.14))
    class Thing:
        def __init__(self) -> None:
            self.x = 10
            self.y = 20
    th = Thing()
    print("vars(th) ->", vars(th))
    names = ["aya", "bo", "cy"]
    scores = [9, 8, 10]
    print("list(zip(names,scores)) ->", list(zip(names, scores)))

# --- Putting it all together ----------------------------------------------------------

def run_all() -> None:
    demo_core_truthiness()
    demo_bytes_and_callables()
    demo_method_kinds()
    demo_dynamic_code()
    demo_collections_and_introspection()
    demo_higher_order_and_meta()
    demo_types_and_introspection2()
    demo_data_access_and_math()
    demo_sequences_and_ordering()
    demo_text_numbers_and_zipping()

if __name__ == "__main__":
    run_all()
