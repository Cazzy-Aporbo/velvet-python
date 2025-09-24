"""
generate_500_advanced_tests.py

Generates 500 advanced Python tests with randomized inputs across multiple topics.
Tests are saved to `advanced_tests_generated.py` and ready for pytest.

Author: ChatGPT
"""

import os
import random
import string
import numpy as np
import pandas as pd

OUTPUT_FILE = "advanced_tests_generated.py"

# Helper functions to generate randomized test inputs
def random_list(n=5, low=1, high=100):
    return [random.randint(low, high) for _ in range(n)]

def random_string(length=6):
    return ''.join(random.choices(string.ascii_letters, k=length))

def random_matrix(rows=3, cols=3):
    return np.random.randint(0, 10, size=(rows, cols))

def random_dataframe(rows=3, cols=3):
    data = {f"col_{i}": np.random.randint(0, 100, rows) for i in range(cols)}
    return pd.DataFrame(data)

# Templates for advanced tests
templates = [
    # Algorithms
    ("Test sorting a list", "lst = {lst}; lst.sort(); assert lst == sorted(lst)"),
    ("Test reversing a list", "lst = {lst}; rev = lst[::-1]; assert rev == list(reversed(lst))"),
    ("Test sum of list", "lst = {lst}; assert sum(lst) == {sum_val}"),
    
    # Recursion
    ("Test factorial function", 
     "def fact(n): return 1 if n<=1 else n*fact(n-1); assert fact({n}) == {fact_val}"),
    
    # OOP
    ("Test class instantiation", "class A: def __init__(self, x): self.x=x; a=A({n}); assert a.x=={n}"),
    
    # Functional programming
    ("Test map function", "lst={lst}; res=list(map(lambda x:x*2,lst)); assert res == [x*2 for x in lst]"),
    ("Test filter function", "lst={lst}; res=list(filter(lambda x:x%2==0,lst)); assert res == [x for x in lst if x%2==0]"),
    
    # NumPy
    ("Test NumPy sum", "mat = np.array({matrix}); assert mat.sum() == {sum_val}"),
    
    # Pandas
    ("Test DataFrame sum", "df = pd.DataFrame({df}); assert df.sum().sum() == {sum_val}"),
    
    # Strings
    ("Test string uppercase", "s='{s}'; assert s.upper() == '{upper_s}'"),
    ("Test string reverse", "s='{s}'; assert s[::-1] == '{rev_s}'"),
    
    # Random / Math
    ("Test random integer in range", "x=random.randint(1,10); assert 1<=x<=10"),
    
    # Error handling
    ("Test ZeroDivisionError", "try: x=1/0; except ZeroDivisionError: x='error'; assert x=='error'"),
    
    # Decorators
    ("Test decorator doubling", 
     "def deco(f): return lambda: f()*2; @deco\ndef f(): return {n}; assert f()=={n2}"),
]

# Expand templates to 500
tests = []
while len(tests) < 500:
    t = random.choice(templates)
    desc = t[0]
    code_template = t[1]
    
    # Randomize placeholders
    lst = random_list()
    n = random.randint(2,6)
    sum_val = sum(lst)
    matrix = random_matrix().tolist()
    df = random_dataframe().to_dict(orient='list')
    s = random_string()
    upper_s = s.upper()
    rev_s = s[::-1]
    n2 = n*2
    
    code = code_template.format(lst=lst, sum_val=sum_val, n=n, matrix=matrix, df=df,
                                s=s, upper_s=upper_s, rev_s=rev_s, n2=n2)
    tests.append((desc, code))

# Write to output file
with open(OUTPUT_FILE, "w") as f:
    f.write("import pytest\n")
    f.write("import numpy as np\n")
    f.write("import pandas as pd\n")
    f.write("import random\n\n")
    for idx, (desc, code) in enumerate(tests, 1):
        func_name = f"test_{idx:03d}"
        f.write(f"def {func_name}():\n")
        f.write(f"    '''{desc}'''\n")
        for line in code.split("; "):
            f.write(f"    {line}\n")
        f.write("\n")

print(f"500 advanced Python tests generated in {OUTPUT_FILE}")