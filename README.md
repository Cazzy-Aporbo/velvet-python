<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,16,18,20,22&height=220&section=header&text=Velvet%20Python&fontSize=80&animation=fadeIn&fontAlignY=35&desc=Where%20curiosity%20becomes%20code&descAlignY=56&descSize=22&fontColor=8B7D8B" />

<br>

[![CI](https://img.shields.io/github/actions/workflow/status/Cazzy-Aporbo/velvet-python/ci.yml?style=for-the-badge&label=CI&color=EDE5FF&labelColor=706B70)](https://github.com/Cazzy-Aporbo/velvet-python/actions)
&nbsp;
[![Python](https://img.shields.io/badge/Python-3.10+-FFE4E1?style=for-the-badge&logo=python&logoColor=706B70)](https://www.python.org/)
&nbsp;
[![License](https://img.shields.io/badge/License-MIT-F0E6FF?style=for-the-badge)](LICENSE)
&nbsp;
[![Last Commit](https://img.shields.io/github/last-commit/Cazzy-Aporbo/velvet-python?style=for-the-badge&color=FFEFD5&labelColor=706B70)](https://github.com/Cazzy-Aporbo/velvet-python/commits)

<br>

*A personal Python laboratory. Classification algorithms built from first principles.*
*Recursive thinking explored six ways. Sankey diagrams from real data.*
*Games that teach chemistry through electron physics.*
*Everything here runs.*

</div>

<br>

## The Idea

Most Python repositories are either tutorials that simplify past the point of usefulness, or production code that's impenetrable without three months of onboarding. This is neither.

Every module in this repo exists because I needed to understand something deeply enough to build it from scratch — no frameworks hiding the math, no black-box imports replacing the thinking. When I implement Naive Bayes, the Bayesian math is in the docstring. When I write a TF-IDF classifier, the vector algebra is visible in every method. When I compute Fibonacci, I do it six different ways and time them all.

Started January 2025. Still going.

<br>

## Architecture

<div align="center">

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#FFE4E1', 'primaryBorderColor':'#8B7D8B', 'primaryTextColor':'#4A4A4A', 'lineColor':'#DDA0DD', 'fontFamily':'Georgia, serif'}}}%%

graph LR
    subgraph Core ["src/"]
        A[ai.py<br/><i>3 classifiers</i>] --> B[ml_pipeline.py<br/><i>TF-IDF + FreqModel</i>]
        B --> C[data_utils.py<br/><i>load · split · validate</i>]
    end

    subgraph Tests ["tests/"]
        D[Unit Tests] --> E[Benchmarks]
        E --> F[Integration]
    end

    subgraph Lab ["pyfiles/"]
        G[recursion_masterclass.py<br/><i>6 approaches</i>]
        H[variable_atlas.py<br/><i>stats → Python bridge</i>]
        I[python_lessons_suite.py<br/><i>complete builtins tour</i>]
    end

    subgraph Games ["games/"]
        J[Molecular Cascade<br/><i>electron physics</i>]
        K[Backprop Adventure<br/><i>neural net learning</i>]
        L[Quantum Navigator<br/><i>maze algorithms</i>]
    end

    subgraph Tools ["scripts/"]
        M[Environment Builders]
        N[Import Encyclopedia]
        O[Sankey Visualizations]
    end

    Core --> Tests
    Lab --> Core
    Games --> Lab
    Tools --> Core

    style Core fill:#FFE4E1,stroke:#8B7D8B,stroke-width:2px
    style Tests fill:#EDE5FF,stroke:#8B7D8B,stroke-width:2px
    style Lab fill:#F0E6FF,stroke:#8B7D8B,stroke-width:2px
    style Games fill:#FFF0F5,stroke:#8B7D8B,stroke-width:2px
    style Tools fill:#FFEFD5,stroke:#8B7D8B,stroke-width:2px
```

</div>

<br>

## What Lives Here

<div align="center">
<table>
<tr>
<td width="50%" valign="top">

### `src/` — The Core Library

**Three classification strategies in `ai.py`:**

| Strategy | Math | Lines |
|:---------|:-----|------:|
| Rule-based | Pattern matching | ~15 |
| Naive Bayes | P(c\|text) ∝ Π P(w\|c) · P(c) | ~70 |
| Cosine Similarity | cos(θ) = A·B / (\|\|A\|\| · \|\|B\|\|) | ~60 |

**Two ML pipelines in `ml_pipeline.py`:**

| Model | Approach | Key Insight |
|:------|:---------|:------------|
| WordFrequencyModel | Token overlap scoring | Baseline — fast, transparent |
| TFIDFModel | TF·IDF weighted cosine | Rare words matter most |

Plus: `confusion_matrix()`, `evaluate()`, `train_test_split()` — all from scratch, no sklearn.

</td>
<td width="50%" valign="top">

### `pyfiles/` — The Sketchbook

**`recursion_masterclass.py`** — Factorial and Fibonacci computed six ways:
- Naive recursion (O(2ⁿ) — feel the pain)
- Memoized with `@lru_cache`
- Bottom-up dynamic programming
- Explicit stack (no recursion at all)
- Tail-call style accumulator
- Mutual recursion (`is_even` ↔ `is_odd`)

Plus: N-Queens backtracking, permutation generation, DFS tree traversal. Every function has inline tests.

**`variable_atlas.py`** — Variables as contracts, not boxes. Maps statistical notation (y, X, β) to Python with dataclasses, validation, and a complete OLS regression built on the standard library.

**`python_lessons_suite.py`** — A 493-line teaching program that tours Python's builtins with live demonstrations.

</td>
</tr>
</table>
</div>

<div align="center">
<table>
<tr>
<td width="50%" valign="top">

### `scripts/` — Standalone Tools

- **Environment Builders** — Audit your Python installation, generate reproducible setups, test package compatibility
- **Import Encyclopedia** — Map every module in Python's standard library
- **Sankey Toolkit** — Plotly-based flow visualizations with Titanic passenger data
- **JH Presentation** — A talk I gave at Johns Hopkins
- **Games** — Backpropagation adventure, pattern recognition challenges, recursive thinking puzzles, chaotic space navigation

</td>
<td width="50%" valign="top">

### `data/` — Real Datasets

| Domain | Contents |
|:-------|:---------|
| Health | Patient records, breast cancer research, supplement studies |
| NLP | Doctor-patient dialogue corpora |
| Finance | Time series, market data |
| Epidemiology | Our World in Data exports |
| Demographics | Census data, fisherman mercury studies |
| Transport | Airline passenger datasets |

Nothing synthetic unless explicitly labeled.

</td>
</tr>
</table>
</div>

<div align="center">
<table>
<tr>
<td width="50%" valign="top">

### `pyfiles/games/` — Learning Through Play

| Game | What It Teaches |
|:-----|:----------------|
| Molecular Cascade | Electron shells, valence bonds, chain reactions — via Pygame |
| Cardiovascular Model | Hemodynamics simulation |
| Quantum Navigator | Maze solving with quantum-inspired algorithms |
| Crystal Engine | Lattice structures and symmetry |
| Aerospace Systems | Orbital mechanics fundamentals |

</td>
<td width="50%" valign="top">

### `tests/` — The Proof

```
tests/
├── test_ai.py                  # 3 classifiers, parametrized
├── test_ml_pipeline.py         # Both models + confusion matrix
├── test_pipeline_integration.py # train → split → evaluate
├── test_benchmark.py           # Speed gates
├── test_benchmark_performance.py
└── test_example.py             # Smoke tests
```

Every `src/` function has a corresponding test.
Benchmarks enforce sub-100ms training and sub-500ms for 9000 predictions.

</td>
</tr>
</table>
</div>

<br>

## Quick Start

```bash
git clone https://github.com/Cazzy-Aporbo/velvet-python.git
cd velvet-python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

Or with the Makefile:

```bash
make install    # pip install -r requirements.txt
make test       # pytest tests/ -v --tb=short
make lint       # ruff check src/ tests/
make check      # lint + test in one pass
make clean      # remove __pycache__, .pytest_cache, etc.
```

<br>

## Code Samples

<div align="center">

### Naive Bayes — The Math Behind Spam Filters

</div>

```python
from src.ai import NaiveBayesClassifier

nb = NaiveBayesClassifier(alpha=1.0)  # Laplace smoothing

nb.train(
    texts=["great movie loved it", "terrible waste of time",
           "amazing performance", "boring and dull"],
    labels=["positive", "negative", "positive", "negative"],
)

# P(positive | "loved the movie") ∝ P("loved"|pos) · P("the"|pos) · P("movie"|pos) · P(pos)
nb.predict("loved the movie")           # → "positive"
nb.predict_proba("loved the movie")     # → {"positive": 0.83, "negative": 0.17}
```

<div align="center">

### TF-IDF — Why Rare Words Matter

</div>

```python
from src.ml_pipeline import TFIDFModel, evaluate
from src.data_utils import load_dataset, train_test_split

data = load_dataset()
train, test = train_test_split(data, test_ratio=0.3, seed=42)

model = TFIDFModel()
model.train(
    texts=[t for t, _ in train],
    labels=[l for _, l in train],
)

# TF("learning", doc) = count("learning") / len(doc)
# IDF("learning") = log(N / docs_containing("learning"))
# Weight = TF × IDF — common words get low weight, distinctive words get high weight

accuracy = evaluate(model, test)
```

<div align="center">

### Recursion — Six Ways to Compute the Same Thing

</div>

```python
# From pyfiles/recursion_masterclass.py

factorial_recursive(6)           # 720  — classic, hits recursion limit at ~1000
factorial_iterative(6)           # 720  — no stack overhead
factorial_with_explicit_stack(6) # 720  — recursion without recursion
factorial_tail_like(6)           # 720  — accumulator pattern (Python doesn't optimize TCO, but the pattern teaches)

fib_recursive_naive(30)          # 832040 — takes ~0.3s (O(2ⁿ) — feel it)
fib_recursive_memo(30)           # 832040 — takes ~0.00001s (O(n) with @lru_cache)
fib_bottom_up(30)                # 832040 — takes ~0.00001s (O(n), O(1) space)
```

<br>

## Philosophy

<div align="center">
<table>
<tr>
<td width="33%" align="center" style="padding: 20px;">

**Build the math first**

If I can't derive it on paper,
I don't trust it in code.
Every classifier in `src/ai.py`
has its formula in the docstring.

</td>
<td width="33%" align="center" style="padding: 20px;">

**Multiple approaches**

One solution means you memorized it.
Three solutions means you understand it.
Fibonacci six ways. Classification three ways.
The comparison *is* the lesson.

</td>
<td width="33%" align="center" style="padding: 20px;">

**Everything runs**

No pseudocode. No placeholder functions.
No `# TODO: implement this`.
If it's in this repo, `python file.py`
produces output.

</td>
</tr>
</table>
</div>

<br>

## Contributing

See [`contributions.md`](contributions.md). The short version: fork, branch, test, PR. Format with ruff. If you add code to `src/`, add a test to `tests/`.

## License

MIT (code) &nbsp;·&nbsp; CC-BY-4.0 (documentation and datasets)

<div align="center">

<br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,16,18,20,22&height=100&section=footer&animation=fadeIn&fontColor=8B7D8B" />

**Cazzy Aporbo** · Started January 2025

<a href="https://github.com/Cazzy-Aporbo/velvet-python/issues">
<img src="https://img.shields.io/badge/Questions-Open_Issue-FFE4E1?style=flat-square&labelColor=706B70" />
</a>
&nbsp;
<a href="https://github.com/Cazzy-Aporbo/velvet-python/stargazers">
<img src="https://img.shields.io/badge/Star-For_Updates-EDE5FF?style=flat-square&labelColor=706B70" />
</a>
&nbsp;
<a href="https://github.com/Cazzy-Aporbo/velvet-python/fork">
<img src="https://img.shields.io/badge/Fork-Contribute-F0E6FF?style=flat-square&labelColor=706B70" />
</a>

</div>
