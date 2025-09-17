# _variable_atlas.py
# -----------------------------------------------------------------------------
# Title: Variable Atlas — Teaching Variables in Python by Building a Model
# Author: Cazandra Aporbo
# Started: February 9 2023
# Goal:    Every step is commented: what I do, why I do
#          it, and alternatives I considered. The goal is to teach "variables"
#          as living objects: names, shapes, contracts, and roles in a model.
# Design: Learning to connect variable names you see in statistics (y, X, beta, ...)
#         to Python variables with strong habits (validation, clarity,
#         reproducibility). Multiple approaches are shown side-by-side.
#         The file runs end-to-end without any third-party installs.
# -----------------------------------------------------------------------------

from __future__ import annotations  # allow forward refs in type hints

# I keep to the standard library to make the script runnable anywhere.
# If NumPy is available, I demonstrate it as an option — but I never require it.
import math
import random
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as _np  # optional acceleration path
except Exception:  # if NumPy isn't available, we set a sentinel
    _np = None

# -----------------------------------------------------------------------------
# Console formatting helpers — so the program reads like a guided lesson.
# -----------------------------------------------------------------------------

def block(title: str) -> None:
    print("\n" + "=" * 78)  # visual separator for sections
    print(title)
    print("-" * 78)


def say(text: str, width: int = 78) -> None:
    print("\n".join(textwrap.wrap(text, width=width)))

# -----------------------------------------------------------------------------
# PART 1 — Variables as *Contracts*: names + shapes + meaning (not just boxes)
# -----------------------------------------------------------------------------
# Many beginners treat variables as boxes. I treat them as contracts:
#   • a name (so humans can talk about it),
#   • a shape (so code can validate it),
#   • a meaning (so the next reader knows what it stands for),
#   • and a value (the actual data right now).
# You can formalize this with a small dataclass.

@dataclass
class VarSpec:
    latex: str                 # pretty math name, e.g., "y" or "X^T X"
    pyname: str                # Python identifier where it lives, e.g., "y"
    meaning: str               # a one-line human description
    shape: str                 # human-friendly shape description, e.g., "(n, )"
    value: Any = None          # the value bound right now
    notes: str = ""            # optional nuance: units, caveats, etc.

    def bind(self, value: Any) -> "VarSpec":
        # I return self so you can chain .bind() while building the table.
        self.value = value
        return self

    def show(self) -> None:
        # I keep the display compact so the table fits the terminal.
        print(f"{self.latex:<12} ← {self.pyname:<12} | shape {self.shape:<8} | {self.meaning}")
        if self.notes:
            say("  notes: " + self.notes)

# -----------------------------------------------------------------------------
# PART 2 — A tiny linear-algebra core (pure Python) so we can compute β two ways
# -----------------------------------------------------------------------------
# I write a minimal matrix toolkit to avoid external dependencies.
# It is not meant to be the fastest; it's meant to be readable.

Matrix = List[List[float]]
Vector = List[float]


def mt(rows: int, cols: int, fill: float = 0.0) -> Matrix:
    """Allocate a rows×cols matrix filled with a constant."""
    return [[fill for _ in range(cols)] for _ in range(rows)]


def transpose(A: Matrix) -> Matrix:
    return [list(row) for row in zip(*A)]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    # I check dimensions explicitly to teach the habit.
    ra, ca = len(A), len(A[0])
    rb, cb = len(B), len(B[0])
    assert ca == rb, "inner dimensions must match for A@B"
    C = mt(ra, cb)
    for i in range(ra):
        for k in range(ca):
            aik = A[i][k]
            for j in range(cb):
                C[i][j] += aik * B[k][j]
    return C


def matvec(A: Matrix, x: Vector) -> Vector:
    assert len(A[0]) == len(x)
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in A]


def dot(u: Vector, v: Vector) -> float:
    assert len(u) == len(v)
    return sum(ui * vi for ui, vi in zip(u, v))


def identity(n: int) -> Matrix:
    I = mt(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def invert(A: Matrix) -> Matrix:
    """Gauss–Jordan inversion with partial pivoting (educational, small n)."""
    n = len(A)
    assert all(len(row) == n for row in A), "A must be square"
    # I copy A so I don't surprise callers by mutating their data.
    M = [row[:] for row in A]
    Inv = identity(n)
    for col in range(n):
        # Pivot: pick the largest absolute value in/under this row to improve stability.
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix: columns are linearly dependent")
        # Swap the pivot row to the current position.
        M[col], M[pivot] = M[pivot], M[col]
        Inv[col], Inv[pivot] = Inv[pivot], Inv[col]
        # Scale the pivot row to make the pivot 1.
        piv = M[col][col]
        M[col] = [v / piv for v in M[col]]
        Inv[col] = [v / piv for v in Inv[col]]
        # Eliminate all other entries in this column.
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor != 0.0:
                M[r] = [rv - factor * cv for rv, cv in zip(M[r], M[col])]
                Inv[r] = [rv - factor * cv for rv, cv in zip(Inv[r], Inv[col])]
    return Inv


def ridge_inverse(XTX: Matrix, lam: float) -> Matrix:
    """Compute (X^T X + λ I)^-1 in pure Python."""
    n = len(XTX)
    A = [row[:] for row in XTX]
    for i in range(n):
        A[i][i] += lam
    return invert(A)

# -----------------------------------------------------------------------------
# PART 3 — A dataset factory: builds y, X with an intercept column 1_n
# -----------------------------------------------------------------------------

@dataclass
class LinearDataset:
    """A tiny dataset class to keep variables together and validated."""
    y: Vector                     # outcome vector (n,)
    X_no_intercept: Matrix        # predictors WITHOUT the intercept (n, p0)
    add_intercept: bool = True    # whether to prepend a ones column

    # I compute these in __post_init__ so they're always coherent.
    n: int = field(init=False)    # number of rows
    p: int = field(init=False)    # number of columns *including* intercept if used
    X: Matrix = field(init=False) # design matrix actually used in modeling

    def __post_init__(self) -> None:
        # Validate shapes first.
        assert len(self.y) > 0, "empty y"
        assert len(self.y) == len(self.X_no_intercept), "y and X row counts differ"
        assert all(len(row) == len(self.X_no_intercept[0]) for row in self.X_no_intercept), "ragged X"
        self.n = len(self.y)
        # Build X by possibly adding a 1_n intercept column.
        if self.add_intercept:
            ones = [[1.0] for _ in range(self.n)]
            self.X = [oi + row for oi, row in zip(ones, self.X_no_intercept)]
        else:
            self.X = [row[:] for row in self.X_no_intercept]
        self.p = len(self.X[0])  # number of columns after intercept decision

    def column(self, j: int) -> Vector:
        # X_j accessor to match the math symbol X_j
        return [row[j] for row in self.X]

# -----------------------------------------------------------------------------
# PART 4 — Two paths to β: (A) pure Python normal equations, (B) NumPy (optional)
# -----------------------------------------------------------------------------

@dataclass
class OLSResult:
    beta: Vector
    fitted: Vector
    residuals: Vector
    sigma2: float
    r2: float
    aic: float
    bic: float
    se_beta: Vector
    loglik: float


def ols_pure_python(ds: LinearDataset, ridge_lambda: float = 0.0) -> OLSResult:
    """Compute OLS (or ridge if λ>0) with pure-Python linear algebra.
    Why this path: It shows exactly how variables y, X, X^T X, (X^T X)^-1, etc.,
    are constructed and used — no magic boxes.
    """
    # 1) Build the key matrices from variables.
    X = ds.X                                   # (n, p)
    y = ds.y                                   # (n,)
    XT = transpose(X)                          # X^T
    XTX = matmul(XT, X)                        # X^T X
    XTy = [dot(col, y) for col in XT]          # X^T y (as a vector)

    # 2) Invert either X^T X or X^T X + λI for ridge.
    if ridge_lambda != 0.0:
        Inv = ridge_inverse(XTX, ridge_lambda)
    else:
        Inv = invert(XTX)

    # 3) β = Inv · X^T y
    beta = matvec(Inv, XTy)

    # 4) Fitted values and residuals
    fitted = matvec(X, beta)
    resid = [yi - fi for yi, fi in zip(y, fitted)]  # y - Xβ

    # 5) Variance σ^2 = (1/n) * residual' residual  (MLE)
    rss = dot(resid, resid)
    sigma2 = rss / ds.n

    # 6) R^2 = 1 - RSS/TSS
    ybar = sum(y) / ds.n
    tss = sum((yi - ybar) ** 2 for yi in y)
    r2 = 1.0 - (rss / tss if tss > 0 else 0.0)

    # 7) Gaussian log-likelihood at MLE sigma^2
    loglik = -0.5 * ds.n * (math.log(2 * math.pi * sigma2) + 1.0)

    # 8) AIC/BIC (count parameters k = p + 1 for σ^2)
    k = ds.p + 1
    aic = 2 * k - 2 * loglik
    bic = math.log(ds.n) * k - 2 * loglik

    # 9) Standard errors: sqrt(σ^2 * diag((X^T X)^-1))
    diag = [Inv[i][i] for i in range(ds.p)]
    se_beta = [math.sqrt(max(0.0, sigma2 * d)) for d in diag]

    return OLSResult(beta, fitted, resid, sigma2, r2, aic, bic, se_beta, loglik)


def ols_numpy(ds: LinearDataset, ridge_lambda: float = 0.0) -> Optional[OLSResult]:
    """Same calculation using NumPy if available. Why include it: to contrast
    ergonomics and performance with the pure-Python path.
    """
    if _np is None:
        return None
    X = _np.array(ds.X, dtype=float)
    y = _np.array(ds.y, dtype=float)
    XT = X.T
    XTX = XT @ X
    if ridge_lambda != 0.0:
        XTX = XTX + ridge_lambda * _np.eye(XTX.shape[0])
    Inv = _np.linalg.inv(XTX)
    beta = Inv @ (XT @ y)
    fitted = X @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    sigma2 = rss / ds.n
    ybar = float(_np.mean(y))
    tss = float(((y - ybar) ** 2).sum())
    r2 = 1.0 - (rss / tss if tss > 0 else 0.0)
    loglik = -0.5 * ds.n * (math.log(2 * math.pi * sigma2) + 1.0)
    k = ds.p + 1
    aic = 2 * k - 2 * loglik
    bic = math.log(ds.n) * k - 2 * loglik
    se_beta = list(_np.sqrt(_np.clip(_np.diag(Inv) * sigma2, 0.0, _np.inf)))
    return OLSResult(list(beta), list(fitted), list(resid), sigma2, r2, aic, bic, se_beta, loglik)

# -----------------------------------------------------------------------------
# PART 5 — Gradient descent: a third path to β (to contrast with closed form)
# -----------------------------------------------------------------------------
# Why: variables aren't just values; they evolve by update rules. I show a very
# small gradient descent for OLS to highlight β as an evolving variable with
# learning rate and stopping criteria. Educational, not production.


def ols_gradient_descent(ds: LinearDataset, lr: float = 0.01, iters: int = 2000) -> Vector:
    # Initialize β with zeros. I choose zeros for clarity; random init is another option.
    beta = [0.0] * ds.p
    X = ds.X
    y = ds.y
    n = ds.n
    for t in range(iters):
        # Gradient of (1/2n)||y - Xβ||^2 is  -(1/n) X^T (y - Xβ)
        pred = matvec(X, beta)
        resid = [yi - pi for yi, pi in zip(y, pred)]
        # Compute gradient g = -(1/n) X^T resid
        XT = transpose(X)
        g = [-(1.0 / n) * dot(col, resid) for col in XT]
        # Update rule: β ← β - lr * g
        beta = [bi - lr * gi for bi, gi in zip(beta, g)]
        # Optional: stop early if gradient is tiny
        if max(abs(gi) for gi in g) < 1e-9:
            break
    return beta

# -----------------------------------------------------------------------------
# PART 6 — Build a tiny synthetic dataset so every symbol in the brief appears
# -----------------------------------------------------------------------------


def make_toy_dataset(n: int = 20, seed: int = 7) -> LinearDataset:
    random.seed(seed)  # deterministic so examples reproduce exactly
    # I simulate y = β0 + β1*x1 + β2*x2 + ε, with small noise.
    beta_true = [2.0, 3.0, -1.5]  # intercept, slope1, slope2
    X_no_intercept: Matrix = []
    y: Vector = []
    for _ in range(n):
        x1 = random.uniform(-2, 2)
        x2 = random.uniform(-1, 3)
        noise = random.gauss(0.0, 0.3)
        y.append(beta_true[0] + beta_true[1] * x1 + beta_true[2] * x2 + noise)
        X_no_intercept.append([x1, x2])
    return LinearDataset(y=y, X_no_intercept=X_no_intercept, add_intercept=True)

# -----------------------------------------------------------------------------
# PART 7 — A variable ledger: show each symbol with its Python face and role
# -----------------------------------------------------------------------------


def show_variable_ledger(ds: LinearDataset, res: OLSResult) -> None:
    block("Variable Ledger — mapping math symbols to Python variables")
    # I compute a few derived variables so the ledger can reference them.
    y = ds.y
    X = ds.X
    beta = res.beta
    fitted = res.fitted
    resid = res.residuals
    n, p = ds.n, ds.p
    XT = transpose(X)
    XTX = matmul(XT, X)
    XTy = [dot(col, y) for col in XT]
    XtX_beta = matvec(XTX, beta)

    ledger: List[VarSpec] = [
        VarSpec("y", "y", "Outcome vector (dependent variable)", "(n,)", y, "Often a list/array.").
            bind(y),
        VarSpec("X", "X", "Design matrix with intercept column 1_n first", "(n,p)", X).
            bind(X),
        VarSpec("beta", "beta", "Coefficient vector including intercept", "(p,)", beta).
            bind(beta),
        VarSpec("epsilon", "residuals", "Errors/residuals y - Xβ", "(n,)", resid).
            bind(resid),
        VarSpec("sigma^2", "sigma2", "Variance of errors (MLE)", "scalar", res.sigma2).
            bind(res.sigma2),
        VarSpec("X_i", "X[:,i]", "i-th explanatory variable (column)", "(n,)", ds.column(1), "Example: i=1 here.").
            bind(ds.column(1)),
        VarSpec("beta_i", "beta[i]", "Coefficient for variable i", "scalar", beta[1]).
            bind(beta[1]),
        VarSpec("n", "n", "Sample size (rows)", "scalar", n).bind(n),
        VarSpec("p", "p", "Number of predictors incl. intercept", "scalar", p).bind(p),
        VarSpec("X^T X", "XTX", "Cross-product matrix", "(p,p)", XTX).bind(XTX),
        VarSpec("X^T y", "XTy", "Predictor–outcome cross term", "(p,)", XTy).bind(XTy),
        VarSpec("(X^T X)^{-1}", "Inv", "Inverse of cross-product", "(p,p)", "see below").
            bind("computed inside OLS"),
        VarSpec("Xβ", "fitted", "Predicted values", "(n,)", fitted).bind(fitted),
        VarSpec("y - Xβ", "residuals", "Residual vector", "(n,)", resid).bind(resid),
        VarSpec("X_j", "column(j)", "Individual column of X", "(n,)", ds.column(2)).bind(ds.column(2)),
        VarSpec("1_n", "ones", "Intercept column of ones", "(n,1)", [1.0]*n).bind([1.0]*n),
        VarSpec("X_new", "X_new", "New data for prediction", "(m,p)", "see demo").
            bind("constructed below"),
        VarSpec("y_new", "y_new", "Predictions for new data", "(m,)", "see demo").
            bind("constructed below"),
        VarSpec("R^2", "r2", "Coefficient of determination", "scalar", res.r2).bind(res.r2),
        VarSpec("AIC", "aic", "Akaike information criterion", "scalar", res.aic).bind(res.aic),
        VarSpec("BIC", "bic", "Bayesian information criterion", "scalar", res.bic).bind(res.bic),
        VarSpec("σ̂(β)", "se_beta", "Std. error of coefficients", "(p,)", res.se_beta).bind(res.se_beta),
        VarSpec("X^T", "XT", "Transpose of X", "(p,n)", XT).bind(XT),
        VarSpec("y^T", "yT", "Transpose of y (viewed as row)", "(1,n)", [y]).bind([y]),
        VarSpec("I", "identity(p)", "Identity matrix", "(p,p)", identity(p)).bind(identity(p)),
        VarSpec("λ", "ridge_lambda", "Regularization strength for Ridge", "scalar", 0.0).bind(0.0),
        VarSpec("y^T y", "yTy", "Sum of squares of y", "scalar", dot(y, y)).bind(dot(y, y)),
        VarSpec("X^T X β", "XtX_beta", "Left-hand of normal equation", "(p,)", XtX_beta).bind(XtX_beta),
        VarSpec("Xβ - y", "neg_resid", "Alternative residual sign", "(n,)", [fi - yi for yi,fi in zip(y,fitted)]).
            bind([fi - yi for yi,fi in zip(y,fitted)]),
    ]
    for vs in ledger:
        vs.show()

# -----------------------------------------------------------------------------
# PART 8 — Lesson: variables, scope, mutability, and safer patterns
# -----------------------------------------------------------------------------

# I anchor the ML variables above with core Python variable behavior here, so
# "variable" means the same thing whether it's a scalar or a whole matrix.

def lesson_core_variable_habits() -> None:
    block("Variables 101 — binding, mutability, and scope")
    # Binding vs. copying: names point to objects; assignment rebinds the name.
    a = [1, 2]
    b = a                     # b points to the same list as a
    b.append(3)               # mutating via b affects a as well
    print("aliasing demo — a is", a, "; b is", b)

    # Defensive copy when you intend independence.
    c = a[:]                  # shallow copy (works for flat lists)
    c.append(4)
    print("after shallow copy — a is", a, "; c is", c)

    # Immutability keeps variables stable through sharing.
    t = (1, 2)
    # t[0] = 99  # would error; tuples are immutable
    print("immutable tuple t:", t)

    # Scope: local, enclosing, global; closures capture *variables*, not values.
    def maker():
        x = 10
        def inc():
            # nonlocal x  # uncomment to modify outer x; without it, x is read-only here
            return x + 1
        return inc
    f = maker()
    print("closure reads x from outer scope — f() ->", f())

# -----------------------------------------------------------------------------
# PART 9 — Demonstration driver tying it all together
# -----------------------------------------------------------------------------

def run_demo() -> None:
    block("Variable Atlas — building β three ways and reading every symbol")
    say("Step 1 — Build a synthetic dataset so y, X, β, ε are concrete.")
    ds = make_toy_dataset(n=25, seed=11)

    say("Step 2 — Compute β via normal equations (pure Python). Why: full clarity.")
    res_py = ols_pure_python(ds)

    say("Step 3 — Compute β via gradient descent. Why: show β as an evolving variable.")
    beta_gd = ols_gradient_descent(ds, lr=0.05, iters=1000)

    if _np is not None:
        say("Step 4 — Compute β via NumPy. Why: contrast ergonomics/performance.")
        res_np = ols_numpy(ds)
        assert res_np is not None
        # I sanity-check that all β estimates broadly agree.
        for b1, b2 in zip(res_py.beta, res_np.beta):
            assert abs(b1 - b2) < 1e-6
    else:
        res_np = None

    # Show the ledger with every symbol connected to a value.
    show_variable_ledger(ds, res_py)

    # New-data prediction demo: build X_new and compute y_new = X_new β
    block("Predict on new data — constructing X_new and y_new explicitly")
    X_new_no_intercept: Matrix = [[-0.2, 0.5], [1.0, -0.3]]
    ds_new = LinearDataset(y=[0.0, 0.0], X_no_intercept=X_new_no_intercept, add_intercept=True)
    y_new = matvec(ds_new.X, res_py.beta)
    print("X_new ->", ds_new.X)
    print("y_new = X_new β ->", y_new)

    # Ridge variant: demonstrate λ through the same variable pipeline.
    block("Ridge variant — β = (X^T X + λI)^{-1} X^T y (λ=1e-2)")
    res_ridge = ols_pure_python(ds, ridge_lambda=1e-2)
    print("beta (OLS)  ->", [round(v, 4) for v in res_py.beta])
    print("beta (Ridge)->", [round(v, 4) for v in res_ridge.beta])

    # Educational printout: compare all three β estimates.
    block("β estimates across methods (OLS, Ridge, Gradient Descent)")
    print("OLS β    ->", [round(v, 6) for v in res_py.beta])
    if res_np:
        print("NumPy β  ->", [round(v, 6) for v in res_np.beta])
    print("GD β     ->", [round(v, 6) for v in beta_gd])

    # Model quality variables
    block("Model diagnostics — σ^2, R^2, AIC, BIC, log-likelihood, SE(β)")
    print("sigma^2 ->", res_py.sigma2)
    print("R^2     ->", res_py.r2)
    print("AIC     ->", res_py.aic)
    print("BIC     ->", res_py.bic)
    print("SE(β)   ->", [round(v, 6) for v in res_py.se_beta])

    # Wrap with a short checklist of variable habits used above.
    block("Variable habits used above — why those choices")
    say("• Named variables after math (y, X, β) to reduce translation cost for readers.")
    say("• Kept a VarSpec ledger so each name has meaning/shape alongside value.")
    say("• Validated shapes at construction to fail early and teach good hygiene.")
    say("• Showed three β methods to teach equivalence of roles, not of tools.")
    say("• Avoided global state; passed variables explicitly to teach scope.")

# -----------------------------------------------------------------------------
# PART 10 — Lightweight tests (so the lesson guards itself)
# -----------------------------------------------------------------------------

def _tests() -> None:
    ds = make_toy_dataset(n=12, seed=3)
    res = ols_pure_python(ds)
    # Sanity: R^2 within [0,1]
    assert 0.0 <= res.r2 <= 1.0
    # Recompute via gradient descent and compare roughly
    beta_gd = ols_gradient_descent(ds, lr=0.05, iters=1500)
    for b1, b2 in zip(res.beta, beta_gd):
        assert abs(b1 - b2) < 0.2  # gradient descent is approximate here
    # Inversion correctness: A*(A^-1) ≈ I for a random SPD example
    X = [[1.0, 2.0], [2.0, 5.0]]
    Inv = invert(X)
    Prod = matmul(X, Inv)
    assert abs(Prod[0][0] - 1.0) < 1e-9 and abs(Prod[1][1] - 1.0) < 1e-9


if __name__ == "__main__":
    _tests()
    run_demo()
