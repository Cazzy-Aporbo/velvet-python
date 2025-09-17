# multiple_regression_commentary.py  # file name that states purpose clearly
# ------------------------------------------------------------------------------------  
# Title: Multiple Regression — A Fully Commented, From-First-Principles Demonstration 
# Author: Cazandra Aporbo  
# Completed: August 2025  
# Promise: Every single line is commented: what the line does, why I'm doing it,    
#          and alternative choices you could make in real projects. No emojis.      
# Design Goals:                                                                       
#   • Show multiple regression with several *variable* types: numeric, indicators,    
#     nonlinear transforms, and an interaction.                                       
#   • Teach both the *math* (normal equations) and the *practice* (cross‑validation,  
#     model comparison by AIC/BIC, and a ridge variant for stability).                
#   • Depend only on NumPy; fall back to a tiny pure‑Python linear algebra path if   
#     NumPy is unavailable (small sizes only).                                        
#   • Zero file I/O and no user input; the script is fully reproducible.              
# ------------------------------------------------------------------------------------ 

from __future__ import annotations  # allows forward references in type hints for older Python

import math  # for log, pi in likelihood calculations
import random  # for deterministic dataset generation with seeds
from dataclasses import dataclass  # to keep return values structured and named
from typing import List, Tuple, Dict, Optional  # type hints for clarity

# Try importing NumPy for robust linear algebra; if not present, keep going.      # optional dependency
try:  # begin try block for optional import
    import numpy as np  # standard numerical array library
    HAS_NUMPY = True  # flag to branch implementations
except Exception:  # if NumPy is not available or import fails
    HAS_NUMPY = False  # set flag so we can use the pure‑Python path

# -----------------------------
# Pure‑Python linear algebra (tiny, educational — used only if NumPy missing)
# -----------------------------
# Notes: This is not optimized. It is here to make the demo runnable anywhere.    

Matrix = List[List[float]]  # alias for readability in pure‑Python code
Vector = List[float]  # alias for 1‑D arrays when NumPy is absent

def _pp_transpose(A: Matrix) -> Matrix:  # small helper: matrix transpose
    return [list(row) for row in zip(*A)]  # zip star trick transposes rows↔columns

def _pp_matmul(A: Matrix, B: Matrix) -> Matrix:  # naive O(n^3) matrix product
    ra, ca = len(A), len(A[0])  # rows and cols of A for dimension checks
    rb, cb = len(B), len(B[0])  # rows and cols of B for dimension checks
    assert ca == rb, "dimension mismatch in _pp_matmul"  # guard against misuse
    C: Matrix = [[0.0 for _ in range(cb)] for _ in range(ra)]  # allocate result
    for i in range(ra):  # iterate result rows
        for k in range(ca):  # iterate shared dimension
            aik = A[i][k]  # cache to reduce indexing overhead in inner loop
            for j in range(cb):  # iterate result columns
                C[i][j] += aik * B[k][j]  # accumulate dot product term
    return C  # return dense product

def _pp_matvec(A: Matrix, x: Vector) -> Vector:  # matrix–vector product
    assert len(A[0]) == len(x), "dimension mismatch in _pp_matvec"  # shape check
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in A]  # list comp

def _pp_identity(n: int) -> Matrix:  # identity matrix constructor
    I = [[0.0 for _ in range(n)] for _ in range(n)]  # start with zeros
    for i in range(n):  # diagonal loop
        I[i][i] = 1.0  # set ones on diagonal
    return I  # return identity

def _pp_invert(A: Matrix) -> Matrix:  # Gauss–Jordan inverse (educational)
    n = len(A)  # square dimension
    assert all(len(r) == n for r in A), "A must be square"  # validation
    M = [row[:] for row in A]  # copy to avoid mutating caller data
    Inv = _pp_identity(n)  # start with identity to build inverse alongside
    for col in range(n):  # pivot each column
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))  # partial pivot
        if abs(M[pivot][col]) < 1e-12:  # numerical singularity guard
            raise ValueError("singular matrix in _pp_invert")  # explain failure
        M[col], M[pivot] = M[pivot], M[col]  # swap rows in M
        Inv[col], Inv[pivot] = Inv[pivot], Inv[col]  # swap same rows in Inv
        piv = M[col][col]  # pivot value
        M[col] = [v / piv for v in M[col]]  # scale pivot row to pivot 1
        Inv[col] = [v / piv for v in Inv[col]]  # mirror scaling in Inv
        for r in range(n):  # eliminate this column in all other rows
            if r == col:  # skip pivot row
                continue  # continue to next row
            factor = M[r][col]  # multiplier to eliminate
            if factor != 0.0:  # skip unnecessary work
                M[r] = [rv - factor * cv for rv, cv in zip(M[r], M[col])]  # row op
                Inv[r] = [rv - factor * cv for rv, cv in zip(Inv[r], Inv[col])]  # row op
    return Inv  # finished inverse

# -----------------------------
# Data generation — small synthetic dataset with: numeric, indicator, nonlinear
# -----------------------------
# I simulate an outcome influenced by square footage, bedrooms, a categorical
# neighborhood (as indicators), and a nonlinear term (time_on_market^2).         # scenario rationale

@dataclass  # container to return both arrays and human‑readable names
class ToyData:  # minimal structured dataset
    X: List[List[float]]  # design matrix (with intercept added later)
    y: List[float]  # outcome vector
    names: List[str]  # feature names matching the columns of X (post‑build)


def make_toy_data(n: int = 120, seed: int = 7) -> ToyData:  # reproducible generator
    random.seed(seed)  # deterministic RNG for consistent results in examples
    y: List[float] = []  # allocate outcome list
    rows: List[List[float]] = []  # allocate raw feature storage (no intercept yet)
    names = [  # feature labels in order for interpretability
        "sqft",  # numeric predictor 1
        "beds",  # numeric predictor 2 (treated as numeric for demo)
        "neigh_B",  # indicator for neighborhood B (A is reference)
        "neigh_C",  # indicator for neighborhood C (A is reference)
        "days",  # numeric: time on market
        "days_sq",  # nonlinear transform: days^2
        "sqft_by_beds",  # interaction between sqft and beds
    ]  # end list of names
    # True coefficients used to synthesize y (include an intercept β0).        
    beta_true = [150.0,  0.35, 12.0, 25.0, -0.8, -0.02, 0.001, 0.05]  # β0..β7
    # Iterate n samples and generate features + outcome with noise.             # loop for observations
    for _ in range(n):  # sample loop
        sqft = random.uniform(600, 3200)  # draw square footage uniformly
        beds = random.choice([1, 2, 3, 4, 5])  # choose discrete bedrooms
        neigh = random.choice(["A", "B", "C"])  # categorical neighborhood
        neigh_B = 1.0 if neigh == "B" else 0.0  # indicator for B
        neigh_C = 1.0 if neigh == "C" else 0.0  # indicator for C
        days = random.uniform(0, 120)  # time on market
        days_sq = days * days  # nonlinear predictor (quadratic time effect)
        sqft_by_beds = sqft * beds  # simple interaction term
        # Build the feature row in the exact order of `names`.                 # consistency matters
        x_row = [sqft, beds, neigh_B, neigh_C, days, days_sq, sqft_by_beds]  # pack features
        rows.append(x_row)  # accumulate row
        # Compute noiseless linear part: β0 + Xβ (without intercept in x_row). # deterministic core
        lin = (beta_true[0]  # intercept term
               + beta_true[1] * sqft
               + beta_true[2] * beds
               + beta_true[3] * neigh_B
               + beta_true[4] * neigh_C
               + beta_true[5] * days
               + beta_true[6] * days_sq
               + beta_true[7] * sqft_by_beds)  # finish linear combination
        noise = random.gauss(0.0, 20.0)  # Gaussian noise for realism
        y.append(lin + noise)  # outcome with noise
    return ToyData(X=rows, y=y, names=names)  # package into dataclass and return

# -----------------------------
# Design matrix builder — add intercept, choose subsets, optionally standardize
# -----------------------------

@dataclass  # return type for design matrix with metadata
class Design:  # container for X matrix, y vector, and column names
    X: List[List[float]]  # final design matrix (with intercept if requested)
    y: List[float]  # outcome (unchanged from ToyData)
    cols: List[str]  # column names matching X


def build_design(data: ToyData, use_cols: Optional[List[str]] = None, add_intercept: bool = True, standardize: bool = False) -> Design:  # builder function
    cols = data.names[:] if use_cols is None else use_cols[:]  # choose feature set
    X = [[row[data.names.index(c)] for c in cols] for row in data.X]  # subset columns
    if standardize:  # feature scaling branch
        means = [sum(col) / len(col) for col in zip(*X)]  # compute column means
        stds = [max(1e-12, (sum((v - m) ** 2 for v in col) / len(col)) ** 0.5) for col, m in zip(zip(*X), means)]  # column stds
        X = [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in X]  # scale each value
        cols = [f"z_{c}" for c in cols]  # rename to indicate z‑scored features
    if add_intercept:  # intercept option
        X = [[1.0] + row for row in X]  # prepend ones column
        cols = ["intercept"] + cols  # reflect intercept in names
    return Design(X=X, y=data.y[:], cols=cols)  # return structured design

# -----------------------------
# OLS estimation — closed form β̂ = (XᵀX)⁻¹ Xᵀy, with diagnostics and metrics
# -----------------------------

@dataclass  # structure to hold fitted model results
class OLSResult:  # results object for clarity
    beta: List[float]  # estimated coefficients (aligned with columns)
    fitted: List[float]  # predicted y values Xβ
    residuals: List[float]  # y − fitted
    sigma2: float  # MLE of error variance (σ²)
    r2: float  # coefficient of determination
    aic: float  # Akaike information criterion
    bic: float  # Bayesian information criterion
    loglik: float  # Gaussian log‑likelihood at σ² MLE


def ols_fit(design: Design) -> OLSResult:  # core OLS routine
    X = design.X  # design matrix as plain lists (or NumPy later)
    y = design.y  # outcome vector
    n = len(y)  # number of observations
    p = len(X[0])  # number of columns (incl. intercept if present)
    if HAS_NUMPY:  # fast path using NumPy when available
        Xa = np.array(X, dtype=float)  # convert to array
        ya = np.array(y, dtype=float)  # convert to array
        XT_X = Xa.T @ Xa  # compute XᵀX
        try:  # try direct solve for numerical stability
            beta = np.linalg.solve(XT_X, Xa.T @ ya)  # β̂ via solve
        except np.linalg.LinAlgError:  # if singular, fall back to pseudo‑inverse
            beta = np.linalg.pinv(XT_X) @ (Xa.T @ ya)  # β̂ via pinv
        fitted = Xa @ beta  # predictions
        resid = ya - fitted  # residuals
        rss = float(resid @ resid)  # residual sum of squares
        sigma2 = rss / n  # MLE σ² (note: unbiased uses n−p)
        ybar = float(np.mean(ya))  # mean of y
        tss = float(((ya - ybar) ** 2).sum())  # total sum of squares
        r2 = 1.0 - (rss / tss if tss > 0 else 0.0)  # R²
        loglik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)  # Gaussian LL
        k = p + 1  # parameters counted: p betas + σ²
        aic = 2 * k - 2 * loglik  # AIC
        bic = math.log(n) * k - 2 * loglik  # BIC
        return OLSResult(beta=beta.tolist(), fitted=fitted.tolist(), residuals=resid.tolist(), sigma2=sigma2, r2=r2, aic=aic, bic=bic, loglik=loglik)  # pack results
    else:  # pure‑Python fallback (ok for small n,p)
        XT = _pp_transpose(X)  # transpose X
        XT_X = _pp_matmul(XT, X)  # XᵀX
        Inv = _pp_invert(XT_X)  # (XᵀX)⁻¹
        XTy = [sum(col[i] * y[i] for i in range(n)) for col in XT]  # Xᵀy
        beta = _pp_matvec(Inv, XTy)  # β̂
        fitted = _pp_matvec(X, beta)  # predictions
        resid = [yi - fi for yi, fi in zip(y, fitted)]  # residuals
        rss = sum(r * r for r in resid)  # RSS
        sigma2 = rss / n  # MLE σ²
        ybar = sum(y) / n  # mean
        tss = sum((yi - ybar) ** 2 for yi in y)  # TSS
        r2 = 1.0 - (rss / tss if tss > 0 else 0.0)  # R²
        loglik = -0.5 * n * (math.log(2 * math.pi * sigma2) + 1.0)  # LL
        k = p + 1  # parameter count
        aic = 2 * k - 2 * loglik  # AIC
        bic = math.log(n) * k - 2 * loglik  # BIC
        return OLSResult(beta=beta, fitted=fitted, residuals=resid, sigma2=sigma2, r2=r2, aic=aic, bic=bic, loglik=loglik)  # pack

# -----------------------------
# Ridge regression — β̂(λ) = (XᵀX + λI)⁻¹ Xᵀy to illustrate regularization
# -----------------------------

@dataclass  # structure to hold ridge results with λ value
class RidgeResult:  # container with diagnostics similar to OLSResult
    lam: float  # regularization strength λ
    beta: List[float]  # coefficients
    r2: float  # R² on training data (for comparability)


def ridge_fit(design: Design, lam: float) -> RidgeResult:  # ridge core
    X = design.X  # design matrix
    y = design.y  # response vector
    n = len(y)  # rows
    p = len(X[0])  # columns
    if HAS_NUMPY:  # NumPy implementation
        Xa = np.array(X, dtype=float)  # to array
        ya = np.array(y, dtype=float)  # to array
        XT_X = Xa.T @ Xa  # XᵀX
        XT_X_lamI = XT_X + lam * np.eye(p)  # XᵀX + λI
        try:  # solve the linear system
            beta = np.linalg.solve(XT_X_lamI, Xa.T @ ya)  # ridge β̂
        except np.linalg.LinAlgError:  # stability fallback
            beta = np.linalg.pinv(XT_X_lamI) @ (Xa.T @ ya)  # pseudo‑inverse
        fitted = Xa @ beta  # predictions
        rss = float(((ya - fitted) ** 2).sum())  # residual sum of squares
        tss = float(((ya - ya.mean()) ** 2).sum())  # total sum of squares
        r2 = 1.0 - (rss / tss if tss > 0 else 0.0)  # R²
        return RidgeResult(lam=lam, beta=beta.tolist(), r2=r2)  # pack
    else:  # pure‑Python path
        XT = _pp_transpose(X)  # transpose X
        XT_X = _pp_matmul(XT, X)  # XᵀX
        for i in range(p):  # add λ to diagonal to form XᵀX+λI
            XT_X[i][i] += lam  # regularize
        Inv = _pp_invert(XT_X)  # invert
        XTy = [sum(col[i] * y[i] for i in range(n)) for col in XT]  # Xᵀy
        beta = _pp_matvec(Inv, XTy)  # coefficients
        fitted = _pp_matvec(X, beta)  # predictions
        rss = sum((yi - fi) ** 2 for yi, fi in zip(y, fitted))  # RSS
        ybar = sum(y) / n  # mean
        tss = sum((yi - ybar) ** 2 for yi in y)  # TSS
        r2 = 1.0 - (rss / tss if tss > 0 else 0.0)  # R²
        return RidgeResult(lam=lam, beta=beta, r2=r2)  # pack

# -----------------------------
# K‑fold cross‑validation — estimate out‑of‑sample MSE to avoid overfitting
# -----------------------------

def kfold_mse(design: Design, k: int = 5, seed: int = 19) -> float:  # CV wrapper
    n = len(design.y)  # number of rows
    idx = list(range(n))  # index list
    random.Random(seed).shuffle(idx)  # fixed shuffle for reproducibility
    folds = [idx[i::k] for i in range(k)]  # round‑robin split into k folds
    mses: List[float] = []  # store fold errors
    for i in range(k):  # iterate held‑out folds
        test_idx = set(folds[i])  # current test set indices as a set
        train_rows = [design.X[j] for j in idx if j not in test_idx]  # X_train
        train_y = [design.y[j] for j in idx if j not in test_idx]  # y_train
        test_rows = [design.X[j] for j in idx if j in test_idx]  # X_test
        test_y = [design.y[j] for j in idx if j in test_idx]  # y_test
        sub = Design(X=train_rows, y=train_y, cols=design.cols)  # design for train
        fit = ols_fit(sub)  # fit OLS on training fold
        # Predict on test: simple matrix‑vector; convert beta to list if needed.  # prediction step
        if HAS_NUMPY:  # use NumPy for convenience if available
            yhat = (np.array(test_rows) @ np.array(fit.beta)).tolist()  # predictions
        else:  # pure‑Python path
            yhat = _pp_matvec(test_rows, fit.beta)  # predictions
        mse = sum((a - b) ** 2 for a, b in zip(test_y, yhat)) / len(test_y)  # fold MSE
        mses.append(mse)  # accumulate
    return sum(mses) / len(mses)  # return average MSE across folds

# -----------------------------
# Pretty printing helpers — readable, aligned output for coefficients and models
# -----------------------------

def print_model_summary(title: str, cols: List[str], beta: List[float], r2: float, aic: float, bic: float) -> None:  # summary printer
    print("\n" + "=" * 78)  # section divider
    print(title)  # model title
    print("-" * 78)  # underline
    for name, b in zip(cols, beta):  # iterate coefficients with names
        print(f"{name:<18} : {b:>12.6f}")  # aligned name/value
    print(f"R^2   : {r2:.4f}")  # R‑squared display
    print(f"AIC   : {aic:.3f}")  # AIC display
    print(f"BIC   : {bic:.3f}")  # BIC display

# -----------------------------
# Main narrative — build variants that reflect the outline and compare them
# -----------------------------

def main() -> int:  # program entry point
    data = make_toy_data(n=160, seed=11)  # build synthetic dataset with structure

    # Model A: Simple baseline — intercept + sqft only (recap of single regression).  # baseline model
    A = build_design(data, use_cols=["sqft"], add_intercept=True, standardize=False)  # design matrix
    resA = ols_fit(A)  # fit OLS
    print_model_summary("Model A — Single predictor (sqft)", A.cols, resA.beta, resA.r2, resA.aic, resA.bic)  # report

    # Model B: Multiple regression — add beds and indicators for neighborhood.    # add indicators
    B = build_design(data, use_cols=["sqft", "beds", "neigh_B", "neigh_C"], add_intercept=True, standardize=False)  # design
    resB = ols_fit(B)  # fit
    print_model_summary("Model B — + beds + neighborhood indicators", B.cols, resB.beta, resB.r2, resB.aic, resB.bic)  # report

    # Model C: Nonlinear + interaction — add days, days^2, and sqft*beds.        # richer model
    C = build_design(data, use_cols=["sqft", "beds", "neigh_B", "neigh_C", "days", "days_sq", "sqft_by_beds"], add_intercept=True, standardize=False)  # design
    resC = ols_fit(C)  # fit
    print_model_summary("Model C — + nonlinear term and interaction", C.cols, resC.beta, resC.r2, resC.aic, resC.bic)  # report

    # Model D: Standardized version of Model C — to compare coefficient scale.   # z‑scoring
    D = build_design(data, use_cols=["sqft", "beds", "neigh_B", "neigh_C", "days", "days_sq", "sqft_by_beds"], add_intercept=True, standardize=True)  # design z‑scored
    resD = ols_fit(D)  # fit
    print_model_summary("Model D — Standardized features (z-scores)", D.cols, resD.beta, resD.r2, resD.aic, resD.bic)  # report

    # Cross‑validation: estimate generalization error (MSE) for B and C.        # CV step
    mseB = kfold_mse(B, k=5, seed=23)  # 5‑fold CV on Model B
    mseC = kfold_mse(C, k=5, seed=23)  # 5‑fold CV on Model C
    print("\nCV (5-fold) MSE — Model B:", round(mseB, 3), "| Model C:", round(mseC, 3))  # compare

    # Ridge variant: stabilize Model C coefficients with a small λ.               # regularization
    ridge = ridge_fit(C, lam=1e-2)  # ridge fit
    print("\nRidge (λ=1e-2) — R^2:", round(ridge.r2, 4))  # quick metric
    for name, b in zip(C.cols, ridge.beta):  # print coefficients
        print(f"ridged {name:<14} : {b:>12.6f}")  # aligned

    # Closing notes (printed) — connect back to the study guide in the prompt.   # pedagogy
    print("\n" + "-" * 78)  # divider
    print("Notes:")  # heading
    print("• Indicator variables: we used neigh_B and neigh_C; neigh_A is reference (implicit).")  # dummy variables
    print("• Nonlinear predictors: 'days_sq' captured curvature in time on market.")  # polynomial term
    print("• Interaction: 'sqft_by_beds' lets the slope of sqft depend on bedrooms.")  # interaction
    print("• Model selection: compare AIC/BIC and CV MSE; pick the trade‑off that fits the goal.")  # selection
    print("• Overfitting vs omitted variables: richer models fit better in‑sample; validate out‑of‑sample.")  # caution
    print("• When to switch to R: when you need formula syntax, complex contrasts, and publication‑grade inference tables quickly.")  # judgment

    return 0  # success status code


if __name__ == "__main__":  # standard module guard
    raise SystemExit(main())  # run main and exit with its status code
