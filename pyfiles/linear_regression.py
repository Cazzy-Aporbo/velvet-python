"""
Linear Regression — Four Ways, with Progressive Visualization

What this program does
----------------------
1) Builds a simple 1D regression dataset with noise + a few outliers so we can
   practice modeling and diagnostics.
2) Fits linear regression in FOUR distinct ways:
   (A) Closed-form Normal Equation (NumPy)
   (B) Batch Gradient Descent from scratch (NumPy)
   (C) scikit-learn's LinearRegression
   (D) statsmodels OLS (with rich statistical summary, if installed)
3) Asks the user how they want to view the charts: Primary / Black & White / Pastel.
   Each chart uses an *ombre* gradient adapted to the chosen palette.
4) Renders four progressively "smarter" figures:
   F1: Basic scatter + fitted line (learn: baselines, slope/intercept)
   F2: Adds axis titles + an on-figure "how to read this" guide
   F3: Highlights outliers via standardized residuals (learn: influence)
   F4: Residual vs Fitted view + quick diagnostics (learn: assumptions)

Skills highlighted
------------------
Math: normal equation, gradient descent update rule, residuals, R^2, MAE/MSE
Data: intercept trick, feature scaling (for GD), outlier detection via z-scores
Programming: top-down orchestration, bottom-up utilities, palette generation,
             reproducibility, clear naming, defensive coding.

Run
---
$ python linear_regression_four_ways.py
(choose a palette at the prompt; images are saved to ./_output)
"""

from __future__ import annotations

# ---- Imports (top-only, as requested) ---------------------------------------
import os
import sys
import math
import textwrap
from dataclasses import dataclass
from typing import Dict, Tuple, List
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# statsmodels is optional; we degrade gracefully if missing
try:
    import statsmodels.api as sm  # type: ignore
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ---- Data structures ---------------------------------------------------------
@dataclass
class FitResult:
    name: str
    intercept: float
    slope: float
    y_pred: np.ndarray
    r2: float
    mse: float
    mae: float


# ---- Utility: palette + ombre ------------------------------------------------
def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """'#RRGGBB' -> (R, G, B) ints."""
    hex_color = hex_color.strip().lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # type: ignore


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """(R, G, B) -> '#RRGGBB'."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _interp(a: int, b: int, t: float) -> int:
    """Linear interpolation for a single color channel."""
    return int(round(a + (b - a) * t))


def make_ombre(start_hex: str, end_hex: str, n: int) -> List[str]:
    """
    Create n-step gradient from start_hex to end_hex inclusive.
    Example: make_ombre('#FFB5CC', '#D4FFE4', 100)
    """
    r1, g1, b1 = _hex_to_rgb(start_hex)
    r2, g2, b2 = _hex_to_rgb(end_hex)
    steps = []
    for i in range(n):
        t = i / max(1, n - 1)
        steps.append(_rgb_to_hex((_interp(r1, r2, t),
                                  _interp(g1, g2, t),
                                  _interp(b1, b2, t))))
    return steps


def choose_palette() -> Dict[str, List[str]]:
    """
    Ask user for a theme. Return a small dictionary of ombre lists we can use.
    """
    print("\nHow do you want to view the graphs?")
    print(" [1] Primary colors (bold)")
    print(" [2] Black & White (publication-style)")
    print(" [3] Pastel (soft)")
    choice = input("Enter 1 / 2 / 3 (default: 1): ").strip() or "1"
    theme = {"1": "primary", "2": "bw", "3": "pastel"}.get(choice, "primary")

    if theme == "primary":
        scatter_ombre = make_ombre("#1F77B4", "#FF7F0E", 200)   # blue -> orange
        line_ombre    = make_ombre("#2CA02C", "#D62728", 50)     # green -> red
    elif theme == "bw":
        scatter_ombre = make_ombre("#111111", "#BBBBBB", 200)    # dark -> light
        line_ombre    = make_ombre("#333333", "#AAAAAA", 50)
    else:  # pastel
        scatter_ombre = make_ombre("#FFB5CC", "#D4FFE4", 200)    # pink -> mint
        line_ombre    = make_ombre("#D8B5D8", "#C8A8C8", 50)     # lavender ombre

    return {"scatter": scatter_ombre, "line": line_ombre}


# ---- Data generation ---------------------------------------------------------
def build_dataset(n: int = 120, random_state: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 1D linear dataset with noise and a few deliberate outliers.
    Returns:
        X (n, 1), y (n,)
    """
    X, y = make_regression(
        n_samples=n,
        n_features=1,
        noise=12.0,
        bias=50.0,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)

    # Add a gentle nonlinearity + heteroskedastic noise to make residuals interesting
    y = y + 0.02 * (X[:, 0] ** 2) + rng.normal(0, 6, size=n)

    # Inject a few outliers (both high and low)
    outlier_idx = rng.choice(n, size=max(3, n // 30), replace=False)
    y[outlier_idx] += rng.choice([-70, 70], size=outlier_idx.size)

    return X.astype(float), y.astype(float)


# ---- Modeling: 4 ways --------------------------------------------------------
def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> FitResult:
    """
    Closed-form OLS via normal equation: beta = (X'X)^(-1) X'y
    Uses explicit intercept column (the "intercept trick").
    """
    X_design = np.column_stack([np.ones_like(X[:, 0]), X[:, 0]])  # [1, x]
    xtx = X_design.T @ X_design
    beta = np.linalg.inv(xtx) @ X_design.T @ y
    y_pred = X_design @ beta
    return _summarize_fit("Normal Equation (NumPy)", y, y_pred, beta)


def fit_gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float = 1e-3, epochs: int = 8000
) -> FitResult:
    """
    Batch Gradient Descent for y ~ a + b*x.
    We standardize x to keep the scale reasonable for a single learning rate.
    Update rule (MSE loss): a <- a - lr * dL/da ;  b <- b - lr * dL/db
    """
    x = X[:, 0]
    x_mu, x_sd = x.mean(), x.std()
    xs = (x - x_mu) / (x_sd + 1e-12)

    a, b = 0.0, 0.0  # intercept, slope (in standardized space)
    n = float(len(y))

    for _ in range(epochs):
        y_hat = a + b * xs
        # derivatives of MSE = (1/n)*sum (y_hat - y)^2
        grad_a = (2.0 / n) * np.sum(y_hat - y)
        grad_b = (2.0 / n) * np.sum((y_hat - y) * xs)
        a -= lr * grad_a
        b -= lr * grad_b

    # Convert back to original x-units:
    # y ≈ a + b * ((x - mu)/sd) = (a - b*mu/sd) + (b/sd) * x
    intercept = a - b * (x_mu / (x_sd + 1e-12))
    slope = b / (x_sd + 1e-12)
    y_pred = intercept + slope * x
    return _summarize_fit("Gradient Descent (NumPy)", y, y_pred, np.array([intercept, slope]))


def fit_sklearn(X: np.ndarray, y: np.ndarray) -> FitResult:
    """scikit-learn's LinearRegression (robust baseline for tabular work)."""
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    y_pred = model.predict(X)
    intercept = float(model.intercept_)
    slope = float(model.coef_[0])
    return _summarize_fit("LinearRegression (scikit-learn)", y, y_pred, np.array([intercept, slope]))


def fit_statsmodels(X: np.ndarray, y: np.ndarray) -> FitResult | None:
    """statsmodels OLS with full summary (optional but very educational)."""
    if not HAS_STATSMODELS:
        return None
    X_design = sm.add_constant(X)  # [1, x]
    model = sm.OLS(y, X_design).fit()
    y_pred = model.predict(X_design)
    intercept = float(model.params[0])
    slope = float(model.params[1])

    # Show a compact summary in the terminal so beginners see p-values, R^2, etc.
    print("\n[statsmodels] OLS summary (trimmed):")
    print(model.summary().as_text().split("\n")[0:20])  # first ~20 lines for brevity
    return _summarize_fit("OLS (statsmodels)", y, y_pred, np.array([intercept, slope]))


# ---- Helpers -----------------------------------------------------------------
def _summarize_fit(name: str, y_true: np.ndarray, y_pred: np.ndarray, beta: np.ndarray) -> FitResult:
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return FitResult(
        name=name,
        intercept=float(beta[0]),
        slope=float(beta[1]),
        y_pred=y_pred.astype(float),
        r2=float(r2),
        mse=float(mse),
        mae=float(mae),
    )


def standardized_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute standardized residuals (z-scores) to spot outliers."""
    resid = y_true - y_pred
    sd = resid.std(ddof=1)
    return (resid - resid.mean()) / (sd + 1e-12)


def ensure_output_dir(path: str | None = None) -> str:
    """
    Create (and return) the output directory.
    Default: <Desktop>/LinearRegression_Outputs
    - Cross-platform (macOS/Windows), including OneDrive-managed Desktops.
    - You may override by passing a path: ensure_output_dir("/some/where")
    """
    if path:
        out = Path(path).expanduser().resolve()
    else:
        home = Path.home()
        candidates = [
            home / "Desktop",                                   # macOS/Linux common
            Path(os.environ.get("USERPROFILE", "")) / "Desktop",# Windows
            Path(os.environ.get("OneDrive", "")) / "Desktop",   # Windows OneDrive
        ]
        # choose the first Desktop that exists; otherwise default to home/Desktop
        desktop = next((p for p in candidates if p and p.exists()), home / "Desktop")
        out = desktop / "LinearRegression_Outputs"

    out.mkdir(parents=True, exist_ok=True)
    return str(out)



# ---- Plotting (four increasingly advanced figures) ---------------------------
def fig1_basic_scatter_and_line(
    X: np.ndarray, y: np.ndarray, result: FitResult, palette: Dict[str, List[str]], outdir: str
) -> None:
    """
    F1: foundational view: scatter + best fit line.
    Teaching goals: what slope/intercept mean; seeing the linear trend.
    """
    x = X[:, 0]
    # Ombre scatter by x-rank
    idx = np.argsort(x)
    colors = np.array(palette["scatter"])[(np.linspace(0, len(palette["scatter"]) - 1, len(x))).astype(int)]
    plt.figure(figsize=(8, 5))
    plt.scatter(x[idx], y[idx], c=colors[idx], s=34, edgecolor="white", linewidth=0.5, alpha=0.92, label="observations")

    # Fitted line (use line palette mid-to-end for contrast)
    x_line = np.linspace(x.min(), x.max(), 250)
    y_line = result.intercept + result.slope * x_line
    line_color = palette["line"][int(0.7 * (len(palette["line"]) - 1))]
    plt.plot(x_line, y_line, lw=2.4, color=line_color, label=f"fit: y = {result.intercept:.2f} + {result.slope:.2f}·x")

    plt.title("Linear Regression — Basic View", pad=14)
    plt.xlabel("Feature x")
    plt.ylabel("Target y")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(outdir, "F1_basic_scatter_and_line.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path} — (slope={result.slope:.3f}, intercept={result.intercept:.3f}, R²={result.r2:.3f})")


def fig2_titled_with_reading_guide(
    X: np.ndarray, y: np.ndarray, result: FitResult, palette: Dict[str, List[str]], outdir: str
) -> None:
    """
    F2: titles, axis labels, and an inline 'how to read' panel.
    Teaching goals: labeling, storytelling, chart literacy.
    """
    x = X[:, 0]
    idx = np.argsort(x)
    colors = np.array(palette["scatter"])[(np.linspace(0, len(palette["scatter"]) - 1, len(x))).astype(int)]
    plt.figure(figsize=(9, 5.5))
    plt.scatter(x[idx], y[idx], c=colors[idx], s=36, edgecolor="white", linewidth=0.6, alpha=0.95)

    x_line = np.linspace(x.min(), x.max(), 250)
    y_line = result.intercept + result.slope * x_line
    line_color = palette["line"][int(0.65 * (len(palette["line"]) - 1))]
    plt.plot(x_line, y_line, lw=2.8, color=line_color)

    plt.title("How to Read This Chart", pad=14)
    plt.xlabel("x (independent variable)")
    plt.ylabel("y (dependent variable)")

    # On-figure reading guide
    guide = textwrap.dedent(f"""
        • Trend line shows average relationship: +{result.slope:.2f} y-units per +1 x.
        • Points above line: under-predicted; below: over-predicted.
        • Spread around line ≈ noise/variance; tighter is better.
        • R²={result.r2:.2f} summarizes fit quality (closer to 1 is stronger).
    """).strip()
    # A translucent textbox
    plt.gca().text(
        0.02, 0.98, guide, transform=plt.gca().transAxes,
        fontsize=9.9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="none", alpha=0.85)
    )

    plt.tight_layout()
    path = os.path.join(outdir, "F2_titled_with_reading_guide.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path} — (learn: labeling + reading the chart)")


def fig3_highlight_outliers(
    X: np.ndarray, y: np.ndarray, result: FitResult, palette: Dict[str, List[str]], outdir: str, z_thresh: float = 2.5
) -> None:
    """
    F3: highlight probable outliers using standardized residuals (|z| >= z_thresh).
    Teaching goals: residuals, influence, visual emphasis without distorting data.
    """
    x = X[:, 0]
    z = standardized_residuals(y, result.y_pred)
    outlier_mask = np.abs(z) >= z_thresh

    # Base scatter
    colors = np.array(palette["scatter"])[(np.linspace(0, len(palette["scatter"]) - 1, len(x))).astype(int)]
    plt.figure(figsize=(9, 5.5))
    plt.scatter(x[~outlier_mask], y[~outlier_mask], c=colors[~outlier_mask], s=36, edgecolor="white", linewidth=0.5, alpha=0.92, label="inliers")

    # Highlight outliers with rings + labels
    plt.scatter(x[outlier_mask], y[outlier_mask], s=80, facecolor="none", edgecolor="#C00000", linewidth=1.8, label=f"outliers (|z|≥{z_thresh})")
    for xi, yi, zi in zip(x[outlier_mask], y[outlier_mask], z[outlier_mask]):
        plt.annotate(f"z={zi:.1f}", (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#9A0000")

    # Fitted line
    x_line = np.linspace(x.min(), x.max(), 250)
    y_line = result.intercept + result.slope * x_line
    line_color = palette["line"][int(0.75 * (len(palette["line"]) - 1))]
    plt.plot(x_line, y_line, lw=2.5, color=line_color)

    plt.title("Outlier Spotlight via Standardized Residuals", pad=14)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(outdir, "F3_highlight_outliers.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path} — (learn: standardized residuals; outlier emphasis)")


def fig4_residuals_vs_fitted(
    X: np.ndarray, y: np.ndarray, result: FitResult, palette: Dict[str, List[str]], outdir: str
) -> None:
    """
    F4: residuals vs fitted — a diagnostic view.
    Teaching goals: linearity, homoscedasticity, and independence (quick checks).
    """
    fitted = result.y_pred
    resid = y - fitted

    # Ombre by fitted order (helps see any structure)
    order = np.argsort(fitted)
    colors = np.array(palette["scatter"])[(np.linspace(0, len(palette["scatter"]) - 1, len(fitted))).astype(int)]

    plt.figure(figsize=(9, 5.5))
    plt.scatter(fitted[order], resid[order], c=colors[order], s=34, edgecolor="white", linewidth=0.5, alpha=0.92)
    plt.axhline(0.0, color=palette["line"][int(0.2 * (len(palette["line"]) - 1))], lw=2, linestyle="--")

    plt.title("Residuals vs Fitted (Diagnostics)", pad=14)
    plt.xlabel("Fitted values (ŷ)")
    plt.ylabel("Residuals (y - ŷ)")

    # Quick checklist
    notes = textwrap.dedent("""
        Expect:
        • Residuals scattered around 0 with no pattern → linearity holds.
        • Roughly constant vertical spread → homoscedasticity.
        • No streaks when ordered by ŷ → independence.
    """).strip()
    plt.gca().text(
        0.98, 0.98, notes, transform=plt.gca().transAxes,
        fontsize=9.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="none", alpha=0.85)
    )

    plt.tight_layout()
    path = os.path.join(outdir, "F4_residuals_vs_fitted.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path} — (learn: model assumptions via residuals)")


# ---- Orchestration -----------------------------------------------------------
def main() -> int:
    # 0) Palette choice and output dir
    palette = choose_palette()
    outdir = ensure_output_dir()

    # 1) Data
    X, y = build_dataset(n=140, random_state=42)

    # 2) Fit models in four ways
    fits: List[FitResult] = []
    fits.append(fit_normal_equation(X, y))
    fits.append(fit_gradient_descent(X, y, lr=8e-4, epochs=12000))
    fits.append(fit_sklearn(X, y))
    sm_fit = fit_statsmodels(X, y)
    if sm_fit is not None:
        fits.append(sm_fit)

    # 3) Report a quick scoreboard
    print("\nModel comparison (lower MSE/MAE; higher R² is better):")
    for fr in fits:
        print(f"  • {fr.name:28s}  R²={fr.r2:6.3f}  MSE={fr.mse:8.2f}  MAE={fr.mae:7.2f}")

    # We'll use the scikit-learn fit for the teaching figures (stable baseline).
    reference = next(fr for fr in fits if "LinearRegression" in fr.name)

    # 4) Progressive figures
    fig1_basic_scatter_and_line(X, y, reference, palette, outdir)
    fig2_titled_with_reading_guide(X, y, reference, palette, outdir)
    fig3_highlight_outliers(X, y, reference, palette, outdir)
    fig4_residuals_vs_fitted(X, y, reference, palette, outdir)

    print(f"\nDone. Open the PNGs in ./{outdir} to review the progression.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
