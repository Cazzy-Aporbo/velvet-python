# logistic_regression_ombre.py
# -----------------------------------------------------------------------------
# Title: Logistic Regression Demo — With Ombre Pink↔Purple Visuals
# Author: Cazandra Aporbo
# Completed: July 2023 (visuals added)
# Last Updated: August 2025
# Intent: A compact, professional demonstration of logistic regression that
#         I would include in a repository to show practical understanding:
#         data prep, fitting, threshold tuning, diagnostics, and visuals.
# Notes:  •  Fully commented on the why, not just the how.
#         • Visuals use a custom ombre (pink→purple) palette only here.
#         • If matplotlib is unavailable or the environment is headless, the
#           script degrades by printing metrics and (if needed) saving PNGs.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math                         # for log/exp in odds/metrics
import sys                          # for environment and argv checks
import warnings                     # to keep output tidy
from dataclasses import dataclass   # to structure results cleanly
from typing import Iterable, List, Optional, Tuple

import numpy as np                  # numerical arrays
from sklearn.datasets import make_classification  # quick, controllable data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)

# Optional classical inference (when I want p-values/LL):
try:
    import statsmodels.api as sm
    HAS_SM = True
except Exception:
    HAS_SM = False

# Optional plotting: I keep it robust to headless environments.
try:
    import matplotlib
    try:
        # If there's clearly no display, prefer Agg so plt.show() won't crash.
        if not (hasattr(matplotlib, "get_backend") and matplotlib.get_backend()):
            matplotlib.use("Agg")
    except Exception:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# -----------------------------------------------------------------------------
# Small data container types so downstream code is explicit and readable
# -----------------------------------------------------------------------------

@dataclass
class FitResult:
    X_train: np.ndarray    # features used for training
    X_test: np.ndarray     # features for evaluation
    y_train: np.ndarray    # training labels
    y_test: np.ndarray     # test labels
    model: Pipeline        # sklearn Pipeline(StandardScaler → LogisticRegression)
    y_proba: np.ndarray    # predicted probabilities (for the positive class)

@dataclass
class Metrics:
    auc_roc: float       # area under ROC (threshold free)
    auc_pr: float        # average precision (area under PR)
    threshold: float     # tuned decision threshold
    cm: np.ndarray       # confusion matrix at threshold

# -----------------------------------------------------------------------------
# Palette and background for the ombre pink↔purple visuals (this file only)
# -----------------------------------------------------------------------------

def _ombre_cmap() -> "LinearSegmentedColormap":
    """Pink→purple custom colormap (rose→magenta→violet)."""
    if not HAS_MPL:
        raise RuntimeError("matplotlib not available")
    return LinearSegmentedColormap.from_list(
        "cazandra_pink_purple",
        ["#ffe2f1", "#ff9ad5", "#d576e8", "#8a5cf6", "#5a3fd8"],
    )


def _ombre_background(ax) -> None:
    """Subtle gradient background to keep focus on the curves/points."""
    if not HAS_MPL:
        return
    grad = np.linspace(0, 1, 256)
    grad = np.vstack([grad, grad])
    ax.imshow(
        grad,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap=_ombre_cmap(),
        alpha=0.18,
        aspect="auto",
        zorder=0,
    )

# -----------------------------------------------------------------------------
# Synthetic dataset that still feels realistic: informative + redundant features
# -----------------------------------------------------------------------------

def make_data(n: int = 1200, seed: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Create a binary classification set with clear signal + some redundancy.
    Why synthetic: commits cleanly in a repo, no external files, reproducible.
    """
    X, y = make_classification(
        n_samples=n,
        n_features=12,
        n_informative=4,    # a handful truly matter
        n_redundant=2,      # correlated distractors
        n_repeated=0,
        n_clusters_per_class=2,
        flip_y=0.03,        # a bit of label noise keeps it honest
        class_sep=1.4,
        random_state=seed,
    )
    return X.astype(float), y.astype(int)

# -----------------------------------------------------------------------------
# Fit logistic regression in a disciplined way (scaling+penalty by default)
# -----------------------------------------------------------------------------

def fit_logreg(X: np.ndarray, y: np.ndarray, seed: int = 11) -> FitResult:
    """Train/evaluate Logistic Regression with scaling and a stable split.
    I use L2-regularized solver by default; it's a good baseline. The Pipeline
    avoids leakage: scaling is fit on train only and applied to test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    # I collect probabilities now so I can do threshold-aware diagnostics later
    proba = pipe.predict_proba(X_test)[:, 1]
    return FitResult(X_train, X_test, y_train, y_test, pipe, proba)

# -----------------------------------------------------------------------------
# Metrics: threshold-free (AUC) and a tuned-threshold confusion matrix
# -----------------------------------------------------------------------------

def evaluate_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> Metrics:
    """Pick a reasonable threshold by maximizing F1 on the PR curve.
    There are other choices (Youden's J on ROC, cost-based thresholds), but F1
    is defensible for imbalanced-ish toy problems.
    """
    auc_roc = float(roc_auc_score(y_true, y_proba))
    auc_pr = float(average_precision_score(y_true, y_proba))
    p, r, thr = precision_recall_curve(y_true, y_proba)
    f1 = (2 * p * r) / np.where((p + r) == 0, 1, (p + r))
    best_idx = int(np.nanargmax(f1))
    # precision_recall_curve returns thresholds of length len(p)-1
    best_thr = float(thr[max(0, best_idx - 1)]) if len(thr) else 0.5
    y_hat = (y_proba >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_hat)
    return Metrics(auc_roc=auc_roc, auc_pr=auc_pr, threshold=best_thr, cm=cm)

# -----------------------------------------------------------------------------
# Visuals — ombre pink↔purple, only for this logistic regression demo
# -----------------------------------------------------------------------------

def render_plots(fit: FitResult, m: Metrics, prefix: str = "logreg_ombre") -> List[str]:
    """Render ROC, PR, probability hist, and confusion matrix with an
    ombre pink→purple aesthetic. Save PNGs (safe for headless) and attempt to
    show if a GUI backend exists. Returns list of saved file paths.
    """
    if not HAS_MPL:
        return []

    saved: List[str] = []
    cmap = _ombre_cmap()

    # 1) ROC
    fpr, tpr, _ = roc_curve(fit.y_test, fit.y_proba)
    fig1, ax1 = plt.subplots(figsize=(6.2, 5.2), dpi=120)
    _ombre_background(ax1)
    ax1.plot([0, 1], [0, 1], "--", lw=1, color="#cda8ff", label="chance")
    ax1.plot(fpr, tpr, lw=2.5, color="#7b4ef0", label=f"ROC AUC = {m.auc_roc:.3f}")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Logistic Regression — ROC Curve")
    ax1.legend(loc="lower right")
    fig1.tight_layout()
    try:
        fig1.canvas.manager.set_window_title("ROC — Logistic Regression")
        plt.show(block=False)
    except Exception:
        pass
    try:
        path = f"{prefix}_roc.png"; fig1.savefig(path, dpi=150); saved.append(path)
    except Exception:
        pass

    # 2) Precision–Recall
    prec, rec, _ = precision_recall_curve(fit.y_test, fit.y_proba)
    fig2, ax2 = plt.subplots(figsize=(6.2, 5.2), dpi=120)
    _ombre_background(ax2)
    ax2.plot(rec, prec, lw=2.5, color="#ff69c8", label=f"AP = {m.auc_pr:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Logistic Regression — Precision–Recall Curve")
    ax2.legend(loc="lower left")
    fig2.tight_layout()
    try:
        fig2.canvas.manager.set_window_title("PR — Logistic Regression")
        plt.show(block=False)
    except Exception:
        pass
    try:
        path = f"{prefix}_pr.png"; fig2.savefig(path, dpi=150); saved.append(path)
    except Exception:
        pass

    # 3) Probability histogram by true class
    fig3, ax3 = plt.subplots(figsize=(6.4, 5.0), dpi=120)
    _ombre_background(ax3)
    proba0 = [p for p, y in zip(fit.y_proba, fit.y_test) if y == 0]
    proba1 = [p for p, y in zip(fit.y_proba, fit.y_test) if y == 1]
    ax3.hist(proba0, bins=np.linspace(0, 1, 21), alpha=0.6, label="class 0", color="#ff9ad5")
    ax3.hist(proba1, bins=np.linspace(0, 1, 21), alpha=0.6, label="class 1", color="#7b4ef0")
    ax3.axvline(m.threshold, ls=":", lw=2, color="#5a3fd8", label=f"threshold = {m.threshold:.2f}")
    ax3.set_xlabel("Predicted probability P(y=1|x)")
    ax3.set_ylabel("Count")
    ax3.set_title("Probability Distributions by True Class")
    ax3.legend(loc="best")
    fig3.tight_layout()
    try:
        fig3.canvas.manager.set_window_title("Probabilities — Logistic Regression")
        plt.show(block=False)
    except Exception:
        pass
    try:
        path = f"{prefix}_proba_hist.png"; fig3.savefig(path, dpi=150); saved.append(path)
    except Exception:
        pass

    # 4) Confusion matrix heatmap
    cm = m.cm
    fig4, ax4 = plt.subplots(figsize=(5.6, 5.0), dpi=120)
    im = ax4.imshow(cm, cmap=cmap)
    # annotate cells with counts; pick text color based on background intensity
    vmax = cm.max() if cm.size else 1
    for (i, j), v in np.ndenumerate(cm):
        ax4.text(j, i, str(v), ha="center", va="center",
                 color=("white" if v > vmax/2 else "black"))
    ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Pred 0", "Pred 1"]) ; ax4.set_yticklabels(["True 0", "True 1"]) 
    ax4.set_title("Confusion Matrix @ tuned threshold")
    fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    fig4.tight_layout()
    try:
        fig4.canvas.manager.set_window_title("Confusion — Logistic Regression")
        plt.show(block=False)
    except Exception:
        pass
    try:
        path = f"{prefix}_cm.png"; fig4.savefig(path, dpi=150); saved.append(path)
    except Exception:
        pass

    return saved

# -----------------------------------------------------------------------------
# Optional: classical inference summary via statsmodels (if available)
# -----------------------------------------------------------------------------

def classical_inference(fit: FitResult) -> Optional[str]:
    """Return a short statsmodels summary table (string) if available.
    I re-fit a simple GLM with a constant to compute standard errors/LL.
    """
    if not HAS_SM:
        return None
    X = fit.X_train
    y = fit.y_train
    Xc = np.c_[np.ones((X.shape[0], 1)), X]
    try:
        model = sm.GLM(y, Xc, family=sm.families.Binomial())
        res = model.fit()
        return res.summary2().as_text()
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Main execution: fit → metrics → visuals → (optional) classical report
# -----------------------------------------------------------------------------

def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("\n" + "=" * 78)
    print("Logistic Regression — Demonstration (with ombre visuals)")
    print("-" * 78)

    # 1) Data
    X, y = make_data(n=1400, seed=5)

    # 2) Fit
    fit = fit_logreg(X, y, seed=13)

    # 3) Threshold-free and thresholded metrics
    metrics = evaluate_thresholds(fit.y_test, fit.y_proba)
    print(f"ROC AUC: {metrics.auc_roc:.3f}  |  PR (AP): {metrics.auc_pr:.3f}")
    print(f"Tuned threshold (F1 on PR): {metrics.threshold:.3f}")
    print("Confusion matrix at threshold:\n", metrics.cm)

    # 4) Human-facing classification report at the tuned threshold
    y_hat = (fit.y_proba >= metrics.threshold).astype(int)
    print("\nClassification report @ tuned threshold\n")
    print(classification_report(fit.y_test, y_hat, digits=3))

    # 5) Visuals — ombre pink↔purple theme
    if HAS_MPL:
        saved = render_plots(fit, metrics, prefix="logreg_ombre")
        if saved:
            print("Saved figures:", ", ".join(saved))
    else:
        print("matplotlib not available; skipping visuals.")

    # 6) Optional: classical inference table (when I want LL/AIC/BIC, SEs)
    summary = classical_inference(fit)
    if summary:
        print("\nClassical GLM summary (statsmodels) — optional)\n")
        print(summary)
    else:
        print("\n(statsmodels not available or failed to fit — skipping classical table)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
