# logistic_regression_demo.py
# -----------------------------------------------------------------------------
# Title: Logistic Regression
# Author: Cazandra Aporbo
# Completed: July 1 2023 
# Last Touched: August 11 2025 
# What this is: a single Python file that *demonstrates competence* with
# logistic regression — when to use it, how to use it well, what trade-offs
# matter, and when I would switch to R. It is not a generic tutorial; it's a
# runnable, opinionated walkthrough with line-by-line commentary.
# All prints are deliberate.
# -----------------------------------------------------------------------------
from __future__ import annotations

# I keep imports explicit so choices are visible.
import math                              # odds⇄log-odds conversions, logs
import sys                               # environment checks; graceful exits
from dataclasses import dataclass         # structured return types for clarity
from typing import List, Optional, Tuple  # type hints for readers

# Core scientific tools. scikit-learn is the only hard dependency.
try:
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        average_precision_score,
        log_loss,
    )
except Exception as e:
    print("This demo requires scikit-learn and numpy. Please install them:")
    print("  pip install scikit-learn numpy")
    raise SystemExit(1)

# Optional: statsmodels for classical inference (SEs, Wald tests). Not required.
try:  # I keep this optional to avoid forcing extra deps.
    import statsmodels.api as sm  # type: ignore
    HAS_SM = True
except Exception:
    HAS_SM = False

# -----------------------------------------------------------------------------
# Utility: clean section headers so the script reads like a notebook in a shell
# -----------------------------------------------------------------------------

def block(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)


# -----------------------------------------------------------------------------
# 0) Problem framing — when is logistic regression a good idea?
# -----------------------------------------------------------------------------
# I write this as a function so it prints in sequence with the rest of the demo.

def intro() -> None:
    block("0) Framing the problem: when logistic regression fits like a glove")
    print("• Task: binary classification with interpretable coefficients.")
    print("• Target: y ∈ {0,1}. Model the *probability* P(y=1|x).")
    print("• Link: logit(p) = log(p/(1-p)) = β₀ + βᵀx. Output is in (0,1).")
    print("• Good fit when: effects are roughly linear on the *log-odds* scale,")
    print("  predictors not perfectly collinear, and decision threshold matters.")

# -----------------------------------------------------------------------------
# 1) Data: synthesize a dataset with numeric + categorical features on purpose
# -----------------------------------------------------------------------------
# Why: demonstrates practical preprocessing (scaling + one-hot), class
# imbalance handling, and clean train/test protocol.

@dataclass
class Dataset:
    X: np.ndarray                   # raw feature matrix (mixed dtypes allowed)
    y: np.ndarray                   # binary targets (0/1)
    feature_names: List[str]        # names for readability
    numeric_idx: List[int]          # column indices for numeric features
    categorical_idx: List[int]      # column indices for categorical features


def make_dataset(n: int = 2500, seed: int = 7) -> Dataset:
    # I create a numeric core with informative/redundant/noisy features.
    X_num, y = make_classification(
        n_samples=n,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        class_sep=1.2,
        flip_y=0.02,
        weights=[0.7, 0.3],   # deliberate imbalance so we can discuss it
        random_state=seed,
    )
    # Now I invent a small categorical feature, correlated but not decisive.
    # I bucket the first numeric feature into low/med/high.
    q = np.quantile(X_num[:, 0], [0.33, 0.66])
    cat = np.where(X_num[:, 0] < q[0], "low",
          np.where(X_num[:, 0] < q[1], "med", "high")).reshape(-1, 1)

    # Concatenate numeric + categorical; I keep types as object for ColumnTransformer.
    X = np.concatenate([X_num.astype(float), cat.astype(object)], axis=1)

    feature_names = [f"x{i}" for i in range(X_num.shape[1])] + ["bucket"]
    numeric_idx = list(range(X_num.shape[1]))
    categorical_idx = [X.shape[1] - 1]

    return Dataset(X=X, y=y, feature_names=feature_names,
                   numeric_idx=numeric_idx, categorical_idx=categorical_idx)

# -----------------------------------------------------------------------------
# 2) Pipeline: preprocessing + logistic regression with sane, explicit defaults
# -----------------------------------------------------------------------------
# Design choices explained inline. I build two pipelines to compare penalties.

@dataclass
class FitResult:
    pipe: Pipeline
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_proba: np.ndarray     # predicted probabilities on X_test
    y_pred: np.ndarray      # hard predictions using a default or tuned threshold
    feature_names_out: List[str]


def build_pipeline(dataset: Dataset, penalty: str = "l2",
                   class_weight: Optional[str] = None,
                   C: float = 1.0,
                   solver: Optional[str] = None) -> Tuple[Pipeline, List[str]]:
    # Preprocessing branch for numeric columns: StandardScaler to stabilize the
    # optimization landscape; it also makes coefficient magnitudes comparable.
    numeric = ("num", StandardScaler(with_mean=True, with_std=True), dataset.numeric_idx)

    # For categorical columns: OneHotEncoder(drop='first') to avoid dummy trap
    # in the *design matrix*. Note: the logistic model itself does not need a
    # separate intercept per category; dropping one level encodes deviations.
    categorical = ("cat", OneHotEncoder(drop="first", sparse_output=False), dataset.categorical_idx)

    pre = ColumnTransformer([numeric, categorical], remainder="drop")

    # Solver choice depends on penalty. I make it explicit to signal intent.
    if solver is None:
        solver = {"l1": "liblinear", "l2": "lbfgs", "elasticnet": "saga"}.get(penalty, "lbfgs")

    logreg = LogisticRegression(
        penalty=penalty,
        C=C,                      # inverse of regularization strength
        solver=solver,
        l1_ratio=0.5 if penalty == "elasticnet" else None,
        max_iter=200,
        n_jobs=None,              # keep deterministic
        class_weight=class_weight # 'balanced' helps on imbalanced datasets
    )

    pipe = Pipeline([("pre", pre), ("clf", logreg)])

    # Feature names after preprocessing for interpretability.
    # ColumnTransformer exposes get_feature_names_out; I call it after fitting
    # below and return them to the caller.
    return pipe, []  # I'll fill names post-fit once the encoders have categories


# -----------------------------------------------------------------------------
# 3) Train/test split, model fit, baseline evaluation, and interpretation
# -----------------------------------------------------------------------------

@dataclass
class Metrics:
    auc_roc: float
    auc_pr: float
    logloss: float
    cm: np.ndarray
    report: str
    threshold: float


def fit_and_evaluate(ds: Dataset, penalty: str = "l2",
                     class_weight: Optional[str] = None,
                     C: float = 1.0) -> Tuple[FitResult, Metrics]:
    # Standard split: stratify to preserve class balance in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=0.25, random_state=11, stratify=ds.y
    )

    pipe, _ = build_pipeline(ds, penalty=penalty, class_weight=class_weight, C=C)

    # Fit the full pipeline: scaling/encoding learned on train only.
    pipe.fit(X_train, y_train)

    # After fit, extract feature names in model space for interpretation.
    pre: ColumnTransformer = pipe.named_steps["pre"]
    feature_names_out = list(pre.get_feature_names_out())

    # Probabilities for ROC/PR; I keep both curves available for thresholding.
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Baseline threshold: 0.5 (we'll tune shortly).
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics — I compute both ROC AUC and average precision (area under PR).
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    ll = log_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, digits=3)

    fitres = FitResult(
        pipe=pipe,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        y_proba=y_proba, y_pred=y_pred, feature_names_out=feature_names_out,
    )
    metrics = Metrics(auc_roc=auc_roc, auc_pr=auc_pr, logloss=ll, cm=cm, report=rep, threshold=0.5)
    return fitres, metrics


# -----------------------------------------------------------------------------
# 4) Threshold tuning — because business costs are rarely symmetric
# -----------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    threshold: float
    fpr: float
    tpr: float
    youden_j: float


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> ThresholdResult:
    # I search ROC points for the threshold maximizing Youden's J = TPR - FPR.
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    k = int(np.argmax(j))
    return ThresholdResult(threshold=float(thr[k]), fpr=float(fpr[k]), tpr=float(tpr[k]), youden_j=float(j[k]))


# -----------------------------------------------------------------------------
# 5) Coefficient interpretation — map model-space weights back to feature names
# -----------------------------------------------------------------------------

@dataclass
class Coefs:
    table: List[Tuple[str, float, float]]  # (name, coef, odds_ratio)


def interpret_coefficients(fit: FitResult) -> Coefs:
    # Extract the fitted LogisticRegression from the pipeline.
    clf: LogisticRegression = fit.pipe.named_steps["clf"]

    # scikit-learn stores coefficients in model space matching feature_names_out.
    beta = clf.coef_.ravel()
    names = fit.feature_names_out

    # Convert to odds ratios: OR = exp(coef).
    orat = np.exp(beta)
    table = [(names[i], float(beta[i]), float(orat[i])) for i in range(len(names))]

    # Sort by absolute effect size to prioritize discussion.
    table.sort(key=lambda t: abs(t[1]), reverse=True)
    return Coefs(table=table)


# -----------------------------------------------------------------------------
# 6) Optional: classical inference via statsmodels GLM if available
# -----------------------------------------------------------------------------
# Why include this: scikit-learn optimizes prediction; it doesn't report
# standard errors, confidence intervals, or p-values. If I need those today, I
# either use statsmodels in Python or switch to R for GLM summaries.

@dataclass
class GLMInference:
    params: np.ndarray
    bse: np.ndarray
    pvalues: np.ndarray


def infer_with_statsmodels(fit: FitResult) -> Optional[GLMInference]:
    if not HAS_SM:
        return None
    # I rebuild the model matrix X̃ by running the preprocessor on train data.
    pre: ColumnTransformer = fit.pipe.named_steps["pre"]
    X_design = pre.transform(fit.X_train)
    y = fit.y_train

    # Add intercept column because statsmodels does not add it by default.
    X_design = sm.add_constant(X_design, has_constant="add")

    model = sm.GLM(y, X_design, family=sm.families.Binomial())
    res = model.fit()
    # I return params/bse/p-values so the caller can print concise highlights.
    return GLMInference(params=res.params, bse=res.bse, pvalues=res.pvalues)


# -----------------------------------------------------------------------------
# 7) Hyperparameter search — a small grid to show regularization choices
# -----------------------------------------------------------------------------

@dataclass
class SearchResult:
    best_params: dict
    best_auc: float


def small_grid_search(ds: Dataset) -> SearchResult:
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=0.25, random_state=11, stratify=ds.y
    )
    pipe, _ = build_pipeline(ds, penalty="l2", class_weight=None)

    # I search only C and class_weight here for speed; expand if needed.
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__class_weight": [None, "balanced"],
        # Note: penalty depends on solver; keeping to L2+lbfgs here.
    }
    cv = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=5, n_jobs=1)
    cv.fit(X_train, y_train)
    proba = cv.best_estimator_.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return SearchResult(best_params=cv.best_params_, best_auc=float(auc))


# -----------------------------------------------------------------------------
# 8) Multicollinearity check — quick VIF on the design matrix
# -----------------------------------------------------------------------------
# High collinearity inflates variance of β; for interpretability, I like to
# glance at VIFs. This is educational; for production, consider domain review.

@dataclass
class VIF:
    names: List[str]
    scores: List[float]


def compute_vif(fit: FitResult) -> VIF:
    pre: ColumnTransformer = fit.pipe.named_steps["pre"]
    X_design = pre.transform(fit.X_train)
    names = fit.feature_names_out

    # VIF_i = 1 / (1 - R^2_i), where R^2_i comes from regressing column i on others
    vifs: List[float] = []
    for i in range(X_design.shape[1]):
        y = X_design[:, i]
        X = np.delete(X_design, i, axis=1)
        # Add intercept term for the auxiliary regression.
        X = np.c_[np.ones(len(X)), X]
        # Closed-form OLS: β = (XᵀX)⁻¹Xᵀy
        XT_X = X.T @ X
        try:
            beta = np.linalg.solve(XT_X, X.T @ y)
        except np.linalg.LinAlgError:
            vifs.append(float("inf")); continue
        y_hat = X @ beta
        ssr = float(np.sum((y - y_hat) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ssr / sst if sst > 0 else 0.0)
        vifs.append(1.0 / max(1e-12, (1.0 - r2)))
    return VIF(names=list(names), scores=vifs)


# -----------------------------------------------------------------------------
# 9) Main narrative — train, evaluate, interpret, tune, and reflect
# -----------------------------------------------------------------------------

def main() -> int:
    intro()

    # Build the dataset; no files required, fully deterministic.
    ds = make_dataset(n=2500, seed=7)

    block("1) Fit a baseline logistic regression (L2, balanced vs not)")
    fit, m = fit_and_evaluate(ds, penalty="l2", class_weight=None, C=1.0)
    print("ROC AUC:", round(m.auc_roc, 4), "| PR AUC:", round(m.auc_pr, 4), "| LogLoss:", round(m.logloss, 4))
    print("Confusion matrix @0.5 threshold:\n", m.cm)
    print(m.report)

    block("2) Tune the decision threshold (maximize Youden's J)")
    thr = tune_threshold(fit.y_test, fit.y_proba)
    print("Best threshold by Youden's J:", round(thr.threshold, 4), "| TPR:", round(thr.tpr, 3), "| FPR:", round(thr.fpr, 3))
    y_pred_tuned = (fit.y_proba >= thr.threshold).astype(int)
    print("Confusion matrix @tuned threshold:\n", confusion_matrix(fit.y_test, y_pred_tuned))

    block("3) Coefficient interpretation (log-odds and odds ratios)")
    coefs = interpret_coefficients(fit)
    for name, b, orat in coefs.table[:10]:  # top 10 by magnitude
        print(f"{name:<24} β={b:+.3f}   OR={orat:.3f}")
    print("Note: coefficients are on the scale *after* scaling/encoding.")

    if HAS_SM:
        block("4) Optional classical inference (statsmodels GLM)")
        inf = infer_with_statsmodels(fit)
        if inf is not None:
            # I display only the first few entries to keep output tidy.
            print("params[:5] ->", np.round(inf.params[:5], 4))
            print("bse[:5]    ->", np.round(inf.bse[:5], 4))
            print("pvalues[:5]->", np.round(inf.pvalues[:5], 4))
            print("If I needed full coefficient tables with CIs/p-values today,\n"
                  "I'd use statsmodels (here) or switch to R for glm().")

    block("5) Small hyperparameter search (C, class_weight)")
    gs = small_grid_search(ds)
    print("Best CV params:", gs.best_params, "| Test ROC AUC from best:", round(gs.best_auc, 4))

    block("6) Multicollinearity quick scan (VIF)")
    vif = compute_vif(fit)
    for nm, score in sorted(zip(vif.names, vif.scores), key=lambda t: t[1], reverse=True)[:8]:
        print(f"VIF {nm:<24} -> {score:.2f}")
    print("Heuristic: VIF≫10 can signal unstable coefficient estimates.")

    block("7) When I would switch to R (judgment, not dogma)")
    print("• I need a formula-driven GLM with complex categorical contrasts,")
    print("  reference level control, and polished inference tables quickly.")
    print("• Small-sample exact tests or Firth bias reduction are required.")
    print("• Mixed-effects logistic regression with rich random-effects structures.")
    print("• The rest of the team is analyzing in R and we must match outputs.")
    print("Python remains excellent for pipelines, deployment, and scale;\n"
          "R remains excellent for *statistical reporting* and certain GLM niceties.")

    block("8) Closing notes: regularization choices")
    print("• L2 (ridge): default; stabilizes coefficients; works with lbfgs.")
    print("• L1 (lasso): sparse solutions; prefer liblinear/saga; helpful for feature selection.")
    print("• ElasticNet: compromise (L1+L2); choose saga; tune l1_ratio.")

    return 0


# -----------------------------------------------------------------------------
# Minimal self-checks so failures are noisy during CI
# -----------------------------------------------------------------------------

def _tests() -> None:
    ds = make_dataset(n=400, seed=3)
    fit, m = fit_and_evaluate(ds, penalty="l2", class_weight=None, C=1.0)
    assert 0.7 <= m.auc_roc <= 1.0  # synthetic set should be learnable
    thr = tune_threshold(fit.y_test, fit.y_proba)
    assert 0.0 <= thr.threshold <= 1.0
    c = interpret_coefficients(fit)
    assert len(c.table) == len(fit.feature_names_out)


if __name__ == "__main__":
    _tests()
    raise SystemExit(main())
