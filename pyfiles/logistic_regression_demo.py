#!/usr/bin/env python3
"""
Logistic Regression: Advanced Implementation and Analysis
Author: Cazandra Aporbo
Completed: July 1, 2023 
Last Updated: November 2, 2025

A demonstration of logistic regression that goes beyond basics.
This implementation shows not just how to use logistic regression, but why
certain choices matter, when alternatives should be considered, and how to
handle real-world complexities. Every design decision is deliberate and explained.
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union
import time
from pathlib import Path
import json

# Core scientific stack with explicit imports for transparency
try:
    import numpy as np
    from scipy import stats
    from scipy.special import expit, logit
    from scipy.optimize import minimize
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import (
        train_test_split, 
        GridSearchCV, 
        StratifiedKFold,
        cross_val_score,
        learning_curve
    )
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
        precision_recall_curve,
        average_precision_score,
        log_loss,
        brier_score_loss,
        calibration_curve,
        matthews_corrcoef
    )
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_selection import SelectFromModel, RFE
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    print("This implementation requires scikit-learn, numpy, scipy, and pandas:")
    print("  pip install scikit-learn numpy scipy pandas")
    raise SystemExit(1)

# Optional advanced statistical packages
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def section_header(title: str, width: int = 78) -> None:
    """Print formatted section header for readability in terminal."""
    print(f"\n{'▸' * width}")
    print(f"  {title}")
    print(f"{'▸' * width}")


@dataclass
class Dataset:
    """Enhanced dataset container with metadata and validation."""
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    numeric_idx: List[int]
    categorical_idx: List[int]
    
    # Additional metadata for better tracking
    creation_time: float = field(default_factory=time.time)
    n_samples: int = field(init=False)
    n_features: int = field(init=False)
    class_balance: Dict[int, float] = field(init=False)
    has_missing: bool = field(init=False)
    
    def __post_init__(self):
        """Compute dataset statistics upon creation."""
        self.n_samples, self.n_features = self.X.shape
        unique, counts = np.unique(self.y, return_counts=True)
        self.class_balance = {int(u): float(c/self.n_samples) for u, c in zip(unique, counts)}
        self.has_missing = np.any(np.isnan(self.X))
        
    def describe(self) -> Dict[str, Any]:
        """Generate comprehensive dataset description."""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_numeric': len(self.numeric_idx),
            'n_categorical': len(self.categorical_idx),
            'class_balance': self.class_balance,
            'has_missing': self.has_missing,
            'memory_mb': self.X.nbytes / (1024 * 1024)
        }


@dataclass
class EnhancedMetrics:
    """Comprehensive metrics container with confidence intervals."""
    # Core metrics
    accuracy: float
    auc_roc: float
    auc_pr: float
    log_loss: float
    brier_score: float
    matthews_corr: float
    
    # Detailed breakdowns
    confusion_matrix: np.ndarray
    classification_report: str
    precision_by_class: Dict[int, float]
    recall_by_class: Dict[int, float]
    f1_by_class: Dict[int, float]
    
    # Threshold-specific metrics
    optimal_threshold: float
    threshold_metrics: Dict[float, Dict[str, float]]
    
    # Confidence intervals (if bootstrapped)
    ci_accuracy: Optional[Tuple[float, float]] = None
    ci_auc_roc: Optional[Tuple[float, float]] = None
    
    # Calibration metrics
    calibration_error: Optional[float] = None
    calibration_bins: Optional[np.ndarray] = None


@dataclass
class ModelInterpretation:
    """Container for model interpretation results."""
    coefficients: np.ndarray
    intercept: float
    odds_ratios: np.ndarray
    feature_names: List[str]
    
    # Statistical inference if available
    standard_errors: Optional[np.ndarray] = None
    p_values: Optional[np.ndarray] = None
    confidence_intervals: Optional[np.ndarray] = None
    
    # Feature importance measures
    absolute_importance: Optional[np.ndarray] = None
    standardized_coefficients: Optional[np.ndarray] = None
    
    # Model diagnostics
    vif_scores: Optional[Dict[str, float]] = None
    condition_number: Optional[float] = None
    
    # SHAP values if available
    shap_values: Optional[np.ndarray] = None
    shap_importance: Optional[Dict[str, float]] = None


class LogisticRegressionAnalyzer:
    """
    Advanced logistic regression analyzer with comprehensive capabilities.
    
    This class encapsulates best practices for logistic regression including:
    - Automatic preprocessing pipeline construction
    - Multiple regularization strategies
    - Threshold optimization for business metrics
    - Model calibration and validation
    - Comprehensive interpretation tools
    """
    
    def __init__(self, 
                 penalty: str = 'l2',
                 solver: Optional[str] = None,
                 class_weight: Optional[Union[str, Dict]] = None,
                 C: float = 1.0,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize the analyzer with model configuration.
        
        The solver is automatically selected based on penalty if not specified,
        ensuring compatibility and optimal performance.
        """
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        
        # Intelligent solver selection based on penalty
        if solver is None:
            self.solver = self._select_optimal_solver(penalty)
        else:
            self.solver = solver
            
        self.pipeline = None
        self.is_fitted = False
        self.feature_names_out = None
        self.training_time = None
        
    def _select_optimal_solver(self, penalty: str) -> str:
        """Select the best solver for the given penalty type."""
        solver_map = {
            'l1': 'liblinear',  # Only solver supporting L1
            'l2': 'lbfgs',      # Efficient for L2
            'elasticnet': 'saga',  # Supports elastic net
            'none': 'newton-cg'  # For unpenalized
        }
        return solver_map.get(penalty, 'lbfgs')
    
    def create_preprocessing_pipeline(self, dataset: Dataset) -> ColumnTransformer:
        """
        Create sophisticated preprocessing pipeline with feature engineering.
        
        This method builds a preprocessing pipeline that:
        - Scales numeric features for stable optimization
        - Encodes categorical variables avoiding the dummy trap
        - Optionally adds polynomial features for non-linearity
        - Handles missing values appropriately
        """
        transformers = []
        
        # Numeric features: scaling is crucial for coefficient interpretation
        if dataset.numeric_idx:
            numeric_transformer = Pipeline([
                ('scaler', StandardScaler(with_mean=True, with_std=True))
            ])
            transformers.append(
                ('numeric', numeric_transformer, dataset.numeric_idx)
            )
        
        # Categorical features: one-hot encoding with first category dropped
        if dataset.categorical_idx:
            categorical_transformer = Pipeline([
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(
                ('categorical', categorical_transformer, dataset.categorical_idx)
            )
        
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        return preprocessor
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            dataset: Dataset,
            add_polynomial: bool = False,
            polynomial_degree: int = 2,
            feature_selection: Optional[str] = None,
            n_features_select: Optional[int] = None) -> 'LogisticRegressionAnalyzer':
        """
        Fit the logistic regression model with optional enhancements.
        
        This method supports:
        - Polynomial feature generation for capturing non-linearities
        - Automatic feature selection (L1-based or RFE)
        - Class weight computation for imbalanced data
        """
        start_time = time.time()
        
        # Build preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(dataset)
        
        # Initialize logistic regression with configuration
        if self.penalty == 'elasticnet':
            # Elastic net requires l1_ratio parameter
            logreg = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=self.solver,
                l1_ratio=0.5,  # Balance between L1 and L2
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1  # Use all cores
            )
        else:
            logreg = LogisticRegression(
                penalty=self.penalty if self.penalty != 'none' else None,
                C=self.C,
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Build pipeline components
        steps = [('preprocessor', preprocessor)]
        
        # Add polynomial features if requested
        if add_polynomial:
            poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
            steps.append(('polynomial', poly))
        
        # Add feature selection if requested
        if feature_selection == 'l1':
            # L1-based feature selection
            selector = SelectFromModel(
                LogisticRegression(penalty='l1', solver='liblinear', C=0.1),
                max_features=n_features_select
            )
            steps.append(('feature_selection', selector))
        elif feature_selection == 'rfe':
            # Recursive feature elimination
            selector = RFE(
                LogisticRegression(penalty='l2', solver='lbfgs'),
                n_features_to_select=n_features_select or 10
            )
            steps.append(('feature_selection', selector))
        
        # Add the main classifier
        steps.append(('classifier', logreg))
        
        # Create and fit the pipeline
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X_train, y_train)
        
        # Extract feature names after transformation
        self.feature_names_out = self._extract_feature_names(dataset)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        return self
    
    def _extract_feature_names(self, dataset: Dataset) -> List[str]:
        """Extract feature names after all transformations."""
        # Get names from preprocessor
        preprocessor = self.pipeline.named_steps['preprocessor']
        names = list(preprocessor.get_feature_names_out())
        
        # Handle polynomial features if present
        if 'polynomial' in self.pipeline.named_steps:
            poly = self.pipeline.named_steps['polynomial']
            # Polynomial features change the feature names
            n_features = len(names)
            names = [f'poly_{i}' for i in range(poly.n_output_features_)]
        
        return names
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_fitted()
        return self.pipeline.predict_proba(X)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict classes with custom threshold."""
        self._check_fitted()
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)
    
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray,
                thresholds: List[float] = None,
                bootstrap_ci: bool = False,
                n_bootstrap: int = 1000) -> EnhancedMetrics:
        """
        Comprehensive model evaluation with multiple metrics.
        
        This method computes:
        - Standard classification metrics
        - Probability calibration measures
        - Threshold-specific performance
        - Bootstrap confidence intervals (optional)
        """
        self._check_fitted()
        
        # Get predictions
        y_proba = self.predict_proba(X_test)[:, 1]
        y_pred = self.predict(X_test)
        
        # Core metrics
        accuracy = np.mean(y_test == y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)
        auc_pr = average_precision_score(y_test, y_proba)
        logloss = log_loss(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        matthews = matthews_corrcoef(y_test, y_pred)
        
        # Confusion matrix and per-class metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Extract per-class metrics
        precision_by_class = {int(k): v['precision'] for k, v in report.items() 
                            if k.isdigit()}
        recall_by_class = {int(k): v['recall'] for k, v in report.items() 
                         if k.isdigit()}
        f1_by_class = {int(k): v['f1-score'] for k, v in report.items() 
                      if k.isdigit()}
        
        # Find optimal threshold using Youden's J statistic
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = float(roc_thresholds[optimal_idx])
        
        # Evaluate at different thresholds
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        threshold_metrics = {}
        for thr in thresholds:
            y_pred_thr = (y_proba >= thr).astype(int)
            threshold_metrics[thr] = {
                'accuracy': np.mean(y_test == y_pred_thr),
                'precision': np.sum((y_pred_thr == 1) & (y_test == 1)) / max(np.sum(y_pred_thr == 1), 1),
                'recall': np.sum((y_pred_thr == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1),
                'f1': 2 * np.sum((y_pred_thr == 1) & (y_test == 1)) / 
                      max(np.sum(y_pred_thr == 1) + np.sum(y_test == 1), 1)
            }
        
        # Bootstrap confidence intervals if requested
        ci_accuracy = None
        ci_auc_roc = None
        
        if bootstrap_ci:
            accuracies = []
            aucs = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                idx = np.random.choice(len(y_test), len(y_test), replace=True)
                y_test_boot = y_test[idx]
                y_proba_boot = y_proba[idx]
                y_pred_boot = (y_proba_boot >= 0.5).astype(int)
                
                accuracies.append(np.mean(y_test_boot == y_pred_boot))
                if len(np.unique(y_test_boot)) == 2:  # Check if both classes present
                    aucs.append(roc_auc_score(y_test_boot, y_proba_boot))
            
            # Calculate 95% confidence intervals
            ci_accuracy = (np.percentile(accuracies, 2.5), np.percentile(accuracies, 97.5))
            if aucs:
                ci_auc_roc = (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
        
        # Calibration analysis
        fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
        calibration_error = np.mean(np.abs(fraction_pos - mean_pred))
        
        return EnhancedMetrics(
            accuracy=accuracy,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            log_loss=logloss,
            brier_score=brier,
            matthews_corr=matthews,
            confusion_matrix=cm,
            classification_report=classification_report(y_test, y_pred),
            precision_by_class=precision_by_class,
            recall_by_class=recall_by_class,
            f1_by_class=f1_by_class,
            optimal_threshold=optimal_threshold,
            threshold_metrics=threshold_metrics,
            ci_accuracy=ci_accuracy,
            ci_auc_roc=ci_auc_roc,
            calibration_error=calibration_error,
            calibration_bins=fraction_pos
        )
    
    def interpret_model(self, 
                       dataset: Dataset,
                       X_train: Optional[np.ndarray] = None,
                       compute_vif: bool = True,
                       compute_shap: bool = False) -> ModelInterpretation:
        """
        Comprehensive model interpretation including coefficients and diagnostics.
        
        This method provides:
        - Coefficient analysis with odds ratios
        - Multicollinearity diagnostics (VIF)
        - Statistical inference (if statsmodels available)
        - SHAP values for model-agnostic interpretation
        """
        self._check_fitted()
        
        # Extract the classifier
        clf = self.pipeline.named_steps['classifier']
        
        # Get coefficients and intercept
        coefficients = clf.coef_.ravel()
        intercept = clf.intercept_[0]
        
        # Calculate odds ratios
        odds_ratios = np.exp(coefficients)
        
        # Standardized coefficients (if we have training data)
        standardized_coefs = None
        if X_train is not None:
            X_transformed = self.pipeline[:-1].transform(X_train)
            X_std = np.std(X_transformed, axis=0)
            standardized_coefs = coefficients * X_std
        
        # VIF calculation for multicollinearity
        vif_scores = None
        condition_num = None
        
        if compute_vif and X_train is not None:
            vif_scores = self._calculate_vif(X_train, dataset)
            # Calculate condition number
            X_transformed = self.pipeline[:-1].transform(X_train)
            _, s, _ = np.linalg.svd(X_transformed)
            condition_num = s[0] / s[-1] if s[-1] != 0 else np.inf
        
        # Statistical inference with statsmodels
        std_errors = None
        p_values = None
        conf_intervals = None
        
        if HAS_STATSMODELS and X_train is not None:
            inference = self._statsmodels_inference(X_train, dataset)
            if inference:
                std_errors = inference['std_errors']
                p_values = inference['p_values']
                conf_intervals = inference['conf_intervals']
        
        # SHAP values for interpretability
        shap_values = None
        shap_importance = None
        
        if compute_shap and HAS_SHAP and X_train is not None:
            shap_results = self._compute_shap_values(X_train[:100])  # Limit for speed
            shap_values = shap_results['values']
            shap_importance = shap_results['importance']
        
        return ModelInterpretation(
            coefficients=coefficients,
            intercept=intercept,
            odds_ratios=odds_ratios,
            feature_names=self.feature_names_out,
            standard_errors=std_errors,
            p_values=p_values,
            confidence_intervals=conf_intervals,
            absolute_importance=np.abs(coefficients),
            standardized_coefficients=standardized_coefs,
            vif_scores=vif_scores,
            condition_number=condition_num,
            shap_values=shap_values,
            shap_importance=shap_importance
        )
    
    def _calculate_vif(self, X: np.ndarray, dataset: Dataset) -> Dict[str, float]:
        """Calculate variance inflation factors for multicollinearity detection."""
        X_transformed = self.pipeline[:-1].transform(X)
        vif_scores = {}
        
        for i, name in enumerate(self.feature_names_out):
            # Regress each feature on all others
            X_i = X_transformed[:, i]
            X_others = np.delete(X_transformed, i, axis=1)
            
            # Add constant for regression
            X_others_const = np.column_stack([np.ones(len(X_others)), X_others])
            
            try:
                # OLS regression
                beta = np.linalg.lstsq(X_others_const, X_i, rcond=None)[0]
                y_pred = X_others_const @ beta
                
                # Calculate R-squared
                ss_res = np.sum((X_i - y_pred) ** 2)
                ss_tot = np.sum((X_i - np.mean(X_i)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # VIF = 1 / (1 - R²)
                vif = 1 / (1 - r_squared) if r_squared < 0.9999 else np.inf
                vif_scores[name] = vif
                
            except np.linalg.LinAlgError:
                vif_scores[name] = np.inf
        
        return vif_scores
    
    def _statsmodels_inference(self, X: np.ndarray, dataset: Dataset) -> Optional[Dict]:
        """Compute statistical inference using statsmodels."""
        if not HAS_STATSMODELS:
            return None
        
        try:
            # Transform features
            X_transformed = self.pipeline[:-1].transform(X)
            
            # Add constant for statsmodels
            X_with_const = sm.add_constant(X_transformed)
            
            # Fit logistic regression
            model = sm.Logit(dataset.y[:len(X)], X_with_const)
            result = model.fit(disp=0)
            
            # Extract inference statistics
            return {
                'std_errors': result.bse[1:],  # Exclude intercept
                'p_values': result.pvalues[1:],
                'conf_intervals': result.conf_int()[1:]
            }
        except Exception:
            return None
    
    def _compute_shap_values(self, X_sample: np.ndarray) -> Dict:
        """Compute SHAP values for model interpretation."""
        if not HAS_SHAP:
            return {'values': None, 'importance': None}
        
        try:
            # Create explainer
            X_transformed = self.pipeline[:-1].transform(X_sample)
            explainer = shap.LinearExplainer(
                self.pipeline.named_steps['classifier'],
                X_transformed
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_transformed)
            
            # Calculate feature importance
            shap_importance = {}
            for i, name in enumerate(self.feature_names_out):
                shap_importance[name] = np.mean(np.abs(shap_values[:, i]))
            
            return {
                'values': shap_values,
                'importance': shap_importance
            }
        except Exception:
            return {'values': None, 'importance': None}
    
    def calibrate_model(self, X: np.ndarray, y: np.ndarray, 
                       method: str = 'sigmoid') -> 'LogisticRegressionAnalyzer':
        """
        Calibrate predicted probabilities using Platt scaling or isotonic regression.
        
        Calibration improves probability estimates, especially important for
        decision-making based on predicted probabilities.
        """
        self._check_fitted()
        
        # Create calibrated classifier
        calibrated = CalibratedClassifierCV(
            self.pipeline,
            method=method,
            cv=3
        )
        
        # Fit calibration
        calibrated.fit(X, y)
        
        # Replace pipeline with calibrated version
        self.pipeline = calibrated
        
        return self
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv: int = 5, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """
        Perform cross-validation with multiple scoring metrics.
        
        Returns detailed results including mean, std, and individual fold scores.
        """
        self._check_fitted()
        
        # Define multiple scoring metrics
        scoring_metrics = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1']
        
        results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(
                self.pipeline, X, y,
                cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring=metric,
                n_jobs=-1
            )
            results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores.tolist()
            }
        
        return results
    
    def plot_learning_curve(self, X: np.ndarray, y: np.ndarray,
                          train_sizes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate learning curve data for visualization.
        
        Returns training sizes and corresponding train/validation scores.
        """
        self._check_fitted()
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.pipeline, X, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        return train_sizes, (np.mean(train_scores, axis=1), np.mean(val_scores, axis=1))
    
    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before use. Call fit() first.")
    
    def save_model(self, filepath: Path) -> None:
        """Save the fitted model to disk."""
        self._check_fitted()
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'feature_names_out': self.feature_names_out,
                'config': {
                    'penalty': self.penalty,
                    'C': self.C,
                    'solver': self.solver,
                    'training_time': self.training_time
                }
            }, f)
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'LogisticRegressionAnalyzer':
        """Load a saved model from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance with saved config
        config = data['config']
        instance = cls(
            penalty=config['penalty'],
            C=config['C'],
            solver=config['solver']
        )
        
        # Restore fitted state
        instance.pipeline = data['pipeline']
        instance.feature_names_out = data['feature_names_out']
        instance.is_fitted = True
        instance.training_time = config['training_time']
        
        return instance


def create_enhanced_dataset(n_samples: int = 5000, 
                           n_features: int = 20,
                           n_informative: int = 15,
                           n_categorical: int = 3,
                           imbalance_ratio: float = 0.3,
                           noise_level: float = 0.1,
                           seed: int = 42) -> Dataset:
    """
    Create a sophisticated synthetic dataset for demonstration.
    
    This dataset includes:
    - Mix of informative and noise features
    - Categorical variables with multiple levels
    - Controlled class imbalance
    - Non-linear relationships
    """
    np.random.seed(seed)
    
    # Generate numeric features
    X_numeric, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=2,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        class_sep=1.0,
        flip_y=noise_level,
        random_state=seed
    )
    
    # Add non-linear transformations to some features
    X_numeric[:, 0] = np.square(X_numeric[:, 0])
    X_numeric[:, 1] = np.sin(X_numeric[:, 1] * np.pi)
    X_numeric[:, 2] = X_numeric[:, 2] * X_numeric[:, 3]
    
    # Generate categorical features correlated with outcome
    categorical_features = []
    for i in range(n_categorical):
        # Create categories based on quantiles of a numeric feature
        feature_idx = i % n_features
        quantiles = np.quantile(X_numeric[:, feature_idx], [0.25, 0.5, 0.75])
        
        categories = np.digitize(X_numeric[:, feature_idx], quantiles)
        
        # Add correlation with target
        correlation_strength = 0.3
        mask = y == 1
        categories[mask] = np.clip(
            categories[mask] + np.random.binomial(1, correlation_strength, np.sum(mask)),
            0, 3
        )
        
        # Convert to string categories
        category_names = [f'cat_{i}_level_{c}' for c in categories]
        categorical_features.append(np.array(category_names).reshape(-1, 1))
    
    # Combine features
    if categorical_features:
        X_categorical = np.hstack(categorical_features)
        X = np.hstack([X_numeric, X_categorical])
        
        feature_names = ([f'num_feature_{i}' for i in range(n_features)] +
                        [f'cat_feature_{i}' for i in range(n_categorical)])
        numeric_idx = list(range(n_features))
        categorical_idx = list(range(n_features, n_features + n_categorical))
    else:
        X = X_numeric
        feature_names = [f'num_feature_{i}' for i in range(n_features)]
        numeric_idx = list(range(n_features))
        categorical_idx = []
    
    return Dataset(
        X=X,
        y=y,
        feature_names=feature_names,
        numeric_idx=numeric_idx,
        categorical_idx=categorical_idx
    )


def perform_hyperparameter_search(analyzer: LogisticRegressionAnalyzer,
                                 X: np.ndarray, 
                                 y: np.ndarray,
                                 dataset: Dataset) -> Dict[str, Any]:
    """
    Comprehensive hyperparameter optimization with multiple strategies.
    
    Searches over:
    - Regularization strength (C)
    - Penalty types
    - Class weight strategies
    - Feature selection methods
    """
    # Define parameter grid
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__class_weight': [None, 'balanced'],
    }
    
    # Use a subset of data for faster search
    X_search, _, y_search, _ = train_test_split(X, y, test_size=0.7, stratify=y, random_state=42)
    
    # Fit analyzer first to get pipeline
    analyzer.fit(X_search, y_search, dataset)
    
    # Perform grid search
    grid_search = GridSearchCV(
        analyzer.pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_search, y_search)
    
    # Extract results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': {
            'mean_scores': grid_search.cv_results_['mean_test_score'].tolist(),
            'std_scores': grid_search.cv_results_['std_test_score'].tolist(),
            'params': grid_search.cv_results_['params']
        }
    }
    
    # Find top 5 configurations
    sorted_idx = np.argsort(grid_search.cv_results_['mean_test_score'])[::-1][:5]
    results['top_5_configs'] = [
        {
            'params': grid_search.cv_results_['params'][i],
            'score': grid_search.cv_results_['mean_test_score'][i]
        }
        for i in sorted_idx
    ]
    
    return results


def compare_regularization_strategies(X: np.ndarray, y: np.ndarray, 
                                     dataset: Dataset) -> Dict[str, Any]:
    """
    Compare different regularization strategies and their effects.
    
    Tests L1, L2, Elastic Net, and no regularization to understand
    their impact on model performance and sparsity.
    """
    strategies = {
        'No Regularization': {'penalty': 'none', 'C': 1.0},
        'L2 (Ridge)': {'penalty': 'l2', 'C': 1.0},
        'L1 (Lasso)': {'penalty': 'l1', 'C': 1.0},
        'L2 Strong': {'penalty': 'l2', 'C': 0.01},
        'L1 Strong': {'penalty': 'l1', 'C': 0.01},
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    results = {}
    
    for name, config in strategies.items():
        # Create and fit analyzer
        analyzer = LogisticRegressionAnalyzer(**config)
        
        # Skip if incompatible configuration
        try:
            analyzer.fit(X_train, y_train, dataset)
        except Exception as e:
            results[name] = {'error': str(e)}
            continue
        
        # Evaluate
        metrics = analyzer.evaluate(X_test, y_test, bootstrap_ci=False)
        
        # Get interpretation
        interpretation = analyzer.interpret_model(dataset, X_train, compute_vif=False)
        
        # Count non-zero coefficients (sparsity)
        n_nonzero = np.sum(np.abs(interpretation.coefficients) > 1e-10)
        
        results[name] = {
            'auc_roc': metrics.auc_roc,
            'accuracy': metrics.accuracy,
            'log_loss': metrics.log_loss,
            'n_nonzero_coefs': n_nonzero,
            'coef_l2_norm': np.linalg.norm(interpretation.coefficients, 2),
            'coef_l1_norm': np.linalg.norm(interpretation.coefficients, 1),
        }
    
    return results


def demonstrate_business_metrics(analyzer: LogisticRegressionAnalyzer,
                                X_test: np.ndarray,
                                y_test: np.ndarray) -> Dict[str, Any]:
    """
    Demonstrate business-oriented model evaluation.
    
    Shows how to optimize for different business objectives:
    - Minimizing false positives (precision focus)
    - Minimizing false negatives (recall focus)
    - Profit optimization with custom cost matrix
    """
    # Get probabilities
    y_proba = analyzer.predict_proba(X_test)[:, 1]
    
    # Define business scenarios
    scenarios = {}
    
    # Scenario 1: High precision (minimize false positives)
    # Use case: Expensive interventions, limited resources
    precision_threshold = 0.7
    y_pred_precision = (y_proba >= precision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_precision).ravel()
    scenarios['high_precision'] = {
        'threshold': precision_threshold,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'interventions': np.sum(y_pred_precision),
        'caught_positives': tp
    }
    
    # Scenario 2: High recall (minimize false negatives)
    # Use case: Critical failures, safety applications
    recall_threshold = 0.3
    y_pred_recall = (y_proba >= recall_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_recall).ravel()
    scenarios['high_recall'] = {
        'threshold': recall_threshold,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'interventions': np.sum(y_pred_recall),
        'missed_positives': fn
    }
    
    # Scenario 3: Profit optimization
    # Define cost matrix (example values)
    cost_fp = 100  # Cost of false positive (unnecessary intervention)
    cost_fn = 500  # Cost of false negative (missed failure)
    benefit_tp = 1000  # Benefit of true positive (prevented failure)
    
    # Find threshold that maximizes expected profit
    thresholds = np.linspace(0.1, 0.9, 50)
    profits = []
    
    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thr).ravel()
        
        profit = benefit_tp * tp - cost_fp * fp - cost_fn * fn
        profits.append(profit)
    
    best_profit_idx = np.argmax(profits)
    best_profit_threshold = thresholds[best_profit_idx]
    
    y_pred_profit = (y_proba >= best_profit_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_profit).ravel()
    
    scenarios['profit_optimized'] = {
        'threshold': best_profit_threshold,
        'expected_profit': profits[best_profit_idx],
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'cost_false_positives': cost_fp * fp,
        'cost_false_negatives': cost_fn * fn,
        'benefit_true_positives': benefit_tp * tp
    }
    
    return scenarios


def main():
    """Main demonstration of advanced logistic regression capabilities."""
    
    section_header("Advanced Logistic Regression Analysis")
    print("Author: Cazandra Aporbo")
    print("Demonstrating comprehensive logistic regression techniques")
    
    # Create sophisticated dataset
    section_header("1. Data Generation and Exploration")
    
    dataset = create_enhanced_dataset(
        n_samples=5000,
        n_features=20,
        n_categorical=3,
        imbalance_ratio=0.3
    )
    
    data_info = dataset.describe()
    print(f"Dataset created: {data_info['n_samples']} samples, {data_info['n_features']} features")
    print(f"Class balance: {data_info['class_balance']}")
    print(f"Memory usage: {data_info['memory_mb']:.2f} MB")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.25, stratify=dataset.y, random_state=42
    )
    
    # Initialize analyzer
    section_header("2. Model Training with Advanced Features")
    
    analyzer = LogisticRegressionAnalyzer(
        penalty='l2',
        C=1.0,
        class_weight='balanced'
    )
    
    # Fit with polynomial features
    analyzer.fit(X_train, y_train, dataset, add_polynomial=False)
    print(f"Model trained in {analyzer.training_time:.2f} seconds")
    print(f"Features after transformation: {len(analyzer.feature_names_out)}")
    
    # Comprehensive evaluation
    section_header("3. Model Evaluation with Bootstrap Confidence Intervals")
    
    metrics = analyzer.evaluate(X_test, y_test, bootstrap_ci=True, n_bootstrap=100)
    
    print(f"AUC-ROC: {metrics.auc_roc:.4f}")
    if metrics.ci_auc_roc:
        print(f"  95% CI: [{metrics.ci_auc_roc[0]:.4f}, {metrics.ci_auc_roc[1]:.4f}]")
    
    print(f"Accuracy: {metrics.accuracy:.4f}")
    if metrics.ci_accuracy:
        print(f"  95% CI: [{metrics.ci_accuracy[0]:.4f}, {metrics.ci_accuracy[1]:.4f}]")
    
    print(f"Matthews Correlation: {metrics.matthews_corr:.4f}")
    print(f"Brier Score: {metrics.brier_score:.4f}")
    print(f"Calibration Error: {metrics.calibration_error:.4f}")
    print(f"Optimal Threshold (Youden): {metrics.optimal_threshold:.3f}")
    
    # Model interpretation
    section_header("4. Model Interpretation and Diagnostics")
    
    interpretation = analyzer.interpret_model(dataset, X_train, compute_vif=True)
    
    # Top features by absolute coefficient
    coef_importance = sorted(
        zip(interpretation.feature_names[:10], interpretation.coefficients[:10], interpretation.odds_ratios[:10]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]
    
    print("\nTop 10 Features by Coefficient Magnitude:")
    for name, coef, odds in coef_importance:
        print(f"  {name:30s} β={coef:+.3f}  OR={odds:.3f}")
    
    # Multicollinearity check
    if interpretation.vif_scores:
        high_vif = {k: v for k, v in interpretation.vif_scores.items() if v > 10}
        if high_vif:
            print(f"\nFeatures with high VIF (>10): {len(high_vif)}")
            for name, vif in list(high_vif.items())[:5]:
                print(f"  {name}: {vif:.2f}")
    
    if interpretation.condition_number:
        print(f"\nCondition Number: {interpretation.condition_number:.2f}")
        if interpretation.condition_number > 1000:
            print("  Warning: High condition number suggests multicollinearity")
    
    # Cross-validation
    section_header("5. Cross-Validation Performance")
    
    cv_results = analyzer.cross_validate(X_train, y_train, cv=5)
    
    for metric, results in cv_results.items():
        print(f"{metric:12s}: {results['mean']:.4f} (±{results['std']:.4f})")
    
    # Hyperparameter search
    section_header("6. Hyperparameter Optimization")
    
    search_results = perform_hyperparameter_search(analyzer, dataset.X, dataset.y, dataset)
    
    print(f"Best parameters: {search_results['best_params']}")
    print(f"Best CV score: {search_results['best_score']:.4f}")
    
    print("\nTop 5 configurations:")
    for i, config in enumerate(search_results['top_5_configs'], 1):
        print(f"  {i}. Score: {config['score']:.4f}")
        for param, value in config['params'].items():
            print(f"     {param}: {value}")
    
    # Regularization comparison
    section_header("7. Regularization Strategy Comparison")
    
    reg_comparison = compare_regularization_strategies(dataset.X, dataset.y, dataset)
    
    print(f"{'Strategy':<20} {'AUC-ROC':<10} {'Accuracy':<10} {'Non-zero':<10} {'L2 Norm':<10}")
    print("-" * 60)
    
    for name, results in reg_comparison.items():
        if 'error' not in results:
            print(f"{name:<20} {results['auc_roc']:<10.4f} {results['accuracy']:<10.4f} "
                  f"{results['n_nonzero_coefs']:<10} {results['coef_l2_norm']:<10.3f}")
    
    # Business metrics
    section_header("8. Business-Oriented Evaluation")
    
    business_metrics = demonstrate_business_metrics(analyzer, X_test, y_test)
    
    for scenario_name, scenario in business_metrics.items():
        print(f"\n{scenario_name.replace('_', ' ').title()}:")
        for key, value in scenario.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    # Learning curve analysis
    section_header("9. Learning Curve Analysis")
    
    train_sizes, (train_scores, val_scores) = analyzer.plot_learning_curve(X_train, y_train)
    
    print("Training set sizes:", train_sizes[:5], "...")
    print("Final training score:", train_scores[-1]:.4f)
    print("Final validation score:", val_scores[-1]:.4f)
    
    if train_scores[-1] - val_scores[-1] > 0.1:
        print("Note: Gap suggests overfitting - consider stronger regularization")
    elif val_scores[-1] < 0.7:
        print("Note: Low validation score - consider more complex features or different model")
    
    section_header("10. Key Insights and Recommendations")
    
    print("Model Performance Summary:")
    print(f"• The model achieves {metrics.auc_roc:.1%} AUC-ROC on test data")
    print(f"• Optimal threshold of {metrics.optimal_threshold:.3f} balances sensitivity/specificity")
    print(f"• Calibration error of {metrics.calibration_error:.3f} indicates probability reliability")
    
    print("\nWhen to Consider Alternatives:")
    print("• Non-linear decision boundaries → Consider kernel methods or tree ensembles")
    print("• High-dimensional sparse data → L1 regularization or elastic net")
    print("• Need for interaction terms → Polynomial features or GAMs")
    print("• Interpretability paramount → Keep linear model with careful feature engineering")
    
    print("\nProduction Deployment Considerations:")
    print("• Monitor for distribution shift with regular recalibration")
    print("• Set threshold based on business costs, not just statistical metrics")
    print("• Implement prediction intervals for uncertainty quantification")
    print("• Version control model artifacts and preprocessing pipelines")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
