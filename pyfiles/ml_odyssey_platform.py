#!/usr/bin/env python3
"""
Machine Learning Odyssey Platform
Author: Cazandra Aporbo: Updated 11/2025
A educational system merging statistical analysis with interactive learning
This program demonstrates production-level ML engineering while teaching concepts through
practical challenges.

"""

from __future__ import annotations
import math
import sys
import time
import random
import re
import json
import hashlib
import pickle
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable, Union, TypeVar, Generic
from functools import lru_cache, wraps, partial
from contextlib import contextmanager
from enum import Enum, auto
from collections import defaultdict, deque, Counter
from pathlib import Path
import threading
import queue
import traceback

# Scientific computing essentials
try:
    import numpy as np
    from scipy import stats, special, optimize
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, log_loss,
        mean_squared_error, mean_absolute_error, r2_score,
        confusion_matrix, classification_report
    )
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("Note: Install numpy, scipy, scikit-learn for full functionality")
    print("Basic features will still work without them")

# Terminal styling for educational clarity
class Colors:
    """Terminal colors optimized for readability across different backgrounds."""
    RESET = "\033[0m"
    HEADER = "\033[95m"
    INFO = "\033[94m"
    SUCCESS = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    MUTED = "\033[90m"
    BOLD = "\033[1m"

T = TypeVar('T')

@dataclass
class ModelCard:
    """Comprehensive model documentation following responsible AI practices."""
    name: str
    version: str
    task_type: str
    performance_metrics: Dict[str, float]
    limitations: List[str]
    training_data_characteristics: Dict[str, Any]
    deployment_considerations: List[str]
    timestamp: float = field(default_factory=time.time)
    
    def to_json(self) -> str:
        """Serialize for model registry storage."""
        return json.dumps({
            'name': self.name,
            'version': self.version,
            'task_type': self.task_type,
            'metrics': self.performance_metrics,
            'limitations': self.limitations,
            'data_profile': self.training_data_characteristics,
            'deployment': self.deployment_considerations,
            'created_at': self.timestamp
        }, indent=2)

class MLConcept(Enum):
    """Core ML concepts for structured learning paths."""
    BIAS_VARIANCE = auto()
    REGULARIZATION = auto()
    CROSS_VALIDATION = auto()
    FEATURE_ENGINEERING = auto()
    HYPERPARAMETER_TUNING = auto()
    ENSEMBLE_METHODS = auto()
    INTERPRETABILITY = auto()
    DEPLOYMENT = auto()

@dataclass
class LearningObjective:
    """Educational objective with measurable outcomes."""
    concept: MLConcept
    description: str
    success_criteria: List[str]
    prerequisite_concepts: List[MLConcept]
    estimated_time_minutes: int
    
    def is_accessible(self, completed: set[MLConcept]) -> bool:
        """Check if prerequisites are met."""
        return all(prereq in completed for prereq in self.prerequisite_concepts)

class PerformanceMonitor:
    """Track computational and statistical performance metrics."""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.memory_snapshots = deque(maxlen=100)
        self.convergence_history = defaultdict(list)
        
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timing_data[operation].append(elapsed)
            if len(self.timing_data[operation]) > 10:
                # Keep running statistics without unbounded growth
                self.timing_data[operation] = self.timing_data[operation][-10:]
    
    def report_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        summary = {}
        for op, times in self.timing_data.items():
            if times:
                summary[op] = {
                    'mean_ms': np.mean(times) * 1000 if DEPS_AVAILABLE else sum(times)/len(times) * 1000,
                    'std_ms': np.std(times) * 1000 if DEPS_AVAILABLE else 0,
                    'calls': len(times)
                }
        return summary

class StatisticalValidator:
    """Rigorous statistical validation beyond basic metrics."""
    
    @staticmethod
    def check_assumptions(X: np.ndarray, y: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Validate statistical assumptions for different model types."""
        if not DEPS_AVAILABLE:
            return {'status': 'dependencies not available'}
        
        results = {}
        
        if model_type == 'linear_regression':
            # Check linearity via residual plots would go here
            # Check homoscedasticity
            residuals = y - np.mean(y)  # Simplified for demo
            _, p_value = stats.levene(residuals[:len(residuals)//2], 
                                     residuals[len(residuals)//2:])
            results['homoscedasticity_p'] = p_value
            
            # Check normality of residuals
            _, p_norm = stats.normaltest(residuals)
            results['normality_p'] = p_norm
            
            # Check for multicollinearity via condition number
            if X.shape[1] > 1:
                cond_number = np.linalg.cond(X.T @ X)
                results['condition_number'] = cond_number
                results['multicollinearity_risk'] = cond_number > 30
        
        elif model_type == 'logistic_regression':
            # Check for complete separation
            for col_idx in range(X.shape[1]):
                unique_pairs = set()
                for xi, yi in zip(X[:, col_idx], y):
                    unique_pairs.add((xi, yi))
                if len(unique_pairs) == len(set(X[:, col_idx])):
                    results['complete_separation_risk'] = True
                    break
            else:
                results['complete_separation_risk'] = False
            
            # Sample size adequacy (rule of thumb: 10 events per variable)
            n_minority = min(np.sum(y == 0), np.sum(y == 1))
            results['events_per_variable'] = n_minority / X.shape[1]
            results['sample_size_adequate'] = results['events_per_variable'] >= 10
        
        return results
    
    @staticmethod
    def bootstrap_confidence_intervals(estimator, X: np.ndarray, y: np.ndarray, 
                                      n_bootstraps: int = 100, ci_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for model parameters."""
        if not DEPS_AVAILABLE:
            return {}
        
        n_samples = X.shape[0]
        bootstrap_coefs = []
        
        for _ in range(n_bootstraps):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit model on bootstrap sample
            estimator_boot = estimator.__class__(**estimator.get_params())
            estimator_boot.fit(X_boot, y_boot)
            
            if hasattr(estimator_boot, 'coef_'):
                bootstrap_coefs.append(estimator_boot.coef_.ravel())
        
        if not bootstrap_coefs:
            return {}
        
        bootstrap_coefs = np.array(bootstrap_coefs)
        alpha = 1 - ci_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_dict = {}
        for i in range(bootstrap_coefs.shape[1]):
            lower = np.percentile(bootstrap_coefs[:, i], lower_percentile)
            upper = np.percentile(bootstrap_coefs[:, i], upper_percentile)
            ci_dict[f'feature_{i}'] = (lower, upper)
        
        return ci_dict

class AdaptiveLearningEngine:
    """Personalized learning path based on performance and concepts mastered."""
    
    def __init__(self):
        self.concept_mastery = defaultdict(float)
        self.attempt_history = defaultdict(list)
        self.learning_velocity = defaultdict(list)
        self.difficulty_calibration = 1.0
        
    def update_mastery(self, concept: MLConcept, success: bool, time_taken: float):
        """Update mastery score using exponential moving average."""
        alpha = 0.3  # Learning rate for EMA
        current_score = 1.0 if success else 0.0
        
        if concept in self.concept_mastery:
            self.concept_mastery[concept] = (1 - alpha) * self.concept_mastery[concept] + alpha * current_score
        else:
            self.concept_mastery[concept] = current_score
        
        self.attempt_history[concept].append((success, time_taken))
        
        # Calculate learning velocity
        if len(self.attempt_history[concept]) >= 3:
            recent = self.attempt_history[concept][-3:]
            velocity = sum(1 for s, _ in recent if s) / len(recent)
            self.learning_velocity[concept].append(velocity)
    
    def recommend_next_concept(self, available_objectives: List[LearningObjective]) -> Optional[LearningObjective]:
        """Recommend next learning objective based on prerequisites and mastery."""
        completed = {concept for concept, mastery in self.concept_mastery.items() if mastery > 0.7}
        
        # Filter accessible objectives
        accessible = [obj for obj in available_objectives if obj.is_accessible(completed)]
        
        if not accessible:
            return None
        
        # Prioritize by prerequisite depth and estimated time
        def priority_score(obj: LearningObjective) -> float:
            novelty = 1.0 - self.concept_mastery.get(obj.concept, 0.0)
            prerequisite_strength = sum(self.concept_mastery.get(p, 0) for p in obj.prerequisite_concepts) / max(len(obj.prerequisite_concepts), 1)
            time_factor = 1.0 / (1.0 + obj.estimated_time_minutes / 30.0)
            return novelty * prerequisite_strength * time_factor * self.difficulty_calibration
        
        return max(accessible, key=priority_score)
    
    def calibrate_difficulty(self):
        """Adjust difficulty based on recent performance."""
        recent_success_rate = np.mean([s for attempts in self.attempt_history.values() 
                                       for s, _ in attempts[-5:] if attempts])
        
        if recent_success_rate > 0.8:
            self.difficulty_calibration = min(2.0, self.difficulty_calibration * 1.1)
        elif recent_success_rate < 0.4:
            self.difficulty_calibration = max(0.5, self.difficulty_calibration * 0.9)

class InteractiveChallenge:
    """Base class for ML challenges with automated evaluation."""
    
    def __init__(self, concept: MLConcept, difficulty: float = 1.0):
        self.concept = concept
        self.difficulty = difficulty
        self.performance_monitor = PerformanceMonitor()
    
    def generate_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate appropriate synthetic data for the challenge."""
        if not DEPS_AVAILABLE:
            # Fallback to simple generated data
            X = [[random.gauss(0, 1) for _ in range(5)] for _ in range(n_samples)]
            y = [int(sum(row) > 0) for row in X]
            return np.array(X), np.array(y)
        
        if self.concept in [MLConcept.BIAS_VARIANCE, MLConcept.REGULARIZATION]:
            # High-dimensional data prone to overfitting
            X, y = make_classification(n_samples=n_samples, n_features=20, 
                                      n_informative=5, n_redundant=5,
                                      n_clusters_per_class=2, flip_y=0.1,
                                      random_state=int(time.time()) % 1000)
        else:
            X, y = make_classification(n_samples=n_samples, n_features=10,
                                      n_informative=7, n_redundant=2,
                                      random_state=int(time.time()) % 1000)
        return X, y
    
    def evaluate_solution(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation of submitted solution."""
        if not DEPS_AVAILABLE:
            return {'error': 'Dependencies not available for evaluation'}
        
        with self.performance_monitor.timer('prediction'):
            y_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        metrics = {}
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
                metrics['avg_precision'] = average_precision_score(y_test, y_proba[:, 1])
                metrics['log_loss'] = log_loss(y_test, y_proba)
        
        metrics['accuracy'] = np.mean(y_pred == y_test)
        
        # Add interpretability analysis if applicable
        if hasattr(model, 'coef_'):
            metrics['sparsity'] = np.mean(np.abs(model.coef_) < 1e-4)
            metrics['coef_variance'] = np.var(model.coef_)
        
        return metrics

class RegularizationChallenge(InteractiveChallenge):
    """Challenge focusing on regularization techniques."""
    
    def __init__(self):
        super().__init__(MLConcept.REGULARIZATION, difficulty=1.5)
        self.target_sparsity = random.uniform(0.3, 0.7)
    
    def create_challenge(self) -> Dict[str, Any]:
        """Create a regularization challenge with specific objectives."""
        X, y = self.generate_dataset(n_samples=500)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        
        # Add polynomial features to increase complexity
        if DEPS_AVAILABLE:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
        else:
            X_train_poly = X_train
            X_test_poly = X_test
        
        challenge = {
            'description': f"""
Regularization Challenge:
Your task is to build a logistic regression model that achieves:
1. Test AUC > 0.75
2. Coefficient sparsity between {self.target_sparsity-0.1:.1f} and {self.target_sparsity+0.1:.1f}
3. No overfitting (train-test accuracy gap < 0.1)

The data has polynomial features, making regularization crucial.
Consider L1, L2, or ElasticNet penalties.
""",
            'X_train': X_train_poly,
            'y_train': y_train,
            'X_test': X_test_poly,
            'y_test': y_test,
            'evaluation_criteria': {
                'min_auc': 0.75,
                'sparsity_range': (self.target_sparsity - 0.1, self.target_sparsity + 0.1),
                'max_overfit': 0.1
            }
        }
        
        return challenge
    
    def check_solution(self, model, challenge_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate if solution meets challenge criteria."""
        X_train = challenge_data['X_train']
        y_train = challenge_data['y_train']
        X_test = challenge_data['X_test']
        y_test = challenge_data['y_test']
        criteria = challenge_data['evaluation_criteria']
        
        # Evaluate performance
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        metrics = self.evaluate_solution(model, X_test, y_test)
        
        # Check criteria
        checks = []
        
        # AUC check
        if 'roc_auc' in metrics:
            auc_pass = metrics['roc_auc'] >= criteria['min_auc']
            checks.append(('AUC', auc_pass, f"{metrics['roc_auc']:.3f}"))
        
        # Sparsity check
        if 'sparsity' in metrics:
            sparsity = metrics['sparsity']
            sparsity_pass = criteria['sparsity_range'][0] <= sparsity <= criteria['sparsity_range'][1]
            checks.append(('Sparsity', sparsity_pass, f"{sparsity:.3f}"))
        
        # Overfitting check
        overfit = train_score - test_score
        overfit_pass = overfit <= criteria['max_overfit']
        checks.append(('Overfitting', overfit_pass, f"{overfit:.3f}"))
        
        # Generate feedback
        all_pass = all(passed for _, passed, _ in checks)
        
        feedback = "Challenge Results:\n"
        for name, passed, value in checks:
            status = "✓" if passed else "✗"
            feedback += f"  {status} {name}: {value}\n"
        
        if all_pass:
            feedback += "\nExcellent! You've successfully applied regularization to prevent overfitting while maintaining performance."
        else:
            feedback += "\nHints:\n"
            if not checks[0][1]:  # AUC failed
                feedback += "- Try adjusting the regularization strength (C parameter)\n"
            if len(checks) > 1 and not checks[1][1]:  # Sparsity failed
                feedback += "- Consider L1 or ElasticNet for sparsity, adjust alpha\n"
            if len(checks) > 2 and not checks[2][1]:  # Overfitting
                feedback += "- Increase regularization to reduce overfitting\n"
        
        return all_pass, feedback

class CrossValidationChallenge(InteractiveChallenge):
    """Challenge focusing on proper cross-validation techniques."""
    
    def __init__(self):
        super().__init__(MLConcept.CROSS_VALIDATION, difficulty=1.3)
        self.imbalance_ratio = random.uniform(0.1, 0.3)
    
    def create_challenge(self) -> Dict[str, Any]:
        """Create a cross-validation challenge with class imbalance."""
        if not DEPS_AVAILABLE:
            return {'error': 'This challenge requires scikit-learn'}
        
        # Generate imbalanced dataset
        X, y = make_classification(n_samples=1000, n_features=15,
                                  n_informative=10, n_redundant=3,
                                  weights=[1-self.imbalance_ratio, self.imbalance_ratio],
                                  flip_y=0.05, random_state=int(time.time()) % 1000)
        
        challenge = {
            'description': f"""
Cross-Validation Challenge:
You have an imbalanced dataset (minority class: {self.imbalance_ratio:.1%}).

Your task:
1. Implement appropriate cross-validation strategy
2. Use stratification to maintain class distribution
3. Report mean and std of AUC across folds
4. Compare at least 2 different models
5. Account for class imbalance in evaluation

The winning model should have:
- Mean CV AUC > 0.80
- Std CV AUC < 0.05
- Proper handling of class imbalance
""",
            'X': X,
            'y': y,
            'evaluation_criteria': {
                'min_mean_auc': 0.80,
                'max_std_auc': 0.05,
                'required_cv_folds': 5
            }
        }
        
        return challenge
    
    def evaluate_cv_solution(self, cv_results: Dict[str, Any], challenge_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate cross-validation implementation."""
        criteria = challenge_data['evaluation_criteria']
        
        feedback = "Cross-Validation Evaluation:\n"
        
        # Check if stratification was used
        if cv_results.get('stratified', False):
            feedback += "  ✓ Stratification used\n"
        else:
            feedback += "  ✗ Warning: Stratification not confirmed\n"
        
        # Check fold count
        n_folds = cv_results.get('n_folds', 0)
        if n_folds >= criteria['required_cv_folds']:
            feedback += f"  ✓ Used {n_folds} folds\n"
        else:
            feedback += f"  ✗ Insufficient folds ({n_folds} < {criteria['required_cv_folds']})\n"
        
        # Check performance metrics
        mean_auc = cv_results.get('mean_auc', 0)
        std_auc = cv_results.get('std_auc', 1)
        
        auc_pass = mean_auc >= criteria['min_mean_auc']
        std_pass = std_auc <= criteria['max_std_auc']
        
        feedback += f"  {'✓' if auc_pass else '✗'} Mean AUC: {mean_auc:.3f} (target: {criteria['min_mean_auc']})\n"
        feedback += f"  {'✓' if std_pass else '✗'} Std AUC: {std_auc:.3f} (target: <{criteria['max_std_auc']})\n"
        
        # Model comparison
        if 'model_comparison' in cv_results:
            feedback += "\nModel Comparison:\n"
            for model_name, model_metrics in cv_results['model_comparison'].items():
                feedback += f"  {model_name}: AUC={model_metrics['auc']:.3f}\n"
        
        all_pass = all([
            cv_results.get('stratified', False),
            n_folds >= criteria['required_cv_folds'],
            auc_pass,
            std_pass,
            len(cv_results.get('model_comparison', {})) >= 2
        ])
        
        if all_pass:
            feedback += "\nOutstanding! You've properly implemented stratified CV with good performance."
        else:
            feedback += "\nSuggestions:\n"
            feedback += "- Use StratifiedKFold for imbalanced data\n"
            feedback += "- Try class_weight='balanced' in classifiers\n"
            feedback += "- Consider SMOTE or other resampling techniques\n"
        
        return all_pass, feedback

class MLOdysseyPlatform:
    """Main platform orchestrating the learning experience."""
    
    def __init__(self):
        self.learning_engine = AdaptiveLearningEngine()
        self.performance_monitor = PerformanceMonitor()
        self.completed_challenges = []
        self.model_registry = []
        
        # Define learning objectives
        self.learning_objectives = [
            LearningObjective(
                MLConcept.BIAS_VARIANCE,
                "Understand bias-variance tradeoff through experimentation",
                ["Identify underfitting", "Identify overfitting", "Find optimal complexity"],
                [],
                20
            ),
            LearningObjective(
                MLConcept.REGULARIZATION,
                "Master L1, L2, and ElasticNet regularization",
                ["Apply appropriate penalty", "Tune hyperparameters", "Achieve sparsity"],
                [MLConcept.BIAS_VARIANCE],
                30
            ),
            LearningObjective(
                MLConcept.CROSS_VALIDATION,
                "Implement robust model evaluation",
                ["Use stratification", "Report confidence intervals", "Compare models"],
                [MLConcept.BIAS_VARIANCE],
                25
            ),
            LearningObjective(
                MLConcept.FEATURE_ENGINEERING,
                "Create informative features",
                ["Generate polynomial features", "Handle categorical variables", "Scale appropriately"],
                [MLConcept.REGULARIZATION],
                35
            )
        ]
    
    def print_header(self, text: str):
        """Display formatted header."""
        width = 80
        print(f"\n{Colors.HEADER}{'=' * width}")
        print(f"{text.center(width)}")
        print(f"{'=' * width}{Colors.RESET}\n")
    
    def print_info(self, text: str):
        """Display informational text."""
        print(f"{Colors.INFO}{text}{Colors.RESET}")
    
    def print_success(self, text: str):
        """Display success message."""
        print(f"{Colors.SUCCESS}{text}{Colors.RESET}")
    
    def print_warning(self, text: str):
        """Display warning message."""
        print(f"{Colors.WARNING}{text}{Colors.RESET}")
    
    def print_error(self, text: str):
        """Display error message."""
        print(f"{Colors.ERROR}{text}{Colors.RESET}")
    
    def run_diagnostic(self):
        """Run system diagnostic and dependency check."""
        self.print_header("System Diagnostic")
        
        print("Checking dependencies:")
        dependencies = [
            ('NumPy', 'numpy', DEPS_AVAILABLE),
            ('SciPy', 'scipy', DEPS_AVAILABLE),
            ('scikit-learn', 'sklearn', DEPS_AVAILABLE)
        ]
        
        for name, module, available in dependencies:
            status = "✓ Available" if available else "✗ Not installed"
            color = Colors.SUCCESS if available else Colors.WARNING
            print(f"  {color}{name:15} {status}{Colors.RESET}")
        
        if not DEPS_AVAILABLE:
            print("\nTo enable full functionality, install:")
            print("  pip install numpy scipy scikit-learn")
        
        print(f"\n{Colors.MUTED}Platform initialized successfully{Colors.RESET}")
    
    def show_learning_path(self):
        """Display personalized learning path."""
        self.print_header("Your Learning Path")
        
        completed = {concept for concept, mastery in self.learning_engine.concept_mastery.items() 
                    if mastery > 0.7}
        
        for obj in self.learning_objectives:
            if obj.concept in completed:
                status = f"{Colors.SUCCESS}✓ Completed{Colors.RESET}"
            elif obj.is_accessible(completed):
                status = f"{Colors.INFO}→ Available{Colors.RESET}"
            else:
                status = f"{Colors.MUTED}  Locked{Colors.RESET}"
            
            print(f"{status} {obj.concept.name:20} {obj.description}")
            
            if obj.concept not in completed and obj.is_accessible(completed):
                mastery = self.learning_engine.concept_mastery.get(obj.concept, 0)
                print(f"    Progress: {'█' * int(mastery * 10)}{'░' * (10 - int(mastery * 10))} {mastery:.0%}")
    
    def run_challenge(self, concept: MLConcept):
        """Execute an interactive challenge for a concept."""
        if concept == MLConcept.REGULARIZATION:
            challenge = RegularizationChallenge()
        elif concept == MLConcept.CROSS_VALIDATION:
            challenge = CrossValidationChallenge()
        else:
            self.print_warning(f"Challenge for {concept.name} not yet implemented")
            return
        
        self.print_header(f"{concept.name} Challenge")
        
        with self.performance_monitor.timer('challenge_generation'):
            challenge_data = challenge.create_challenge()
        
        if 'error' in challenge_data:
            self.print_error(challenge_data['error'])
            return
        
        print(challenge_data['description'])
        
        # Here we would normally get user's solution
        # For demo, we'll create an example solution
        if concept == MLConcept.REGULARIZATION and DEPS_AVAILABLE:
            print("\n[Demo Mode: Creating example solution]")
            
            # Create a solution that meets criteria
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=0.7,
                C=0.1,
                max_iter=1000,
                random_state=42
            )
            
            model.fit(challenge_data['X_train'], challenge_data['y_train'])
            
            success, feedback = challenge.check_solution(model, challenge_data)
            print(feedback)
            
            # Update learning engine
            self.learning_engine.update_mastery(concept, success, time.time())
            
            if success:
                self.completed_challenges.append(concept)
                
                # Create model card
                metrics = challenge.evaluate_solution(model, 
                                                     challenge_data['X_test'], 
                                                     challenge_data['y_test'])
                
                model_card = ModelCard(
                    name=f"{concept.name}_model",
                    version="1.0",
                    task_type="binary_classification",
                    performance_metrics=metrics,
                    limitations=["Synthetic data", "Limited feature engineering"],
                    training_data_characteristics={
                        'n_samples': len(challenge_data['y_train']),
                        'n_features': challenge_data['X_train'].shape[1]
                    },
                    deployment_considerations=["Requires feature scaling", "Monitor for drift"]
                )
                
                self.model_registry.append(model_card)
                self.print_success("\nModel registered successfully!")
                
        elif concept == MLConcept.CROSS_VALIDATION and DEPS_AVAILABLE:
            print("\n[Demo Mode: Creating CV solution]")
            
            # Demonstrate proper CV
            from sklearn.model_selection import StratifiedKFold
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            
            X = challenge_data['X']
            y = challenge_data['y']
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            models = {
                'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
                'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
            }
            
            cv_results = {
                'stratified': True,
                'n_folds': 5,
                'model_comparison': {}
            }
            
            for name, model in models.items():
                scores = cross_validate(model, X, y, cv=cv, 
                                      scoring='roc_auc', 
                                      return_train_score=True)
                
                cv_results['model_comparison'][name] = {
                    'auc': scores['test_score'].mean(),
                    'std': scores['test_score'].std()
                }
                
                if name == 'RandomForest':  # Use RF as primary model
                    cv_results['mean_auc'] = scores['test_score'].mean()
                    cv_results['std_auc'] = scores['test_score'].std()
            
            success, feedback = challenge.evaluate_cv_solution(cv_results, challenge_data)
            print(feedback)
            
            self.learning_engine.update_mastery(concept, success, time.time())
    
    def show_performance_report(self):
        """Display comprehensive performance analytics."""
        self.print_header("Performance Report")
        
        # Learning progress
        print("Learning Progress:")
        for concept, mastery in self.learning_engine.concept_mastery.items():
            progress_bar = '█' * int(mastery * 20) + '░' * (20 - int(mastery * 20))
            print(f"  {concept.name:20} {progress_bar} {mastery:.0%}")
        
        # Computational performance
        print("\nComputational Performance:")
        perf_summary = self.performance_monitor.report_summary()
        for operation, stats in perf_summary.items():
            print(f"  {operation:20} Mean: {stats['mean_ms']:.2f}ms (n={stats['calls']})")
        
        # Model registry
        if self.model_registry:
            print(f"\nModels in Registry: {len(self.model_registry)}")
            for card in self.model_registry[-3:]:  # Show last 3
                print(f"  • {card.name} v{card.version}")
                if card.performance_metrics:
                    metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in 
                                          list(card.performance_metrics.items())[:3])
                    print(f"    {metrics_str}")
    
    def interactive_session(self):
        """Run interactive learning session."""
        self.print_header("ML Odyssey Platform")
        print("Author: Cazandra Aporbo")
        print("Advanced Machine Learning Education System")
        
        self.run_diagnostic()
        
        while True:
            print(f"\n{Colors.BOLD}Commands:{Colors.RESET}")
            print("  1. Show learning path")
            print("  2. Start challenge")
            print("  3. Performance report")
            print("  4. Exit")
            
            try:
                choice = input(f"\n{Colors.INFO}Your choice: {Colors.RESET}")
                
                if choice == '1':
                    self.show_learning_path()
                    
                elif choice == '2':
                    # Get next recommended challenge
                    next_obj = self.learning_engine.recommend_next_concept(self.learning_objectives)
                    if next_obj:
                        print(f"\nRecommended: {next_obj.concept.name}")
                        proceed = input("Start this challenge? (y/n): ")
                        if proceed.lower() == 'y':
                            self.run_challenge(next_obj.concept)
                    else:
                        print("No challenges currently available")
                        
                elif choice == '3':
                    self.show_performance_report()
                    
                elif choice == '4':
                    print(f"\n{Colors.SUCCESS}Thank you for using ML Odyssey Platform!{Colors.RESET}")
                    break
                    
                else:
                    print(f"{Colors.WARNING}Invalid choice{Colors.RESET}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.WARNING}Session interrupted{Colors.RESET}")
                break
            except Exception as e:
                print(f"{Colors.ERROR}Error: {e}{Colors.RESET}")
                traceback.print_exc()

def main():
    """Main entry point."""
    platform = MLOdysseyPlatform()
    platform.interactive_session()

if __name__ == "__main__":
    main()
