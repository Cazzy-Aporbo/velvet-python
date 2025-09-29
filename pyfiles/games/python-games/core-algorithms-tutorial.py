"""
Core Algorithms for Data Science, AI, and ML Engineers
A comprehensive implementation and tutorial of essential algorithms
Each algorithm includes detailed documentation on when to use it,
what to look for, and practical implementation considerations.

Author: Cazzy Aporbo
Version: 1.0.0
Python: 3.8+
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Union, Callable
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Linear Regression
class LinearRegressionTutorial:
    """
    Linear Regression: Models linear relationships between features and continuous targets
    
    When to use:
    - Baseline model for regression problems
    - When interpretability is crucial (coefficients show feature importance)
    - Testing for linear trends in data
    - When you need confidence intervals for predictions
    
    What to look for:
    - Residual patterns (should be random, not systematic)
    - R-squared value (proportion of variance explained)
    - Multicollinearity (VIF > 10 indicates issues)
    - Homoscedasticity (constant variance in residuals)
    - Normal distribution of residuals for inference
    
    Similar algorithms: Ridge (L2 penalty), Lasso (L1 penalty), Elastic Net (L1+L2)
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        # Initialize hyperparameters
        self.learning_rate = learning_rate  # Step size for gradient descent
        self.n_iterations = n_iterations    # Number of optimization iterations
        self.weights = None                 # Model parameters (coefficients)
        self.bias = None                    # Intercept term
        self.losses = []                    # Track training loss over iterations
        
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Fit linear regression using gradient descent
        
        Mathematical foundation:
        - Hypothesis: h(x) = w^T * x + b
        - Cost function: J = (1/2m) * sum((h(x) - y)^2)
        - Gradient: dJ/dw = (1/m) * X^T * (predictions - y)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters with small random values
        # Xavier initialization scaled by input dimension
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        # Gradient descent optimization
        for iteration in range(self.n_iterations):
            # Forward pass: compute predictions
            y_predicted = self.predict(X)
            
            # Compute loss (Mean Squared Error)
            loss = np.mean((y_predicted - y) ** 2)
            self.losses.append(loss)
            
            # Backward pass: compute gradients
            # Partial derivative with respect to weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # Partial derivative with respect to bias
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress every 100 iterations
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using learned parameters"""
        return np.dot(X, self.weights) + self.bias
    
    def evaluate_assumptions(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Check key assumptions of linear regression
        Returns dictionary with diagnostic information
        """
        predictions = self.predict(X)
        residuals = y - predictions
        
        diagnostics = {
            'r_squared': 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2)),
            'mean_residual': np.mean(residuals),  # Should be close to 0
            'residual_std': np.std(residuals),
            'rmse': np.sqrt(np.mean(residuals**2))
        }
        
        # Check for patterns in residuals (simplified)
        # In practice, use statistical tests like Durbin-Watson
        diagnostics['residual_correlation'] = np.corrcoef(predictions, residuals)[0, 1]
        
        return diagnostics


# Logistic Regression
class LogisticRegressionTutorial:
    """
    Logistic Regression: Binary classification with calibrated probabilities
    
    When to use:
    - Binary classification problems
    - When you need probability estimates (not just class labels)
    - As a baseline classifier before trying complex models
    - When interpretability matters (odds ratios from coefficients)
    
    What to look for:
    - Convergence of log-likelihood during training
    - Calibration of probabilities (reliability diagrams)
    - Class imbalance (may need class weights)
    - Separation in data (perfect separation causes issues)
    - Multicollinearity between features
    
    Similar algorithms: Probit regression, Softmax regression (multiclass)
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function: maps any value to [0, 1]
        sigma(z) = 1 / (1 + exp(-z))
        
        Numerical stability: clip values to prevent overflow
        """
        z = np.clip(z, -500, 500)  # Prevent overflow in exp
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Fit logistic regression using gradient descent
        
        Mathematical foundation:
        - Hypothesis: h(x) = sigmoid(w^T * x + b)
        - Log loss: J = -1/m * sum(y*log(h) + (1-y)*log(1-h))
        - Gradient similar to linear regression due to sigmoid derivative
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for iteration in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(z)
            
            # Compute binary cross-entropy loss
            # Add small epsilon to prevent log(0)
            epsilon = 1e-7
            loss = -np.mean(y * np.log(y_predicted + epsilon) + 
                           (1 - y) * np.log(1 - y_predicted + epsilon))
            self.losses.append(loss)
            
            # Backward pass (gradient computation)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}, Log Loss: {loss:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates for the positive class"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions
        Threshold can be adjusted based on precision-recall tradeoff
        """
        return (self.predict_proba(X) >= threshold).astype(int)


# Decision Tree
class DecisionTreeTutorial:
    """
    Decision Tree: Rule-based recursive partitioning for classification/regression
    
    When to use:
    - Non-linear patterns in data
    - Mixed feature types (numerical and categorical)
    - When you need interpretable rules
    - Feature interactions are important
    - No need for feature scaling
    
    What to look for:
    - Tree depth (too deep = overfitting)
    - Leaf node samples (too few = overfitting)
    - Feature importance scores
    - Pruning parameters (min_samples_split, min_samples_leaf)
    - Cross-validation performance vs training performance
    
    Similar algorithms: Random Forest, Gradient Boosted Trees, ExtraTrees
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth  # Maximum tree depth to prevent overfitting
        self.min_samples_split = min_samples_split  # Minimum samples to split a node
        self.tree = None
    
    def gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity for a node
        Gini = 1 - sum(p_i^2) where p_i is proportion of class i
        
        Lower Gini = more pure node (one class dominates)
        """
        if len(y) == 0:
            return 0
        
        # Calculate proportion of each class
        proportions = np.bincount(y) / len(y)
        # Gini impurity formula
        return 1 - np.sum(proportions ** 2)
    
    def information_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """
        Calculate information gain from a split
        IG = impurity(parent) - weighted_average(impurity(children))
        
        Higher information gain = better split
        """
        n = len(parent)
        n_left, n_right = len(left), len(right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        # Calculate weighted average impurity of children
        parent_impurity = self.gini_impurity(parent)
        left_impurity = self.gini_impurity(left)
        right_impurity = self.gini_impurity(right)
        
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        return parent_impurity - weighted_impurity
    
    def find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """
        Find the best feature and threshold to split on
        Exhaustive search over all features and possible thresholds
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            # Get unique values as potential thresholds
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                # Create split
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                # Skip if split doesn't divide samples
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate information gain
                gain = self.information_gain(
                    y,
                    y[left_mask],
                    y[right_mask]
                )
                
                # Update best split if needed
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    class Node:
        """Tree node structure"""
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature      # Feature index for splitting
            self.threshold = threshold  # Threshold value for splitting
            self.left = left           # Left child node
            self.right = right         # Right child node
            self.value = value         # Prediction value for leaf nodes
    
    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build decision tree
        Stop conditions: max depth reached, pure node, or min samples
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            # Create leaf node with majority class
            leaf_value = np.bincount(y).argmax()
            return self.Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self.find_best_split(X, y)
        
        if best_feature is None:
            # No valid split found
            leaf_value = np.bincount(y).argmax()
            return self.Node(value=leaf_value)
        
        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return self.Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Build the decision tree"""
        self.tree = self.build_tree(X, y)


# Random Forest
class RandomForestTutorial:
    """
    Random Forest: Ensemble of decision trees with bagging and feature randomness
    
    When to use:
    - Robust performance on tabular data
    - Reduced overfitting compared to single tree
    - Built-in feature importance
    - Handles missing values well
    - No feature scaling required
    
    What to look for:
    - Number of trees (more = better, but diminishing returns)
    - Max features per split (sqrt(n_features) for classification)
    - Out-of-bag (OOB) error as validation metric
    - Feature importance consistency
    - Training time vs accuracy tradeoff
    
    Similar algorithms: ExtraTrees, Gradient Boosted Trees, Isolation Forest
    """
    
    def __init__(self, n_trees: int = 100, max_depth: int = 10, max_features: str = 'sqrt'):
        self.n_trees = n_trees          # Number of trees in forest
        self.max_depth = max_depth      # Maximum depth per tree
        self.max_features = max_features  # Features to consider at each split
        self.trees = []
        self.feature_importances_ = None
        
    def bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create bootstrap sample (sampling with replacement)
        This introduces diversity among trees
        """
        n_samples = X.shape[0]
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def get_random_features(self, n_features: int) -> np.ndarray:
        """
        Select random subset of features for each split
        This decorrelates trees and improves ensemble performance
        """
        if self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_selected = int(np.log2(n_features))
        else:
            n_selected = n_features
        
        return np.random.choice(n_features, size=n_selected, replace=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Train random forest by building multiple trees on bootstrap samples
        
        Key concepts:
        1. Bagging: Each tree trained on bootstrap sample
        2. Feature randomness: Random features at each split
        3. Ensemble: Aggregate predictions from all trees
        """
        n_samples, n_features = X.shape
        self.trees = []
        
        for i in range(self.n_trees):
            if verbose and i % 20 == 0:
                print(f"Building tree {i+1}/{self.n_trees}")
            
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            
            # Build tree on bootstrap sample
            # In practice, would use modified DecisionTree with feature randomness
            tree = DecisionTreeTutorial(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            
            self.trees.append(tree)
        
        # Calculate feature importances (simplified)
        # In practice, aggregate importances from all trees
        self.feature_importances_ = np.random.rand(n_features)
        self.feature_importances_ /= self.feature_importances_.sum()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions by majority voting (classification)
        For regression, would use mean instead
        """
        # Collect predictions from all trees
        predictions = []
        for tree in self.trees:
            # Note: Actual implementation would need tree.predict(X)
            pred = np.random.randint(0, 2, size=X.shape[0])  # Placeholder
            predictions.append(pred)
        
        predictions = np.array(predictions)
        # Majority voting
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, predictions)


# Gradient Boosting
class GradientBoostingTutorial:
    """
    Gradient Boosting: Sequential ensemble that corrects previous errors
    
    When to use:
    - State-of-the-art accuracy on tabular data
    - When you can afford longer training time
    - Complex non-linear patterns
    - Feature interactions are important
    - Competitions (XGBoost, LightGBM, CatBoost)
    
    What to look for:
    - Learning rate vs number of trees tradeoff
    - Early stopping based on validation score
    - Overfitting (use regularization parameters)
    - Feature importance stability
    - Optimal number of boosting rounds
    
    Similar algorithms: AdaBoost, Random Forest, XGBoost, LightGBM, CatBoost
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators    # Number of boosting rounds
        self.learning_rate = learning_rate  # Shrinkage parameter
        self.max_depth = max_depth         # Depth of weak learners
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """
        Train gradient boosting model
        
        Algorithm:
        1. Start with initial prediction (mean for regression)
        2. For each boosting round:
           a. Calculate residuals (negative gradient)
           b. Fit tree to residuals
           c. Update predictions with shrinkage
        """
        n_samples = X.shape[0]
        
        # Initialize with mean prediction
        self.initial_prediction = np.mean(y)
        predictions = np.full(n_samples, self.initial_prediction)
        
        for i in range(self.n_estimators):
            # Calculate pseudo-residuals (negative gradient)
            residuals = y - predictions
            
            # Fit weak learner to residuals
            tree = DecisionTreeTutorial(max_depth=self.max_depth)
            # In practice, fit tree to predict residuals
            # tree.fit(X, residuals)
            
            # Update predictions with learning rate
            # tree_predictions = tree.predict(X)
            tree_predictions = residuals * 0.1  # Placeholder
            predictions += self.learning_rate * tree_predictions
            
            self.trees.append(tree)
            
            if verbose and i % 20 == 0:
                mse = np.mean((y - predictions) ** 2)
                print(f"Round {i+1}, MSE: {mse:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions by summing all tree contributions
        """
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            # Add each tree's contribution
            # predictions += self.learning_rate * tree.predict(X)
            predictions += self.learning_rate * np.random.randn(X.shape[0]) * 0.1  # Placeholder
        
        return predictions


# K-Nearest Neighbors
class KNNTutorial:
    """
    K-Nearest Neighbors: Instance-based learning using distance metrics
    
    When to use:
    - Simple non-parametric baseline
    - Low-dimensional data (curse of dimensionality)
    - Local patterns matter more than global
    - Non-linear decision boundaries
    - When you have sufficient training data
    
    What to look for:
    - Optimal K through cross-validation
    - Distance metric choice (Euclidean, Manhattan, Minkowski)
    - Feature scaling is CRITICAL
    - Computational cost for large datasets
    - Performance degrades in high dimensions
    
    Similar algorithms: Kernel Density Estimation, RBF-kernel SVM, Local Outlier Factor
    """
    
    def __init__(self, k: int = 5, distance_metric: str = 'euclidean'):
        self.k = k  # Number of neighbors to consider
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Euclidean distance: L2 norm
        d = sqrt(sum((x1_i - x2_i)^2))
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Manhattan distance: L1 norm
        d = sum(|x1_i - x2_i|)
        Better for high-dimensional sparse data
        """
        return np.sum(np.abs(x1 - x2))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        KNN doesn't actually 'train' - just stores the data
        This is why it's called lazy learning
        """
        self.X_train = X
        self.y_train = y
        print(f"Stored {len(X)} training samples")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by finding K nearest neighbors and voting
        
        Time complexity: O(n*m*d) where n=test samples, m=train samples, d=dimensions
        This is why KNN can be slow for large datasets
        """
        predictions = []
        
        for test_point in X:
            # Calculate distances to all training points
            distances = []
            for i, train_point in enumerate(self.X_train):
                if self.distance_metric == 'euclidean':
                    dist = self.euclidean_distance(test_point, train_point)
                else:
                    dist = self.manhattan_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            
            # Sort by distance and get k nearest
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Vote for classification (mode)
            # For regression, would use mean
            k_nearest_labels = [label for _, label in k_nearest]
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)
        
        return np.array(predictions)


# Support Vector Machine
class SVMTutorial:
    """
    Support Vector Machine: Maximum margin classifier with kernel trick
    
    When to use:
    - Binary classification with clear margin
    - High-dimensional data (text, genomics)
    - Non-linear patterns (with kernel trick)
    - Robust to outliers (only support vectors matter)
    - Medium-sized datasets (O(n^2) to O(n^3) complexity)
    
    What to look for:
    - C parameter (regularization vs margin tradeoff)
    - Kernel choice (linear, RBF, polynomial)
    - Gamma for RBF kernel (controls influence radius)
    - Number of support vectors (fewer is better)
    - Feature scaling is ESSENTIAL
    
    Similar algorithms: Logistic Regression (linear), Neural Networks (non-linear)
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'linear', gamma: float = 0.1):
        self.C = C              # Regularization parameter
        self.kernel = kernel    # Kernel type
        self.gamma = gamma      # RBF kernel parameter
        self.support_vectors = None
        self.dual_coef = None
        
    def linear_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Linear kernel: simple dot product"""
        return np.dot(x1, x2)
    
    def rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        RBF (Gaussian) kernel: maps to infinite dimensional space
        K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
        
        Gamma controls the influence of single training example:
        - High gamma: nearby points have high influence (overfitting risk)
        - Low gamma: far away points also influence (underfitting risk)
        """
        return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
    
    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix K[i,j] = kernel(X1[i], X2[j])
        This is the key to kernel trick - work in feature space without explicit mapping
        """
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                if self.kernel == 'linear':
                    K[i, j] = self.linear_kernel(X1[i], X2[j])
                elif self.kernel == 'rbf':
                    K[i, j] = self.rbf_kernel(X1[i], X2[j])
        
        return K
    
    def objective_function(self, alpha: np.ndarray, K: np.ndarray, y: np.ndarray) -> float:
        """
        SVM dual objective function to maximize:
        L(alpha) = sum(alpha) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K[i,j])
        
        Subject to constraints:
        - 0 <= alpha_i <= C
        - sum(alpha_i * y_i) = 0
        """
        return np.sum(alpha) - 0.5 * np.sum(np.outer(alpha * y, alpha * y) * K)
    
    def fit_simplified(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Simplified SVM training (actual implementation uses SMO algorithm)
        
        Key concepts:
        1. Find support vectors (points on or within margin)
        2. Only support vectors determine decision boundary
        3. Dual formulation allows kernel trick
        """
        n_samples = X.shape[0]
        
        # Compute kernel matrix
        K = self.compute_kernel_matrix(X, X)
        
        # Initialize Lagrange multipliers
        alpha = np.zeros(n_samples)
        
        # Simplified optimization (actual SVM uses SMO)
        print("Training SVM (simplified demonstration)")
        print(f"Kernel: {self.kernel}, C: {self.C}")
        
        # In practice, use quadratic programming solver
        # Here we just demonstrate the concept
        self.support_vectors = X[alpha > 1e-5]  # Non-zero alphas
        self.dual_coef = alpha[alpha > 1e-5] * y[alpha > 1e-5]
        
        print(f"Found {len(self.support_vectors)} support vectors")


# Naive Bayes
class NaiveBayesTutorial:
    """
    Naive Bayes: Probabilistic classifier based on Bayes' theorem
    
    When to use:
    - Text classification (spam, sentiment)
    - High-dimensional sparse features
    - Real-time predictions needed (very fast)
    - Small training datasets
    - Features are roughly independent
    
    What to look for:
    - Feature independence assumption validity
    - Smoothing parameter (Laplace smoothing)
    - Class prior probabilities
    - Zero frequency problem
    - Log probabilities for numerical stability
    
    Similar algorithms: Logistic Regression, Linear Discriminant Analysis
    """
    
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing  # Laplace smoothing parameter
        self.class_priors = {}      # P(class)
        self.feature_probs = {}     # P(feature|class)
        self.classes = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Learn class priors and feature probabilities
        
        Bayes theorem: P(class|features) = P(features|class) * P(class) / P(features)
        Naive assumption: Features are conditionally independent given class
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Calculate class priors P(class)
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
            print(f"Class {c} prior: {self.class_priors[c]:.3f}")
        
        # Calculate feature probabilities P(feature|class)
        for c in self.classes:
            class_mask = (y == c)
            X_class = X[class_mask]
            
            # For Gaussian Naive Bayes
            self.feature_probs[c] = {
                'mean': np.mean(X_class, axis=0),
                'var': np.var(X_class, axis=0) + 1e-9  # Add small value for stability
            }
    
    def gaussian_probability(self, x: float, mean: float, var: float) -> float:
        """
        Calculate Gaussian probability density
        P(x|mean,var) = 1/sqrt(2*pi*var) * exp(-(x-mean)^2 / (2*var))
        """
        eps = 1e-9  # Numerical stability
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-(x - mean) ** 2 / (2 * var + eps))
        return coeff * exponent
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class using maximum a posteriori (MAP) estimation
        
        For each class:
        1. Start with log prior: log P(class)
        2. Add log likelihoods: sum(log P(feature_i|class))
        3. Select class with highest score
        """
        predictions = []
        
        for sample in X:
            posteriors = {}
            
            for c in self.classes:
                # Start with log prior (use log for numerical stability)
                posterior = np.log(self.class_priors[c])
                
                # Multiply by feature likelihoods (add in log space)
                for i, feature_val in enumerate(sample):
                    mean = self.feature_probs[c]['mean'][i]
                    var = self.feature_probs[c]['var'][i]
                    likelihood = self.gaussian_probability(feature_val, mean, var)
                    posterior += np.log(likelihood + 1e-9)  # Avoid log(0)
                
                posteriors[c] = posterior
            
            # Predict class with maximum posterior
            predictions.append(max(posteriors, key=posteriors.get))
        
        return np.array(predictions)


# Principal Component Analysis (PCA)
class PCATutorial:
    """
    PCA: Orthogonal linear transformation for dimensionality reduction
    
    When to use:
    - Dimensionality reduction before modeling
    - Data visualization (2D/3D projection)
    - Noise reduction
    - Feature decorrelation
    - Compression
    
    What to look for:
    - Explained variance ratio (cumulative)
    - Scree plot elbow
    - Loadings interpretation
    - Data should be centered
    - Scale features if different units
    
    Similar algorithms: SVD, ICA, t-SNE, UMAP, Autoencoders
    """
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None  # Principal components (eigenvectors)
        self.explained_variance_ = None  # Eigenvalues
        self.mean_ = None  # Feature means for centering
        
    def fit(self, X: np.ndarray) -> None:
        """
        Find principal components using eigendecomposition
        
        Steps:
        1. Center the data (subtract mean)
        2. Compute covariance matrix
        3. Find eigenvectors and eigenvalues
        4. Sort by eigenvalues (descending)
        5. Select top k eigenvectors
        """
        # Step 1: Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        # Cov = 1/(n-1) * X^T * X
        n_samples = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        print(f"Covariance matrix shape: {cov_matrix.shape}")
        
        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Step 4: Sort by eigenvalues (descending order)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Select top k components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        explained_ratio = self.explained_variance_ / total_variance
        cumulative_ratio = np.cumsum(explained_ratio)
        
        print(f"Explained variance ratio: {explained_ratio}")
        print(f"Cumulative explained variance: {cumulative_ratio}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components
        X_transformed = (X - mean) @ components^T
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Reconstruct original data from principal components
        Used to assess reconstruction error
        """
        return np.dot(X_transformed, self.components_) + self.mean_


# K-Means Clustering
class KMeansTutorial:
    """
    K-Means: Partition data into K clusters by minimizing within-cluster variance
    
    When to use:
    - Known or estimated number of clusters
    - Spherical/globular clusters
    - Similar cluster sizes
    - Fast baseline clustering
    - Large datasets
    
    What to look for:
    - Optimal K (elbow method, silhouette score)
    - Initialization sensitivity (use k-means++)
    - Convergence speed
    - Empty clusters
    - Feature scaling importance
    
    Similar algorithms: GMM (soft clusters), DBSCAN (density-based), Hierarchical
    """
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None  # Sum of squared distances to nearest centroid
        
    def initialize_centroids(self, X: np.ndarray, method: str = 'random') -> np.ndarray:
        """
        Initialize cluster centroids
        
        Methods:
        - random: Randomly select k points from data
        - k-means++: Smart initialization for faster convergence
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        if method == 'random':
            # Randomly select k points as initial centroids
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centroids = X[indices].copy()
        
        elif method == 'k-means++':
            centroids = []
            # Choose first centroid randomly
            centroids.append(X[np.random.randint(n_samples)])
            
            # Choose remaining centroids with probability proportional to squared distance
            for _ in range(1, self.n_clusters):
                distances = []
                for point in X:
                    # Distance to nearest centroid
                    min_dist = min([np.sum((point - c) ** 2) for c in centroids])
                    distances.append(min_dist)
                
                # Convert to probabilities
                distances = np.array(distances)
                probabilities = distances / distances.sum()
                
                # Choose next centroid
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()
                for j, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids.append(X[j])
                        break
            
            centroids = np.array(centroids)
        
        return centroids
    
    def assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each point to nearest centroid
        This is the 'assignment step' in K-means
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i, point in enumerate(X):
            # Calculate distance to each centroid
            distances = [np.sum((point - centroid) ** 2) for centroid in self.centroids]
            # Assign to nearest centroid
            labels[i] = np.argmin(distances)
        
        return labels
    
    def update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids as mean of assigned points
        This is the 'update step' in K-means
        """
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            # Get points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # New centroid is mean of cluster points
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Empty cluster: keep old centroid or reinitialize
                print(f"Warning: Cluster {k} is empty")
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    
    def fit(self, X: np.ndarray, verbose: bool = True) -> None:
        """
        Fit K-means using Lloyd's algorithm
        
        Algorithm:
        1. Initialize centroids
        2. Repeat until convergence:
           a. Assign points to nearest centroid
           b. Update centroids as cluster means
        """
        # Initialize centroids using k-means++
        self.centroids = self.initialize_centroids(X, method='k-means++')
        
        for iteration in range(self.max_iters):
            # Assignment step
            labels = self.assign_clusters(X)
            
            # Update step
            new_centroids = self.update_centroids(X, labels)
            
            # Check convergence (centroids don't change)
            if np.allclose(self.centroids, new_centroids):
                if verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
            
            self.centroids = new_centroids
            
            if verbose and iteration % 10 == 0:
                # Calculate inertia (within-cluster sum of squares)
                inertia = self.calculate_inertia(X, labels)
                print(f"Iteration {iteration + 1}, Inertia: {inertia:.2f}")
        
        self.labels_ = labels
        self.inertia_ = self.calculate_inertia(X, labels)
    
    def calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate within-cluster sum of squared distances
        Lower inertia = tighter clusters
        """
        inertia = 0
        for i, point in enumerate(X):
            centroid = self.centroids[labels[i]]
            inertia += np.sum((point - centroid) ** 2)
        return inertia
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to nearest centroid"""
        return self.assign_clusters(X)


# Expectation-Maximization for Gaussian Mixture Model
class GMMTutorial:
    """
    Gaussian Mixture Model: Soft clustering using mixture of Gaussians
    
    When to use:
    - Overlapping clusters
    - Different cluster shapes and sizes
    - Soft cluster assignments needed
    - Density estimation
    - More flexible than K-means
    
    What to look for:
    - Number of components (BIC, AIC)
    - Convergence of log-likelihood
    - Covariance type (full, diagonal, spherical)
    - Singularity issues (add regularization)
    - Initialization sensitivity
    
    Similar algorithms: K-means, DBSCAN, Variational Inference
    """
    
    def __init__(self, n_components: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol  # Convergence tolerance
        self.weights_ = None  # Mixture weights
        self.means_ = None   # Component means
        self.covariances_ = None  # Component covariances
        self.log_likelihood_ = []
        
    def initialize_parameters(self, X: np.ndarray) -> None:
        """
        Initialize GMM parameters
        - weights: uniform
        - means: random points from data
        - covariances: identity matrices
        """
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Initialize means using random samples
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices].copy()
        
        # Initialize covariances as identity matrices
        self.covariances_ = [np.eye(n_features) for _ in range(self.n_components)]
    
    def multivariate_gaussian(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Compute multivariate Gaussian probability density
        
        PDF = 1/((2π)^(d/2)|Σ|^(1/2)) * exp(-1/2 * (x-μ)^T * Σ^(-1) * (x-μ))
        """
        n_features = X.shape[1]
        diff = X - mean
        
        # Add small regularization to diagonal for numerical stability
        cov_reg = cov + 1e-6 * np.eye(n_features)
        
        # Compute inverse and determinant
        try:
            cov_inv = np.linalg.inv(cov_reg)
            cov_det = np.linalg.det(cov_reg)
        except np.linalg.LinAlgError:
            # Singular matrix, return small probability
            return np.ones(X.shape[0]) * 1e-10
        
        # Normalization constant
        norm_const = 1 / np.sqrt((2 * np.pi) ** n_features * cov_det)
        
        # Mahalanobis distance
        mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)
        
        return norm_const * np.exp(-0.5 * mahalanobis)
    
    def e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Expectation step: Calculate posterior probabilities (responsibilities)
        
        γ_ik = P(z_i = k|x_i) = π_k * N(x_i|μ_k,Σ_k) / Σ_j π_j * N(x_i|μ_j,Σ_j)
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Calculate likelihood for each component
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self.multivariate_gaussian(
                X, self.means_[k], self.covariances_[k]
            )
        
        # Normalize to get posteriors
        row_sums = responsibilities.sum(axis=1, keepdims=True)
        responsibilities = responsibilities / (row_sums + 1e-10)
        
        return responsibilities
    
    def m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        Maximization step: Update parameters to maximize expected log-likelihood
        
        π_k = 1/N * Σ_i γ_ik
        μ_k = Σ_i γ_ik * x_i / Σ_i γ_ik
        Σ_k = Σ_i γ_ik * (x_i - μ_k)(x_i - μ_k)^T / Σ_i γ_ik
        """
        n_samples = X.shape[0]
        
        for k in range(self.n_components):
            # Effective number of points assigned to cluster k
            n_k = responsibilities[:, k].sum()
            
            # Update weight
            self.weights_[k] = n_k / n_samples
            
            # Update mean
            self.means_[k] = (responsibilities[:, k, np.newaxis] * X).sum(axis=0) / n_k
            
            # Update covariance
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k, np.newaxis] * diff
            self.covariances_[k] = (weighted_diff.T @ diff) / n_k
    
    def fit(self, X: np.ndarray, verbose: bool = True) -> None:
        """
        Fit GMM using Expectation-Maximization algorithm
        
        Algorithm:
        1. Initialize parameters
        2. Repeat until convergence:
           a. E-step: Calculate responsibilities
           b. M-step: Update parameters
           c. Calculate log-likelihood
        """
        self.initialize_parameters(X)
        
        for iteration in range(self.max_iters):
            # E-step
            responsibilities = self.e_step(X)
            
            # M-step
            self.m_step(X, responsibilities)
            
            # Calculate log-likelihood
            log_likelihood = self.calculate_log_likelihood(X)
            self.log_likelihood_.append(log_likelihood)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}, Log-likelihood: {log_likelihood:.2f}")
            
            # Check convergence
            if iteration > 0:
                if abs(self.log_likelihood_[-1] - self.log_likelihood_[-2]) < self.tol:
                    if verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break
    
    def calculate_log_likelihood(self, X: np.ndarray) -> float:
        """
        Calculate log-likelihood of data under current model
        log L = Σ_i log(Σ_k π_k * N(x_i|μ_k,Σ_k))
        """
        n_samples = X.shape[0]
        likelihoods = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            likelihoods[:, k] = self.weights_[k] * self.multivariate_gaussian(
                X, self.means_[k], self.covariances_[k]
            )
        
        return np.sum(np.log(likelihoods.sum(axis=1) + 1e-10))


# Gradient Descent Optimization
class GradientDescentTutorial:
    """
    Gradient Descent: First-order optimization for continuous functions
    
    When to use:
    - Differentiable objective functions
    - Large-scale optimization problems
    - Neural network training
    - When analytical solution doesn't exist
    - Online learning scenarios
    
    What to look for:
    - Learning rate selection (too high = divergence, too low = slow)
    - Convergence criteria
    - Local vs global minima
    - Gradient vanishing/exploding
    - Batch size (SGD vs mini-batch vs full batch)
    
    Variants: SGD, Momentum, RMSProp, Adam, AdaGrad, L-BFGS
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.history = {'loss': [], 'gradients': []}
    
    def standard_gradient_descent(self, gradient_fn: Callable, params: np.ndarray, 
                                 n_iterations: int = 100) -> np.ndarray:
        """
        Standard (batch) gradient descent
        θ_t+1 = θ_t - α * ∇f(θ_t)
        """
        for i in range(n_iterations):
            # Compute gradient
            grad = gradient_fn(params)
            
            # Update parameters
            params = params - self.learning_rate * grad
            
            self.history['gradients'].append(np.linalg.norm(grad))
            
            if i % 20 == 0:
                print(f"Iteration {i}, Gradient norm: {np.linalg.norm(grad):.6f}")
        
        return params
    
    def momentum_gradient_descent(self, gradient_fn: Callable, params: np.ndarray,
                                n_iterations: int = 100) -> np.ndarray:
        """
        Gradient descent with momentum
        v_t = β * v_t-1 + α * ∇f(θ_t)
        θ_t+1 = θ_t - v_t
        
        Momentum helps escape local minima and accelerate convergence
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        for i in range(n_iterations):
            # Compute gradient
            grad = gradient_fn(params)
            
            # Update velocity with momentum
            self.velocity = self.momentum * self.velocity + self.learning_rate * grad
            
            # Update parameters
            params = params - self.velocity
            
            if i % 20 == 0:
                print(f"Iteration {i}, Velocity norm: {np.linalg.norm(self.velocity):.6f}")
        
        return params
    
    def adam_optimizer(self, gradient_fn: Callable, params: np.ndarray,
                      n_iterations: int = 100, beta1: float = 0.9, 
                      beta2: float = 0.999, epsilon: float = 1e-8) -> np.ndarray:
        """
        Adam optimizer: Adaptive moment estimation
        Combines momentum with adaptive learning rates
        
        m_t = β1 * m_t-1 + (1-β1) * ∇f(θ_t)  # First moment (mean)
        v_t = β2 * v_t-1 + (1-β2) * ∇f(θ_t)² # Second moment (variance)
        m̂_t = m_t / (1 - β1^t)               # Bias correction
        v̂_t = v_t / (1 - β2^t)               # Bias correction
        θ_t+1 = θ_t - α * m̂_t / (√v̂_t + ε)
        """
        m = np.zeros_like(params)  # First moment
        v = np.zeros_like(params)  # Second moment
        
        for i in range(n_iterations):
            t = i + 1  # Timestep (starts at 1 for bias correction)
            
            # Compute gradient
            grad = gradient_fn(params)
            
            # Update biased first moment
            m = beta1 * m + (1 - beta1) * grad
            
            # Update biased second moment
            v = beta2 * v + (1 - beta2) * grad ** 2
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters
            params = params - self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            if i % 20 == 0:
                print(f"Iteration {i}, Adaptive LR: {self.learning_rate / (np.sqrt(v_hat.mean()) + epsilon):.6f}")
        
        return params


# Neural Network (Multilayer Perceptron)
class NeuralNetworkTutorial:
    """
    Neural Network: Universal function approximator with hidden layers
    
    When to use:
    - Complex non-linear patterns
    - Large amounts of data available
    - Feature engineering is difficult
    - Image, text, or sequential data
    - End-to-end learning needed
    
    What to look for:
    - Network architecture (depth vs width)
    - Activation functions (ReLU, sigmoid, tanh)
    - Weight initialization (Xavier, He)
    - Overfitting (use dropout, regularization)
    - Vanishing/exploding gradients
    - Learning rate scheduling
    - Batch normalization benefits
    
    Similar algorithms: CNN (images), RNN (sequences), Transformers (attention)
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu'):
        self.layer_sizes = layer_sizes  # [input_size, hidden1, hidden2, ..., output_size]
        self.activation = activation
        self.weights = []
        self.biases = []
        self.initialize_parameters()
        
    def initialize_parameters(self) -> None:
        """
        Initialize weights using Xavier/He initialization
        
        Xavier: Good for sigmoid/tanh
        He: Good for ReLU
        """
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            if self.activation == 'relu':
                # He initialization for ReLU
                weight = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
            else:
                # Xavier initialization for sigmoid/tanh
                weight = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
            
            bias = np.zeros((1, output_size))
            
            self.weights.append(weight)
            self.biases.append(bias)
            
            print(f"Layer {i+1}: {input_size} -> {output_size}")
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU: max(0, x) - Most common activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid: 1/(1+e^(-x)) - Output layer for binary classification"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax: exp(x_i)/sum(exp(x)) - Output layer for multiclass"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray, training: bool = False) -> List[np.ndarray]:
        """
        Forward propagation through the network
        Store activations for backpropagation if training
        """
        activations = [X]
        current = X
        
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            z = np.dot(current, weight) + bias
            
            # Apply activation function
            if i < len(self.weights) - 1:  # Hidden layers
                if self.activation == 'relu':
                    current = self.relu(z)
                elif self.activation == 'sigmoid':
                    current = self.sigmoid(z)
            else:  # Output layer
                current = z  # Linear for regression, or apply sigmoid/softmax
            
            if training:
                activations.append(current)
        
        return activations if training else current
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray],
                learning_rate: float = 0.01) -> float:
        """
        Backpropagation algorithm
        
        Key concepts:
        1. Compute output error
        2. Propagate error backwards through network
        3. Update weights using gradients
        """
        m = X.shape[0]  # Number of samples
        
        # Output layer error (assuming MSE loss for simplicity)
        delta = activations[-1] - y
        loss = np.mean(delta ** 2)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient for weights: input^T @ delta
            grad_weight = np.dot(activations[i].T, delta) / m
            grad_bias = np.mean(delta, axis=0, keepdims=True)
            
            # Update parameters
            self.weights[i] -= learning_rate * grad_weight
            self.biases[i] -= learning_rate * grad_bias
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                
                # Apply activation derivative
                if self.activation == 'relu':
                    delta *= self.relu_derivative(activations[i])
        
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
             learning_rate: float = 0.01, batch_size: int = 32) -> None:
        """
        Train neural network using mini-batch gradient descent
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                activations = self.forward(X_batch, training=True)
                
                # Backward pass
                loss = self.backward(X_batch, y_batch, activations, learning_rate)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


def demonstrate_all_algorithms():
    """
    Comprehensive demonstration of all algorithms with real examples
    Shows when to use each algorithm and what to look for in results
    """
    print("CORE ALGORITHMS DEMONSTRATION")
    print("Teaching guide for DS/AI/ML Engineers\n")
    
    # Generate sample datasets for demonstrations
    np.random.seed(42)
    
    # Regression dataset
    X_reg = np.random.randn(100, 3)
    y_reg = 2 * X_reg[:, 0] - 3 * X_reg[:, 1] + X_reg[:, 2] + np.random.randn(100) * 0.5
    
    # Classification dataset  
    X_class = np.random.randn(150, 2)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)
    
    # Clustering dataset
    X_cluster = np.vstack([
        np.random.randn(50, 2) + [2, 2],
        np.random.randn(50, 2) + [-2, -2],
        np.random.randn(50, 2) + [2, -2]
    ])
    
    print("1. LINEAR REGRESSION")
    print("-" * 50)
    print("Use when: Linear relationships, interpretability needed, baseline model")
    print("Watch for: R-squared, residual patterns, multicollinearity\n")
    
    lin_reg = LinearRegressionTutorial(learning_rate=0.01, n_iterations=100)
    lin_reg.fit(X_reg, y_reg, verbose=False)
    diagnostics = lin_reg.evaluate_assumptions(X_reg, y_reg)
    
    print(f"R-squared: {diagnostics['r_squared']:.4f}")
    print(f"RMSE: {diagnostics['rmse']:.4f}")
    print(f"Mean residual (should be ~0): {diagnostics['mean_residual']:.4f}")
    print(f"Residual correlation (should be ~0): {diagnostics['residual_correlation']:.4f}")
    print("Interpretation: Higher R-squared = better fit, watch for patterns in residuals\n")
    
    print("2. LOGISTIC REGRESSION")
    print("-" * 50)
    print("Use when: Binary classification, need probabilities, interpretable baseline")
    print("Watch for: Log loss convergence, probability calibration, class imbalance\n")
    
    log_reg = LogisticRegressionTutorial(learning_rate=0.1, n_iterations=100)
    log_reg.fit(X_class, y_class, verbose=False)
    probabilities = log_reg.predict_proba(X_class)
    predictions = log_reg.predict(X_class)
    accuracy = np.mean(predictions == y_class)
    
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Final log loss: {log_reg.losses[-1]:.4f}")
    print(f"Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print("Interpretation: Well-calibrated probabilities should match actual frequencies\n")
    
    print("3. DECISION TREE")
    print("-" * 50)
    print("Use when: Non-linear patterns, mixed features, need interpretable rules")
    print("Watch for: Tree depth (overfitting), min samples per leaf, feature importance\n")
    
    tree = DecisionTreeTutorial(max_depth=3, min_samples_split=5)
    tree.fit(X_class, y_class)
    
    print(f"Tree depth: {tree.max_depth}")
    print(f"Min samples to split: {tree.min_samples_split}")
    print("Best practices: Use pruning, cross-validate depth, check feature importance")
    print("Warning signs: 100% training accuracy often means overfitting\n")
    
    print("4. RANDOM FOREST")
    print("-" * 50)
    print("Use when: Robust tabular performance, reduce overfitting, feature importance")
    print("Watch for: Number of trees, max features, OOB error, training time\n")
    
    rf = RandomForestTutorial(n_trees=10, max_depth=5)
    rf.fit(X_class, y_class, verbose=False)
    
    print(f"Number of trees: {rf.n_trees}")
    print(f"Max features per split: {rf.max_features}")
    print("Feature importances (example):", rf.feature_importances_[:3])
    print("Best practices: More trees = better (but diminishing returns)")
    print("Tip: Use OOB score for validation without separate set\n")
    
    print("5. GRADIENT BOOSTING")
    print("-" * 50)
    print("Use when: Maximum accuracy needed, competitions, complex patterns")
    print("Watch for: Learning rate vs n_estimators, early stopping, overfitting\n")
    
    gbm = GradientBoostingTutorial(n_estimators=20, learning_rate=0.1, max_depth=3)
    gbm.fit(X_reg, y_reg, verbose=False)
    
    print(f"Number of boosting rounds: {gbm.n_estimators}")
    print(f"Learning rate: {gbm.learning_rate}")
    print(f"Weak learner depth: {gbm.max_depth}")
    print("Trade-off: Lower learning rate + more trees = better but slower")
    print("XGBoost/LightGBM/CatBoost offer optimized implementations\n")
    
    print("6. K-NEAREST NEIGHBORS")
    print("-" * 50)
    print("Use when: Simple baseline, low dimensions, local patterns matter")
    print("Watch for: Optimal K, distance metric, MUST scale features, curse of dimensionality\n")
    
    knn = KNNTutorial(k=5, distance_metric='euclidean')
    knn.fit(X_class[:100], y_class[:100])
    knn_predictions = knn.predict(X_class[100:110])
    
    print(f"K value: {knn.k}")
    print(f"Distance metric: {knn.distance_metric}")
    print("Critical: Always standardize features before KNN!")
    print("Warning: Performance degrades badly in high dimensions (>20)\n")
    
    print("7. SUPPORT VECTOR MACHINE")
    print("-" * 50)
    print("Use when: Clear margin exists, high dimensions OK, robust to outliers")
    print("Watch for: C parameter, kernel choice, gamma (RBF), support vectors count\n")
    
    svm = SVMTutorial(C=1.0, kernel='rbf', gamma=0.1)
    svm.fit_simplified(X_class, y_class)
    
    print(f"Kernel: {svm.kernel}")
    print(f"C (regularization): {svm.C}")
    print(f"Gamma (RBF width): {svm.gamma}")
    print("Rule of thumb: Start with RBF kernel, tune C and gamma via grid search")
    print("Scaling: ESSENTIAL for SVM - features must be standardized\n")
    
    print("8. NAIVE BAYES")
    print("-" * 50)
    print("Use when: Text classification, high dimensions, speed matters, small data")
    print("Watch for: Feature independence assumption, smoothing parameter\n")
    
    nb = NaiveBayesTutorial(smoothing=1.0)
    nb.fit(X_class, y_class)
    nb_predictions = nb.predict(X_class[:10])
    
    print(f"Laplace smoothing: {nb.smoothing}")
    print(f"Number of classes: {len(nb.classes)}")
    print("Strength: Very fast training and prediction")
    print("Weakness: Independence assumption rarely holds exactly\n")
    
    print("9. PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("-" * 50)
    print("Use when: Dimensionality reduction, visualization, decorrelation, compression")
    print("Watch for: Explained variance, scree plot elbow, interpretability loss\n")
    
    pca = PCATutorial(n_components=2)
    pca.fit(X_reg)
    X_transformed = pca.transform(X_reg[:5])
    
    print(f"Original dimensions: {X_reg.shape[1]}")
    print(f"Reduced dimensions: {pca.n_components}")
    print(f"Explained variance: {pca.explained_variance_}")
    print("Rule: Keep components explaining 95% variance")
    print("Remember: Always center data, consider scaling if different units\n")
    
    print("10. K-MEANS CLUSTERING")
    print("-" * 50)
    print("Use when: Known K, spherical clusters, similar sizes, fast clustering")
    print("Watch for: Elbow method for K, silhouette score, initialization sensitivity\n")
    
    kmeans = KMeansTutorial(n_clusters=3, max_iters=50)
    kmeans.fit(X_cluster, verbose=False)
    
    print(f"Number of clusters: {kmeans.n_clusters}")
    print(f"Final inertia: {kmeans.inertia_:.2f}")
    print("Finding K: Use elbow method (plot inertia vs K)")
    print("Validation: Silhouette score measures cluster quality")
    print("Tip: Run multiple times with different seeds, k-means++ helps\n")
    
    print("11. GAUSSIAN MIXTURE MODEL (EM)")
    print("-" * 50)
    print("Use when: Overlapping clusters, soft assignments, different shapes/sizes")
    print("Watch for: Number of components (BIC/AIC), covariance type, convergence\n")
    
    gmm = GMMTutorial(n_components=3, max_iters=50)
    gmm.fit(X_cluster, verbose=False)
    
    print(f"Number of components: {gmm.n_components}")
    print(f"Final log-likelihood: {gmm.log_likelihood_[-1]:.2f}")
    print("Advantage over K-means: Soft assignments, elliptical clusters")
    print("Model selection: Use BIC/AIC to choose number of components\n")
    
    print("12. GRADIENT DESCENT OPTIMIZATION")
    print("-" * 50)
    print("Use when: Differentiable objectives, large-scale problems, neural nets")
    print("Watch for: Learning rate, convergence, local minima, gradient explosion\n")
    
    gd = GradientDescentTutorial(learning_rate=0.01, momentum=0.9)
    
    # Example: minimize f(x) = x^2
    def simple_gradient(x):
        return 2 * x  # derivative of x^2
    
    initial_params = np.array([10.0])
    final_params = gd.standard_gradient_descent(simple_gradient, initial_params, n_iterations=50)
    
    print(f"Initial value: {initial_params[0]:.2f}")
    print(f"Final value: {final_params[0]:.6f} (should approach 0)")
    print("Learning rate tips: Start with 0.01, decay over time")
    print("Modern optimizers: Adam combines momentum + adaptive LR\n")
    
    print("13. NEURAL NETWORK (MLP)")
    print("-" * 50)
    print("Use when: Complex patterns, lots of data, end-to-end learning")
    print("Watch for: Architecture, activation functions, overfitting, gradients\n")
    
    nn = NeuralNetworkTutorial(layer_sizes=[2, 10, 5, 1], activation='relu')
    nn.train(X_class, y_class.reshape(-1, 1), epochs=20, learning_rate=0.01)
    
    print("\nArchitecture guidelines:")
    print("- Start simple, add complexity if needed")
    print("- Hidden units: Often between input and output size")
    print("- Depth vs width: Deeper = more complex features")
    print("- Regularization: Dropout, L2, early stopping")
    print("- Batch norm: Helps with deep networks\n")
    
    print("ALGORITHM SELECTION GUIDE")
    print("=" * 50)
    print("\nFor REGRESSION:")
    print("  Start with: Linear Regression (baseline)")
    print("  Then try: Random Forest, Gradient Boosting")
    print("  Complex: Neural Networks")
    
    print("\nFor CLASSIFICATION:")
    print("  Start with: Logistic Regression (baseline)")
    print("  Then try: Random Forest, Gradient Boosting")
    print("  High-dim: SVM, Naive Bayes")
    print("  Complex: Neural Networks")
    
    print("\nFor CLUSTERING:")
    print("  Start with: K-means (fast, simple)")
    print("  Overlapping: Gaussian Mixture Model")
    print("  Unknown K: DBSCAN, Hierarchical")
    
    print("\nFor DIMENSIONALITY REDUCTION:")
    print("  Linear: PCA, LDA")
    print("  Non-linear: t-SNE, UMAP, Autoencoders")
    
    print("\nPERFORMANCE vs INTERPRETABILITY TRADEOFF:")
    print("  High Interpretability: Linear/Logistic Regression, Decision Trees")
    print("  Moderate: Random Forest (feature importance), SVM")
    print("  Low: Neural Networks, Gradient Boosting")
    
    print("\nCOMPUTATIONAL CONSIDERATIONS:")
    print("  Fast training: Naive Bayes, Linear models")
    print("  Moderate: Random Forest, K-means")
    print("  Slow: SVM (non-linear), Neural Networks, Gradient Boosting")
    print("  Fast inference: All except KNN (lazy learning)")
    
    print("\nFEATURE SCALING REQUIREMENTS:")
    print("  REQUIRED: KNN, SVM, Neural Networks, K-means, PCA")
    print("  NOT REQUIRED: Tree-based (RF, GBM), Naive Bayes")
    
    print("\nHANDLING MISSING DATA:")
    print("  Native support: Tree-based methods (some implementations)")
    print("  Requires imputation: Most others")
    
    print("\nSAMPLE SIZE GUIDELINES:")
    print("  Small (<1000): Simple models, Naive Bayes, KNN")
    print("  Medium (1K-100K): Random Forest, SVM, Gradient Boosting")
    print("  Large (>100K): Neural Networks, SGD variants")
    
    print("\nFEATURE ENGINEERING IMPORTANCE:")
    print("  Critical: Linear models, KNN, Naive Bayes")
    print("  Important: SVM, Clustering")
    print("  Less critical: Tree-based, Neural Networks")
    
    return True


class ModelEvaluationGuide:
    """
    Comprehensive guide for evaluating and comparing models
    Teaches what metrics to use and how to interpret them
    """
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate and interpret regression metrics
        
        Metrics explained:
        - MSE: Average squared error (penalizes large errors)
        - RMSE: Root MSE (same units as target)
        - MAE: Average absolute error (robust to outliers)
        - R²: Proportion of variance explained (1 = perfect, 0 = baseline)
        - MAPE: Mean absolute percentage error
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Avoid division by zero in MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate and interpret classification metrics
        
        Metrics explained:
        - Accuracy: Overall correct predictions (can be misleading if imbalanced)
        - Precision: Of predicted positives, how many are correct?
        - Recall: Of actual positives, how many did we find?
        - F1: Harmonic mean of precision and recall
        - AUC-ROC: Area under ROC curve (probability ranking quality)
        """
        accuracy = np.mean(y_true == y_pred)
        
        # Binary classification metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Confusion Matrix': [[tn, fp], [fn, tp]]
        }
        
        # Add AUC if probabilities provided
        if y_proba is not None:
            # Simplified AUC calculation
            # In practice, use sklearn.metrics.roc_auc_score
            metrics['AUC-ROC'] = 'Use sklearn for proper calculation'
        
        return metrics
    
    @staticmethod
    def clustering_metrics(X: np.ndarray, labels: np.ndarray, 
                          true_labels: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate and interpret clustering metrics
        
        Metrics explained:
        - Silhouette: Measures how similar a point is to its cluster (-1 to 1, higher is better)
        - Calinski-Harabasz: Ratio of between-cluster to within-cluster variance
        - Davies-Bouldin: Average similarity between clusters (lower is better)
        - Adjusted Rand Index: Similarity to true labels (if available)
        """
        from sklearn.metrics import silhouette_score
        
        metrics = {}
        
        # Silhouette coefficient
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            metrics['Silhouette Score'] = silhouette
        
        # Inertia (within-cluster sum of squares)
        inertia = 0
        for k in np.unique(labels):
            cluster_points = X[labels == k]
            centroid = np.mean(cluster_points, axis=0)
            inertia += np.sum((cluster_points - centroid) ** 2)
        metrics['Inertia'] = inertia
        
        # If true labels available
        if true_labels is not None:
            # Adjusted Rand Index (requires sklearn)
            metrics['ARI'] = 'Use sklearn.metrics.adjusted_rand_score'
        
        return metrics


class AdvancedTips:
    """
    Advanced tips and tricks for each algorithm
    Real-world insights from production ML systems
    """
    
    @staticmethod
    def feature_engineering_tips():
        """Advanced feature engineering strategies"""
        tips = """
        FEATURE ENGINEERING BEST PRACTICES:
        
        1. NUMERIC FEATURES:
           - Log transform: Skewed distributions (income, counts)
           - Polynomial features: Capture non-linear relationships
           - Binning: Convert continuous to categorical (age groups)
           - Scaling: StandardScaler for normal, MinMaxScaler for bounded
           - Clipping: Handle outliers (e.g., clip at 99th percentile)
        
        2. CATEGORICAL FEATURES:
           - One-hot: Low cardinality (<50 unique values)
           - Target encoding: High cardinality, but watch for leakage
           - Frequency encoding: When frequency matters
           - Embedding: Neural networks, high cardinality
           - Leave-one-out: Robust target encoding
        
        3. INTERACTION FEATURES:
           - Multiplication: feature1 * feature2
           - Division: Create ratios (debt-to-income)
           - Differences: temporal features (days_since_event)
        
        4. DOMAIN-SPECIFIC:
           - Time: Hour of day, day of week, is_weekend
           - Text: TF-IDF, word embeddings, n-grams
           - Geographic: Distance to key points, clustering
        
        5. FEATURE SELECTION:
           - Filter: Correlation, mutual information, chi-square
           - Wrapper: Forward/backward selection, RFE
           - Embedded: L1 regularization, tree importance
        """
        return tips
    
    @staticmethod
    def hyperparameter_tuning_tips():
        """Hyperparameter tuning strategies"""
        tips = """
        HYPERPARAMETER TUNING STRATEGIES:
        
        1. SEARCH STRATEGIES:
           - Grid Search: Exhaustive, good for small spaces
           - Random Search: Often better than grid for many params
           - Bayesian Optimization: Efficient for expensive models
           - Halving: Progressively eliminate poor performers
        
        2. ALGORITHM-SPECIFIC PARAMETERS:
        
        Random Forest:
           - n_estimators: Start 100, increase until no improvement
           - max_depth: None for full trees, constrain if overfitting
           - min_samples_split: 2-10, higher prevents overfitting
           - max_features: 'sqrt' for classification, 'log2' alternative
        
        Gradient Boosting:
           - learning_rate: 0.01-0.1, lower = need more trees
           - n_estimators: 100-1000, use early stopping
           - max_depth: 3-10, shallow trees often work well
           - subsample: 0.8, adds randomness to prevent overfitting
        
        SVM:
           - C: 0.001 to 1000 (log scale), regularization
           - gamma: 'scale' or 0.001 to 1 (log scale) for RBF
           - kernel: Try linear first, then RBF
        
        Neural Networks:
           - learning_rate: 0.001 typical start, decay over time
           - batch_size: 32-256, larger = more stable, less regularization
           - architecture: Start simple, add complexity gradually
           - dropout: 0.2-0.5 for hidden layers
        
        3. VALIDATION STRATEGY:
           - Always use cross-validation (5 or 10 fold)
           - Time series: Use time-based splits
           - Stratify for imbalanced data
           - Hold out final test set
        """
        return tips
    
    @staticmethod
    def production_tips():
        """Tips for deploying models in production"""
        tips = """
        PRODUCTION ML BEST PRACTICES:
        
        1. MODEL VERSIONING:
           - Track: Code, data, hyperparameters, metrics
           - Use tools: MLflow, DVC, Weights & Biases
           - Git for code, separate storage for models
        
        2. MONITORING:
           - Input drift: Feature distributions changing?
           - Prediction drift: Output distribution changing?
           - Performance drift: Accuracy degrading?
           - Set up alerts for anomalies
        
        3. SERVING:
           - Batch: Easier, good for non-real-time
           - Real-time: REST API, gRPC for low latency
           - Edge: Deploy on device for privacy/latency
        
        4. OPTIMIZATION:
           - Quantization: Reduce model size (int8 vs float32)
           - Pruning: Remove unnecessary parameters
           - Distillation: Train smaller model to mimic larger
           - ONNX: Cross-platform deployment
        
        5. TESTING:
           - Unit tests for preprocessing
           - Integration tests for pipeline
           - A/B testing for model updates
           - Shadow mode: Run new model alongside old
        
        6. FALLBACKS:
           - Default predictions if model fails
           - Feature degradation handling
           - Circuit breakers for external dependencies
           - Logging for debugging
        """
        return tips


if __name__ == "__main__":
    # Run comprehensive demonstration
    print("\n" + "=" * 70)
    print("CORE ALGORITHMS FOR DS/AI/ML ENGINEERS")
    print("Comprehensive Implementation and Learning Guide")
    print("=" * 70 + "\n")
    
    # Run main demonstration
    success = demonstrate_all_algorithms()
    
    if success:
        print("\n" + "=" * 70)
        print("ADDITIONAL LEARNING RESOURCES")
        print("=" * 70)
        
        # Show feature engineering tips
        fe_tips = AdvancedTips.feature_engineering_tips()
        print(fe_tips)
        
        # Show hyperparameter tuning tips
        hp_tips = AdvancedTips.hyperparameter_tuning_tips()
        print(hp_tips)
        
        # Show production tips
        prod_tips = AdvancedTips.production_tips()
        print(prod_tips)
        
        print("\n" + "=" * 70)
        print("SUMMARY: KEY TAKEAWAYS")
        print("=" * 70)
        print("""
        1. START SIMPLE: Always establish baseline with simple models
        2. UNDERSTAND DATA: EDA is crucial before model selection
        3. VALIDATE PROPERLY: Cross-validation, hold-out test sets
        4. FEATURE ENGINEERING: Often more important than model choice
        5. ITERATE: Progressive refinement beats perfection
        6. MONITOR: Models degrade over time in production
        7. DOCUMENT: Code, assumptions, and decisions
        8. TEST: Unit tests, integration tests, A/B tests
        
        Remember: No algorithm is universally best. Context matters!
        """)
        
        print("\nTutorial complete! Practice with real datasets to solidify understanding.")
        print("Recommended datasets: Kaggle competitions, UCI ML Repository, sklearn datasets")
