import numpy as np
from base_class import BaseMLModel

class Model(BaseMLModel):
    """
    Gaussian Naive Bayes with Fisher Score and Soft Class Balancing.
    """

    def __init__(self):
        super().__init__()
        # Since prelim.py showed NO correlation, we can safely keep more features.
        # 100 is a robust number to capture signal without overfitting.
        self.k_best_features = 100 
        
        self.selected_indices = None
        self.scaler_mean = None
        self.scaler_std = None
        self.class_priors = None
        self.theta = None
        self.sigma = None
        self.epsilon = 1e-9

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
        """
        Train GNB using Square-Root Priors to handle imbalance gently.
        """
        # 1. Standardize (Z-Score)
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        X_scaled = (X - self.scaler_mean) / self.scaler_std

        # 2. Fisher Score Feature Selection
        self.classes = np.unique(y)
        mean_global = np.mean(X_scaled, axis=0)
        
        numerator = np.zeros(X_scaled.shape[1])
        denominator = np.zeros(X_scaled.shape[1])
        
        for c in self.classes:
            X_c = X_scaled[y == c]
            mean_c = np.mean(X_c, axis=0)
            var_c = np.var(X_c, axis=0)
            numerator += X_c.shape[0] * (mean_c - mean_global) ** 2
            denominator += X_c.shape[0] * var_c

        denominator[denominator == 0] = 1.0
        fisher_scores = numerator / denominator
        
        # Select Top 100
        sorted_indices = np.argsort(fisher_scores)[::-1]
        self.selected_indices = sorted_indices[:self.k_best_features]
        X_selected = X_scaled[:, self.selected_indices]

        # 3. Train GNB with Square-Root Priors
        n_classes = len(self.classes)
        self.theta = np.zeros((n_classes, self.k_best_features))
        self.sigma = np.zeros((n_classes, self.k_best_features))

        # --- THE FIX ---
        # Instead of "Natural" (81% / 2%) or "Uniform" (25% / 25%),
        # we use Square Root: P(c) ~ sqrt(Count).
        # This boosts minority visibility without causing hallucination.
        counts = np.array([np.sum(y == c) for c in self.classes])
        sqrt_counts = np.sqrt(counts)
        self.class_priors = sqrt_counts / np.sum(sqrt_counts)

        for idx, c in enumerate(self.classes):
            X_c = X_selected[y == c]
            # Calculate Mean and Variance for each class
            self.theta[idx, :] = np.mean(X_c, axis=0)
            self.sigma[idx, :] = np.var(X_c, axis=0) + self.epsilon

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 1. Impute & Scale
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_selected = X_scaled[:, self.selected_indices]

        # 2. Calculate Log-Likelihood
        n_samples = X_selected.shape[0]
        n_classes = len(self.classes)
        log_posteriors = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            prior = np.log(self.class_priors[idx])
            term1 = -0.5 * np.sum(np.log(2 * np.pi * self.sigma[idx, :]))
            diff = X_selected - self.theta[idx, :]
            term2 = -0.5 * np.sum((diff ** 2) / self.sigma[idx, :], axis=1)
            log_posteriors[:, idx] = prior + term1 + term2

        return self.classes[np.argmax(log_posteriors, axis=1)]