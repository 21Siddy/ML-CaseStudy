"""
Abstract Base Class for Machine Learning Models

This module provides an abstract base class that defines the essential interface
for machine learning models in the course. You must implement all abstract
methods to create a working ML model.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseMLModel(ABC):
    """
    Abstract base class for machine learning models.

    This class defines the essential interface that all machine learning models
    must implement. It enforces a consistent API across different model types
    (classification, regression, etc.) and ensures students implement all
    necessary functionality.
    """

    def __init__(self):
        """
        Initialize the base model.

        Subclasses should call super().__init__() and then initialize their
        own parameters and hyperparameters.
        """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMLModel':
        """
        Train the machine learning model on the provided data.

        This method must be implemented by all subclasses. It should:
        1. Validate input data
        2. Store necessary information about the training data
        3. Learn model parameters from the training data

        Args:
            X (np.ndarray): Training features with shape (n_samples, n_features)
                           Each row represents a sample, each column a feature
            y (np.ndarray): Training targets with shape (n_samples,) for single output
                           or (n_samples, n_outputs) for multi-output problems
                           For classification: integer class labels (0, 1, 2, ...)
                           For regression: continuous target values

        Returns:
            BaseMLModel: Returns self to allow method chaining (e.g., model.fit(X, y).predict(X_test))

        Raises:
            ValueError: If input data has invalid shape or contains invalid values
            TypeError: If input data is not numpy arrays

        Example:
            >>> model = YourModelClass()
            >>> model.fit(X_train, y_train)
            >>> # Model is now trained and ready for prediction
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data using the trained model.

        This method must be implemented by all subclasses. It should:
        1. Validate input data format and shape
        2. Generate predictions using the learned parameters

        Args:
            X (np.ndarray): Input features with shape (n_samples, n_features)
                           Must have the same number of features as training data
                           Each row represents a sample to predict

        Returns:
            np.ndarray: Predictions with shape (n_samples,) for single output
                       or (n_samples, n_outputs) for multi-output problems
                       For classification: predicted class labels
                       For regression: predicted continuous values

        Raises:
            ValueError: If model hasn't been fitted or input has wrong shape
            TypeError: If input is not a numpy array

        Example:
            >>> predictions = model.predict(X_test)
            >>> print(f"Predicted classes: {predictions}")
        """
        pass
