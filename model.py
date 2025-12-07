from base_class import BaseMLModel
import pandas as pd
import numpy as np

class Model(BaseMLModel):
    def __init__(self):
        super().__init__()
        self.config = {
            'data_path': 'train_df.csv'
        }
        # Attributes to store model state
        self.train_min = None
        self.train_max = None
        self.X_train_normalized = None
        self.y_train = None

    def compute_min_max(self, data: np.ndarray):
        """Helper to calculate min and max scaling parameters."""
        self.train_min = data.min(axis=0)
        self.train_max = data.max(axis=0)
    
    def normalise_data(self, data: np.ndarray) -> np.ndarray:
        """Helper to apply min-max scaling."""
        if self.train_min is None or self.train_max is None:
            raise ValueError("Model is not fitted. Min and Max values are missing.")
        
        # Avoid division by zero if max == min (constant feature)
        # We add a tiny epsilon or handle it, but for this dataset simple subtraction is likely fine.
        denominator = self.train_max - self.train_min
        denominator[denominator == 0] = 1.0  # Prevent division by zero
        
        return (data - self.train_min) / denominator

    def calculate_euclidean_distance(self, x_single):
        """Helper to calculate distance from one test point to all training points."""
        # Vectorized Euclidean distance calculation
        # Sum of squared differences along axis 1 (features), then sqrt
        distances = np.sqrt(np.sum((self.X_train_normalized - x_single) ** 2, axis=1))
        return distances
    
    def get_k_nearest_neighbors(self, x_single, k=5):
        """Helper to find the k nearest neighbors and vote."""
        distances = self.calculate_euclidean_distance(x_single)
        
        # Get indices of the k smallest distances
        neighbor_indices = np.argsort(distances)[:k]
        
        # Get the labels for these indices
        neighbor_labels = self.y_train[neighbor_indices]
        
        # Majority voting
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        
        return majority_label

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseMLModel':
        """
        Train the model. Implements rigorous validation as per BaseMLModel.
        """
        # 1. Validate Input Types
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Input data X and y must be numpy arrays.")
        
        # 2. Validate Input Shapes
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (n_samples, n_features), got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) must match.")

        # 3. Learn Parameters (Scaling)
        self.compute_min_max(X)
        
        # 4. Store Training Data (Normalized)
        self.X_train_normalized = self.normalise_data(X)
        self.y_train = y
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions. Implements rigorous validation as per BaseMLModel.
        """
        # 1. Check if model is fitted
        if self.X_train_normalized is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # 2. Validate Input Types
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data X must be a numpy array.")
            
        # 3. Validate Input Shapes
        if X.ndim != 2:
             raise ValueError(f"X must be 2D array, got shape {X.shape}")
        if X.shape[1] != self.X_train_normalized.shape[1]:
            raise ValueError(f"Feature mismatch: Model expects {self.X_train_normalized.shape[1]} features, got {X.shape[1]}")

        # 4. Normalise Input Data
        X_normalized = self.normalise_data(X)
        
        # 5. Generate Predictions
        predictions = []
        for i in range(X_normalized.shape[0]):
            # Prediction logic for each sample
            pred_label = self.get_k_nearest_neighbors(X_normalized[i], k=5)
            predictions.append(pred_label)
            
        return np.array(predictions)

if __name__ == '__main__':
    # 1. Load Data
    try:
        train_df = pd.read_csv('train_df.csv')
        test_df = pd.read_csv('test_df.csv')
    except FileNotFoundError:
        print("Error: Data files not found. Make sure train_df.csv and test_df.csv are in the same directory.")
        exit(1)

    # 2. Prepare Training Data
    # Drop the target column to get features (X)
    X_train = train_df.drop(columns=['category']).values
    # Select only the target column to get labels (y)
    y_train = train_df['category'].values

    # 3. Prepare Test Data
    X_test = test_df.values

    # 4. Initialize and Train Model
    print("Initializing and training model...")
    model = Model()
    try:
        model.fit(X_train, y_train)
        print("Model trained successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        exit(1)

    # 5. Predict on Test Data
    print("Generating predictions...")
    try:
        predictions = model.predict(X_test)
        print(f"Generated {len(predictions)} predictions.")
        
        # 6. Save Predictions
        np.savetxt('predictions.txt', predictions, fmt='%d')
        print("Predictions saved to 'predictions.txt'.")
    except Exception as e:
        print(f"Prediction failed: {e}")