from base_class import BaseMLModel
import pandas as pd
import numpy as np

class Model(BaseMLModel):
    def __init__(self):
        super().__init__()
        # hyperparameters
        self.learning_rate = 0.1
        self.num_epochs = 1000
        self.reg_lambda = 0.01

        # learned parameters (initialized later)
        self.W = None  # shape (n_features, n_classes)
        self.b = None  # shape (n_classes,)

        # for feature scaling
        self.mean_ = None  # shape (n_features,)
        self.std_ = None   # shape (n_features,)
    
    def fit(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")

        if X.ndim != 2:
            raise ValueError("X must be 2D with shape (n_samples, n_features)")

        y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be 1D with shape (n_samples,)")

        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("X and y must have the same number of samples")

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_

        classes = np.unique(y)
        self.classes_ = classes
        n_classes = classes.shape[0]

        class_to_index = {c: i for i, c in enumerate(classes)}
        y_idx = np.vectorize(class_to_index.get)(y)
        Y_onehot = np.eye(n_classes)[y_idx]

        rng = np.random.default_rng(42)
        if self.W is None:
            self.W = rng.normal(0, 0.01, size=(n_features, n_classes))
        if self.b is None:
            self.b = np.zeros(n_classes)

        lr = float(self.learning_rate)
        epochs = int(self.num_epochs)
        reg = float(self.reg_lambda)

        for _ in range(epochs):
            scores = Xs @ self.W + self.b
            scores -= scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

            # optional loss (not printed): cross-entropy + L2
            # loss = (-np.log(probs[np.arange(n_samples), y_idx]).mean() 
            #         + 0.5 * reg * np.sum(self.W * self.W))

            dscores = (probs - Y_onehot) / n_samples
            dW = Xs.T @ dscores + reg * self.W
            db = dscores.sum(axis=0)

            self.W -= lr * dW
            self.b -= lr * db

        return self

    def predict(self, X):
        if self.W is None or self.b is None or self.mean_ is None or self.std_ is None:
            raise ValueError("Model is not fitted yet")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError("X must be 2D with shape (n_samples, n_features)")
        if X.shape[1] != self.W.shape[0]:
            raise ValueError("Number of features in X does not match the model")

        Xs = (X - self.mean_) / self.std_
        scores = Xs @ self.W + self.b
        preds_idx = scores.argmax(axis=1)
        return self.classes_[preds_idx]

    def evaluate(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        y_pred = self.predict(X)
        y_true = y.ravel()
        accuracy = (y_pred == y_true).mean()
        # build confusion matrix
        classes = self.classes_
        class_to_idx = {c: i for i, c in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for yt, yp in zip(y_true, y_pred):
            cm[class_to_idx[yt], class_to_idx[yp]] += 1
        return accuracy, cm

if __name__ == '__main__':
    model = Model()

    # Load training data
    train_df = pd.read_csv('train_df.csv')
    X_train = train_df.drop(columns=['category']).values.astype(float)
    y_train = train_df['category'].values

    # Simple train/validation split
    rng = np.random.default_rng(0)
    idx = np.arange(X_train.shape[0])
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, val_idx = idx[:split], idx[split:]
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # Train the model
    model.fit(X_tr, y_tr)

    # Evaluate on validation set
    acc, cm = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Load test data and make predictions
    test_df = pd.read_csv('test_df.csv')
    X_test = test_df.values.astype(float)
    predictions = model.predict(X_test)

    # Show a quick preview of predictions
    print(predictions[:10])
    print(predictions)
    
    # write predictions.txt
    with open("predictions.txt", "w") as f:
        for label in predictions:
            f.write(f"{int(label)}\n")