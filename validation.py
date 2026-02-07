import numpy as np
import pandas as pd
from model import Model

# --- Metric Helper Functions (No Scikit-Learn allowed) ---

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def calculate_confusion_matrix(y_true, y_pred, n_classes=4):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix

def calculate_f1_macro(y_true, y_pred, n_classes=4):
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Macro F1 = Average of F1 scores for each class
    
    f1_scores = []
    
    for c in range(n_classes):
        # True Positives, False Positives, False Negatives
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        # Precision & Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 for this class
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        f1_scores.append(f1)
        
    return np.mean(f1_scores), f1_scores

# --- Main Validation Routine ---

def main():
    print("--- STARTING VALIDATION RUN ---")
    
    # 1. Load Data
    df = pd.read_csv('train_df.csv')
    
    # Separate Features and Target
    # Drop ID (col 0) and num_errors (target, last col)
    # Assuming 'ID' is the first column and 'num_errors' is the target
    X = df.drop(columns=['ID', 'num_errors']).values
    y = df['num_errors'].values
    
    # 2. Shuffle and Split (80% Train, 20% Validation)
    np.random.seed(42) # Fixed seed for reproducibility
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"Training Set: {X_train.shape[0]} samples")
    print(f"Validation Set: {X_val.shape[0]} samples")
    print("-" * 30)

    # 3. Train Model
    print("Training Model...")
    model = Model()
    model.fit(X_train, y_train)
    
    # 4. Predict
    print("Generating Predictions...")
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # 5. Compute Metrics for TRAIN
    train_acc = calculate_accuracy(y_train, y_train_pred)
    train_f1, _ = calculate_f1_macro(y_train, y_train_pred)
    
    # 6. Compute Metrics for VALIDATION
    val_acc = calculate_accuracy(y_val, y_val_pred)
    val_f1, val_f1_per_class = calculate_f1_macro(y_val, y_val_pred)
    val_cm = calculate_confusion_matrix(y_val, y_val_pred)
    
    # 7. Report Results
    print("\n" + "="*30)
    print("       RESULTS REPORT       ")
    print("="*30)
    
    print(f"\n[TRAINING SET]")
    print(f"Accuracy:  {train_acc:.4f}")
    print(f"Macro F1:  {train_f1:.4f}")
    
    print(f"\n[VALIDATION SET] (The important one)")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Macro F1:  {val_f1:.4f}  <-- Needs to be > 0.5 for bonus")
    
    print("\nValidation F1 per Class:")
    for i, score in enumerate(val_f1_per_class):
        print(f"Class {i}: {score:.4f}")
        
    print("\nConfusion Matrix (Validation):")
    # Pretty print confusion matrix
    print("      Pred 0  Pred 1  Pred 2  Pred 3")
    for i, row in enumerate(val_cm):
        print(f"True {i}  {row[0]:<7} {row[1]:<7} {row[2]:<7} {row[3]:<7}")

if __name__ == "__main__":
    main()