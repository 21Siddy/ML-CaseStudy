import pandas as pd
import numpy as np

def main():
    print("--- 1. LOADING DATA ---")
    df = pd.read_csv('train_df.csv') # Using the full dataset
    X = df.drop(columns=['ID', 'num_errors']).values
    y = df['num_errors'].values
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features.")

    print("\n--- 2. CLASS BALANCE CHECK ---")
    classes, counts = np.unique(y, return_counts=True)
    for c, count in zip(classes, counts):
        print(f"Class {c}: {count} samples ({count/len(y)*100:.2f}%)")
    print("-> DIAGNOSIS: Severe imbalance. 'Uniform' oversampling is too aggressive.")

    print("\n--- 3. CORRELATION CHECK (The Hidden Killer) ---")
    # We take a random sample of 1000 rows to speed up correlation check
    idx = np.random.choice(len(X), 1000, replace=False)
    X_sample = X[idx]
    
    # Normalize first
    X_sample = (X_sample - np.mean(X_sample, axis=0)) / (np.std(X_sample, axis=0) + 1e-9)
    
    # Calculate Correlation Matrix
    corr_matrix = np.corrcoef(X_sample, rowvar=False)
    np.fill_diagonal(corr_matrix, 0) # Ignore self-correlation
    
    # Count pairs with correlation > 0.90
    high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.90) / 2
    print(f"Number of feature pairs with Correlation > 0.90: {int(high_corr_pairs)}")
    
    if high_corr_pairs > 100:
        print("-> DIAGNOSIS: High Redundancy. Naive Bayes is 'double counting' evidence.")
        print("-> FIX: We MUST implement a Correlation Filter.")
    else:
        print("-> DIAGNOSIS: Correlation is low. Feature selection is safe.")

if __name__ == "__main__":
    main()