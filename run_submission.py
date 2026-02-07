import pandas as pd
import numpy as np
from model import Model

def main():
    # 1. Load Data
    print("Loading datasets...")
    train_df = pd.read_csv('train_df.csv')
    test_df = pd.read_csv('test_df.csv')
    submission_df = pd.read_csv('submission.csv')

    # 2. Preprocess
    # Drop ID columns to match matrix format
    X_train = train_df.drop(columns=['ID', 'num_errors']).values
    y_train = train_df['num_errors'].values
    X_test = test_df.drop(columns=['index']).values

    # 3. Train
    print("Training model...")
    model = Model()
    model.fit(X_train, y_train)

    # 4. Predict
    print("Predicting test set...")
    predictions = model.predict(X_test)

    # 5. Save
    submission_df['Predicted'] = predictions
    submission_df.to_csv('submission_new_2.csv', index=False)
    print("Success! 'submission_new_2.csv' has been updated.")

if __name__ == "__main__":
    main()