from base_class import BaseMLModel
import pandas as pd
import numpy as np

class Model(BaseMLModel):
    def __init__(self):
        self.config = {
            'data_path': 'train_df.csv'
        }
        self.model = None
    
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([])

if __name__ == '__main__':
    model = Model()
    print("works for me as well")