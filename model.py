from base_class import BaseModel
import pandas as pd
import numpy as np

class Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
    
    def fit(self):
        df = pd.read_csv(self.config['data_path'])
        print(df.head())

if '__name__' == '__main__':
    config = {
        'data_path': 'train_df.csv'
    }
    model = Model(config)
    model.fit()