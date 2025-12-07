import pandas as pd
import numpy as np

df = pd.read_csv('train_df.csv')
print(df.head())
print('#'*40)
print(f'the shape of the dataframe {df.shape}')
print('#'*40)
print(f'the value count of the category column is \n{df["category"].value_counts()}')
print(f'the description of the dataframe \n{df.describe()}')
