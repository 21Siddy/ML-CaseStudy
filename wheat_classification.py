import numpy as np
import pandas as pd

def load_wheat_data(file_path):
    """
    Load wheat classification data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the wheat data.

    Returns:
    pd.DataFrame: A DataFrame containing the wheat data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None