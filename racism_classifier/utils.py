import pandas as pd
from datasets import Dataset

def load_data(file_path:str):
    """Loads the data from an xls file and returns a Dataset object"""
    df = pd.read_excel(file_path)
    return Dataset.from_pandas(df)
