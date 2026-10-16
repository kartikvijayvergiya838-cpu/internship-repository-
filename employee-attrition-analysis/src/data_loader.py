import pandas as pd
import os

def load_data(filepath="data/WA_Fn-UseC_-HR-Employee-Attrition.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at '{filepath}'. Please check the data folder.")
    df = pd.read_csv(filepath)
    return df