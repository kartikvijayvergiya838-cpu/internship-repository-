import pandas as pd
import os

def load_data(filepath="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please download it from Kaggle.")
    df = pd.read_csv(filepath)
    return df
    