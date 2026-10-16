import pandas as pd

def inspect_dataset(df):
    print("--- Dataset Overview ---")
    print(f"Total Records: {df.shape[0]} | Features: {df.shape[1]}")
    print("\nMissing Values Count:\n", df.isnull().sum()[df.isnull().sum() > 0])
    
    attrition_rate = (df['Attrition'].value_counts(normalize=True) * 100)
    print("\nTarget Distribution (Attrition %):")
    print(attrition_rate.round(2))
    
    # Key HR metrics summary
    print("\n--- Key Factor Averages by Attrition ---")
    metrics = ['Age', 'MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany']
    print(df.groupby('Attrition')[metrics].mean().round(2))