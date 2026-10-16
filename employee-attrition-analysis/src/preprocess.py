import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_and_preprocess(df):
    df = df.copy()
    
    # 1. Drop redundant columns with zero variance or identifiers
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    # 2. Target Encoding: Attrition (Yes -> 1, No -> 0)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # 3. Categorical Encoding (One-Hot)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 4. Split Features and Target
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    feature_names = X.columns.tolist()
    
    # 5. Train-Test Split (Stratified due to class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. Scale numerical features
    num_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 
                'MonthlyIncome', 'MonthlyRate', 'TotalWorkingYears', 
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
                'YearsWithCurrManager']
    
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, feature_names