import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            scale_pos_weight=5,  # Handles class imbalance
            eval_metric='logloss'
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def save_artifacts(model, feature_names, model_path="models/attrition_model.pkl", features_path="models/feature_names.pkl"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)
    print(f"\nModel and features successfully saved to 'models/' directory.")