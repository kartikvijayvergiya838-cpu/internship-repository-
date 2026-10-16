from src.data_loader import load_data
from src.eda import inspect_dataset
from src.preprocess import clean_and_preprocess
from src.train import train_models, save_artifacts
from src.evaluate import evaluate_models

def main():
    print("\n--- Step 1: Loading HR Dataset ---")
    df = load_data()
    
    print("\n--- Step 2: Running Exploratory Analysis ---")
    inspect_dataset(df)
    
    print("\n--- Step 3: Cleaning & Preprocessing ---")
    X_train, X_test, y_train, y_test, feature_names = clean_and_preprocess(df)
    print(f"Features Generated: {len(feature_names)} | Training Set: {X_train.shape[0]} samples")
    
    print("\n--- Step 4: Training Classifiers ---")
    models = train_models(X_train, y_train)
    
    print("\n--- Step 5: Evaluating Model Benchmarks ---")
    results = evaluate_models(models, X_test, y_test)
    print("\nBenchmark Summary:")
    print(results.to_string(index=False))
    
    best_model_name = results.iloc[0]['Model']
    print(f"\nTop Performing Model: {best_model_name}")
    save_artifacts(models[best_model_name], feature_names)

if __name__ == "__main__":
    main()