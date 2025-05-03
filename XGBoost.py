#!/usr/bin/env python3
# train_xgboost.py
# Complete XGBoost regression training script
# Sources: MachineLearningMastery :contentReference[oaicite:0]{index=0}, DataCamp :contentReference[oaicite:1]{index=1}, RandomRealizations :contentReference[oaicite:2]{index=2}

import os
import time  # Add time for timing operations
import argparse
import pandas as pd                               # data handling :contentReference[oaicite:3]{index=3}
from sklearn.model_selection import train_test_split, GridSearchCV  # splitting & tuning :contentReference[oaicite:4]{index=4}
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error  # Add MAE
import xgboost as xgb                                                       # XGBoost API :contentReference[oaicite:6]{index=6}
import matplotlib.pyplot as plt                                             # plotting feature importance :contentReference[oaicite:7]{index=7}
import psutil # For system resource monitoring

def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost regression on weather–yield data")
    parser.add_argument("--data", type=str, default="algeria_weather_yield.csv",
                        help="Path to CSV dataset (default: algeria_weather_yield.csv)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of validation set (default: 0.2)")
    parser.add_argument("--early_stop", type=int, default=10,
                        help="Early stopping rounds (default: 10)")
    return parser.parse_args()

def load_data(path):
    """Load CSV into pandas DataFrame and select numeric features"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    if 'value' not in df.columns:
        raise ValueError("Expected a 'value' column in the dataset")

    # Select only numeric columns as features (excluding the target 'Value')
    X = df.select_dtypes(include=['number']).drop('value', axis=1, errors='ignore')
    y = df['value']

    if X.empty:
        raise ValueError("No numeric features found in the dataset after excluding 'Value'. Check your CSV.")

    print(f"Using features: {X.columns.tolist()}") # Log the features being used

    return X, y

def train_xgboost(X_train, y_train, X_val, y_val, params, num_rounds, early_stop):
    """Train using native XGBoost API with early stopping"""
    dtrain = xgb.DMatrix(X_train, label=y_train)     # optimized DMatrix :contentReference[oaicite:9]{index=9}
    dval   = xgb.DMatrix(X_val,   label=y_val)
    evals  = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=early_stop,
        verbose_eval=True
    )                                                # training loop :contentReference[oaicite:10]{index=10}
    return model

def evaluate(model, X_test, y_test):
    """Predict and print RMSE, R², MAE, MSE, and Inference Time"""
    dtest = xgb.DMatrix(X_test)

    start_time = time.time()
    y_pred = model.predict(dtest)
    end_time = time.time()
    inference_time = end_time - start_time

    # Remove squared=False as it's not a valid argument for root_mean_squared_error in this sklearn version
    rmse = root_mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = rmse**2 # MSE is RMSE squared

    print(f"  Inference Time: {inference_time:.4f} seconds")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    # Return metrics for potential further use
    return {'rmse': rmse, 'r2': r2, 'mae': mae, 'mse': mse, 'inference_time': inference_time}

def plot_feature_importance(model, fmap=None):
    """Plot the top 10 feature importances"""
    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(model, fmap=fmap, max_num_features=10, ax=ax)
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()                                       # visualize key features :contentReference[oaicite:11]{index=11}

def grid_search_cv(X, y):
    """Optional: Grid search over key hyperparameters"""
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=4, random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.7, 0.9]
    }
    gs = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1
    )                                                # grid search setup :contentReference[oaicite:12]{index=12}
    gs.fit(X, y)
    print("Best parameters:", gs.best_params_)
    print("Best CV RMSE:   ", -gs.best_score_)
    return gs.best_estimator_

def main():
    # --- Resource Monitoring Setup ---
    process = psutil.Process(os.getpid())
    mem_before_train = process.memory_info().rss / (1024 * 1024) # Resident Set Size in MB
    cpu_percent_before_train = process.cpu_percent(interval=None) # Get initial CPU%

    args = parse_args()  # Parse command-line arguments
    X, y = load_data(args.data) # Use args.data instead of hardcoded path
    feature_names = X.columns.tolist() # Get feature names

    # Split data directly into training and validation sets
    # No separate test set due to small dataset size
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # Default hyperparameters :contentReference[oaicite:14]{index=14}
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    num_rounds = 100

    # --- Train model ---
    print("\n--- Training Model ---")
    start_train_time = time.time()
    model = train_xgboost(X_train, y_train, X_val, y_val, params, num_rounds, args.early_stop)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # --- Resource Monitoring After Training ---
    mem_after_train = process.memory_info().rss / (1024 * 1024) # MB
    # Call cpu_percent again after the interval to get meaningful reading
    cpu_percent_during_train = process.cpu_percent(interval=None)
    # Note: cpu_percent measures the usage since the last call or process start.
    # This value represents the average CPU usage during the script execution up to this point.
    # For peak usage during training specifically, more advanced monitoring (e.g., threading) is needed.

    print(f"Training completed in {training_time:.4f} seconds.")

    # --- Save Model ---
    model_filename = "xgboost_model.json" # Or use .ubj for binary format
    model.save_model(model_filename)
    print(f"Model saved to {model_filename}")
    try:
        model_size_bytes = os.path.getsize(model_filename)
        model_size_kb = model_size_bytes / 1024
        print(f"Model file size: {model_size_kb:.2f} KB")
    except OSError as e:
        print(f"Could not get model file size: {e}")
        model_size_kb = None


    # --- Evaluate ---
    print("\n--- Evaluating on Validation Set ---")
    eval_metrics = evaluate(model, X_val, y_val)

    # --- Feature importance ---
    print("\n--- Plotting Feature Importance ---")
    plot_feature_importance(model, fmap='') # fmap='' might not be needed if feature names are in DMatrix

    # --- Reporting Statistics ---
    print("\n\n--- MODEL STATISTICS SUMMARY ---")
    print("\n1. Performance Metrics (Validation Set):")
    print(f"   RMSE: {eval_metrics['rmse']:.4f}")
    print(f"   MSE:  {eval_metrics['mse']:.4f}")
    print(f"   MAE:  {eval_metrics['mae']:.4f}")
    print(f"   R² Score: {eval_metrics['r2']:.4f}")
    print(f"   Training Time: {training_time:.4f} seconds")
    print(f"   Inference Time (Validation): {eval_metrics['inference_time']:.4f} seconds")

    print("\n2. Resource Requirements:")
    if model_size_kb is not None:
        print(f"   Model Storage Size: {model_size_kb:.2f} KB")
    print(f"   Memory Usage (Training): ~{mem_after_train - mem_before_train:.2f} MB increase (End: {mem_after_train:.2f} MB)")
    print(f"   Avg. CPU Usage (During Script Run): {cpu_percent_during_train:.2f}%")
    print(f"   Dependencies: pandas, scikit-learn, xgboost, matplotlib, psutil") # Updated dependencies

    print("\n3. Model Characteristics:")
    print(f"   Model Type: XGBoost Regressor")
    print(f"   Number of Features Used: {len(feature_names)}")
    print(f"   Features Used: {feature_names}")
    print(f"   Hyperparameters Used:")
    for key, value in params.items():
        print(f"     {key}: {value}")
    print(f"   Number of Boosting Rounds (Default): {num_rounds}")
    if hasattr(model, 'best_iteration'): # Check if early stopping occurred
         print(f"   Actual Boosting Rounds (Early Stopping): {model.best_iteration}")
    else:
         print(f"   Actual Boosting Rounds: {num_rounds} (Early stopping not triggered or not used effectively)")


    # Optional Grid Search Section (remains unchanged)
    # Uncomment to run grid search (may be slow)
    # Note: Grid search typically uses cross-validation on the training data,
    # so it doesn't directly use the validation set split here.
    # best_model = grid_search_cv(X_train, y_train) # Pass only training data
    # print("\n--- Evaluating Best GridSearch Model on Validation Set ---")
    # evaluate(best_model, X_val, y_val)
    # plot_feature_importance(best_model)

if __name__ == "__main__":
    main()
