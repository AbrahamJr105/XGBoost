#!/usr/bin/env python3
# predict_xgboost.py
# Load a saved XGBoost model and make predictions on new data.

import argparse
import pandas as pd
import xgboost as xgb
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Predict using a saved XGBoost model")
    parser.add_argument("--model", type=str, default="xgboost_model.json",
                        help="Path to the saved XGBoost model file (default: xgboost_model.json)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the CSV file containing data for prediction (must have same features as training data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save predictions to a CSV file.")
    return parser.parse_args()

def load_prediction_data(path):
    """Load CSV into pandas DataFrame and select numeric features for prediction."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prediction data file not found: {path}")
    df = pd.read_csv(path)

    # Select only numeric columns - assumes same features as training
    # Exclude 'value' if it exists, as it's the target we want to predict
    X = df.select_dtypes(include=['number']).drop('value', axis=1, errors='ignore')

    if X.empty:
        raise ValueError("No numeric features found in the prediction data. Check your CSV.")

    print(f"Using features for prediction: {X.columns.tolist()}")
    return X, df # Return original df too if saving output

def load_model(model_path):
    """Load the XGBoost model from file."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    bst = xgb.Booster()
    bst.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return bst

def main():
    args = parse_args()

    # Load the model
    model = load_model(args.model)

    # Load data for prediction
    X_predict, original_df = load_prediction_data(args.data)

    # Ensure feature names match the model's expected features if possible
    # Note: XGBoost DMatrix doesn't strictly require matching names if order is correct,
    # but it's good practice. The model object itself stores feature names if saved from XGBoost >= 1.0
    if model.feature_names is not None:
        if list(X_predict.columns) != model.feature_names:
            print("Warning: Feature names in prediction data do not match model's feature names.")
            print(f"Prediction data features: {list(X_predict.columns)}")
            print(f"Model expected features: {model.feature_names}")
            # Optional: Reorder columns if necessary and possible
            try:
                X_predict = X_predict[model.feature_names]
                print("Reordered prediction data columns to match model.")
            except KeyError as e:
                raise ValueError(f"Prediction data is missing feature required by the model: {e}")

    # Create DMatrix for prediction
    dpredict = xgb.DMatrix(X_predict)

    # Make predictions
    predictions = model.predict(dpredict)

    print("\n--- Predictions ---")
    print(predictions)

    # Optionally save predictions
    if args.output:
        output_df = original_df.copy()
        output_df['predicted_value'] = predictions
        output_df.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")

if __name__ == "__main__":
    main()
