# src/inference.py
import os
import joblib
import pandas as pd
from data_processing import basic_cleaning, encode_object_cols

# Paths to artifacts
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_PATH, "best_model.joblib")
SCALER_PATH = os.path.join(ARTIFACTS_PATH, "scaler.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_PATH, "X_columns.joblib")

# Load model, scaler, and feature columns
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X_columns = joblib.load(FEATURES_PATH)

def predict_autism(new_data: pd.DataFrame) -> pd.Series:
    """
    Input: new_data (DataFrame) with same columns as training features (excluding 'result')
    Output: predictions (0 = No ASD, 1 = ASD)
    """
    # Remove 'result' column if present
    if "result" in new_data.columns:
        new_data = new_data.drop(columns=["result"])

    # 1️⃣ Clean & encode
    df = basic_cleaning(new_data)
    df = encode_object_cols(df)

    # 2️⃣ Reindex to match training columns
    df = df.reindex(columns=X_columns, fill_value=0)

    # 3️⃣ Scale and predict
    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    return pd.Series(predictions)

def manual_input_prediction():
    """
    Let user enter values manually for each feature column (excluding target) and get prediction.
    """
    # Load feature columns
    X_columns = joblib.load(FEATURES_PATH)
    
    # Remove 'result' if present
    X_columns = [col for col in X_columns if col != "result"]

    print("\nEnter values for the following features:")
    input_data = {}
    for col in X_columns:
        while True:
            val = input(f"{col}: ")
            try:
                # Try numeric
                input_data[col] = float(val)
                break
            except ValueError:
                # If categorical, keep as string
                input_data[col] = val
                break
    
    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Predict
    prediction = predict_autism(df_input).iloc[0]
    print(f"\nPredicted Class: {prediction} ({'ASD' if prediction==1 else 'No ASD'})\n")

# Main execution
if __name__ == "__main__":
    print("Choose input mode:")
    print("1. CSV file (sample_input.csv)")
    print("2. Manual input via terminal")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_input.csv")
        if os.path.exists(sample_path):
            new_df = pd.read_csv(sample_path)
            # Drop 'result' column if present in CSV
            if "result" in new_df.columns:
                new_df = new_df.drop(columns=["result"])
            preds = predict_autism(new_df)
            print("\nPredictions:", preds.tolist())
        else:
            print("Place a CSV file named 'sample_input.csv' in the data folder for testing.")
    elif choice == "2":
        manual_input_prediction()
    else:
        print("Invalid choice.")
