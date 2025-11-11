# frontend/app.py
import os
import sys
from pathlib import Path

# --- Fix import path ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import joblib
from src.data_processing import basic_cleaning, encode_object_cols

# --- Load model artifacts ---
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_PATH / "best_model.joblib"
SCALER_PATH = ARTIFACTS_PATH / "scaler.joblib"
FEATURES_PATH = ARTIFACTS_PATH / "X_columns.joblib"

if not ARTIFACTS_PATH.exists():
    st.error(f"Artifacts folder not found at: {ARTIFACTS_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_COLUMNS = joblib.load(FEATURES_PATH)
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# --- Mapping model columns to user-friendly questions ---
QUESTIONS_MAPPING = {
    "A1_Score": "Does your child have difficulty making eye contact?",
    "A2_Score": "Does your child respond to their name being called?",
    "A3_Score": "Does your child show interest in other children?",
    "A4_Score": "Does your child engage in imaginative play?",
    "A5_Score": "Does your child repeat words or phrases?",
    "A6_Score": "Does your child follow instructions appropriately?",
    "A7_Score": "Does your child get upset by minor changes?",
    "A8_Score": "Does your child have unusual sensory interests?",
    "A9_Score": "Does your child have difficulty with social cues?",
    "A10_Score": "Does your child engage in repetitive movements?",
    "ethnicity": "Select Ethnicity",
    "country_of_res": "Select Country of Residence",
    "relation": "Select Relation",
    "age": "Enter the age of the individual",
    "gender": "Select Gender"
}

# --- Dropdown options for categorical features ---
DROPDOWN_OPTIONS = {
    "ethnicity": ["White-European", "Latino", "Middle Eastern", "Black", "South Asian", "East Asian", "Other"],
    "country_of_res": ["United States", "United Kingdom", "India", "Canada", "Australia", "Other"],
    "relation": ["Self", "Parent", "Relative", "Health Care Professional", "Other"],
    "gender": ["male", "female", "other"]
}

# --- Streamlit Page Config ---
st.set_page_config(page_title="Autism Prediction App", page_icon="🧠", layout="centered")
st.title("🧠 Autism Spectrum Disorder (ASD) Prediction")
st.write("Enter responses manually or upload a CSV file to predict ASD likelihood.")

# --- Prediction Function ---
def predict_df(df: pd.DataFrame):
    """Cleans, encodes, scales, and predicts."""
    df = basic_cleaning(df)
    df = encode_object_cols(df)
    df = df.reindex(columns=X_COLUMNS, fill_value=0)
    arr = scaler.transform(df)
    preds = model.predict(arr)
    return preds

# --- Input Mode ---
mode = st.sidebar.radio("Input Mode", ["Manual Entry", "CSV Upload"])

# --- Manual Entry Mode ---
if mode == "Manual Entry":
    st.subheader("🧾 Manual Entry for a Single Individual")

    with st.form(key="manual_form"):
        inputs = {}
        for col in X_COLUMNS:
            question = QUESTIONS_MAPPING.get(col, col)

            if "_Score" in col:
                inputs[col] = st.selectbox(question, ["No", "Yes"], index=0)
            elif col.lower() in DROPDOWN_OPTIONS:
                inputs[col] = st.selectbox(question, DROPDOWN_OPTIONS[col.lower()])
            elif "age" in col.lower():
                inputs[col] = st.number_input("Enter Age (in years)", min_value=1, max_value=120, value=25)
            else:
                inputs[col] = st.text_input(question, value="0")

        submit = st.form_submit_button("Predict")

    if submit:
        # Convert inputs to DataFrame
        row = {}
        for k, v in inputs.items():
            if v == "Yes":
                row[k] = 1
            elif v == "No":
                row[k] = 0
            else:
                try:
                    row[k] = float(v)
                except Exception:
                    row[k] = v

        df_input = pd.DataFrame([row])
        preds = predict_df(df_input)
        result = int(preds[0])

        st.success(f"🧩 Prediction Result: {result} — {'ASD Detected' if result == 1 else 'No ASD Detected'}")
        st.write("🔍 Input Processed for Prediction:")
        st.dataframe(df_input.reindex(columns=X_COLUMNS, fill_value=0).head())

# --- CSV Upload Mode ---
else:
    st.subheader("📂 Upload a CSV File for Batch Predictions")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None

        if df is not None:
            st.write("📊 Uploaded Data Preview:")
            st.dataframe(df.head())

            if st.button("Run Predictions"):
                preds = predict_df(df)
                output_df = df.copy()
                output_df["Predicted_ASD"] = preds.astype(int)

                st.success("✅ Predictions Completed")
                st.dataframe(output_df.head())

                csv = output_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Predictions CSV", csv, file_name="predictions.csv", mime="text/csv")
