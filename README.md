# 🧠 Autism Prediction Using Machine Learning

This project is designed to support **early screening for Autism Spectrum Disorder (ASD)** using machine learning models.  
The system analyzes questionnaire-based behavioral data and demographic information to predict whether an individual is likely to show autistic traits.

It includes:
- A **trained ML model** (Logistic Regression selected based on performance)
- A **Streamlit web interface** for easy interaction
- **Manual input** mode and **CSV batch testing** mode

---

## ✅ Features

| Feature | Description |
|--------|-------------|
| **Machine Learning Model** | Predicts ASD likelihood based on questionnaire data |
| **Streamlit UI** | Easy-to-use web interface (no coding required) |
| **Manual Data Entry** | Fill the form and get instant prediction |
| **CSV Upload Support** | Predict ASD for multiple individuals at once |
| **Preprocessed & Encoded Input** | Ensures correct prediction pipeline |

---

## 🧑‍💻 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **Imbalanced-learn**
- **Streamlit**
- **Joblib**

---

## 📁 Project Structure

Autism Prediction/
├── src/
│ ├── data_processing.py
│ ├── train_model.py
│ ├── inference.py
│ └── utils.py
├── frontend/
│ └── app.py
├── artifacts/
│ ├── best_model.joblib
│ ├── scaler.joblib
│ └── X_columns.joblib
├── data/
│ └── sample_input.csv
├── requirements.txt
└── README.md
