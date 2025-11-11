# src/train_model.py
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler

from utils import load_data
from data_processing import basic_cleaning, encode_object_cols, feature_target_split

def main():
    # 1️⃣ Load dataset
    df = load_data()

    # 2️⃣ Clean & encode
    df = basic_cleaning(df)
    df = encode_object_cols(df)

    # 3️⃣ Split features and target
    X, y = feature_target_split(df)

    # 4️⃣ Save feature columns for inference
    ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts")
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    joblib.dump(list(X.columns), os.path.join(ARTIFACTS_PATH, "X_columns.joblib"))

    # 5️⃣ Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6️⃣ Balance training set
    ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    # 7️⃣ Scale features
    scaler = StandardScaler().fit(X_train_res)
    X_train_scaled = scaler.transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)

    # 8️⃣ Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "SVC": SVC(kernel='rbf', probability=True)
    }

    results = {}
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train_res)

        # Evaluate
        train_auc = metrics.roc_auc_score(y_train_res, model.predict(X_train_scaled))
        val_auc = metrics.roc_auc_score(y_val, model.predict(X_val_scaled))

        results[name] = {"model": model, "train_auc": train_auc, "val_auc": val_auc}
        print(f"{name}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}")

    # 9️⃣ Save the best model (highest validation AUC)
    best_name = max(results, key=lambda n: results[n]["val_auc"])
    best_model = results[best_name]["model"]

    # Save model & scaler
    joblib.dump(best_model, os.path.join(ARTIFACTS_PATH, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_PATH, "scaler.joblib"))
    print(f"✅ Best model saved: {best_name}")

if __name__ == "__main__":
    main()
