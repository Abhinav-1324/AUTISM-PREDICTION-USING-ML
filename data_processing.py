# src/data_processing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset:
    - Replace yes/no with 1/0
    - Replace '?' and 'others' with 'Others'
    """
    df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})
    return df

def encode_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical (object type) columns into numbers
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def feature_target_split(df: pd.DataFrame):
    """
    Split dataset into features (X) and target (y).
    - Remove ID, age_desc, used_app_before, and austim columns.
    - Target is Class/ASD.
    """
    removal = ['ID', 'age_desc', 'used_app_before', 'austim']
    X = df.drop(removal + ['Class/ASD'], axis=1)
    y = df['Class/ASD']
    return X, y

# quick test
if __name__ == "__main__":
    from utils import load_data

    df = load_data()
    print("Original shape:", df.shape)

    df = basic_cleaning(df)
    df = encode_object_cols(df)

    X, y = feature_target_split(df)

    print("✅ Cleaned dataset shape:", X.shape, "Target shape:", y.shape)
    print(X.head())
    print(y.head())
