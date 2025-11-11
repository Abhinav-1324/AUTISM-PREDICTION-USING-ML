# src/utils.py
import pandas as pd
from pathlib import Path

# path to dataset
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"

def load_data(path=DATA_PATH):
    """Load the autism dataset from CSV"""
    df = pd.read_csv(path)
    return df

# quick test
if __name__ == "__main__":
    df = load_data()
    print("✅ Dataset loaded successfully! Shape:", df.shape)
    print(df.head())
