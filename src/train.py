import gc
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from process_data import load_and_merge_dataset

def prepare_data_and_train():
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[+] 1. Loading and merging dataset from data directory...")
    df = load_and_merge_dataset(DATA_DIR)

    feature_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["label", "attack_type"]
    ]

    X = df[feature_cols]
    y = df["label"]

    TOTAL_ROWS = len(df)
    print(f"\n[+] Total rows available: {TOTAL_ROWS:,}")

    print("\n[+] 2. Performing Stratified Split (160,000 Train / 40,000 Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=160_000,
        test_size=40_000,
        stratify=y,
        random_state=42
    )

    del df, X, y
    gc.collect()

    print(f"  |-- Train set shape: {X_train.shape}")
    print(f"  |-- Test set shape : {X_test.shape}")

    print("\n[+] 3. Training Random Forest Classifier with balanced weights...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        verbose=1
    )
    rf_model.fit(X_train, y_train)
    print("[+] Model training finished successfully.")

    model_artifact = {
        "model": rf_model,
        "features": feature_cols,
        "classes": rf_model.classes_.tolist()
    }

    model_path = MODELS_DIR / "netids_model.pkl"
    joblib.dump(model_artifact, model_path, compress=3)
    print(f"[+] Saved model to: {model_path}")

    test_data_path = MODELS_DIR / "test_data.pkl"
    joblib.dump((X_test, y_test), test_data_path, compress=3)
    print(f"[+] Saved test data to: {test_data_path}")

    print("\n" + "=" * 60)
    print("  [SUCCESS] Training completed and artifacts saved.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    prepare_data_and_train()