import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

def evaluate_system():
    model_path = MODELS_DIR / "netids_model.pkl"
    test_data_path = MODELS_DIR / "test_data.pkl"

    if not model_path.exists() or not test_data_path.exists():
        raise FileNotFoundError("فایل مدل یا داده‌های تست در پوشه models یافت نشد!")

    print("\n[+] 1. Loading model and test dataset...")
    bundle = joblib.load(model_path)
    rf_model = bundle["model"]
    feature_names = bundle["features"]

    X_test, y_test = joblib.load(test_data_path)
    print(f"  |-- Test samples count: {len(X_test):,}")

    print("\n[+] 2. Performing inference on test set...")
    y_pred = rf_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

    print("\n" + "=" * 60)
    print("           NIDS PERFORMANCE EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy : {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(clf_report)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60)

    # ۱. ذخیره گزارش متنی ارزیابی با استاندارد UTF-8
    metrics_file = MODELS_DIR / "metrics_evaluation.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("=== Network AI Project: Performance Evaluation Report ===\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(clf_report + "\n\n")
        f.write("Confusion Matrix (Labels: " + ", ".join(rf_model.classes_) + "):\n")
        f.write(str(cm) + "\n")
    print(f"\n[+] Saved metrics text file to: {metrics_file.name}")

    # ۲. رسم و ذخیره ماتریس آشفتگی استاندارد
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
    disp.plot(cmap=plt.cm.Blues, values_format="d", ax=ax, colorbar=False)
    plt.title("NIDS - Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_img_path = MODELS_DIR / "confusion_matrix.png"
    plt.savefig(cm_img_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix plot to: {cm_img_path.name}")

    # ۳. استخراج و رسم ۲۰ ویژگی حیاتی ترافیک شبکه
    try:
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

        fi_csv_path = MODELS_DIR / "feature_importances.csv"
        fi_df.to_csv(fi_csv_path, index=False, encoding="utf-8")
        print(f"[+] Saved feature importances CSV to: {fi_csv_path.name}")

        top20 = fi_df.head(20)
        plt.figure(figsize=(10, 8))
        plt.barh(top20["Feature"][::-1], top20["Importance"][::-1], color="#1f77b4")
        plt.xlabel("Importance Score", fontsize=11)
        plt.title("Top-20 Network Traffic Features", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fi_img_path = MODELS_DIR / "feature_importances_top20.png"
        plt.savefig(fi_img_path, dpi=300)
        plt.close()
        print(f"[+] Saved top-20 features plot to: {fi_img_path.name}")
    except Exception as e:
        print(f"[-] Feature importance extraction error: {e}")

    print("\n" + "=" * 60)
    print("  [SUCCESS] Evaluation and visualization artifacts generated.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    evaluate_system()