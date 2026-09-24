import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

class NetworkIntrusionPredictor:
    """
    موتور استنتاج مستقل تشخیص نفوذ شبکه (Inference Engine)
    قابلیت بارگذاری آرتیفکت مدل و پیش‌بینی برچسب و احتمال وقوع نفوذ
    """
    def __init__(self, model_filename="netids_model.pkl"):
        model_path = MODELS_DIR / model_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"فایل مدل در مسیر {model_path} یافت نشد! ابتدا مدل را آموزش دهید."
            )
        
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.classes = bundle["classes"]

    def predict_packet(self, packet_features: pd.DataFrame):
        """
        دریافت ویژگی‌های یک پکت شبکه و صدور حکم امنیتی همراه با زمان پاسخ
        """
        missing_cols = set(self.features) - set(packet_features.columns)
        if missing_cols:
            raise ValueError(f"ستون‌های ورودی ناقص هستند! موارد مفقود: {missing_cols}")

        packet_aligned = packet_features[self.features]

        start_time = time.perf_counter()
        prediction = self.model.predict(packet_aligned)[0]
        probabilities = self.model.predict_proba(packet_aligned)[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        confidence = float(np.max(probabilities) * 100)
        prob_dict = {str(cls_name): float(prob * 100) for cls_name, prob in zip(self.classes, probabilities)}

        return {
            "prediction": str(prediction),
            "confidence": confidence,
            "probabilities": prob_dict,
            "latency_ms": elapsed_ms
        }

def run_demonstration():
    print("\n" + "=" * 65)
    print("      NIDS INFERENCE ENGINE - LIVE PACKET EVALUATION DEMO")
    print("=" * 65)
    
    predictor = NetworkIntrusionPredictor()
    print(f"[+] مدل با موفقیت در حافظه بارگذاری شد. تعداد ویژگی‌ها: {len(predictor.features)}")

    test_data_path = MODELS_DIR / "test_data.pkl"
    if not test_data_path.exists():
        print("[-] داده‌های آزمون برای تست نمونه در دسترس نیست.")
        return

    X_test, y_test = joblib.load(test_data_path)

    benign_indices = np.where(y_test == "benign")[0]
    attack_indices = np.where(y_test == "attack")[0]

    test_cases = []
    if len(benign_indices) > 0:
        idx = benign_indices[0]
        test_cases.append(("نمونه بسته عادی (Normal Benign Traffic)", X_test.iloc[[idx]], y_test.iloc[idx]))
    
    if len(attack_indices) > 0:
        idx = attack_indices[0]
        test_cases.append(("نمونه بسته نفوذ/حمله (Malicious Attack Packet)", X_test.iloc[[idx]], y_test.iloc[idx]))

    print("\n[+] در حال اجرای آزمون استنتاج روی بسته‌های نمونه:")
    for title, packet_df, ground_truth in test_cases:
        result = predictor.predict_packet(packet_df)
        
        status_icon = "🚨 ATTACK DETECTED" if result["prediction"] == "attack" else "✅ BENIGN / SAFE"
        print("\n" + "-" * 65)
        print(f"  سناریو         : {title}")
        print(f"  برچسب واقعی    : {ground_truth}")
        print(f"  نتیجه تشخیص    : {result['prediction'].upper()} -> {status_icon}")
        print(f"  میزان اطمینان   : {result['confidence']:.2f}%")
        print(f"  توزیع احتمالات : {result['probabilities']}")
        print(f"  زمان پاسخ‌دهی  : {result['latency_ms']:.3f} میلی‌ثانیه")
    
    print("-" * 65)
    print("\n[+] تست پکت با بردار تمام‌صفر (Baseline Zero-Vector Test):")
    zero_df = pd.DataFrame([np.zeros(len(predictor.features))], columns=predictor.features)
    zero_res = predictor.predict_packet(zero_df)
    print(f"  نتیجه ورودی صفر: {zero_res['prediction']} (اطمینان: {zero_res['confidence']:.2f}%)")
    print("=" * 65)
    print("  [SUCCESS] ماژول استنتاج بدون نقص اجرا شد و آماده یکپارچه‌سازی است.\n")

if __name__ == "__main__":
    run_demonstration()