import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="NIDS - AI Security Dashboard",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_nids_assets():
    model_path = MODELS_DIR / "netids_model.pkl"
    test_path = MODELS_DIR / "test_data.pkl"

    if not model_path.exists():
        st.error(f"مدل در مسیر {model_path} یافت نشد! ابتدا مدل را آموزش دهید.")
        st.stop()

    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    classes = bundle["classes"]

    test_data = None
    if test_path.exists():
        test_data = joblib.load(test_path)

    return model, features, classes, test_data

model, feature_names, class_names, test_data = load_nids_assets()

# --- هدر و مشخصات سیستم ---
st.title("🛡️ سامانه هوشمند پایش و تشخیص نفوذ شبکه (NIDS)")
st.markdown(
    "**پروژه نرم‌افزار دانشگاهی** | پیاده‌سازی مبتنی بر یادگیری ماشین با مجموعه داده **CICIDS2017**"
)
st.divider()

# --- ستون‌های آماری وضعیت مدل ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("وضعیت موتور استنتاج", "فعال (Online)", delta="Ready")
col_m2.metric("الگوریتم یادگیری", "Random Forest", delta="100 Trees")
col_m3.metric("تعداد ویژگی‌های جریان شبکه", len(feature_names))
col_m4.metric("کلاس‌های تشخیصی", " / ".join(class_names))

st.write("")

# --- پنل تست بسته‌ها ---
tabs = st.tabs(["🔍 شبیه‌سازی و تست زنده پکت‌ها", "📊 تحلیل ماتریس آشفتگی و ویژگی‌ها", "📁 متادیتای سیستم"])

with tabs[0]:
    st.subheader("تست و ارزیابی جریان ترافیک ورودی")
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("##### سناریوی ترافیک مورد آزمایش را انتخاب کنید:")
        scenario = st.radio(
            "نوع ترافیک:",
            [
                "نمونه ترافیک پاک و عادی (Benign Flow Sample)",
                "نمونه جریان حمله و نفوذ سایبری (Attack/Malicious Sample)",
                "جریان خنثی با مقادیر صفر (Baseline Zero-Vector)"
            ]
        )

        sample_packet = None
        ground_truth = "نامشخص"

        if test_data is not None:
            X_test, y_test = test_data
            if scenario.startswith("نمونه ترافیک پاک"):
                benign_idxs = np.where(y_test == "benign")[0]
                if len(benign_idxs) > 0:
                    idx = benign_idxs[0]
                    sample_packet = X_test.iloc[[idx]]
                    ground_truth = "benign"
            elif scenario.startswith("نمونه جریان حمله"):
                attack_idxs = np.where(y_test == "attack")[0]
                if len(attack_idxs) > 0:
                    idx = attack_idxs[0]
                    sample_packet = X_test.iloc[[idx]]
                    ground_truth = "attack"
            else:
                sample_packet = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
                ground_truth = "baseline"

        inspect_features = st.checkbox("مشاهده جزئیات فیچرهای بسته انتخابی")
        if inspect_features and sample_packet is not None:
            st.dataframe(sample_packet.T.rename(columns={sample_packet.index[0]: "مقدار عددی"}))

        run_btn = st.button("🚀 اجرای ارزیابی بلادرنگ بسته", type="primary", use_container_width=True)

    with col_result:
        st.markdown("##### حکم امنیتی صادره توسط هوش مصنوعی:")
        if run_btn and sample_packet is not None:
            aligned_packet = sample_packet[feature_names]
            
            t_start = time.perf_counter()
            prediction = model.predict(aligned_packet)[0]
            probs = model.predict_proba(aligned_packet)[0]
            latency_ms = (time.perf_counter() - t_start) * 1000

            confidence = np.max(probs) * 100

            if prediction == "attack":
                st.error("### 🚨 هشدار: حمله سایبری کشف شد (ATTACK DETECTED)")
            else:
                st.success("### ✅ ترافیک امن و عادی تایید شد (BENIGN / NORMAL)")

            st.write(f"**برچسب واقعی بسته در دیتاست:** `{ground_truth}`")
            st.write(f"**ضریب اطمینان مدل (Confidence):** `{confidence:.2f}%`")
            st.write(f"**تاخیر زمانی پردازش (Latency):** `{latency_ms:.3f} میلی‌ثانیه`")

            st.progress(int(confidence))

            st.markdown("###### تفکیک احتمالاتی:")
            prob_df = pd.DataFrame({
                "کلاس": class_names,
                "درصد احتمال": [p * 100 for p in probs]
            })
            st.bar_chart(prob_df.set_index("کلاس"))
        else:
            st.info("یکی از گزینه‌های سمت چپ را انتخاب کرده و روی دکمه اجرای ارزیابی کلیک کنید.")

with tabs[1]:
    st.subheader("شاخص‌های عملکردی و ساختار تصمیم‌گیری مدل")
    c_img1, c_img2 = st.columns(2)

    cm_file = MODELS_DIR / "confusion_matrix.png"
    fi_file = MODELS_DIR / "feature_importances_top20.png"

    with c_img1:
        if cm_file.exists():
            st.image(str(cm_file), caption="ماتریس درهم‌ریختگی (Confusion Matrix)", use_container_width=True)
        else:
            st.warning("تصویر ماتریس آشفتگی هنوز ایجاد نشده است.")

    with c_img2:
        if fi_file.exists():
            st.image(str(fi_file), caption="۲۰ ویژگی برتر پکت‌ها در تصمیم‌گیری مدل", use_container_width=True)
        else:
            st.warning("تصویر اهمیت ویژگی‌ها هنوز ایجاد نشده است.")

with tabs[2]:
    st.subheader("مستندات و فایل‌های سیستم")
    metrics_txt = MODELS_DIR / "metrics_evaluation.txt"
    if metrics_txt.exists():
        st.markdown("##### آخرین گزارش ارزیابی ذخیره‌شده:")
        with open(metrics_txt, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.info("فایل گزارش متنی یافت نشد.")