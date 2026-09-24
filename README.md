# 🛡️ AI-Powered Network Intrusion Detection System (NIDS)

An end-to-end Machine Learning pipeline designed for network intrusion detection and cyber threat classification, featuring modular MLOps architecture and an interactive Streamlit dashboard.

---

## 📌 Executive Summary
With the rapid increase in sophisticated cyber attacks, signature-based intrusion detection systems fall short against zero-day and anomalous network activities. This project implements a production-grade machine learning system to analyze network packet flows, classify malicious intrusions, and provide explainable AI insights (Feature Importance & Confusion Matrix) for SOC (Security Operations Center) analysts.

---

## 🏗️ System Architecture & Engineering Pipeline
The project is decoupled into clean, modular Python components following production software engineering best practices:

- **Data Processing (`src/process_data.py`)**: Automated cleaning, missing value imputation, categorical encoding, and feature scaling.
- **Model Training (`src/train.py`)**: Supervised model training with hyperparameter optimization.
- **Evaluation & Diagnostics (`src/evaluate.py`)**: Extraction of classification metrics (Precision, Recall, F1-Score) and export of decision diagnostic plots.
- **Inference Service (`src/predict.py`)**: Standalone inference engine designed for batch and real-time packet scoring.
- **Security Dashboard (`src/app.py`)**: Interactive UI developed with **Streamlit** for real-time visualization of model outputs and threat analytics.

---

## 📂 Repository Structure

```text
├── models/                     # Evaluation artifacts and diagnostic plots
│   ├── confusion_matrix.png
│   ├── feature_importances_top20.png
│   └── metrics_evaluation.txt
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── app.py                  # Streamlit Web Dashboard
│   ├── evaluate.py             # Evaluation & report generation
│   ├── predict.py              # Inference pipeline
│   ├── process_data.py         # ETL & feature preprocessing
│   └── train.py                # Model training script
├── .gitignore                  # Production Git ignore rules
├── requirements.txt            # Reproducible dependencies
├── verify_env.py               # Pre-flight environment check
└── README.md                   # System documentationgit add README.md


## 🚀 Quickstart & Reproduction

```bash
git clone [https://github.com/supernatural963/network-intrusion-detection-ml.git](https://github.com/supernatural963/network-intrusion-detection-ml.git)
cd network-intrusion-detection-ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py

📊 Key Results & Diagnostics
Comprehensive Evaluation: Metrics and classification reports are exported to models/metrics_evaluation.txt.

Explainability: Top 20 network features influencing model decisions are visualized in models/feature_importances_top20.png.

Classification Performance: Detailed multi-class / binary misclassification analysis available via models/confusion_matrix.png.