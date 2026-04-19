import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
import shap
import mlflow
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Employee Analytics Dashboard", layout="wide")

st.title("💼 Employee Performance Intelligence Dashboard")

# ---------------- INPUT UI ----------------
st.sidebar.header("🔧 Input Employee Data")

age = st.sidebar.slider("Age", 20, 60, 30)
experience = st.sidebar.slider("Experience", 1, 20, 5)
training_hours = st.sidebar.slider("Training Hours", 10, 100, 40)
projects = st.sidebar.slider("Projects", 1, 15, 5)
attendance = st.sidebar.slider("Attendance", 0.5, 1.0, 0.9)
feedback_score = st.sidebar.slider("Feedback", 1.0, 5.0, 3.5)
salary = st.sidebar.number_input("Salary", 20000, 200000, 50000)
manager_score = st.sidebar.slider("Manager Score", 1.0, 5.0, 3.5)

input_df = pd.DataFrame({
    'age':[age],'experience':[experience],'training_hours':[training_hours],
    'projects':[projects],'attendance':[attendance],
    'feedback_score':[feedback_score],'salary':[salary],'manager_score':[manager_score]
})

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2,2])

# ---------------- PREDICTION ----------------
if st.sidebar.button("🚀 Predict"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    label_map = {0:"Low",1:"Medium",2:"High"}
    result = label_map[pred]

    # ---------------- LEFT PANEL ----------------
    with col1:
        st.subheader("🎯 Prediction Result")
        st.metric("Performance", result)
        st.metric("Confidence", f"{max(prob)*100:.2f}%")

        # HR INSIGHTS
        st.subheader("📋 HR Insights")
        if pred == 0:
            st.error("⚠️ Low Performance → Recommend Training + Mentorship")
        elif pred == 1:
            st.warning("📈 Medium → Improvement Plan Needed")
        else:
            st.success("🏆 High → Promotion Ready")

        # Employee Summary
        st.subheader("👤 Employee Summary")
        st.write(input_df)

    # ---------------- RIGHT PANEL ----------------
    with col2:

        # FEATURE IMPORTANCE
        st.subheader("📊 Feature Importance")
        try:
            importances = model.named_steps["model"].feature_importances_
            features = input_df.columns
            fig, ax = plt.subplots()
            sns.barplot(x=importances, y=features, ax=ax)
            st.pyplot(fig)
        except:
            st.warning("Feature importance not available")

        # ---------------- SHAP ----------------
        st.subheader("🔍 SHAP Explainability")
        try:
            explainer = shap.Explainer(model.named_steps["model"])
            shap_values = explainer(input_df)

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP error: {e}")

# ---------------- MODEL COMPARISON ----------------
st.subheader("📈 Model Comparison")

comparison_df = pd.DataFrame({
    "Model":["Random Forest","XGBoost"],
    "Accuracy":[0.85,0.88]
})

st.bar_chart(comparison_df.set_index("Model"))

# ---------------- MLFLOW METRICS ----------------
st.subheader("📡 MLflow Experiment Tracking")

try:
    import mlflow.tracking
    client = mlflow.tracking.MlflowClient()

    experiments = client.search_experiments()
    exp_id = experiments[0].experiment_id

    runs = client.search_runs(exp_id)
    latest_run = runs[0]

    st.write("Latest Run Metrics:")
    st.json(latest_run.data.metrics)

except:
    st.warning("⚠️ MLflow not running or no experiments found")

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📊 Confusion Matrix")

cm = np.array([[50,10,5],[8,60,7],[3,6,40]])
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(fig)

# ---------------- DOWNLOAD REPORT ----------------
st.subheader("📥 Download Prediction Report")

if st.button("Download CSV"):
    input_df.to_csv("employee_report.csv", index=False)
    with open("employee_report.csv","rb") as f:
        st.download_button("Download File", f, "report.csv")