import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from xgboost import XGBClassifier
from src.pipeline import create_pipeline

# ---------------- DATA ----------------
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'age': np.random.randint(22, 60, n),
    'experience': np.random.randint(1, 20, n),
    'training_hours': np.random.randint(10, 100, n),
    'projects': np.random.randint(1, 15, n),
    'attendance': np.random.uniform(0.6, 1.0, n),
    'feedback_score': np.random.uniform(1, 5, n),
    'salary': np.random.randint(20000, 150000, n),
    'manager_score': np.random.uniform(1, 5, n)
})

data['performance'] = np.where(
    (data['feedback_score'] > 4) & (data['attendance'] > 0.9),
    2,
    np.where(data['feedback_score'] > 3, 1, 0)
)

X = data.drop('performance', axis=1)
y = data['performance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
xgb = XGBClassifier(eval_metric='mlogloss')

rf_pipe = create_pipeline(rf)
xgb_pipe = create_pipeline(xgb)

mlflow.set_experiment("Model Comparison")

results = {}

for name, model in {"RandomForest": rf_pipe, "XGBoost": xgb_pipe}.items():

    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"{name} Accuracy:", acc)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, name)

# ---------------- SAVE BEST MODEL ----------------
best_model_name = max(results, key=results.get)
best_model = {"RandomForest": rf_pipe, "XGBoost": xgb_pipe}[best_model_name]

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")

print(f"✅ Best Model: {best_model_name}")