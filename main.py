# ===============================
# TRAINING SCRIPT (FIXED)
# ===============================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ===============================
# CREATE DATA
# ===============================
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

# ===============================
# TARGET VARIABLE
# ===============================
data['performance'] = np.where(
    (data['feedback_score'] > 4) & (data['attendance'] > 0.9),
    2,  # High
    np.where(
        (data['feedback_score'] > 3),
        1,  # Medium
        0   # Low
    )
)

print("✅ Data created")

# ===============================
# SPLIT DATA (STRATIFIED)
# ===============================
X = data.drop('performance', axis=1)
y = data['performance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# MODEL (WITH IMBALANCE FIX)
# ===============================
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained")

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("✅ Model saved at models/model.pkl")