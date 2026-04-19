import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_pipeline(model):

    num_features = [
        'age','experience','training_hours','projects',
        'attendance','feedback_score','salary','manager_score'
    ]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features)
    ])

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return full_pipeline