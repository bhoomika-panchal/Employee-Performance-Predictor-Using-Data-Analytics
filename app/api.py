import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}