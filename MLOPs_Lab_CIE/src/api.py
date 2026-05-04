from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("../models/best_model.pkl")

class InputData(BaseModel):
    request_size_kb: float = Field(..., ge=1, le=500)
    server_load: float = Field(..., ge=0.1, le=1.0)
    is_cached: int = Field(..., ge=0, le=1)
    region_latency: float = Field(..., ge=10, le=200)

@app.get("/health")
def health():
    return {"status": "operational", "service": "CloudPulse API"}

@app.post("/estimate")
def predict(data: InputData):
    arr = np.array([[data.request_size_kb, data.server_load, data.is_cached, data.region_latency]])
    pred = model.predict(arr)[0]
    return {"prediction": float(pred)}