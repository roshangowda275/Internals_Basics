"""FastAPI service for CloudPulse response time estimates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
STEP2_PATH = ROOT / "results" / "step2_s4.json"

app = FastAPI()
model = joblib.load(MODEL_PATH)


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
    arr = np.array(
        [
            [
                data.request_size_kb,
                data.server_load,
                data.is_cached,
                data.region_latency,
            ]
        ]
    )
    pred = model.predict(arr)[0]
    return {"prediction": float(pred)}


def write_step2_json() -> None:
    client = TestClient(app)
    health = client.get("/health").json()
    test_input = {
        "request_size_kb": 242.0,
        "server_load": 0.5,
        "is_cached": 1,
        "region_latency": 127,
    }
    pred_body = client.post("/estimate", json=test_input).json()
    out = {
        "health_endpoint": "/health",
        "predict_endpoint": "/estimate",
        "port": 8080,
        "health_response": health,
        "test_input": test_input,
        "prediction": float(pred_body["prediction"]),
    }
    STEP2_PATH.parent.mkdir(parents=True, exist_ok=True)
    STEP2_PATH.write_text(json.dumps(out, indent=4))
    print("Wrote", STEP2_PATH)


if __name__ == "__main__":
    if "--write-step2" in sys.argv:
        write_step2_json()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8080)
