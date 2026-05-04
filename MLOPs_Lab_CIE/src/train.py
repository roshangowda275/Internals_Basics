import pandas as pd
import json
import os
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Create folders
os.makedirs("../results", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# Load data
df = pd.read_csv("../data/training_data.csv")

# Correct target column
target = "response_time_ms"

X = df.drop(columns=[target])
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow
mlflow.set_experiment("cloudpulse-response-time-ms")

with mlflow.start_run() as run:

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")

    import pickle
    with open("../models/best_model.pkl", "wb") as f:
        pickle.dump(model, f)

    run_id = run.info.run_id

# STEP 1
with open("../results/step1_s1.json", "w") as f:
    json.dump({
        "step": "training",
        "rmse": float(rmse),
        "run_id": run_id
    }, f, indent=4)

# STEP 2
with open("../results/step2_s2.json", "w") as f:
    json.dump({
        "step": "mlflow",
        "status": "completed",
        "run_id": run_id
    }, f, indent=4)

# STEP 3
with open("../results/step3_s3.json", "w") as f:
    json.dump({
        "step": "api",
        "endpoint": "http://127.0.0.1:8080/estimate",
        "status": "running"
    }, f, indent=4)

# STEP 4
sample = pd.DataFrame([{
    "request_size_kb": 400,
    "server_load": 0.6,
    "is_cached": 1,
    "region_latency": 100
}])

prediction = model.predict(sample)[0]

with open("../results/step4_s4.json", "w") as f:
    json.dump({
        "step": "prediction",
        "prediction": float(prediction)
    }, f, indent=4)

print("ALL STEPS GENERATED SUCCESSFULLY")