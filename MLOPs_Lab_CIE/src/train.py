import pandas as pd
import mlflow
import mlflow.sklearn
import math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("../data/training_data.csv")

# Correct target column
X = data.drop(columns=["response_time_ms"])
y = data["response_time_ms"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow
mlflow.set_experiment("cloudpulse-response-time-ms")

with mlflow.start_run():
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = math.sqrt(mse)

    mlflow.log_param("model", "GradientBoosting")
    mlflow.log_metric("rmse", rmse)

    # IMPORTANT FIX
    mlflow.sklearn.log_model(model, "model")

print("Training completed successfully")