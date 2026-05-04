"""Train Lasso and GradientBoosting with MLflow; pick best by MAE; save model for API."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "training_data.csv"
RESULTS_PATH = ROOT / "results" / "step1_s1.json"
MODELS_DIR = ROOT / "models"
EXPERIMENT_NAME = "cloudpulse-response-time-ms"


def _tracking_uri() -> str:
    (ROOT / "mlruns").mkdir(parents=True, exist_ok=True)
    return (ROOT / "mlruns").resolve().as_uri()


def _metrics(y_true, y_pred) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def _log_sklearn_params(model) -> None:
    for key, value in model.get_params().items():
        mlflow.log_param(key, str(value))


def main() -> None:
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=["response_time_ms"])
    y = data["response_time_ms"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []

    configs = [
        ("Lasso", Lasso(random_state=42)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ]

    for name, estimator in configs:
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("domain", "cloud_saas")
            mlflow.set_tag("model_name", name)
            mlflow.log_param("model_type", name)
            _log_sklearn_params(estimator)

            estimator.fit(X_train, y_train)
            preds = estimator.predict(X_test)
            m = _metrics(y_test, preds)
            for k, v in m.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(estimator, name="model")
            summaries.append({"name": name, **m})

    best = min(summaries, key=lambda x: x["mae"])
    best_name = best["name"]

    best_est = (
        Lasso(random_state=42)
        if best_name == "Lasso"
        else GradientBoostingRegressor(random_state=42)
    )
    best_est.fit(X_train, y_train)
    joblib.dump(best_est, MODELS_DIR / "best_model.pkl")

    out = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [
            {
                "name": s["name"],
                "mae": s["mae"],
                "rmse": s["rmse"],
                "r2": s["r2"],
                "mape": s["mape"],
            }
            for s in summaries
        ],
        "best_model": best_name,
        "best_metric_name": "mae",
        "best_metric_value": best["mae"],
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=4))
    print("Training completed; wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
