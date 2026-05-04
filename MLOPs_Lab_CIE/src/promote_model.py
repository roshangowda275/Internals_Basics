"""Champion alias, challenger (random_state=99), register v2, compare MAE, promote or keep."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "training_data.csv"
STEP1_PATH = ROOT / "results" / "step1_s1.json"
STEP4_PATH = ROOT / "results" / "step4_s7.json"
EXPERIMENT_NAME = "cloudpulse-response-time-ms"
REGISTERED_NAME = "cloudpulse-response-time-ms-predictor"


def _tracking_uri() -> str:
    return (ROOT / "mlruns").resolve().as_uri()


def _build_estimator(model_name: str, random_state: int):
    if model_name == "Lasso":
        return Lasso(random_state=random_state)
    if model_name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=random_state)
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient()

    step1 = json.loads(STEP1_PATH.read_text())
    best_name = step1["best_model"]

    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=["response_time_ms"])
    y = data["response_time_ms"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mv1_list = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    if not mv1_list:
        raise SystemExit("No registered model versions. Run register_model.py first.")
    v1_num = min(int(v.version) for v in mv1_list)

    client.set_registered_model_alias(REGISTERED_NAME, "champion", v1_num)

    mv1 = client.get_model_version(REGISTERED_NAME, str(v1_num))
    run1 = client.get_run(mv1.run_id)
    mae_v1 = float(run1.data.metrics["mae"])

    challenger = _build_estimator(best_name, random_state=99)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{best_name}_challenger_rs99") as run:
        mlflow.set_tag("domain", "cloud_saas")
        mlflow.set_tag("model_name", best_name)
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("random_state", "99")
        challenger.fit(X_train, y_train)
        preds = challenger.predict(X_test)
        mae_v2 = float(mean_absolute_error(y_test, preds))
        mlflow.log_metric("mae", mae_v2)
        mlflow.sklearn.log_model(challenger, name="model")
        run_id = run.info.run_id

    mv2 = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model", name=REGISTERED_NAME
    )
    v2_num = int(mv2.version)

    if mae_v2 < mae_v1:
        client.set_registered_model_alias(REGISTERED_NAME, "champion", v2_num)
        action = "promoted"
        champion_ver = v2_num
    else:
        client.set_registered_model_alias(REGISTERED_NAME, "champion", v1_num)
        action = "kept"
        champion_ver = v1_num

    out = {
        "registered_model_name": REGISTERED_NAME,
        "alias_name": "champion",
        "champion_version": champion_ver,
        "challenger_version": v2_num,
        "action": action,
    }
    STEP4_PATH.parent.mkdir(parents=True, exist_ok=True)
    STEP4_PATH.write_text(json.dumps(out, indent=4))
    print("Promotion workflow done; wrote", STEP4_PATH)


if __name__ == "__main__":
    main()
