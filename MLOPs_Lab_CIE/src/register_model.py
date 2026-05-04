"""Register best MAE run from Task 1 in MLflow Model Registry."""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "step3_s6.json"
EXPERIMENT_NAME = "cloudpulse-response-time-ms"
REGISTERED_NAME = "cloudpulse-response-time-ms-predictor"


def _tracking_uri() -> str:
    return (ROOT / "mlruns").resolve().as_uri()


def main() -> None:
    mlflow.set_tracking_uri(_tracking_uri())
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment {EXPERIMENT_NAME!r} not found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.mae ASC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No runs found.")
    run = runs[0]
    mae = float(run.data.metrics["mae"])
    model_uri = f"runs:/{run.info.run_id}/model"

    registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_NAME)
    latest_ver = int(registered.version)

    out = {
        "registered_model_name": REGISTERED_NAME,
        "version": latest_ver,
        "run_id": run.info.run_id,
        "source_metric": "mae",
        "source_metric_value": mae,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=4))
    print("Registered model; wrote", RESULTS_PATH)


if __name__ == "__main__":
    main()
