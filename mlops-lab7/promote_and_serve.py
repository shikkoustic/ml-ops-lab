"""
Practical 07 Task (addendum) - Registry and Native Serving.

Promotes the cancer-classifier model (registered in Practical 06) to the
Production stage, then serves it directly from the MLflow Model Registry
with MLflow's built-in server - no custom FastAPI app involved.

Run from mlops-lab6/ (needs its mlflow.db, where cancer-classifier was
registered):

    python3 ../mlops-lab7/promote_and_serve.py

Then, in another terminal, from the same folder:

    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    mlflow models serve -m "models:/cancer-classifier/1" --port 5005 \
        --env-manager local --host 127.0.0.1

Test it:

    curl -X POST http://127.0.0.1:5005/invocations \
        -H "Content-Type: application/json" \
        -d @sample_request.json
"""
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

client.transition_model_version_stage(
    name="cancer-classifier",
    version=1,
    stage="Production",
    archive_existing_versions=False,
)

version = client.get_model_version("cancer-classifier", "1")
print(f"{version.name} v{version.version} -> stage: {version.current_stage}")
print(f"Source run: {version.run_id}")
