# Practical 07 — Serving a Model as a Web Service with FastAPI

**Course:** SCSE3040 MLOps (ODD Semester 2026-27)

## What this practical does
Trains a Linear Regression model on a synthetic food-delivery dataset and
saves it with `joblib`, then wraps it in a FastAPI service (`work/app.py`)
with a `GET /health` check and a `POST /predict` endpoint. Pydantic `Field`
constraints reject out-of-range input (e.g. `traffic_level=9`) with a
`422` response before the model ever runs. The service is exercised two
ways: in-notebook with FastAPI's `TestClient` (no server needed), and as a
real `uvicorn` server with the interactive `/docs` page.

## Task — Iris Flower Classification Service
Same pattern applied to a classification problem:

- `LogisticRegression(max_iter=1000)` trained on `sklearn.datasets.load_iris`,
  saved to `task/iris_model.joblib` (plus the class names).
- `task/iris_app.py` — `GET /health` and `POST /predict`, accepting
  `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (each
  `Field(gt=0, le=10)` cm) and returning the predicted species **name**
  (e.g. `"setosa"`).
- Verified with `TestClient`: a valid sample predicts correctly, and a
  negative measurement is rejected with status `422`.
- Run with `uvicorn` on port 8001; `/docs` screenshot shows a successful
  `POST /predict` call returning `{"species": "setosa"}`.

## Addendum — Registry and Native Serving (P07 submission)
The course's actual P07 assignment asks for a different serving path than
the FastAPI notebook above: promoting a registered model to the
**Production** stage and serving it directly from the **MLflow Model
Registry** with MLflow's own built-in server, rather than a custom API.

`promote_and_serve.py` promotes `cancer-classifier` (registered in
Practical 06, in `mlops-lab6/mlflow.db`) to Production, then it's served
with `mlflow models serve -m "models:/cancer-classifier/1" --port 5005
--env-manager local` and tested with `curl` against `/invocations` using
`sample_request.json` — a real held-out test sample. The served model
correctly predicted its true label (benign).

## Files
- `lab7.ipynb` — notebook, run top to bottom with outputs.
- `work/app.py`, `work/model.joblib` — the delivery-time service (Steps 1-5).
- `task/iris_app.py`, `task/iris_model.joblib`, `task/iris_target_names.joblib`
  — the Iris classification service.
- `promote_and_serve.py`, `sample_request.json` — the registry/native-serving
  addendum (see above).

## How to run
```
pip install fastapi uvicorn httpx scikit-learn pandas numpy joblib mlflow
jupyter lab lab7.ipynb                    # run all cells (trains + saves both models)
python -m uvicorn app:app --port 8000     # from work/   — delivery service
python -m uvicorn iris_app:app --port 8001  # from task/ — iris service
```
Then open `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8001/docs`.

For the registry/native-serving addendum, from `mlops-lab6/` (where
`cancer-classifier` is registered):
```
python3 ../mlops-lab7/promote_and_serve.py
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow models serve -m "models:/cancer-classifier/1" --port 5005 --env-manager local
curl -X POST http://127.0.0.1:5005/invocations -H "Content-Type: application/json" -d @../mlops-lab7/sample_request.json
```

## AI Assistance Disclosure
Claude AI used for experiment setup and helping with execution where I struggled.
