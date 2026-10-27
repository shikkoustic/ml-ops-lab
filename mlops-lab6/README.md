# Practical 06 — Experiment Tracking and Model Registry with MLflow

**Course:** SCSE3040 MLOps (ODD Semester 2026-27)

## What this practical does
Connects MLflow to a local SQLite store (`sqlite:///mlflow.db`), tracks every
training run as it happens (parameters, metrics, tags, the saved model as an
artifact), and shows how to pull the best run back out of the database with
`mlflow.search_runs()` instead of reading printed output.

Steps 1–12 follow the given delivery-time regression example: training
without tracking, connecting MLflow, logging single runs, logging a
`max_depth` sweep (6 runs), querying for the best run by MAE, registering the
winning model as `delivery-time-model`, loading it back by registry URI, and
browsing everything in the MLflow UI.

## Task 13 — Breast Cancer Diagnosis
Same workflow applied to `sklearn.datasets.load_breast_cancer`, tracked in a
separate experiment, `cancer-diagnosis`:

- `logreg-baseline` — a single Logistic Regression run.
- A 3×3 Random Forest sweep over `n_estimators` (10, 50, 100) and `max_depth`
  (3, 5, 10) — 9 runs, all tagged `sweep = rf`.
- Every run logs its parameters, **accuracy**, and **F1-score**, and saves the
  trained model as an artifact.
- The best run is found by F1-score via `mlflow.search_runs()` (not by
  reading earlier output), registered as `cancer-classifier`, loaded back
  with `models:/cancer-classifier/1`, and used to predict the first five test
  samples — predictions matched the true labels.

Result: best run was a Random Forest (`n_estimators=100`, `max_depth=10`),
F1-score ≈ 0.972, beating the Logistic Regression baseline (F1 ≈ 0.966).

## Files
- `lab6.ipynb` — notebook, run top to bottom with outputs.
- `mlflow.db` — SQLite tracking store (all runs + the model registry).
- `mlruns/` — MLflow's artifact store (saved model files per run).

## How to run
```
pip install mlflow scikit-learn pandas numpy
jupyter lab lab6.ipynb      # run all cells
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Then open http://127.0.0.1:5001 to browse the experiments and the registry.

## AI Assistance Disclosure
Claude AI used for experiment setup and helping with execution where I struggled.
