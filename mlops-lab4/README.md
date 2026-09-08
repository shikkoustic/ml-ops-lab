# Practical 04 — Refactoring into a Python Package

**Course:** SCSE3040 MLOps (ODD Semester 2026-27)

## What this practical does
Refactors the scattered notebook code into a proper Python package
(`delivery/`) with separate modules — `data.py`, `features.py`, `model.py`,
`validate.py`, tied together by `__init__.py`. The package is imported like a
library to train and save a model, and the pipeline is run from the command
line via standalone scripts (`train.py`, `predict.py`).

The To-Do repeats the pattern for the **classification** workflow as a second
package (`cancer/`):
- `model.py` uses cross-validation to compare LogisticRegression,
  DecisionTree, and RandomForest and auto-saves the best model.
- `validate.py` rejects an order unless at least 3 measurements are realistic.
- `predict_cancer.py` prints the prediction (malignant/benign) and the model's
  confidence via `predict_proba`.
- `metrics.py` prints a confusion matrix and classification report for the
  saved model.

## Files
- `lab4.ipynb` — notebook, run top to bottom with outputs.
- `delivery/` — regression package (data, features, model, validate, __init__).
- `cancer/` — classification package (data, features, model, validate, metrics, __init__).
- `train.py`, `predict.py` — delivery command-line scripts.
- `train_cancer.py`, `predict_cancer.py` — classification command-line scripts.

## AI Assistance Disclosure
I used an AI assistant (Claude) for project setup and for verifying the
submission format. I have reviewed, understand, and can explain and modify all
code in this submission.
