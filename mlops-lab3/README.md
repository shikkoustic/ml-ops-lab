# Practical 03 — Model Comparison & Cross-Validation

**Course:** SCSE3040 MLOps (ODD Semester 2026-27)

## What this practical does
Trains and compares several regression models (LinearRegression, DecisionTree,
RandomForest) on the delivery-time dataset, measuring train vs test MAE to see
overfitting (the train/test gap). Uses 5-fold cross-validation for a more
reliable error estimate and sweeps decision-tree depths to find the best one.

The To-Do extends this to a **classification** problem (Breast Cancer dataset):
LogisticRegression, DecisionTree, and RandomForest are compared by train/test
accuracy and cross-validation, and the best model is reported with a confusion
matrix and classification metrics.

## Files
- `lab3.ipynb` — the notebook, run top to bottom with outputs.
- `data/` — generated delivery dataset (seed 42).

## AI Assistance Disclosure
AI assistance (Claude) was used for environment setup and formatting.
