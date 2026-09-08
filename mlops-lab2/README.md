# Practical 02 — First Model

**Course:** SCSE3040 MLOps (ODD Semester 2026-27)
**Submission:** Practical 02

## Contents
- `lab2.ipynb` — notebook run top to bottom with outputs, ending in the
  results/self-check table.
- `data/delivery_times.csv` — generated dataset (seed 42).

## What this practical does
Splits the delivery dataset into train/test, trains a LinearRegression model,
and measures error (MAE, RMSE) on the held-out test set, comparing it against
mean- and median-guess baselines.

## Results (test set)
| approach            | test MAE |
|---------------------|----------|
| mean baseline       | 10.32    |
| median baseline     | 10.33    |
| linear regression   | 1.92     |

## Tasks
- **T1** — median baseline (MAE 10.33) vs mean baseline (MAE 10.32); mean is
  marginally better, but both are far worse than the model.
- **T2** — 70/30 split, seed 7: model test MAE moved 1.92 -> 2.10; conclusion
  unchanged (model still beats baseline). A single split is a noisy estimate.
- **T3** — `predict_delivery(...)` returns a single rounded prediction for one order.

## Environment
- Python 3.12 (conda env `mlops`)
- numpy, pandas, scikit-learn, matplotlib

## AI Assistance Disclosure
AI assistance (Claude) was used for environment setup and formatting.
