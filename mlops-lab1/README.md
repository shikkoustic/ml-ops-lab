# Practical 01 — MLOps Workbench

**Course:** SCSE3040 MLOps
**Submission:** Practical 01

## Contents
- `lab1.ipynb` — the notebook, run top to bottom with outputs.
- `work/my_requirements.txt` — pinned library versions (Task T2).
- `work/requirements.txt`, `work/run_info.json` — run record.
- `data/delivery_times.csv` — generated dataset (seed 42).
- Git repo in `work/` — commit record (`git log --oneline`).

## What this practical does
Identifies the exact Python running the notebook, pins the library
versions the project needs, and demonstrates reproducibility: a fixed
seed (42) produces byte-for-byte identical datasets across runs
(verified with SHA-256 hashes), and the run is recorded in
`run_info.json` and committed to git.

## Tasks
- **T1** — rebuilt distance values with seed 7, reported first three.
- **T2** — wrote `work/my_requirements.txt` with numpy, pandas,
  scikit-learn pinned to installed versions.
- **T3** — `fingerprint(path)` returning rows, sha256, and seed.

## Environment
- Python 3.12 (conda env `mlops`)
- numpy, pandas, scikit-learn, matplotlib (versions pinned in requirements.txt)

## AI Assistance Disclosure
AI assistance (Claude) was used for environment setup and formatting.
