import sys
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from cancer import is_valid_order

# 4 key measurements from the command line
radius_mean = float(sys.argv[1])
texture_mean = float(sys.argv[2])
perimeter_mean = float(sys.argv[3])
area_mean = float(sys.argv[4])

if not is_valid_order(radius_mean, texture_mean, perimeter_mean, area_mean):
    print("Invalid order")
    sys.exit()

# build a full 30-feature row: fill the rest with dataset means, override the 4 given
bc = load_breast_cancer()
row = pd.DataFrame(bc.data, columns=bc.feature_names).mean()
row["mean radius"] = radius_mean
row["mean texture"] = texture_mean
row["mean perimeter"] = perimeter_mean
row["mean area"] = area_mean
order = pd.DataFrame([row])

model = joblib.load("cancer_model.joblib")
pred = int(model.predict(order)[0])
proba = model.predict_proba(order)[0]

label = "benign" if pred == 1 else "malignant"
confidence = proba[pred] * 100
# T3: print prediction AND confidence
print("Prediction:", label)
print("Confidence:", round(confidence, 1), "%")
