import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

HERE = os.path.dirname(__file__)
model = joblib.load(os.path.join(HERE, "iris_model.joblib"))
target_names = joblib.load(os.path.join(HERE, "iris_target_names.joblib"))

app = FastAPI(title="Iris Flower Classification Service")


class IrisMeasurements(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width:  float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width:  float = Field(gt=0, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(sample: IrisMeasurements):
    row = pd.DataFrame([[
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width,
    ]], columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

    predicted_class = model.predict(row)[0]
    species = target_names[predicted_class]
    return {"species": species}
