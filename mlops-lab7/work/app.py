import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the saved model once, when the service starts
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))

app = FastAPI(title="Delivery Time Prediction Service")


# The shape of a valid request (ranges match the training data)
class Order(BaseModel):
    distance_km:   float = Field(gt=0, le=15)
    items:         int   = Field(ge=1, le=9)
    traffic_level: int   = Field(ge=1, le=3)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(order: Order):
    # Convert the order into a one-row table, as the model expects
    row = pd.DataFrame([order.model_dump()])
    prediction = model.predict(row)[0]
    return {"predicted_minutes": round(float(prediction), 1)}
