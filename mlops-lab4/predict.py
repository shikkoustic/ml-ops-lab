import sys
import joblib
import pandas as pd
from delivery import is_valid_order

distance_km = float(sys.argv[1])
prep_time_min = float(sys.argv[2])
traffic_level = int(sys.argv[3])
rain = int(sys.argv[4])

if not is_valid_order(distance_km, prep_time_min, traffic_level, rain):
    print("Invalid order")
else:
    model = joblib.load("delivery_model.joblib")
    order = pd.DataFrame([[distance_km, prep_time_min, traffic_level, rain]],
                          columns=["distance_km", "prep_time_min", "traffic_level", "rain"])
    prediction = model.predict(order)[0]
    print("Predicted delivery time:", round(prediction, 1), "minutes")
