import pandas as pd
import numpy as np
import os

def load_data():
    path = "data/delivery_times.csv"

    if not os.path.exists(path):
        np.random.seed(42)
        n = 600
        distance_km = np.random.uniform(0.5, 12, n)
        prep_time_min = np.random.uniform(5, 30, n)
        traffic_level = np.random.randint(1, 4, n)
        rain = np.random.randint(0, 2, n)
        noise = np.random.normal(0, 2, n)

        delivery_min = 6 + 3 * distance_km + 0.6 * prep_time_min + 4 * traffic_level + 5 * rain + noise

        df = pd.DataFrame({
            "distance_km": distance_km,
            "prep_time_min": prep_time_min,
            "traffic_level": traffic_level,
            "rain": rain,
            "delivery_min": delivery_min
        })
        df.to_csv(path, index=False)

    return pd.read_csv(path)
