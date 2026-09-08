def get_features_and_target(df):
    X = df[["distance_km", "prep_time_min", "traffic_level", "rain"]]
    y = df["delivery_min"]
    return X, y

# T1
def average_speed_kmph(distance_km, delivery_min):
    hours = delivery_min / 60
    return distance_km / hours
