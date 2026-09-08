# T2
def is_valid_order(distance_km, prep_time_min, traffic_level, rain):
    if distance_km <= 0:
        return False
    if prep_time_min <= 0:
        return False
    if traffic_level not in [1, 2, 3]:
        return False
    if rain not in [0, 1]:
        return False
    return True
