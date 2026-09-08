# T2: reject unless at least 3 measurements are in a realistic range
def is_valid_order(radius_mean, texture_mean, perimeter_mean, area_mean):
    checks = [
        0 < radius_mean < 50,
        0 < texture_mean < 60,
        0 < perimeter_mean < 300,
        0 < area_mean < 5000,
    ]
    return sum(checks) >= 3
