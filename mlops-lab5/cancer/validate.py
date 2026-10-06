# To Do T2: is_valid_sample takes a dict of measurements and rejects it
# unless at least 3 of them are in a realistic range
REALISTIC_RANGES = {
    "radius_mean": (0, 50),
    "texture_mean": (0, 60),
    "perimeter_mean": (0, 300),
    "area_mean": (0, 5000),
}


def is_valid_sample(sample):
    checks = [
        low < sample.get(name, -1) < high
        for name, (low, high) in REALISTIC_RANGES.items()
    ]
    return sum(checks) >= 3


# kept from Practical 04 so the old positional call still works
def is_valid_order(radius_mean, texture_mean, perimeter_mean, area_mean):
    return is_valid_sample({
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "perimeter_mean": perimeter_mean,
        "area_mean": area_mean,
    })
