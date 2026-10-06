import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cancer import is_valid_sample


def test_realistic_sample_is_accepted():
    good = {
        "radius_mean": 14.0,
        "texture_mean": 20.0,
        "perimeter_mean": 90.0,
        "area_mean": 600.0,
    }
    assert is_valid_sample(good) is True


def test_unrealistic_sample_is_rejected():
    bad = {
        "radius_mean": 999.0,
        "texture_mean": 999.0,
        "perimeter_mean": 999.0,
        "area_mean": 600.0,
    }
    assert is_valid_sample(bad) is False
