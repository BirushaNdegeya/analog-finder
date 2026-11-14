"""Unit tests for multi-temporal differential indicator computation and ingestion."""
import numpy as np
import pytest

from analog_finder.indicators import compute_differential_indicators


def test_compute_differential_indicators_dem():
    """Verify elevation and slope differential calculations between T0 and T1."""
    # Create flat DEM at T0 (elevation 100)
    dem_t0 = np.full((30, 30), 100.0, dtype="float32")
    # T1 has a 10m subsidence crater in the center
    dem_t1 = dem_t0.copy()
    dem_t1[10:20, 10:20] -= 10.0

    res = compute_differential_indicators(dem_t0=dem_t0, dem_t1=dem_t1)
    assert "t0" in res and "t1" in res and "pixel_diffs" in res
    assert "delta_dem" in res["pixel_diffs"]
    assert "delta_pente" in res["pixel_diffs"]

    # Minimum elevation change should be -10.0
    assert pytest.approx(res["pixel_diffs"]["delta_dem"].min(), 0.1) == -10.0


def test_compute_differential_indicators_s2():
    """Verify Delta NDVI computation from multi-temporal Sentinel-2 bands."""
    # Synthetic S2 bands at T0 (healthy vegetation: B08 high, B04 low)
    s2_t0 = {
        "B02": np.full((20, 20), 0.1, dtype="float32"),
        "B04": np.full((20, 20), 0.1, dtype="float32"),
        "B08": np.full((20, 20), 0.6, dtype="float32"),
        "B11": np.full((20, 20), 0.2, dtype="float32"),
        "B12": np.full((20, 20), 0.1, dtype="float32"),
    }
    # T1: stripped bare soil / scar (B08 low, B04 high)
    s2_t1 = {
        "B02": np.full((20, 20), 0.2, dtype="float32"),
        "B04": np.full((20, 20), 0.4, dtype="float32"),
        "B08": np.full((20, 20), 0.15, dtype="float32"),
        "B11": np.full((20, 20), 0.3, dtype="float32"),
        "B12": np.full((20, 20), 0.2, dtype="float32"),
    }

    res = compute_differential_indicators(s2_t0=s2_t0, s2_t1=s2_t1)
    assert "delta_ndvi" in res["pixel_diffs"]
    # NDVI at T0 was (0.6 - 0.1) / (0.6 + 0.1) = ~0.71
    # NDVI at T1 is (0.15 - 0.4) / (0.15 + 0.4) = -0.45
    # delta_ndvi should be strongly negative (< -0.8)
    assert np.mean(res["pixel_diffs"]["delta_ndvi"]) < -0.8
