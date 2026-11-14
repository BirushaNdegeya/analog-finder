"""Unit tests for geospatial indicator computations.

Tests run on small synthetic numpy arrays — no satellite data required.
"""
import numpy as np
import pytest

from analog_finder.indicators import (
    slope_deg,
    local_std,
    local_fraction,
    closed_depressions,
    sample,
    dem_indicators,
    compute_site_indicators,
)
from analog_finder import config as C


# ---------------------------------------------------------------------------
# slope_deg
# ---------------------------------------------------------------------------

def test_slope_flat_dem():
    """A perfectly flat DEM must return slope = 0 everywhere."""
    dem = np.zeros((20, 20), dtype="float32")
    result = slope_deg(dem, res_m=100.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


def test_slope_tilted_dem():
    """A uniformly tilted plane (1 m / 100 m) must return ~0.57°."""
    dem = np.tile(np.arange(20, dtype="float32") * 1.0, (20, 1))
    result = slope_deg(dem, res_m=100.0)
    expected = np.degrees(np.arctan(1.0 / 100.0))
    # Border pixels have edge-gradient artefacts — only check interior
    np.testing.assert_allclose(result[2:-2, 2:-2], expected, atol=0.01)


# ---------------------------------------------------------------------------
# local_std
# ---------------------------------------------------------------------------

def test_local_std_constant():
    """Std of a constant array is 0."""
    arr = np.full((30, 30), 5.0, dtype="float32")
    result = local_std(arr, window_px=5)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# local_fraction
# ---------------------------------------------------------------------------

def test_local_fraction_all_true():
    """Fraction of an all-True mask must be 1.0 everywhere."""
    mask = np.ones((20, 20), dtype="bool")
    result = local_fraction(mask, window_px=5)
    np.testing.assert_allclose(result, 1.0, atol=1e-5)


def test_local_fraction_all_false():
    """Fraction of an all-False mask must be 0.0 everywhere."""
    mask = np.zeros((20, 20), dtype="bool")
    result = local_fraction(mask, window_px=5)
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# closed_depressions
# ---------------------------------------------------------------------------

def test_closed_depressions_flat():
    """A flat DEM has no closed depressions."""
    dem = np.zeros((30, 30), dtype="float32")
    result = closed_depressions(dem, min_depth_m=1.0)
    assert not result.any(), "Flat DEM should have no closed depressions"


def test_closed_depressions_pit():
    """A single sunken pixel should be detected as a closed depression."""
    dem = np.zeros((30, 30), dtype="float32")
    dem[15, 15] = -10.0  # pit 10 m below surroundings
    result = closed_depressions(dem, min_depth_m=1.0)
    assert result[15, 15], "Central pit should be flagged as closed depression"


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def test_sample_max_n():
    """sample() must return at most MAX_SAMPLES values."""
    arr = np.arange(C.MAX_SAMPLES * 3, dtype="float32")
    result = sample(arr)
    assert result.size <= C.MAX_SAMPLES


def test_sample_excludes_nan():
    """sample() must exclude NaN and Inf values."""
    arr = np.array([1.0, np.nan, 2.0, np.inf, 3.0], dtype="float32")
    result = sample(arr)
    assert np.all(np.isfinite(result))
    assert result.size == 3


# ---------------------------------------------------------------------------
# dem_indicators
# ---------------------------------------------------------------------------

def test_dem_indicators_keys():
    """dem_indicators must return 'pente' and 'rugosite' keys."""
    dem = np.random.default_rng(0).uniform(0, 100, (40, 40)).astype("float32")
    result = dem_indicators(dem, res_m=100.0)
    assert "pente" in result
    assert "rugosite" in result


def test_dem_indicators_non_empty():
    """Both sample arrays from dem_indicators must be non-empty."""
    dem = np.random.default_rng(1).uniform(0, 100, (40, 40)).astype("float32")
    result = dem_indicators(dem, res_m=100.0)
    assert result["pente"].size > 0
    assert result["rugosite"].size > 0


# ---------------------------------------------------------------------------
# compute_site_indicators
# ---------------------------------------------------------------------------

def test_compute_site_indicators_no_data():
    """With no inputs, the function must return an empty dict (not crash)."""
    result = compute_site_indicators()
    assert result == {}


def test_compute_site_indicators_dem_only():
    """With only a DEM, only topographic indicators are computed."""
    dem = np.random.default_rng(2).uniform(0, 500, (50, 50)).astype("float32")
    result = compute_site_indicators(dem100=dem)
    assert "pente" in result
    assert "rugosite" in result
    # Sentinel-2 / precip / basalt indicators must be absent
    for key in ("sol_nu", "oxydes_fer", "argiles", "aridite", "basalte"):
        assert key not in result, f"Unexpected key '{key}' without S2/basalt/precip input"


def test_compute_site_indicators_missing_indicator_not_zero():
    """Missing indicators must not be returned as 0.0 — they must be absent."""
    dem = np.ones((30, 30), dtype="float32") * 100.0
    result = compute_site_indicators(dem100=dem)
    # No Sentinel-2 → no sol_nu indicator at all
    assert "sol_nu" not in result
