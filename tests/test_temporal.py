"""Unit tests for the temporal geohazard & deformation engine (Mode B)."""
import numpy as np
import pytest

from analog_finder import config as C
from analog_finder.schemas import AlertLevel
from analog_finder.temporal import (
    compute_temporal_wasserstein,
    statistical_significance,
    analyze_indicator_change,
    temporal_criteria_scores,
    temporal_global_criticality,
    classify_alert_level,
    patch_wasserstein_grid,
    temporal_validity_card,
)


def test_temporal_wasserstein_identical():
    """Identical distributions must have Wasserstein distance 0.0."""
    d0 = np.linspace(0, 10, 100)
    d1 = np.linspace(0, 10, 100)
    assert compute_temporal_wasserstein(d0, d1) == 0.0


def test_temporal_wasserstein_shifted():
    """Shifted distributions must return expected distance."""
    d0 = np.zeros(100)
    d1 = np.ones(100) * 5.0
    assert pytest.approx(compute_temporal_wasserstein(d0, d1), 0.01) == 5.0


def test_statistical_significance_null_change():
    """Draws from identical distribution should not be flagged as significant."""
    rng = np.random.default_rng(42)
    d0 = rng.normal(10.0, 1.0, 500)
    d1 = rng.normal(10.0, 1.0, 500)
    p_val, is_sig = statistical_significance(d0, d1, alpha=0.01)
    assert is_sig is False
    assert p_val > 0.01


def test_statistical_significance_major_change():
    """Draws from widely different distributions must be flagged as significant."""
    rng = np.random.default_rng(42)
    d0 = rng.normal(10.0, 1.0, 500)
    d1 = rng.normal(25.0, 1.0, 500)
    p_val, is_sig = statistical_significance(d0, d1, alpha=0.05)
    assert is_sig is True
    assert p_val < 0.001


def test_analyze_indicator_change():
    """Verify change analysis metrics and interpretation output."""
    d0 = np.linspace(5.0, 7.0, 30)
    d1 = np.linspace(12.0, 14.0, 30)
    res = analyze_indicator_change("pente", d0, d1, norm_scale=15.0)
    assert res["indicator_id"] == "pente"
    assert res["delta_median"] > 0
    assert res["score"] > 0
    assert "steepening" in res["interpretation"].lower() or "slope" in res["interpretation"].lower()


def test_classify_alert_levels():
    """Verify alert level classification across the 4 tiers."""
    assert classify_alert_level(0.10) == AlertLevel.STABLE
    assert classify_alert_level(0.25) == AlertLevel.WATCH
    assert classify_alert_level(0.50) == AlertLevel.WARNING
    assert classify_alert_level(0.75) == AlertLevel.CRITICAL


def test_patch_wasserstein_grid_localization():
    """Verify 2D patch-wise localized deformation anomaly detection."""
    # Create 40x40 raster (4x4 patches of 10x10)
    arr_t0 = np.zeros((40, 40), dtype="float32")
    arr_t1 = np.zeros((40, 40), dtype="float32")

    # Introduce localized deformation only in patch (row 2, col 2)
    arr_t1[20:30, 20:30] = 8.0

    heatmap = patch_wasserstein_grid(arr_t0, arr_t1, patch_size_px=10)
    assert heatmap.shape == (4, 4)
    # The anomaly patch should be high (> 7)
    assert pytest.approx(heatmap[2, 2], 0.1) == 8.0
    # Other patches should be 0.0
    assert heatmap[0, 0] == 0.0
    assert heatmap[1, 1] == 0.0
    assert heatmap[3, 3] == 0.0


def test_temporal_validity_card():
    """Verify qualitative report generation."""
    criteria_scores = {
        "topographie": 0.70,  # Critical
        "geologie": 0.10,     # Stable
        "environnement": 0.30 # Watch/drift
    }
    card = temporal_validity_card(criteria_scores, AlertLevel.CRITICAL)
    assert any("Topography" in s for s in card["divergent_or_critical"])
    assert any("Geology" in s for s in card["reproduced_or_stable"])
    assert len(card["recommended_actions"]) > 0
