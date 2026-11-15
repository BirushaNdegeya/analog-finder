"""Tests for interactive custom AOI evaluation."""
from __future__ import annotations

import pytest
from analog_finder import interactive as IA


def test_custom_analog_study_synthetic():
    bbox = (29.1, -1.6, 29.4, -1.35)
    res = IA.run_custom_analog_study(
        bbox=bbox,
        target_id="marius_hills",
        custom_name="Test Virunga AOI",
        synthetic=True,
    )
    assert res is not None
    assert "custom_site" in res
    assert res["custom_site"]["name"] == "Test Virunga AOI"
    assert res["rank"] >= 1
    assert res["total_sites"] >= 1
    assert "criteria" in res["custom_site"]
    assert "validity" in res["custom_site"]
    assert 0.0 <= res["custom_site"]["global"] <= 1.0


def test_custom_temporal_study_synthetic():
    bbox = (28.84, -2.53, 28.86, -2.51)
    res = IA.run_custom_temporal_study(
        bbox=bbox,
        baseline_t0="2019-06-01/2019-09-01",
        observation_t1="2024-06-01/2024-09-01",
        zone_name="Test Bukavu Landslide AOI",
        hazard_type="landslide",
        synthetic=True,
    )
    assert res is not None
    assert res["name"] == "Test Bukavu Landslide AOI"
    assert res["alert_level"] in ["critical", "warning", "watch", "stable"]
    assert 0.0 <= res["criticality_index"] <= 1.0
    assert "indicators" in res
    assert "validity_card" in res
