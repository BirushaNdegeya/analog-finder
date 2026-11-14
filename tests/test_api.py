"""Unit and integration tests for the FastAPI dual-mode REST endpoints."""
from fastapi.testclient import TestClient
import pytest

from analog_finder.api.app import app
from analog_finder.schemas import AnalysisMode, AlertLevel

client = TestClient(app)


def test_api_health():
    """Verify health endpoint returns healthy status and engine information."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "wasserstein" in data["engine"]
    assert "analog" in data["modes"]
    assert "temporal" in data["modes"]


def test_api_catalogs():
    """Verify catalog endpoints (sites, targets, temporal-pairs, indicators)."""
    # Sites
    r_sites = client.get("/v1/sites")
    assert r_sites.status_code == 200
    assert len(r_sites.json()["sites"]) >= 8

    # Targets
    r_targets = client.get("/v1/targets")
    assert r_targets.status_code == 200
    assert "marius_hills" in r_targets.json()

    # Temporal pairs
    r_temporal = client.get("/v1/temporal-pairs")
    assert r_temporal.status_code == 200
    assert "bukavu_kadutu" in r_temporal.json()["temporal_pairs"]

    # Indicators
    r_indicators = client.get("/v1/indicators")
    assert r_indicators.status_code == 200
    assert len(r_indicators.json()) == 9


def test_api_analyze_analog_mode():
    """Test POST /v1/analyze in analog mode."""
    payload = {
        "mode": "analog",
        "site_id": "virunga",
        "target_ref": {"target_id": "marius_hills"},
        "options": {"synthetic": True},
    }
    response = client.post("/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "analog"
    assert data["site_id"] == "virunga"
    assert "global_score" in data
    assert 0.0 <= data["global_score"] <= 1.0
    assert data["analog_tier"] in ["strong", "partial", "weak"]
    assert "criteria" in data
    assert "indicators" in data


def test_api_analyze_temporal_mode():
    """Test POST /v1/analyze in temporal geohazard mode."""
    payload = {
        "mode": "temporal",
        "site_id": "bukavu_kadutu",
        "temporal_ref": {
            "date_t0": "2018-06-01/2018-09-01",
            "date_t1": "2024-06-01/2024-09-01",
            "pair_id": "bukavu_kadutu",
        },
        "options": {"synthetic": True},
    }
    response = client.post("/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "temporal"
    assert data["site_id"] == "bukavu_kadutu"
    assert "global_score" in data
    assert data["alert_level"] in ["stable", "watch", "warning", "critical"]
    assert data["hazard_type"] == "landslide"
    assert len(data["indicators"]) == 9
    assert "validity_card" in data
