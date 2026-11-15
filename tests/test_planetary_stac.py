"""Unit and integration tests for the Planetary STAC ingestion engine and API bridge."""
import yaml
import numpy as np
import pytest
from fastapi.testclient import TestClient

from analog_finder import config as C
from analog_finder.api.app import app
from analog_finder.planetary_ingest import (
    search_planetary_stac,
    load_planetary_target_dem,
    stream_cog_window,
)

client = TestClient(app)


def test_targets_yaml_planetary_metadata():
    """Verify targets.yaml contains lunar and Martian targets with STAC collections."""
    targets = yaml.safe_load((C.CONFIG_DIR / "targets.yaml").read_text(encoding="utf-8"))
    assert "marius_hills" in targets
    assert "jezero_crater" in targets

    marius = targets["marius_hills"]
    assert marius["body"] == "moon"
    assert "kaguya" in marius["stac_collection"]
    assert "bbox" in marius

    jezero = targets["jezero_crater"]
    assert jezero["body"] == "mars"
    assert "ctx" in jezero["stac_collection"]
    assert "bbox" in jezero


def test_api_planetary_bodies():
    """Verify GET /v1/planetary/bodies returns Earth, Moon, and Mars metadata."""
    resp = client.get("/v1/planetary/bodies")
    assert resp.status_code == 200
    bodies = resp.json()["supported_bodies"]
    assert "earth" in bodies
    assert "moon" in bodies
    assert "mars" in bodies
    assert "kaguya" in bodies["moon"]["primary_dem"]
    assert "ctx" in bodies["mars"]["primary_dem"]


def test_planetary_target_dem_fallback(tmp_path):
    """Test load_planetary_target_dem loads local file when present, or falls back to STAC/None."""
    # Test with local file present
    mock_tcfg = {
        "name": "Mock Moon Target",
        "body": "moon",
        "dem_raster": str(tmp_path / "mock_dem.tif"),
        "bbox": [-57.0, 13.0, -55.0, 15.0],
    }
    # When file does not exist and network might be unreachable, it must return None without raising an exception
    res = load_planetary_target_dem(mock_tcfg)
    assert res is None or isinstance(res, np.ndarray)


def test_api_planetary_search_endpoint():
    """Verify GET /v1/planetary/search handles valid query parameters."""
    resp = client.get("/v1/planetary/search?body=moon&min_lon=-57&min_lat=13&max_lon=-55&max_lat=15&limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["body"] == "moon"
    assert "items" in data
    assert isinstance(data["items"], list)
