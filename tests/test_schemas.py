"""Unit tests for Pydantic schemas and configuration loading for the dual-mode API."""
import yaml
import pytest

from analog_finder import config as C
from analog_finder.schemas import (
    AnalysisMode,
    AlertLevel,
    AnalogTier,
    HazardType,
    AnalysisRequest,
    TargetReference,
    TemporalReference,
    AnalysisResponse,
)


def test_load_temporal_pairs_yaml():
    """Verify temporal_pairs.yaml exists, loads, and contains essential hazard sites."""
    assert C.TEMPORAL_PAIRS_FILE.exists()
    data = yaml.safe_load(C.TEMPORAL_PAIRS_FILE.read_text(encoding="utf-8"))
    assert "temporal_pairs" in data
    pairs = data["temporal_pairs"]
    assert "nyiragongo_2021" in pairs
    assert "bukavu_kadutu" in pairs
    assert "bukavu_ibanda" in pairs
    assert pairs["nyiragongo_2021"]["hazard_type"] == "volcanic"
    assert pairs["bukavu_kadutu"]["hazard_type"] == "landslide"


def test_schemas_analog_request():
    """Test instantiating a valid analog request schema."""
    req = AnalysisRequest(
        mode=AnalysisMode.ANALOG,
        site_id="virunga",
        target_ref=TargetReference(target_id="marius_hills"),
        criterion_weights={"topography": 0.3, "geology": 0.4, "environment": 0.1, "resources": 0.2},
    )
    assert req.mode == AnalysisMode.ANALOG
    assert req.target_ref.target_id == "marius_hills"
    assert req.options.synthetic is False


def test_schemas_temporal_request():
    """Test instantiating a valid temporal request schema."""
    req = AnalysisRequest(
        mode=AnalysisMode.TEMPORAL,
        site_id="bukavu_kadutu",
        temporal_ref=TemporalReference(
            date_t0="2018-06-01/2018-09-01",
            date_t1="2024-06-01/2024-09-01",
            pair_id="bukavu_kadutu",
        ),
    )
    assert req.mode == AnalysisMode.TEMPORAL
    assert req.temporal_ref.pair_id == "bukavu_kadutu"
