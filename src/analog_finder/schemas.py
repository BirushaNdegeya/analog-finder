"""Unified data schemas and validation contracts for Analog Finder Dual-Mode Engine.

Supports both:
  - Mode A: 'analog' (planetary mission preparation, Earth vs Moon/Mars target)
  - Mode B: 'temporal' (geohazard early warning, territorial deformation T0 vs T1)
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalysisMode(str, Enum):
    """Analysis operational mode."""
    ANALOG = "analog"
    TEMPORAL = "temporal"


class HazardType(str, Enum):
    """Geohazard classification for temporal surveillance."""
    VOLCANIC = "volcanic"
    LANDSLIDE = "landslide"
    EROSION = "erosion"
    FLOODING = "flooding"
    GENERAL = "general"


class AlertLevel(str, Enum):
    """Early-warning critical alert level."""
    STABLE = "stable"        # Normal seasonal drift (Wasserstein < 0.20)
    WATCH = "watch"          # Detectable structural deviation (0.20 <= W < 0.40)
    WARNING = "warning"      # Substantial geomorphological change (0.40 <= W < 0.65)
    CRITICAL = "critical"    # Extreme destabilization / imminent hazard (W >= 0.65)


class AnalogTier(str, Enum):
    """Planetary analog suitability tier."""
    STRONG = "strong"        # Score >= 0.70
    PARTIAL = "partial"      # 0.40 <= Score < 0.70
    WEAK = "weak"            # Score < 0.40


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class TargetReference(BaseModel):
    """Planetary target specifications for mode='analog'."""
    target_id: str = Field(..., description="Target planetary ID defined in targets.yaml (e.g. 'marius_hills')")


class TemporalReference(BaseModel):
    """Time-series reference windows for mode='temporal'."""
    date_t0: str = Field(..., description="Baseline reference ISO 8601 date range, e.g. '2021-01-01/2021-05-15'")
    date_t1: str = Field(..., description="Observation surveillance ISO 8601 date range, e.g. '2021-06-01/2021-09-30'")
    pair_id: Optional[str] = Field(None, description="Optional pre-configured pair ID from temporal_pairs.yaml")


class AnalysisOptions(BaseModel):
    """Execution options for analysis requests."""
    synthetic: bool = Field(False, description="Use deterministic synthetic distributions for offline testing")
    patch_size_m: int = Field(500, description="Spatial window in metres for localized patch-wise deformation heatmaps")
    significance_alpha: float = Field(0.05, description="Significance threshold for statistical hypothesis testing")


class AnalysisRequest(BaseModel):
    """Unified request model for POST /v1/analyze."""
    mode: AnalysisMode = Field(..., description="Operational mode: 'analog' or 'temporal'")
    site_id: str = Field(..., description="Candidate site ID or temporal monitoring zone identifier")
    bbox: Optional[List[float]] = Field(
        None,
        description="Bounding box [west, south, east, north] in EPSG:4326. Defaults to configured site bbox."
    )
    target_ref: Optional[TargetReference] = Field(None, description="Required when mode='analog'")
    temporal_ref: Optional[TemporalReference] = Field(None, description="Required when mode='temporal'")
    criterion_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Custom criterion weights. Keys: 'topography', 'geology', 'environment', 'resources'"
    )
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class IndicatorMetric(BaseModel):
    """Per-indicator comparison metrics."""
    indicator_id: str
    label: str
    unit: str
    # Common fields
    wasserstein_distance: float
    score: float = Field(..., description="In [0, 1]. For analog: 1=identical; For temporal: 1=maximum critical change")
    # Temporal-specific fields
    delta_median: Optional[float] = None
    p_value: Optional[float] = None
    is_significant: Optional[bool] = None
    interpretation: str


class CriterionScore(BaseModel):
    """Per-criterion aggregate score."""
    criterion_id: str
    label: str
    score: Optional[float] = None
    confidence: float = Field(..., description="Fraction of non-null indicators evaluated in this criterion")


class ValidityCard(BaseModel):
    """Qualitative validity breakdown."""
    reproduced_or_stable: List[str] = Field(default_factory=list)
    partial_drift: List[str] = Field(default_factory=list)
    divergent_or_critical: List[str] = Field(default_factory=list)
    not_evaluated: List[str] = Field(default_factory=list)
    never_reproducible: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Unified standardized output of the dual-mode engine."""
    mode: AnalysisMode
    site_id: str
    site_name: str
    hazard_type: Optional[HazardType] = None
    reference_summary: str
    # Scores
    global_score: float = Field(..., description="Overall similarity score (analog) or criticality index (temporal)")
    alert_level: Optional[AlertLevel] = None
    analog_tier: Optional[AnalogTier] = None
    confidence: float
    # Detail
    criteria: Dict[str, CriterionScore]
    indicators: Dict[str, IndicatorMetric]
    validity_card: ValidityCard
    explanation: str
    # Metadata
    synthetic: bool = False
    spatial_heatmap_available: bool = False
    heatmap_geojson_url: Optional[str] = None
