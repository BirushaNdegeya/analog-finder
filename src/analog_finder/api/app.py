"""FastAPI REST application exposing the Analog Finder Dual-Mode Engine.

Provides unified endpoints for:
  - Mode A: Planetary mission analog assessment (Earth vs Moon/Mars)
  - Mode B: Geohazard early warning & temporal deformation monitoring (T0 vs T1)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from .. import config as C
from .. import scoring as S
from .. import temporal as T
from ..schemas import (
    AnalysisMode,
    AnalysisRequest,
    AnalysisResponse,
    AlertLevel,
    AnalogTier,
    CriterionScore,
    HazardType,
    IndicatorMetric,
    ValidityCard,
)

# ---------------------------------------------------------------------------
# Application Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Analog Finder — Dual-Mode Geospatial API",
    version="1.1.0",
    description=(
        "Unified Optimal Transport (Wasserstein Distance) engine providing:\n"
        "- Mode A: Planetary exploration analog site assessment (NASA Space Apps)\n"
        "- Mode B: Multi-temporal geohazard early warning (Bukavu landslides & Virunga volcanism)"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _load_yaml(filename: str) -> dict:
    path = C.CONFIG_DIR / filename
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration file {filename} not found."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _synthetic_distributions(seed: int = 42) -> Tuple[dict, dict]:
    """Generate synthetic baseline distributions for offline testing."""
    from ..pipeline import _SYNTHETIC_BASELINE
    rng = np.random.default_rng(seed)
    target = {k: S.samples_from_expected(v, abs(v) * 0.1 + 1e-3) for k, v in _SYNTHETIC_BASELINE.items()}
    site = {k: rng.normal(v, abs(v) * 0.15 + 1e-3, 500) for k, v in _SYNTHETIC_BASELINE.items()}
    return target, site


# ---------------------------------------------------------------------------
# Discovery & Metadata Endpoints
# ---------------------------------------------------------------------------
@app.get("/v1/health", tags=["System"])
def get_health() -> dict:
    """Return API health status, version, and supported engines."""
    return {
        "status": "healthy",
        "version": "1.1.0",
        "engine": "wasserstein_optimal_transport",
        "modes": ["analog", "temporal"],
    }


@app.get("/v1/sites", tags=["Catalogs"])
def list_sites() -> dict:
    """List configured terrestrial analog candidate sites."""
    cfg = _load_yaml("sites.yaml")
    return {"target_default": cfg.get("target"), "sites": cfg.get("sites", [])}


@app.get("/v1/targets", tags=["Catalogs"])
def list_targets() -> dict:
    """List configured planetary targets (Moon, Mars)."""
    return _load_yaml("targets.yaml")


@app.get("/v1/temporal-pairs", tags=["Catalogs"])
def list_temporal_pairs() -> dict:
    """List configured geohazard monitoring zones and historical baseline dates."""
    return _load_yaml("temporal_pairs.yaml")


@app.get("/v1/indicators", tags=["Catalogs"])
def list_indicators() -> List[dict]:
    """List all 8 multi-criteria indicators, descriptions, and measurement units."""
    catalog = []
    for crit, inds in C.CRITERIA.items():
        for ind in inds:
            catalog.append({
                "id": ind,
                "label": C.INDICATOR_LABELS.get(ind, ind),
                "criterion": crit,
                "criterion_label": C.CRITERIA_LABELS.get(crit, crit),
                "unit": C.UNITS.get(ind, ""),
            })
    return catalog


# ---------------------------------------------------------------------------
# Core Unified Analysis Endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Execute analysis in either Mode A (analog) or Mode B (temporal).

    - **Mode A (analog)**: Compares site distribution vs target planetary distribution.
    - **Mode B (temporal)**: Compares site distribution between baseline T0 and surveillance T1.
    """
    if request.mode == AnalysisMode.ANALOG:
        return _handle_analog_mode(request)
    elif request.mode == AnalysisMode.TEMPORAL:
        return _handle_temporal_mode(request)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported analysis mode: {request.mode}")


def _handle_analog_mode(req: AnalysisRequest) -> AnalysisResponse:
    sites_cfg = _load_yaml("sites.yaml")
    site_dict = next((s for s in sites_cfg["sites"] if s["id"] == req.site_id), None)
    site_name = site_dict["name"] if site_dict else req.site_id

    target_id = req.target_ref.target_id if req.target_ref else sites_cfg.get("target", "marius_hills")
    targets_cfg = _load_yaml("targets.yaml")
    target_info = targets_cfg.get(target_id, {"name": target_id})

    # Synthetic or pre-computed data extraction
    target_dist, site_dist = _synthetic_distributions(seed=hash(req.site_id) % 10000)
    sims = S.indicator_similarities(target_dist, {req.site_id: site_dist})[req.site_id]

    crit_scores = S.criterion_scores(sims)
    weights = req.criterion_weights or C.DEFAULT_WEIGHTS
    g_score = round(S.global_score(crit_scores, weights), 4)

    tier = AnalogTier.STRONG if g_score >= C.STRONG else AnalogTier.PARTIAL if g_score >= C.WEAK else AnalogTier.WEAK
    v_card = S.validity_card(crit_scores)

    indicators_resp = {}
    for ind, sc in sims.items():
        w_d = float(np.abs(np.median(site_dist[ind]) - np.median(target_dist[ind])))
        indicators_resp[ind] = IndicatorMetric(
            indicator_id=ind,
            label=C.INDICATOR_LABELS.get(ind, ind),
            unit=C.UNITS.get(ind, ""),
            wasserstein_distance=round(w_d, 4),
            score=round(sc, 4),
            delta_median=round(float(np.median(site_dist[ind]) - np.median(target_dist[ind])), 4),
            interpretation=f"Similarity score {sc:.2f} relative to target {target_info.get('name')}",
        )

    criteria_resp = {
        c: CriterionScore(
            criterion_id=c,
            label=C.CRITERIA_LABELS.get(c, c),
            score=round(val, 4) if val is not None else None,
            confidence=1.0 if val is not None else 0.0,
        )
        for c, val in crit_scores.items()
    }

    return AnalysisResponse(
        mode=AnalysisMode.ANALOG,
        site_id=req.site_id,
        site_name=site_name,
        reference_summary=f"Planetary Target: {target_info.get('name')}",
        global_score=g_score,
        analog_tier=tier,
        confidence=round(S.confidence(sims), 4),
        criteria=criteria_resp,
        indicators=indicators_resp,
        validity_card=ValidityCard(
            reproduced_or_stable=v_card.get("reproduit", []),
            partial_drift=v_card.get("partiel", []),
            divergent_or_critical=v_card.get("non_reproduit", []),
            not_evaluated=v_card.get("non_evalue", []),
            never_reproducible=v_card.get("jamais_reproductible", []),
            recommended_actions=v_card.get("usages_adaptes", []),
        ),
        explanation=S.explain(crit_scores),
        synthetic=req.options.synthetic or True,
    )


def _handle_temporal_mode(req: AnalysisRequest) -> AnalysisResponse:
    temporal_cfg = _load_yaml("temporal_pairs.yaml")
    pair_id = req.temporal_ref.pair_id if req.temporal_ref and req.temporal_ref.pair_id else req.site_id
    pair_info = temporal_cfg.get("temporal_pairs", {}).get(pair_id, {})

    site_name = pair_info.get("name", req.site_id)
    hazard_str = pair_info.get("hazard_type", "general")
    try:
        h_type = HazardType(hazard_str)
    except ValueError:
        h_type = HazardType.GENERAL

    t0_range = req.temporal_ref.date_t0 if req.temporal_ref else pair_info.get("baseline_t0", "T0")
    t1_range = req.temporal_ref.date_t1 if req.temporal_ref else pair_info.get("observation_t1", "T1")

    # Generate synthetic time-shifted distributions
    rng = np.random.default_rng(hash(req.site_id) % 10000)
    from ..pipeline import _SYNTHETIC_BASELINE
    dist_t0 = {k: rng.normal(v, abs(v) * 0.1, 500) for k, v in _SYNTHETIC_BASELINE.items()}
    # Introduce acute slope and bare soil perturbation for high hazard demo
    dist_t1 = {
        k: rng.normal(v * (1.4 if k in ["pente", "sol_nu"] else 1.05), abs(v) * 0.12, 500)
        for k, v in _SYNTHETIC_BASELINE.items()
    }

    ind_metrics = {}
    for ind in C.INDICATORS:
        ind_metrics[ind] = T.analyze_indicator_change(ind, dist_t0[ind], dist_t1[ind])

    crit_scores = T.temporal_criteria_scores(ind_metrics)
    weights = req.criterion_weights or C.DEFAULT_WEIGHTS
    g_criticality = round(T.temporal_global_criticality(crit_scores, weights), 4)

    alert = T.classify_alert_level(g_criticality)
    v_card = T.temporal_validity_card(crit_scores, alert)

    indicators_resp = {
        ind: IndicatorMetric(
            indicator_id=m["indicator_id"],
            label=m["label"],
            unit=m["unit"],
            wasserstein_distance=m["wasserstein_distance"],
            score=m["score"],
            delta_median=m["delta_median"],
            p_value=m["p_value"],
            is_significant=m["is_significant"],
            interpretation=m["interpretation"],
        )
        for ind, m in ind_metrics.items()
    }

    criteria_resp = {
        c: CriterionScore(
            criterion_id=c,
            label=C.CRITERIA_LABELS.get(c, c),
            score=round(val, 4) if val is not None else None,
            confidence=1.0 if val is not None else 0.0,
        )
        for c, val in crit_scores.items()
    }

    return AnalysisResponse(
        mode=AnalysisMode.TEMPORAL,
        site_id=req.site_id,
        site_name=site_name,
        hazard_type=h_type,
        reference_summary=f"Temporal Baseline: {t0_range} vs Surveillance: {t1_range}",
        global_score=g_criticality,
        alert_level=alert,
        confidence=1.0,
        criteria=criteria_resp,
        indicators=indicators_resp,
        validity_card=ValidityCard(
            reproduced_or_stable=v_card.get("reproduced_or_stable", []),
            partial_drift=v_card.get("partial_drift", []),
            divergent_or_critical=v_card.get("divergent_or_critical", []),
            not_evaluated=v_card.get("not_evaluated", []),
            never_reproducible=v_card.get("never_reproducible", []),
            recommended_actions=v_card.get("recommended_actions", []),
        ),
        explanation=f"Criticality index {g_criticality:.2f} — Alert Status: {alert.value.upper()}.",
        synthetic=True,
        spatial_heatmap_available=True,
    )
