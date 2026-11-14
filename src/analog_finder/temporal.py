"""Temporal geohazard monitoring and environmental change detection engine (Mode B).

Methodology:
While Mode A (analog) evaluates planetary resemblance against a reference target:
    S = 1 - W_1(Target, Site) / W_max

Mode B (temporal) measures the energy of physical deformation of a single territory
between baseline T_0 and surveillance T_1:
    Delta_crit = W_1(Site_T0, Site_T1) / W_max_temporal

Statistical significance is tested via two-sample Kolmogorov-Smirnov (KS) tests:
    H_0: P_{T_0} == P_{T_1} (no true physical deformation, sensor noise)
    Reject H_0 if p_value < alpha (e.g. 0.05)

Spatial localization is maintained through sliding-window patch Wasserstein
analysis, producing a 2-D deformation heatmap.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from . import config as C
from .schemas import AlertLevel


# ---------------------------------------------------------------------------
# Elementary distribution comparisons
# ---------------------------------------------------------------------------

def compute_temporal_wasserstein(dist_t0: np.ndarray, dist_t1: np.ndarray) -> float:
    """Compute the 1st Wasserstein distance (Earth Mover's Distance) between T0 and T1."""
    v0 = np.asarray(dist_t0, dtype="float32").ravel()
    v1 = np.asarray(dist_t1, dtype="float32").ravel()
    v0 = v0[np.isfinite(v0)]
    v1 = v1[np.isfinite(v1)]
    if len(v0) == 0 or len(v1) == 0:
        return 0.0
    return float(wasserstein_distance(v0, v1))


def statistical_significance(
    dist_t0: np.ndarray,
    dist_t1: np.ndarray,
    alpha: float = C.SIGNIFICANCE_ALPHA,
) -> Tuple[float, bool]:
    """Test whether observed change between T0 and T1 is statistically significant.

    Uses a two-sample Kolmogorov-Smirnov test.
    Returns (p_value, is_significant).
    """
    v0 = np.asarray(dist_t0, dtype="float32").ravel()
    v1 = np.asarray(dist_t1, dtype="float32").ravel()
    v0 = v0[np.isfinite(v0)]
    v1 = v1[np.isfinite(v1)]
    if len(v0) < 5 or len(v1) < 5:
        return 1.0, False

    # Subsample if large to maintain fast test execution
    if len(v0) > 1000:
        v0 = np.random.default_rng(0).choice(v0, 1000, replace=False)
    if len(v1) > 1000:
        v1 = np.random.default_rng(0).choice(v1, 1000, replace=False)

    stat = ks_2samp(v0, v1)
    p_val = float(stat.pvalue)
    return p_val, bool(p_val < alpha)


# ---------------------------------------------------------------------------
# Indicator & Criteria temporal scoring
# ---------------------------------------------------------------------------

# Expected maximum observable physical shifts for single-site normalization
DEFAULT_TEMPORAL_SCALE: Dict[str, float] = {
    "pente": 15.0,              # 15° change in slope indicates massive collapse
    "rugosite": 6.0,            # 6° shift in local roughness
    "basalte": 0.50,            # 50% shift in basalt coverage (e.g. lava flow)
    "oxydes_fer": 1.0,          # Mineral index alteration
    "argiles": 0.8,             # Clay mineral index alteration
    "sol_nu": 0.60,             # 60% loss of vegetation / bare ground expansion
    "aridite": 500.0,           # 500 mm/yr precipitation variation
    "cavites": 0.05,            # Surface cavity / pit density variation
    "basalte_accessible": 0.50, # Change in exposed volcanic material
}


def interpret_indicator_change(ind: str, delta: float, is_sig: bool) -> str:
    """Generate human-readable domain interpretation for observed change."""
    if not is_sig:
        return "Stable — no statistically significant deformation."

    unit = C.UNITS.get(ind, "")
    sign = "+" if delta > 0 else ""
    if ind == "pente":
        direction = "Slope steepening (potential scarp formation)" if delta > 0 else "Slope flattening (deposition/debris accumulation)"
        return f"{direction} ({sign}{delta:.2f}{unit})"
    if ind == "rugosite":
        direction = "Surface terrain roughening (rockfall/fissuring)" if delta > 0 else "Terrain smoothing (fine sediment deposit)"
        return f"{direction} ({sign}{delta:.2f}{unit})"
    if ind == "sol_nu":
        direction = "Vegetation removal & bare soil exposure" if delta > 0 else "Vegetative recovery / regreening"
        return f"{direction} ({sign}{delta:.1%})"
    if ind == "basalte" or ind == "basalte_accessible":
        direction = "New basaltic lava flow deposition" if delta > 0 else "Burial or weathering of surface basalt"
        return f"{direction} ({sign}{delta:.1%})"
    if ind == "argiles":
        return f"Alteration in hydrous clay signatures ({sign}{delta:.2f})"
    if ind == "oxydes_fer":
        return f"Shift in oxidized mineral spectral signatures ({sign}{delta:.2f})"
    return f"Observed shift of {sign}{delta:.2f} {unit}"


def analyze_indicator_change(
    ind: str,
    dist_t0: np.ndarray,
    dist_t1: np.ndarray,
    norm_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute comprehensive change metrics for a single indicator."""
    v0 = np.asarray(dist_t0, dtype="float32").ravel()
    v1 = np.asarray(dist_t1, dtype="float32").ravel()
    v0 = v0[np.isfinite(v0)]
    v1 = v1[np.isfinite(v1)]

    if len(v0) == 0 or len(v1) == 0:
        return {
            "indicator_id": ind,
            "label": C.INDICATOR_LABELS.get(ind, ind),
            "unit": C.UNITS.get(ind, ""),
            "wasserstein_distance": 0.0,
            "score": 0.0,
            "delta_median": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "interpretation": "No data available for temporal comparison.",
        }

    w_dist = compute_temporal_wasserstein(v0, v1)
    p_val, is_sig = statistical_significance(v0, v1)

    scale = norm_scale or DEFAULT_TEMPORAL_SCALE.get(ind, 1.0)
    score = float(np.clip(w_dist / scale, 0.0, 1.0))
    delta_med = float(np.median(v1) - np.median(v0))

    return {
        "indicator_id": ind,
        "label": C.INDICATOR_LABELS.get(ind, ind),
        "unit": C.UNITS.get(ind, ""),
        "wasserstein_distance": round(w_dist, 4),
        "score": round(score, 4),
        "delta_median": round(delta_med, 4),
        "p_value": round(p_val, 6),
        "is_significant": is_sig,
        "interpretation": interpret_indicator_change(ind, delta_med, is_sig),
    }


def temporal_criteria_scores(indicator_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Aggregate per-indicator temporal criticality into criterion-level scores.

    A criterion with no non-empty indicators evaluates to None.
    """
    crit_scores = {}
    for crit, inds in C.CRITERIA.items():
        scores = [
            indicator_metrics[i]["score"]
            for i in inds
            if i in indicator_metrics and indicator_metrics[i]["wasserstein_distance"] > 0
        ]
        crit_scores[crit] = float(np.mean(scores)) if scores else None
    return crit_scores


def temporal_global_criticality(
    criteria_scores: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute overall territorial criticality index in [0, 1].

    0.0 = total stability
    1.0 = maximum catastrophic deformation
    """
    w = weights or C.DEFAULT_WEIGHTS
    valid_crits = {c: criteria_scores[c] for c in criteria_scores if criteria_scores[c] is not None}
    if not valid_crits:
        return 0.0

    total_w = sum(w.get(c, 0.0) for c in valid_crits)
    if total_w <= 0:
        return 0.0

    weighted_sum = sum(valid_crits[c] * w.get(c, 0.0) for c in valid_crits)
    return float(np.clip(weighted_sum / total_w, 0.0, 1.0))


def classify_alert_level(criticality: float) -> AlertLevel:
    """Classify global criticality into standardized 4-tier early warning alert level."""
    if criticality >= C.CRITICALITY_CRITICAL:
        return AlertLevel.CRITICAL
    if criticality >= C.CRITICALITY_WARNING:
        return AlertLevel.WARNING
    if criticality >= C.CRITICALITY_WATCH:
        return AlertLevel.WATCH
    return AlertLevel.STABLE


# ---------------------------------------------------------------------------
# Spatial patch-wise Wasserstein heatmap (Preserves 2D localization)
# ---------------------------------------------------------------------------

def patch_wasserstein_grid(
    arr_t0: np.ndarray,
    arr_t1: np.ndarray,
    patch_size_px: int = 10,
) -> np.ndarray:
    """Compute localized 2-D patch-wise Wasserstein distance grid.

    Divides both arrays into blocks of (patch_size_px x patch_size_px) and
    calculates the Wasserstein distance between each corresponding block.

    Resolves the 'spatial blind spot' by preserving exact spatial coordinates
    of localized deformation (e.g. a localized scarp or landslide toe).

    Parameters
    ----------
    arr_t0 : 2D float32 array at baseline T0
    arr_t1 : 2D float32 array at observation T1 (must have same shape)
    patch_size_px : size of the square patch in pixels

    Returns
    -------
    2D float32 array of shape (arr_t0.shape[0] // patch_size_px, arr_t0.shape[1] // patch_size_px)
    """
    if arr_t0.shape != arr_t1.shape:
        raise ValueError(f"Array shape mismatch: {arr_t0.shape} vs {arr_t1.shape}")

    h, w = arr_t0.shape
    rows = h // patch_size_px
    cols = w // patch_size_px

    heatmap = np.zeros((rows, cols), dtype="float32")

    for r in range(rows):
        r_start = r * patch_size_px
        r_end = r_start + patch_size_px
        for c in range(cols):
            c_start = c * patch_size_px
            c_end = c_start + patch_size_px

            p0 = arr_t0[r_start:r_end, c_start:c_end].ravel()
            p1 = arr_t1[r_start:r_end, c_start:c_end].ravel()

            p0 = p0[np.isfinite(p0)]
            p1 = p1[np.isfinite(p1)]

            if len(p0) >= 3 and len(p1) >= 3:
                heatmap[r, c] = float(wasserstein_distance(p0, p1))
            else:
                heatmap[r, c] = 0.0

    return heatmap


# ---------------------------------------------------------------------------
# Qualitative temporal validity & action recommendations
# ---------------------------------------------------------------------------

def temporal_validity_card(
    criteria_scores: Dict[str, Optional[float]],
    alert_lvl: AlertLevel,
) -> Dict[str, List[str]]:
    """Build standardized qualitative report and risk mitigation recommendations."""
    stable, drift, critical = [], [], []

    for crit, score in criteria_scores.items():
        label = C.CRITERIA_LABELS.get(crit, crit)
        if score is None:
            continue
        if score >= C.CRITICALITY_WARNING:
            critical.append(f"{label} (Critical deformation index: {score:.2f})")
        elif score >= C.CRITICALITY_WATCH:
            drift.append(f"{label} (Moderate drift: {score:.2f})")
        else:
            stable.append(f"{label} (Stable: {score:.2f})")

    actions = [C.TEMPORAL_ACTIONS.get(alert_lvl.value, "Continue routine monitoring.")]

    return {
        "reproduced_or_stable": stable,
        "partial_drift": drift,
        "divergent_or_critical": critical,
        "not_evaluated": [
            "Deep subterranean shear planes (> 50 m depth)",
            "Groundwater pore-pressure variations",
            "Micro-seismic activity frequency",
        ],
        "never_reproducible": [
            "Human socio-economic informal building dynamics (not captured in satellite physics)",
        ],
        "recommended_actions": actions,
    }
