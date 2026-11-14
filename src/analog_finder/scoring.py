"""Similarity scoring engine and scientific validity assessment.

Methodology:
For each indicator, computes the first Wasserstein distance between the target
empirical distribution and the terrestrial site distribution, normalized by the
maximum observed distance across the candidate panel:
    S = 1 - W / W_max   (1 = identical distribution to target, 0 = farthest site in panel)

SCIENTIFIC LIMITATION NOTICE:
Scores are relative to the comparative panel of candidate sites, rather than an
absolute measure of fitness. No trained model or arbitrary weights are embedded
in the raw similarity scores.
"""
from typing import Dict, Optional
import numpy as np
from scipy.stats import norm, wasserstein_distance

from . import config as C


def samples_from_expected(mean: float, std: float, n: int = 200) -> np.ndarray:
    """Generate a deterministic sample from a normal distribution for a sourced expected value."""
    return mean + std * norm.ppf(np.linspace(0.005, 0.995, n))


def indicator_similarities(target: Dict[str, np.ndarray], sites: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    """Compute per-indicator similarity scores between target and candidate sites.

    target: {indicator_id: 1D array of samples}
    sites: {site_id: {indicator_id: 1D array of samples}}

    Returns {site_id: {indicator_id: similarity_score in [0, 1]}}.
    Missing indicators are completely omitted from the dictionary (not set to 0.0).
    """
    sims = {s: {} for s in sites}
    for ind in C.INDICATORS:
        t = target.get(ind)
        if t is None or len(t) == 0:
            continue
        dist = {}
        for s, samples in sites.items():
            v = samples.get(ind)
            if v is not None and len(v) > 0:
                dist[s] = float(wasserstein_distance(t, v))
        if not dist:
            continue
        w_max = max(dist.values())
        for s, w in dist.items():
            sims[s][ind] = 1.0 if w_max == 0 else float(1.0 - w / w_max)
    return sims


def criterion_scores(sim: Dict[str, float]) -> Dict[str, Optional[float]]:
    """Compute mean similarity for available indicators within each criterion.

    Returns None for any criterion where no indicators are available.
    """
    out = {}
    for crit, inds in C.CRITERIA.items():
        vals = [sim[i] for i in inds if i in sim]
        out[crit] = float(np.mean(vals)) if vals else None
    return out


def global_score(criteria: Dict[str, Optional[float]], weights: Optional[Dict[str, float]] = None) -> float:
    """Compute weighted average of available criteria scores with dynamically renormalized weights.

    Criteria with None (missing data) are cleanly ignored without skewing the score.
    Returns NaN if all weights or all available criteria weights sum to zero.
    """
    w = weights or C.DEFAULT_WEIGHTS
    pairs = [(criteria[c], w.get(c, 0.0)) for c in criteria if criteria[c] is not None]
    total = sum(p for _, p in pairs)
    return float(sum(s * p for s, p in pairs) / total) if total > 0 else float("nan")


def confidence(sim: Dict[str, float]) -> float:
    """Proportion of the 8+ indicators successfully computed for this site (0.0 to 1.0)."""
    return len(sim) / len(C.INDICATORS)


def explain(criteria: Dict[str, Optional[float]]) -> str:
    """Generate human-readable scientific summary of criterion-level fidelity."""
    strong = [C.CRITERIA_LABELS.get(c, c) for c, s in criteria.items() if s is not None and s >= C.STRONG]
    weak = [C.CRITERIA_LABELS.get(c, c) for c, s in criteria.items() if s is not None and s < C.WEAK]
    missing = [C.CRITERIA_LABELS.get(c, c) for c, s in criteria.items() if s is None]

    parts = []
    if strong:
        parts.append(f"Strong analog match: {', '.join(strong)}.")
    else:
        parts.append("No criteria show strong analog match.")

    if weak:
        parts.append(f"Low resemblance: {', '.join(weak)}.")
    if missing:
        parts.append(f"Not evaluated (missing data): {', '.join(missing)}.")

    return " ".join(parts)


def validity_card(criteria: Dict[str, Optional[float]]) -> Dict[str, list]:
    """Build the scientific validity breakdown for the analog site.

    Identifies where the analogy holds, where it partially applies, where it collapses,
    what unmeasured parameters exist, and what cannot be simulated on Earth.
    """
    card = {
        "reproduit": [],
        "partiel": [],
        "non_reproduit": [],
        "non_evalue": list(C.NOT_EVALUATED),
        "jamais_reproductible": list(C.NEVER_REPRODUCED),
        "usages_adaptes": [],
        "usages_partiels": [],
    }
    for crit, score in criteria.items():
        label = C.CRITERIA_LABELS.get(crit, crit)
        usage = C.USAGES.get(crit, "")
        if score is None:
            card["non_evalue"].append(f"{label} (insufficient data)")
        elif score >= C.STRONG:
            card["reproduit"].append(label)
            card["usages_adaptes"].append(f"{label}: {usage}")
        elif score >= C.WEAK:
            card["partiel"].append(label)
            card["usages_partiels"].append(f"{label}: {usage}")
        else:
            card["non_reproduit"].append(label)
    return card
